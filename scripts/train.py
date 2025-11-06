import torch
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import negative_sampling
from src.gnn import UserSubredditSAGE, DotPredictor
from tqdm.auto import tqdm
import wandb
from omegaconf import OmegaConf
import argparse
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/train.yaml")
args = parser.parse_args()

cfg = OmegaConf.load(args.config)

# Reproducibility
torch.manual_seed(int(cfg.seed))
random.seed(int(cfg.seed))
np.random.seed(int(cfg.seed))

DATA_PATH = str(cfg.data.dataset_path)
DEVICE = str(cfg.device)
HIDDEN_DIM = int(cfg.model.hidden_dim)
FANOUT = list(cfg.loader.fanout)
BATCH_SIZE = int(cfg.loader.batch_size)
EPOCHS = int(cfg.train.epochs)
LR = float(cfg.train.lr)
EDGE_TYPE = tuple(cfg.loader.edge_type)
WANDB_PROJECT = str(cfg.logging.wandb_project)
AMP_ENABLED = bool(cfg.amp.enabled) and DEVICE.startswith("cuda")
AMP_DTYPE = torch.bfloat16 if str(cfg.amp.dtype).lower() == "bfloat16" else torch.float16

print("Loading dataset...")
data = torch.load(DATA_PATH, map_location="cpu", weights_only=False).pin_memory()

model = UserSubredditSAGE(input_dim=int(cfg.model.input_dim), hidden_dim=HIDDEN_DIM).to(DEVICE)
predictor = DotPredictor().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler('cuda', enabled=AMP_ENABLED)

# Initialize Weights & Biases for experiment tracking
wandb.init(
    project=WANDB_PROJECT,
    config={
        "data_path": DATA_PATH,
        "device": DEVICE,
        "hidden_dim": HIDDEN_DIM,
        "fanout": FANOUT,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "model": "UserSubredditSAGE",
        "predictor": "DotPredictor",
        "amp_enabled": AMP_ENABLED,
        "amp_dtype": str(cfg.amp.dtype),
    },
)
# Track gradients/parameters lightly
wandb.watch(model, log="gradients", log_freq=100, log_graph=False)

global_step = 0

loader = LinkNeighborLoader(
    data,
    num_neighbors=FANOUT,
    edge_label_index=EDGE_TYPE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
)

def train_one_epoch(loader, epoch):
    global global_step
    total_loss = 0
    total_pos = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} | EdgeType: interacts", unit="batch")
    for batch in pbar:
        batch = batch.to(DEVICE, non_blocking=True)
        pos_src, pos_dst = batch[EDGE_TYPE].edge_label_index
        neg_src, neg_dst = negative_sampling(
            edge_index=batch[EDGE_TYPE].edge_index,
            num_nodes=(batch["user"].num_nodes, batch["subreddit"].num_nodes),
            num_neg_samples=pos_src.size(0),
            method='sparse'
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=AMP_DTYPE, enabled=AMP_ENABLED):
            # Provide edge attributes for rev_interacts to compute MLP-based weights
            rev_store = batch[("subreddit", "rev_interacts", "user")]
            edge_attr_dict = {
                ("subreddit", "rev_interacts", "user"): rev_store.edge_attr
            } if hasattr(rev_store, "edge_attr") else None
            x_dict = model(batch.x_dict, batch.edge_index_dict, edge_attr_dict=edge_attr_dict)
            pos_score = predictor(x_dict["user"][pos_src], x_dict["subreddit"][pos_dst])
            neg_score = predictor(x_dict["user"][neg_src], x_dict["subreddit"][neg_dst])
            loss = (
                -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
                - torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * pos_src.size(0)
        total_pos += pos_src.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        # Log per-batch training loss and progress
        wandb.log(
            {
                "train/batch_loss": loss.item(),
                "train/pos_edges": int(pos_src.size(0)),
                "epoch": epoch,
            },
            step=global_step,
        )
        global_step += 1
    avg_loss = total_loss / max(total_pos, 1)
    # Log per-epoch average loss
    wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch})
    return avg_loss

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_one_epoch(loader, epoch)

torch.save(model.state_dict(), "model.pt")
print("✅ Model saved.")
