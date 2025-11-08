import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
from src.gnn import UserSubredditSAGE, DotPredictor
from tqdm.auto import tqdm
import wandb
from omegaconf import OmegaConf

parser = argparse.ArgumentParser(description="Train GNN with YAML config.")
parser.add_argument(
    "--config",
    type=str,
    default="configs/default.yaml",
    help="Path to YAML config file",
)
args = parser.parse_args()

cfg = OmegaConf.load(args.config)
cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # for wandb

if cfg.get("wandb", {}).get("enabled", True):
    wandb.init(
        project=cfg.get("wandb", {}).get("project", "zip-gnn"),
        config=cfg_dict,
    )

print("Loading dataset...")
data = torch.load(cfg.data_path, map_location="cpu", weights_only=False).pin_memory()

device = cfg.device
model = UserSubredditSAGE(
    input_dim=cfg.model.input_dim,
    hidden_dim=cfg.model.hidden_dim,
).to(device)
predictor = DotPredictor(proj_dim=cfg.model.hidden_dim).to(device)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(predictor.parameters()),
    lr=cfg.training.lr,
)
scaler = torch.amp.GradScaler('cuda', enabled=True)

# LR scheduler (epoch-wise StepLR)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=1,
    gamma=float(cfg.training.get("lr_gamma", 0.5)),
)

# Checkpointing setup
ckpt_cfg = cfg.get("checkpoint", {})
ckpt_dir = Path(ckpt_cfg.get("dir", "checkpoints"))
ckpt_dir.mkdir(parents=True, exist_ok=True)
save_every_steps: int = int(ckpt_cfg.get("save_every_steps", 100))

edge_type = ("user", "interacts", "subreddit")
# For heterogeneous graphs, provide per-edge-type fanout explicitly to avoid sampler mapping asserts
hetero_fanout = {et: list(cfg.sampler.fanout) for et in data.edge_types}

def sample_one_positive_per_user(edge_index: torch.Tensor, num_users: int) -> torch.Tensor:
    """Uniform over users: pick one positive subreddit per user (if degree>0)."""
    src, dst = edge_index
    deg = torch.bincount(src, minlength=num_users)
    valid_users = (deg > 0).nonzero(as_tuple=True)[0]
    if valid_users.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long)
    order = src.argsort()
    dst_sorted = dst[order]
    # start index of each user's block in the sorted edge list
    starts = torch.empty_like(deg)
    starts[0] = 0
    if deg.numel() > 1:
        starts[1:] = deg.cumsum(0)[:-1]
    deg_valid = deg[valid_users]
    # random offset within each user's block
    r = (torch.rand_like(deg_valid, dtype=torch.float32) * deg_valid.to(torch.float32)).to(torch.long)
    idx = starts[valid_users] + r
    pos_src = valid_users
    pos_dst = dst_sorted[idx]
    return torch.stack([pos_src, pos_dst], dim=0)

def train_one_epoch(
    loader: LinkNeighborLoader,
    epoch: int,
    total_pos_edges: int,
    global_step: int,
    save_every_steps: int,
    ckpt_dir: Path,
) -> Tuple[float, int]:
    tau: float = float(cfg.training.get("temperature", 0.07))
    total_loss = 0
    total_pos = 0
    pbar = tqdm(total=total_pos_edges, desc=f"Epoch {epoch:02d}", unit="edge")
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        pos_src, pos_dst = batch[("user", "interacts", "subreddit")].edge_label_index

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Provide edge attributes for rev_interacts to compute MLP-based weights
            rev_store = batch[("subreddit", "rev_interacts", "user")]
            edge_attr_dict = {
                ("subreddit", "rev_interacts", "user"): rev_store.edge_attr
            } if hasattr(rev_store, "edge_attr") else None
            x_dict = model(batch.x_dict, batch.edge_index_dict, edge_attr_dict=edge_attr_dict)
            # InfoNCE with in-batch negatives (user→subreddit) using learnable projection heads + temperature
            u = x_dict["user"][pos_src]
            v = x_dict["subreddit"][pos_dst]
            logits = predictor.compute_logits(u, v, base_tau=tau)
            labels = torch.arange(u.size(0), device=u.device)
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Increment global step and checkpoint periodically
        global_step += 1
        if save_every_steps > 0 and (global_step % save_every_steps == 0):
            latest_path = ckpt_dir / "latest.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "config": cfg_dict,
                },
                latest_path,
            )

        total_loss += loss.item() * pos_src.size(0)
        total_pos += pos_src.size(0)
        remaining = max(total_pos_edges - total_pos, 0)
        pbar.update(pos_src.size(0))
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "remaining": f"{remaining:,}"})
        if cfg.get("wandb", {}).get("enabled", True):
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
            )
    avg_loss = total_loss / max(total_pos, 1)
    if cfg.get("wandb", {}).get("enabled", True):
        wandb.log(
            {
                "train/epoch_avg_loss": avg_loss,
                "train/epoch": epoch,
            }
        )
    return avg_loss, global_step

best_epoch_avg_loss = float("inf")
global_step = 0

for epoch in range(1, cfg.training.epochs + 1):
    model.train()
    # Rebuild a loader that samples one positive edge per user uniformly
    full_pos_edge_index = data[edge_type].edge_index
    num_users = int(data["user"].num_nodes)
    epoch_edge_label_index = sample_one_positive_per_user(full_pos_edge_index, num_users)
    TOTAL_POS_EDGES = int(epoch_edge_label_index.size(1))
    loader = LinkNeighborLoader(
        data,
        num_neighbors=hetero_fanout,
        edge_label_index=(edge_type, epoch_edge_label_index),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    avg_loss, global_step = train_one_epoch(loader, epoch, TOTAL_POS_EDGES, global_step, save_every_steps, ckpt_dir)

    # Save best checkpoint based on epoch average loss
    if ckpt_cfg.get("keep_best", True) and avg_loss < best_epoch_avg_loss:
        best_epoch_avg_loss = avg_loss
        best_path = ckpt_dir / "best.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_epoch_avg_loss": best_epoch_avg_loss,
                "config": cfg_dict,
            },
            best_path,
        )

    # Step LR scheduler at epoch boundary
    scheduler.step()

torch.save(model.state_dict(), "model.pt")
print("✅ Model saved.")
if cfg.get("wandb", {}).get("enabled", True):
    wandb.finish()
