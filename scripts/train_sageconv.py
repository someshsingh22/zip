import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from omegaconf import OmegaConf


# Make src/ importable without installation
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.hetero_graph import (  # noqa: E402
    build_hetero_from_bipartite,
    load_bipartite_graph_payload,
    load_multi_rel_graph_payload,
    build_hetero_from_multi_rel,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("train_sageconv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GraphSAGE (link prediction) on hetero bipartite graph")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_init_wandb(cfg: Any) -> None:
    project = cfg.logging.get("wandb_project", None)
    if not project:
        return
    try:
        import wandb  # type: ignore

        wandb.init(project=project, config=OmegaConf.to_container(cfg, resolve=True))
    except Exception as e:
        logger.warning("WandB init failed: %s", e)


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg.seed))
    maybe_init_wandb(cfg)

    # Load tensors and build hetero graph with two relations when available
    graph_path = str(Path(cfg.paths.out_dir) / "graph.pt")
    payload = load_multi_rel_graph_payload(graph_path)
    num_users = int(payload["num_users"])  # type: ignore
    num_subs = int(payload["num_subs"])  # type: ignore
    data = build_hetero_from_multi_rel(
        edge_index_comments=payload["edge_index_comments"],
        edge_index_posts=payload["edge_index_posts"],
        edge_weight_comments=payload["edge_weight_comments"],
        edge_weight_posts=payload["edge_weight_posts"],
        num_users=num_users,
        num_subs=num_subs,
        add_reverse=True,
    )

    # Optional transforms: random link split for supervision
    import torch_geometric.transforms as T
    from torch_geometric.loader import LinkNeighborLoader
    from torch_geometric.nn import to_hetero
    from torch_geometric.nn.models import GraphSAGE

    split = T.RandomLinkSplit(
        num_val=float(cfg.split.num_val_ratio),
        num_test=float(cfg.split.num_test_ratio),
        is_undirected=True,
        add_negative_train_samples=True,
        edge_types=[("user", "comments", "sub"), ("user", "posts", "sub")],
        rev_edge_types=[("sub", "rev_comments", "user"), ("sub", "rev_posts", "user")],
    )
    train_data, val_data, test_data = split(data)

    # Trainable input features when none are available: per-node embeddings
    in_channels = int(cfg.model.in_channels)
    user_emb = torch.nn.Embedding(num_users, in_channels).to(device)
    sub_emb = torch.nn.Embedding(num_subs, in_channels).to(device)
    torch.nn.init.normal_(user_emb.weight, mean=0.0, std=0.02)
    torch.nn.init.normal_(sub_emb.weight, mean=0.0, std=0.02)

    # Base GraphSAGE, then lift to hetero with shared weights
    hidden = int(cfg.model.hidden_channels)
    num_layers = int(cfg.model.num_layers)
    dropout = float(cfg.model.get("dropout", 0.0))
    base = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=hidden,
        num_layers=num_layers,
        out_channels=hidden,
        dropout=dropout,
    )
    model = to_hetero(base, metadata=train_data.metadata(), aggr="sum").to(device)

    # Neighbor loaders for link prediction
    num_neighbors = list(map(int, cfg.loader.num_neighbors))
    batch_size = int(cfg.loader.batch_size)
    target_edge = getattr(cfg.loader, "target_edge", "comments")
    assert target_edge in ("comments", "posts"), "target_edge must be 'comments' or 'posts'"
    edge_type = ("user", target_edge, "sub")
    train_edge_label_index = train_data[edge_type].edge_label_index
    train_edge_label = train_data[edge_type].edge_label
    val_edge_label_index = val_data[edge_type].edge_label_index
    val_edge_label = val_data[edge_type].edge_label

    train_loader = LinkNeighborLoader(
        train_data,
        num_neighbors={
            ("user", "comments", "sub"): num_neighbors,
            ("sub", "rev_comments", "user"): num_neighbors,
            ("user", "posts", "sub"): num_neighbors,
            ("sub", "rev_posts", "user"): num_neighbors,
        },
        batch_size=batch_size,
        edge_label_index=(edge_type, train_edge_label_index),
        edge_label=train_edge_label,
        shuffle=True,
    )
    val_loader = LinkNeighborLoader(
        val_data,
        num_neighbors={
            ("user", "comments", "sub"): num_neighbors,
            ("sub", "rev_comments", "user"): num_neighbors,
            ("user", "posts", "sub"): num_neighbors,
            ("sub", "rev_posts", "user"): num_neighbors,
        },
        batch_size=batch_size,
        edge_label_index=(edge_type, val_edge_label_index),
        edge_label=val_edge_label,
        shuffle=False,
    )

    # Optimizer includes both model and input embeddings
    lr = float(cfg.train.lr)
    wd = float(cfg.train.get("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters()},
            {"params": user_emb.parameters()},
            {"params": sub_emb.parameters()},
        ],
        lr=lr,
        weight_decay=wd,
    )
    bce = torch.nn.BCEWithLogitsLoss()

    def forward_and_loss(batch: Any) -> torch.Tensor:
        batch = batch.to(device)
        # Build mini-batch features via embedding tables
        x_dict = {
            "user": user_emb(batch["user"].n_id),
            "sub": sub_emb(batch["sub"].n_id),
        }
        z_dict: Dict[str, torch.Tensor] = model(x_dict, batch.edge_index_dict)

        # Handle both hetero-typed and top-level label storage
        edge_label_index = getattr(batch, "edge_label_index", None)
        edge_label = getattr(batch, "edge_label", None)
        if edge_label_index is None or edge_label is None:
            edge_label_index = batch[("user", target_edge, "sub")].edge_label_index
            edge_label = batch[("user", target_edge, "sub")].edge_label
        edge_label = edge_label.float()

        z_user = z_dict["user"][edge_label_index[0]]
        z_sub = z_dict["sub"][edge_label_index[1]]
        pred = (z_user * z_sub).sum(dim=-1)
        return bce(pred, edge_label)

    # Training loop
    epochs = int(cfg.train.epochs)
    log_every = int(cfg.logging.get("log_every", 50))

    global_step = 0
    for epoch in range(epochs):
        model.train()
        user_emb.train()
        sub_emb.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            loss = forward_and_loss(batch)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu().item())
            if (step + 1) % log_every == 0:
                logger.info("epoch %d step %d/%d | loss %.4f", epoch, step + 1, len(train_loader), running / log_every)
                try:
                    import wandb  # type: ignore

                    wandb.log({"train/loss": running / log_every, "epoch": epoch, "step": global_step})
                except Exception:
                    pass
                running = 0.0
            global_step += 1

        # Light validation
        model.eval()
        user_emb.eval()
        sub_emb.eval()
        with torch.no_grad():
            val_loss = 0.0
            n_batches = 0
            for batch in val_loader:
                val_loss += float(forward_and_loss(batch).cpu().item())
                n_batches += 1
            val_loss = val_loss / max(1, n_batches)
        logger.info("epoch %d | val_loss %.4f", epoch, val_loss)
        try:
            import wandb  # type: ignore

            wandb.log({"val/loss": val_loss, "epoch": epoch})
        except Exception:
            pass

    # Save final checkpoint
    out_dir = Path(cfg.paths.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "user_emb": user_emb.weight.detach().cpu(),
        "sub_emb": sub_emb.weight.detach().cpu(),
        "model": model.state_dict(),
        "meta": {
            "num_users": int(num_users),
            "num_subs": int(num_subs),
            "in_channels": in_channels,
            "hidden_channels": hidden,
            "num_layers": num_layers,
        },
    }
    torch.save(state, str(out_dir / "sageconv_latest.pt"))
    logger.info("Saved checkpoint to %s", out_dir / "sageconv_latest.pt")


if __name__ == "__main__":
    main()


