import argparse
import logging
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


# Make src/ importable without installation
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.hetero_graph import (  # noqa: E402
    load_multi_rel_graph_payload,
    build_hetero_from_multi_rel,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("train_metapath2vec")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MetaPath2Vec on heterogeneous bipartite graph")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_init_wandb(cfg):
    project = cfg.logging.get("wandb_project", None)
    if not project:
        return
    try:
        import wandb  # type: ignore

        from omegaconf import OmegaConf as _OC

        wandb.init(project=project, config=_OC.to_container(cfg, resolve=True))
    except Exception as e:
        logger.warning("WandB init failed: %s", e)


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg.seed))
    maybe_init_wandb(cfg)

    # Build hetero graph (we only need edge_index_dict & num_nodes)
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

    # MetaPath2Vec with the simple user–sub–user meta-path
    from torch_geometric.nn.models import MetaPath2Vec

    embedding_dim = int(cfg.model.embedding_dim)
    walk_length = int(cfg.model.walk_length)
    context_size = int(cfg.model.context_size)
    walks_per_node = int(cfg.model.walks_per_node)
    num_negative_samples = int(cfg.model.num_negative_samples)

    # Combine both relations in a longer meta-path to incorporate both edge types
    metapath = [
        ("user", "comments", "sub"),
        ("sub", "rev_comments", "user"),
        ("user", "posts", "sub"),
        ("sub", "rev_posts", "user"),
    ]
    model = MetaPath2Vec(
        edge_index_dict=data.edge_index_dict,
        embedding_dim=embedding_dim,
        metapath=metapath,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=num_negative_samples,
        sparse=True,
    ).to(device)

    # Use the built-in random-walk mini-batch loader
    loader = model.loader(batch_size=int(cfg.loader.batch_size), shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=float(cfg.train.lr))

    epochs = int(cfg.train.epochs)
    log_every = int(cfg.logging.get("log_every", 100))

    global_step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())

            if (step + 1) % log_every == 0:
                logger.info("epoch %d step %d/%d | loss %.4f", epoch, step + 1, len(loader), total_loss / log_every)
                try:
                    import wandb  # type: ignore

                    wandb.log({"train/loss": total_loss / log_every, "epoch": epoch, "step": global_step})
                except Exception:
                    pass
                total_loss = 0.0
            global_step += 1

    # Save trained model checkpoint and metadata for later embedding extraction
    out_dir = Path(cfg.paths.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "state_dict": model.state_dict(),
        "meta": {
            "embedding_dim": embedding_dim,
            "num_users": int(num_users),
            "num_subs": int(num_subs),
            "metapath": metapath,
        },
    }
    torch.save(state, str(out_dir / "metapath2vec_latest.pt"))
    logger.info("Saved checkpoint to %s", out_dir / "metapath2vec_latest.pt")


if __name__ == "__main__":
    main()


