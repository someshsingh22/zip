import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from omegaconf import OmegaConf


# Make src/ importable without installation
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.bine import BipartiteEmbeddingModel  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("train_bine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BiNE embeddings with DDP")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EdgeDataset(Dataset):
    def __init__(self, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> None:
        self.user_ids = edge_index[0].long()
        self.sub_ids = edge_index[1].long()
        self.weights = edge_weight.float()

    def __len__(self) -> int:
        return self.user_ids.numel()

    def __getitem__(self, idx: int) -> Tuple[int, int, float]:
        return (
            int(self.user_ids[idx].item()),
            int(self.sub_ids[idx].item()),
            float(self.weights[idx].item()),
        )


def cosine_with_warmup(total_steps: int, warmup_ratio: float = 0.1):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def _lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return _lambda


def init_distributed(backend: str) -> Tuple[int, int, int]:
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend=backend)
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def maybe_init_wandb(cfg: Any, rank: int) -> None:
    if rank != 0:
        return
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

    rank, world_size, local_rank = init_distributed(cfg.distributed.backend)
    set_seed(int(cfg.seed))
    device = torch.device(f"cuda:{local_rank}")

    # Load graph tensors
    graph_path = Path(cfg.paths.out_dir) / "graph.pt"
    payload: Dict[str, Any] = torch.load(str(graph_path), map_location="cpu")
    edge_index = payload["edge_index"]
    edge_weight = payload["edge_weight"]
    sub_degree = payload["sub_degree"]
    num_users = int(payload["num_users"])
    num_subs = int(payload["num_subs"])

    dataset = EdgeDataset(edge_index=edge_index, edge_weight=edge_weight)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    def collate(batch: Iterable[Tuple[int, int, float]]):
        users, subs, w = zip(*batch)
        return (
            torch.tensor(users, dtype=torch.long),
            torch.tensor(subs, dtype=torch.long),
            torch.tensor(w, dtype=torch.float32),
        )

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
        drop_last=True,
    )

    # Initialize model; optionally from text embeddings
    init_sub = None
    if bool(cfg.model.init_sub_from_text):
        sub_text_path = Path(cfg.paths.out_dir) / "sub_text_emb.pt"
        if sub_text_path.exists():
            init_sub = torch.load(str(sub_text_path), map_location="cpu")
            assert init_sub.shape[0] == num_subs, "sub_text_emb rows must match #subreddits"
            assert (
                init_sub.shape[1] == int(cfg.model.embedding_dim)
            ), "sub_text_emb dim must match model.embedding_dim"
        else:
            if rank == 0:
                logger.warning("Subreddit text embeddings not found at %s; using random init.", sub_text_path)

    model = BipartiteEmbeddingModel(
        num_users=num_users,
        num_subs=num_subs,
        embedding_dim=int(cfg.model.embedding_dim),
        init_sub_weight=init_sub,
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Broadcast subreddit init from rank 0 to others for consistency
    dist.broadcast(model.module.sub_emb.weight.data, src=0)

    # Optimizer with param groups (faster LR for users)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.module.user_emb.parameters(), "lr": float(cfg.train.lr_user)},
            {"params": model.module.sub_emb.parameters(), "lr": float(cfg.train.lr_sub)},
        ],
        weight_decay=float(cfg.train.weight_decay),
    )

    # Scheduler preserving LR ratio
    total_steps = int(len(loader) * int(cfg.train.epochs))
    lr_lambda = cosine_with_warmup(total_steps, warmup_ratio=float(cfg.train.get("warmup_ratio", 0.1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.train.amp))

    # Negative sampling distribution on device
    neg_dist = (sub_degree.float() ** 0.75)
    neg_dist = (neg_dist / neg_dist.sum().clamp_min(1e-8)).to(device)
    K = int(cfg.model.num_negative)

    maybe_init_wandb(cfg, rank)

    # Training loop
    for epoch in range(int(cfg.train.epochs)):
        # Optionally freeze subreddits for initial epochs
        freeze_sub = epoch < int(cfg.model.freeze_sub_epochs)
        for p in model.module.sub_emb.parameters():
            p.requires_grad = not freeze_sub

        sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        for step, (u, s, w) in enumerate(loader):
            u = u.to(device, non_blocking=True)
            s = s.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            neg = torch.multinomial(neg_dist, num_samples=u.size(0) * K, replacement=True).view(u.size(0), K)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg.train.amp)):
                pos_logits, neg_logits = model(u, s, neg)
                loss = model.module.loss_fn(pos_logits, neg_logits, w)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.detach().item()
            if (step + 1) % int(cfg.logging.log_every) == 0 and rank == 0:
                try:
                    import wandb  # type: ignore

                    wandb.log({"loss": running_loss / int(cfg.logging.log_every), "epoch": epoch})
                except Exception:
                    pass
                logger.info(
                    "epoch %d step %d/%d | loss %.4f | lr_user %.2e lr_sub %.2e",
                    epoch,
                    step + 1,
                    len(loader),
                    running_loss / int(cfg.logging.log_every),
                    optimizer.param_groups[0]["lr"],
                    optimizer.param_groups[1]["lr"],
                )
                running_loss = 0.0

        # Save checkpoint on rank 0
        if rank == 0:
            ckpt_dir = Path(cfg.paths.out_dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            state = {
                "epoch": epoch,
                "user_emb": model.module.user_emb.weight.detach().cpu(),
                "sub_emb": model.module.sub_emb.weight.detach().cpu(),
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            torch.save(state, str(ckpt_dir / f"epoch_{epoch}.pt"))
            torch.save(state, str(ckpt_dir / "latest.pt"))

    # Save final embeddings for convenience
    if rank == 0:
        out_dir = Path(cfg.paths.out_dir)
        torch.save(model.module.user_emb.weight.detach().cpu(), str(out_dir / "user_emb.pt"))
        torch.save(model.module.sub_emb.weight.detach().cpu(), str(out_dir / "sub_emb.pt"))
        logger.info("Training finished. Saved embeddings to %s", out_dir)


if __name__ == "__main__":
    main()


