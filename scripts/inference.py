import argparse
import json
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Local imports
from src.gnn import UserSubredditSAGE


logger = logging.getLogger(__name__)


@dataclass
class UserInteractions:
    """Container for a single user's subreddit interaction counts.

    Attributes:
        subreddit_ids: Global subreddit ids (as used in dataset.pt).
        post_counts: Number of posts to each subreddit (same length as subreddit_ids).
        comment_counts: Number of comments to each subreddit (same length as subreddit_ids).
    """

    subreddit_ids: Sequence[int]
    post_counts: Sequence[int]
    comment_counts: Sequence[int]

    def __post_init__(self) -> None:
        assert len(self.subreddit_ids) == len(self.post_counts) == len(
            self.comment_counts
        ), "All fields must have equal length."

    @property
    def num_edges(self) -> int:
        return len(self.subreddit_ids)


def load_subreddit_features(dataset_path: str, device: torch.device) -> torch.Tensor:
    """Load subreddit embedding matrix from dataset.pt.

    Args:
        dataset_path: Path to dataset file produced by scripts/prepare_data.py.
        device: Target device for the returned tensor.

    Returns:
        Tensor of shape [num_subreddits, emb_dim] in float32 on the given device.
    """
    logger.info("Loading dataset from %s", dataset_path)
    data = torch.load(dataset_path, map_location=device)
    sub_x = data["subreddit"].x
    # Ensure float32 for Linear layers
    sub_x = sub_x.to(device=device).float()
    logger.info("Loaded subreddit features: shape=%s dtype=%s", tuple(sub_x.shape), sub_x.dtype)
    return sub_x


def _build_batch_graph(
    sub_x_full: torch.Tensor,
    batch_interactions: Sequence[UserInteractions],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor], List[bool]]:
    """Construct a minimal hetero-graph dicts for a batch of users.

    We keep only the subreddits referenced by this batch to reduce compute,
    remapping their global ids to a local contiguous range.

    Edge attributes follow the training shape (2D):
      - feature[0] = log(1 + total_count) / max(sqrt(user_total), 1e-12)
      - feature[1] = comment_frac = comment_count / max(total_count, 1)

    Note: We omit the global subreddit normalization factor (sqrt(sub_total)).
    This preserves feature semantics and shape while avoiding additional
    dataset scans during inference.

    Args:
        sub_x_full: Subreddit embedding matrix from dataset.pt (float32).
        batch_interactions: List of user interaction objects for this batch.
        device: Device to place tensors on.

    Returns:
        x_dict: Dict with keys "subreddit" and optionally "user" (initialized in model).
        edge_index_dict: Dict with a single key ("subreddit","rev_interacts","user").
        edge_attr_dict: Dict mapping same key to [E,2] float32 edge attributes.
        has_edges_per_user: Boolean list indicating if each user has any edges.
    """
    # Collect unique subreddit ids used in this batch and build local index map
    unique_sub_ids: List[int] = []
    seen: set[int] = set()
    for u in batch_interactions:
        for sid in u.subreddit_ids:
            if sid not in seen:
                seen.add(int(sid))
                unique_sub_ids.append(int(sid))

    if len(unique_sub_ids) == 0:
        # Degenerate case: no edges in entire batch; return minimal placeholders
        x_dict = {"subreddit": sub_x_full.new_empty((0, sub_x_full.size(-1)))}
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {("subreddit", "rev_interacts", "user"): torch.empty((2, 0), dtype=torch.long, device=device)}
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor] = {("subreddit", "rev_interacts", "user"): torch.empty((0, 2), dtype=torch.float32, device=device)}
        has_edges = [False for _ in batch_interactions]
        return x_dict, edge_index_dict, edge_attr_dict, has_edges

    global_to_local: Dict[int, int] = {sid: i for i, sid in enumerate(unique_sub_ids)}
    sub_x = sub_x_full[unique_sub_ids]  # [S_local, D]

    # Build edges
    src_list: List[int] = []
    dst_list: List[int] = []
    edge_feat_list: List[Tuple[float, float]] = []
    has_edges: List[bool] = []

    for user_idx, inter in enumerate(batch_interactions):
        totals = [int(p) + int(c) for p, c in zip(inter.post_counts, inter.comment_counts)]
        user_total = float(sum(totals))
        denom = float(np.sqrt(user_total)) if user_total > 0.0 else 1e-12
        user_has_edges = False

        for sid, p, c, t in zip(inter.subreddit_ids, inter.post_counts, inter.comment_counts, totals):
            t_int = int(t)
            if t_int <= 0:
                continue
            user_has_edges = True
            src_list.append(global_to_local[int(sid)])
            dst_list.append(user_idx)
            log_norm = float(np.log1p(t_int)) / denom
            cfrac = float(c) / float(t_int) if t_int > 0 else 0.0
            edge_feat_list.append((log_norm, cfrac))

        has_edges.append(user_has_edges)

    if len(src_list) == 0:
        # All users had zero totals; return empty edges and track has_edges flags
        x_dict = {"subreddit": sub_x}
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 2), dtype=torch.float32, device=device)
        edge_index_dict = {("subreddit", "rev_interacts", "user"): edge_index}
        edge_attr_dict = {("subreddit", "rev_interacts", "user"): edge_attr}
        return x_dict, edge_index_dict, edge_attr_dict, has_edges

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)  # [2, E]
    edge_attr = torch.tensor(edge_feat_list, dtype=torch.float32, device=device)  # [E, 2]

    x_dict = {"subreddit": sub_x}
    edge_index_dict = {("subreddit", "rev_interacts", "user"): edge_index}
    edge_attr_dict = {("subreddit", "rev_interacts", "user"): edge_attr}
    return x_dict, edge_index_dict, edge_attr_dict, has_edges


@torch.no_grad()
def batch_user_embeddings(
    model: UserSubredditSAGE,
    subreddit_features: torch.Tensor,
    interactions: Sequence[UserInteractions],
    device: Optional[torch.device] = None,
    max_users_per_batch: int = 1024,
) -> torch.Tensor:
    """Compute user embeddings for multiple users using batched mini-graphs.

    This function constructs compact per-batch graphs from the provided
    subreddit interaction counts, runs the GNN, and returns the normalized
    user embeddings as computed by the model.

    Args:
        model: Trained UserSubredditSAGE model (evaluation mode recommended).
        subreddit_features: Full subreddit embedding matrix from dataset.pt (float32).
        interactions: Iterable of user interaction objects, one per user.
        device: Torch device for inference. Defaults to model's device if None.
        max_users_per_batch: Maximum number of users per forward pass.

    Returns:
        Tensor of shape [num_users, hidden_dim] containing normalized user embeddings (float32, on CPU).
        Users with zero interactions receive a zero vector.
    """
    if device is None:
        device = next(model.parameters()).device
    model_was_training = model.training
    model.eval()

    sub_x_full = subreddit_features.to(device=device).float()
    hidden_dim = int(getattr(model, "hidden_dim", sub_x_full.size(-1)))
    results: List[torch.Tensor] = []

    # Process in user batches
    start = 0
    total_users = len(interactions)
    while start < total_users:
        end = min(start + max_users_per_batch, total_users)
        batch = interactions[start:end]

        x_dict, eidx_dict, eattr_dict, has_edges = _build_batch_graph(sub_x_full, batch, device)
        out_x = model(x_dict, eidx_dict, eattr_dict)  # returns updated x_dict with "user"
        user_emb = out_x["user"]  # [B, hidden_dim]
        # Ensure float32 on CPU for return
        user_emb = user_emb.detach().to("cpu", dtype=torch.float32)

        # Zero-out embeddings for users with no edges to avoid random init leakage
        for i, flag in enumerate(has_edges):
            if not flag:
                user_emb[i].zero_()

        results.append(user_emb)
        start = end

    if model_was_training:
        model.train()

    return torch.cat(results, dim=0) if len(results) > 0 else torch.empty((0, hidden_dim), dtype=torch.float32)


def _load_model_from_checkpoint(
    checkpoint_path: str,
    input_dim: int,
    hidden_dim: int,
    residual: bool,
    device: torch.device,
) -> UserSubredditSAGE:
    """Utility to instantiate and load a UserSubredditSAGE from a checkpoint."""
    model = UserSubredditSAGE(input_dim=input_dim, hidden_dim=hidden_dim, residual=residual).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    # Allow either raw state_dict or wrapper
    state_dict = state.get("model", state)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _read_json_interactions(path: str) -> List[UserInteractions]:
    """Read a JSON file containing a list of users' interactions.

    Expected schema:
      [
        {
          "subreddit_ids": [12, 57, ...],
          "post_counts": [3, 1, ...],
          "comment_counts": [7, 0, ...]
        },
        ...
      ]
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    interactions: List[UserInteractions] = []
    for item in raw:
        interactions.append(
            UserInteractions(
                subreddit_ids=item["subreddit_ids"],
                post_counts=item["post_counts"],
                comment_counts=item["comment_counts"],
            )
        )
    return interactions


def main() -> None:
    parser = argparse.ArgumentParser(description="Batched user embedding inference from subreddit interactions.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset.pt produced by prepare_data.py")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--hidden-dim", type=int, required=True, help="Model hidden size used during training")
    parser.add_argument("--residual", action="store_true", help="Enable residual connections (match training)")
    parser.add_argument("--input-dim", type=int, default=3072, help="Subreddit embedding input dimension (default 3072)")
    parser.add_argument("--json", type=str, required=True, help="Path to JSON file with user interactions")
    parser.add_argument("--out", type=str, required=True, help="Output .npy file to save embeddings")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")

    device = torch.device(args.device)
    sub_x = load_subreddit_features(args.dataset, device=device)
    model = _load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        residual=bool(args.residual),
        device=device,
    )
    inter = _read_json_interactions(args.json)
    emb = batch_user_embeddings(
        model=model,
        subreddit_features=sub_x,
        interactions=inter,
        device=device,
        max_users_per_batch=int(args.batch_size),
    )
    np.save(args.out, emb.numpy())
    logger.info("Saved embeddings: %s (shape=%s)", args.out, tuple(emb.shape))


if __name__ == "__main__":
    main()


