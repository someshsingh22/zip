import argparse
import json
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import torch
from tqdm.auto import tqdm
from omegaconf import OmegaConf

# Local imports
from src.gnn import RedditGATv2


logger = logging.getLogger(__name__)


@dataclass
class UserInteractions:
    """Container for a single user's subreddit interaction counts.

    Attributes:
        subreddit_ids: Global subreddit ids (contiguous [0..n_subs-1] for current parquet inputs).
        post_counts: Number of posts to each subreddit (same length as subreddit_ids).
        comment_counts: Number of comments to each subreddit (same length as subreddit_ids).
    """

    subreddit_ids: Sequence[int]
    post_counts: Sequence[int]
    comment_counts: Sequence[int]

    def __post_init__(self) -> None:
        assert (
            len(self.subreddit_ids) == len(self.post_counts) == len(self.comment_counts)
        ), "All fields must have equal length."

    @property
    def num_edges(self) -> int:
        return len(self.subreddit_ids)


def _build_batch_graph(
    sub_x_full: torch.Tensor,
    batch_interactions: Sequence[UserInteractions],
    device: torch.device,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[Tuple[str, str, str], torch.Tensor],
    Dict[Tuple[str, str, str], torch.Tensor],
    List[bool],
]:
    """Construct a minimal hetero-graph dicts for a batch of users.

    We keep only the subreddits referenced by this batch to reduce compute,
    remapping their global ids to a local contiguous range.

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
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {
            ("subreddit", "rev_interacts", "user"): torch.empty(
                (2, 0), dtype=torch.long, device=device
            )
        }
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor] = {
            ("subreddit", "rev_interacts", "user"): torch.empty(
                (0, 2), dtype=torch.float32, device=device
            )
        }
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
        totals = [
            int(p) + int(c) for p, c in zip(inter.post_counts, inter.comment_counts)
        ]
        user_total = float(sum(totals))
        denom = float(np.sqrt(user_total)) if user_total > 0.0 else 1e-12
        user_has_edges = False

        for sid, p, c, t in zip(
            inter.subreddit_ids, inter.post_counts, inter.comment_counts, totals
        ):
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

    edge_index = torch.tensor(
        [src_list, dst_list], dtype=torch.long, device=device
    )  # [2, E]
    edge_attr = torch.tensor(
        edge_feat_list, dtype=torch.float32, device=device
    )  # [E, 2]

    x_dict = {"subreddit": sub_x}
    edge_index_dict = {("subreddit", "rev_interacts", "user"): edge_index}
    edge_attr_dict = {("subreddit", "rev_interacts", "user"): edge_attr}
    return x_dict, edge_index_dict, edge_attr_dict, has_edges


@torch.no_grad()
def batch_user_embeddings(
    model: RedditGATv2,
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
        model: Trained RedditGATv2 model (evaluation mode recommended).
        subreddit_features: Full subreddit embedding matrix (float32).
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
    pbar = tqdm(total=total_users, desc="Embedding users", unit="user")
    while start < total_users:
        end = min(start + max_users_per_batch, total_users)
        batch_size = end - start
        batch = interactions[start:end]

        x_dict, eidx_dict, eattr_dict, has_edges = _build_batch_graph(
            sub_x_full, batch, device
        )
        out_x = model(
            x_dict, eidx_dict, eattr_dict
        )  # returns updated x_dict with "user"
        user_emb = out_x["user"]  # [B, hidden_dim]
        # Ensure float32 on CPU for return
        user_emb = user_emb.detach().to("cpu", dtype=torch.float32)

        # Zero-out embeddings for users with no edges to avoid random init leakage
        for i, flag in enumerate(has_edges):
            if not flag:
                user_emb[i].zero_()

        results.append(user_emb)
        start = end
        pbar.update(batch_size)
    pbar.close()

    if model_was_training:
        model.train()

    return (
        torch.cat(results, dim=0)
        if len(results) > 0
        else torch.empty((0, hidden_dim), dtype=torch.float32)
    )


def _load_model_from_checkpoint(
    checkpoint_path: str,
    input_dim: int,
    hidden_dim: int,
    residual: bool,
    heads: int,
    device: torch.device,
) -> RedditGATv2:
    """Utility to instantiate and load a RedditGATv2 from a checkpoint."""
    model = RedditGATv2(
        input_dim=input_dim, hidden_dim=hidden_dim, residual=residual, heads=heads
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    # Allow either raw state_dict or wrapper
    state_dict = state.get("model", state)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _read_parquets_build_graph_inputs(
    posts_file: str,
    comments_file: str,
) -> Tuple[List[UserInteractions], List[str], List[str], np.ndarray, np.ndarray]:
    """Load two parquet files and build per-user interaction lists and totals.

    The two parquets must have columns: 'author', 'subreddit', 'total_count'.

    Returns:
        interactions: List[UserInteractions] ordered by user_id ascending (only users with edges).
        authors: List of author names aligned with 'interactions' order.
        all_subs: List of subreddit names ordered by sub_id ascending.
        user_total_by_user: np.ndarray [n_users] of total activity per user.
        sub_total_by_sub: np.ndarray [n_subs] of total activity per subreddit.
    """
    logger.info(
        "Loading Parquet files: posts=%s, comments=%s", posts_file, comments_file
    )
    posts_df = pl.read_parquet(posts_file)
    comments_df = pl.read_parquet(comments_file)

    # Build categorical mappings (contiguous ids)
    all_users = pl.concat([posts_df["author"], comments_df["author"]]).unique()
    user_map = pl.DataFrame(
        {
            "author": all_users,
            "user_id": pl.arange(0, all_users.len(), eager=True, dtype=pl.Int32),
        }
    )

    all_subs = pl.concat([posts_df["subreddit"], comments_df["subreddit"]]).unique()
    sub_map = pl.DataFrame(
        {
            "subreddit": all_subs,
            "sub_id": pl.arange(0, all_subs.len(), eager=True, dtype=pl.Int32),
        }
    )

    def _map_edges(df: pl.DataFrame, count_col: str) -> pl.DataFrame:
        return (
            df.join(user_map, on="author", how="inner")
            .join(sub_map, on="subreddit", how="inner")
            .select(
                [
                    pl.col("user_id").alias("user_id"),
                    pl.col("sub_id").alias("sub_id"),
                    pl.col("total_count").alias(count_col),
                ]
            )
        )

    post_edges = _map_edges(posts_df, "post_count")
    comm_edges = _map_edges(comments_df, "comment_count")
    combined = post_edges.join(
        comm_edges, on=["user_id", "sub_id"], how="outer"
    ).fill_null(0)
    combined = combined.with_columns(
        [(pl.col("post_count") + pl.col("comment_count")).alias("total_count")]
    )
    combined = combined.with_columns(
        [
            pl.when(pl.col("total_count") > 0)
            .then(pl.col("comment_count") / pl.col("total_count"))
            .otherwise(0.0)
            .alias("comment_frac")
        ]
    )

    # Totals per user and per subreddit
    user_activity = combined.group_by("user_id").agg(
        pl.col("total_count").sum().alias("user_total")
    )
    sub_activity = combined.group_by("sub_id").agg(
        pl.col("total_count").sum().alias("sub_total")
    )
    combined = combined.join(user_activity, on="user_id", how="left").join(
        sub_activity, on="sub_id", how="left"
    )

    # Create per-user lists
    grouped = (
        combined.sort(["user_id", "sub_id"])
        .group_by("user_id")
        .agg(
            [
                pl.col("sub_id").alias("sub_id_list"),
                pl.col("post_count").alias("post_count_list"),
                pl.col("comment_count").alias("comment_count_list"),
                pl.col("total_count").alias("total_count_list"),
                pl.col("user_total").first().alias("user_total_scalar"),
            ]
        )
        .sort("user_id")
    )

    interactions: List[UserInteractions] = []
    authors: List[str] = []
    all_users_list = list(all_users)
    # Convert to lists for Python side
    for row in grouped.iter_rows(named=True):
        uid = int(row["user_id"])
        authors.append(str(all_users_list[uid]))
        subs = list(row["sub_id_list"])
        posts = list(row["post_count_list"])
        comms = list(row["comment_count_list"])
        interactions.append(
            UserInteractions(
                subreddit_ids=subs, post_counts=posts, comment_counts=comms
            )
        )

    # Extract totals arrays aligned with id ordering
    n_users = all_users.len()
    n_subs = all_subs.len()
    user_total_by_user = np.zeros((n_users,), dtype=np.float32)
    # Build user_total array
    user_totals_df = user_activity.sort("user_id")
    user_total_by_user[: user_totals_df.height] = (
        user_totals_df["user_total"].to_numpy().astype(np.float32)
    )
    # Build sub_total array
    sub_total_by_sub = np.zeros((n_subs,), dtype=np.float32)
    sub_totals_df = sub_activity.sort("sub_id")
    sub_total_by_sub[: sub_totals_df.height] = (
        sub_totals_df["sub_total"].to_numpy().astype(np.float32)
    )

    # Return also the subreddit names in id order
    all_subs_list = list(all_subs)
    return interactions, authors, all_subs_list, user_total_by_user, sub_total_by_sub


def _load_subreddit_embeddings_from_dict(
    all_subs: Sequence[str],
    emb_npy_path: str,
    device: torch.device,
) -> torch.Tensor:
    """Load subreddit embeddings dict (.npy) and build matrix aligned to sub_id ordering."""
    logger.info("Loading subreddit embeddings dict from %s", emb_npy_path)
    sub_emb_dict = np.load(emb_npy_path, allow_pickle=True).item()
    dim = None
    # Probe first present embedding to get dimension
    for name in all_subs:
        emb = sub_emb_dict.get(name, None)
        if emb is not None:
            dim = int(np.asarray(emb).shape[-1])
            break
    if dim is None:
        raise ValueError("No subreddit embeddings found for provided subreddit list.")

    mat = np.zeros((len(all_subs), dim), dtype=np.float32)
    missing = 0
    for i, name in enumerate(all_subs):
        emb = sub_emb_dict.get(name, None)
        if emb is not None:
            mat[i] = np.asarray(emb, dtype=np.float32)
        else:
            missing += 1
    if missing:
        logger.warning("%d subreddits missing embeddings; filled with zeros.", missing)
    return torch.from_numpy(mat).to(device=device, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batched user embedding inference from two parquet files (posts/comments)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="YAML config with model settings (model.input_dim, model.hidden_dim, model.residual).",
    )
    parser.add_argument(
        "--posts",
        type=str,
        default="/dev/shm/zip/data/merged_submissions_filtered_gt1_dsp.parquet",
        help="Path to posts parquet (columns: author, subreddit, total_count)",
    )
    parser.add_argument(
        "--comments",
        type=str,
        default="/dev/shm/zip/data/merged_comments_filtered_3x3_dsp.parquet",
        help="Path to comments parquet (columns: author, subreddit, total_count)",
    )
    parser.add_argument(
        "--subreddit-emb",
        type=str,
        default="/dev/shm/zip/data/processed/subreddit_embeddings.npy",
        help="Path to subreddit_embeddings.npy (dict: name -> vector)",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output .npy file with dict: author -> embedding (float32)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:3" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=100000)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    cfg = OmegaConf.load(args.config)
    input_dim = int(cfg.model.input_dim)
    hidden_dim = int(cfg.model.hidden_dim)
    residual = bool(cfg.model.get("residual", True))
    heads = int(cfg.model.get("heads", 6))
    device = torch.device(args.device)

    # Read parquet inputs and construct interactions + totals
    interactions, authors, all_subs, user_total_by_user, sub_total_by_sub = (
        _read_parquets_build_graph_inputs(args.posts, args.comments)
    )
    # Load subreddit embeddings aligned with sub_id ordering
    sub_x = _load_subreddit_embeddings_from_dict(
        all_subs, args.subreddit_emb, device=device
    )

    model = _load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        residual=residual,
        device=device,
        heads=heads,
    )

    emb = batch_user_embeddings(
        model=model,
        subreddit_features=sub_x,
        interactions=interactions,
        device=device,
        max_users_per_batch=int(args.batch_size),
    )
    # Build mapping author -> embedding (float32 numpy array)
    author_to_emb: Dict[str, np.ndarray] = {}
    for i, author in enumerate(authors):
        author_to_emb[author] = emb[i].numpy()
    np.save(args.out, author_to_emb, allow_pickle=True)
    logger.info("Saved author->embedding dict: %s (num_authors=%d, dim=%d)", args.out, len(author_to_emb), int(emb.shape[1]) if emb.numel() > 0 else 0)


if __name__ == "__main__":
    main()
