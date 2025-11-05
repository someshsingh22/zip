import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import polars as pl
import torch


logger = logging.getLogger(__name__)


def load_counts_parquet(parquet_path: str) -> pl.DataFrame:
    """Load the user–subreddit count interactions parquet.

    Args:
        parquet_path: Path to parquet with columns: author, subreddit, total_count.

    Returns:
        Polars DataFrame with required columns.
    """
    df = pl.read_parquet(parquet_path)
    expected = {"author", "subreddit", "total_count"}
    assert expected.issubset(set(df.columns)), (
        f"Parquet missing required columns. Found: {df.columns}, expected: {expected}"
    )
    return df


def build_id_maps(df: pl.DataFrame) -> Tuple[Dict[str, int], Dict[str, int], List[str], List[str]]:
    """Build stable id maps for authors and subreddits.

    Order is the first-seen order in the parquet (stable unique).

    Args:
        df: DataFrame with columns `author`, `subreddit`.

    Returns:
        user_to_id, sub_to_id, id_to_user, id_to_sub lists aligned to ids.
    """
    users = (
        df.select(pl.col("author")).unique(maintain_order=True).get_column("author").to_list()
    )
    subs = (
        df.select(pl.col("subreddit")).unique(maintain_order=True).get_column("subreddit").to_list()
    )
    user_to_id = {u: i for i, u in enumerate(users)}
    sub_to_id = {s: i for i, s in enumerate(subs)}
    return user_to_id, sub_to_id, users, subs


def build_id_maps_from_multiple(
    dfs: List[pl.DataFrame],
    user_col: str = "author",
    sub_col: str = "subreddit",
) -> Tuple[Dict[str, int], Dict[str, int], List[str], List[str]]:
    """Build id maps from the union of users/subreddits across multiple frames.

    Args:
        dfs: List of DataFrames each containing user and subreddit columns.
        user_col: Column name for users.
        sub_col: Column name for subreddits.

    Returns:
        user_to_id, sub_to_id, id_to_user, id_to_sub
    """
    users: List[str] = []
    subs: List[str] = []
    for df in dfs:
        if df is None:
            continue
        if user_col not in df.columns or sub_col not in df.columns:
            continue
        users.extend(df.get_column(user_col).unique().to_list())
        subs.extend(df.get_column(sub_col).unique().to_list())
    # stable (first occurrence) order
    user_seen: Dict[str, int] = {}
    sub_seen: Dict[str, int] = {}
    id_to_user: List[str] = []
    id_to_sub: List[str] = []
    for u in users:
        if u not in user_seen:
            user_seen[u] = len(id_to_user)
            id_to_user.append(u)
    for s in subs:
        if s not in sub_seen:
            sub_seen[s] = len(id_to_sub)
            id_to_sub.append(s)
    return user_seen, sub_seen, id_to_user, id_to_sub


def to_counts_df(
    df: pl.DataFrame,
    user_col: str = "author",
    sub_col: str = "subreddit",
    count_col: Optional[str] = None,
) -> pl.DataFrame:
    """Normalize an events table into a counts table [author, subreddit, total_count].

    If `count_col` is provided and exists, it is summed, otherwise counts are computed via len().
    """
    if count_col and count_col in df.columns:
        grouped = (
            df.group_by([user_col, sub_col])
            .agg(pl.col(count_col).sum().alias("total_count"))
            .with_columns(
                pl.col(user_col).cast(pl.Utf8),
                pl.col(sub_col).cast(pl.Utf8),
                pl.col("total_count").cast(pl.Int64),
            )
        )
    else:
        grouped = (
            df.group_by([user_col, sub_col])
            .len()
            .rename({"len": "total_count"})
            .with_columns(
                pl.col(user_col).cast(pl.Utf8),
                pl.col(sub_col).cast(pl.Utf8),
                pl.col("total_count").cast(pl.Int64),
            )
        )
    # rename to expected schema
    return grouped.rename({user_col: "author", sub_col: "subreddit"})


def compute_power_tfidf_weights(
    df: pl.DataFrame,
    user_to_id: Dict[str, int],
    sub_to_id: Dict[str, int],
    alpha: float = 0.75,
    tfidf_smooth: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute power-law + TF–IDF weighted edges.

    Weight per edge (u, s): w_{u,s} = tf_{u,s} * idf_s with
    tf from power-law compressed counts and idf with smoothing.

    Args:
        df: Input interactions with columns (author, subreddit, total_count).
        user_to_id: Mapping from author to user id.
        sub_to_id: Mapping from subreddit to subreddit id.
        alpha: Power-law compression exponent.
        tfidf_smooth: Smoothing constant for IDF.

    Returns:
        edge_index [2, E] (long), edge_weight [E] (float32), sub_degree [num_subs] (float32).
    """
    # Power-law compress counts
    df = df.with_columns(
        (pl.col("total_count").cast(pl.Float64) ** alpha).alias("count_pow")
    )

    # Term-frequency per user
    user_sum = df.group_by("author").agg(pl.col("count_pow").sum().alias("user_sum"))
    df = df.join(user_sum, on="author", how="left")
    df = df.with_columns((pl.col("count_pow") / pl.col("user_sum")).alias("tf"))

    # IDF per subreddit with smoothing
    N_users = int(df.select(pl.col("author").n_unique().alias("n")).to_series().item())
    df_counts = df.group_by("subreddit").agg(pl.col("author").n_unique().alias("df"))
    smooth = float(tfidf_smooth)
    # idf = log((N + smooth) / (df + smooth)) + 1
    df_counts = df_counts.with_columns(
        (pl.lit(N_users + smooth).log() - (pl.col("df") + smooth).log() + 1.0).alias("idf")
    )
    df = df.join(df_counts.select(["subreddit", "idf"]), on="subreddit", how="left")
    df = df.with_columns((pl.col("tf") * pl.col("idf")).cast(pl.Float32).alias("weight"))

    # Map to ids (CPU lists are fine; saved once)
    user_ids = [user_to_id[a] for a in df.get_column("author").to_list()]
    sub_ids = [sub_to_id[s] for s in df.get_column("subreddit").to_list()]
    weights = df.get_column("weight").to_list()

    edge_index = torch.tensor([user_ids, sub_ids], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    # Subreddit degree (sum of weights) for negative sampling distribution
    sub_degree_df = df.group_by("subreddit").agg(pl.col("weight").sum().alias("deg"))
    num_subs = len(sub_to_id)
    sub_degree = torch.zeros(num_subs, dtype=torch.float32)
    for s, d in zip(
        sub_degree_df.get_column("subreddit").to_list(),
        sub_degree_df.get_column("deg").to_list(),
    ):
        sub_degree[sub_to_id[s]] = float(d)

    return edge_index, edge_weight, sub_degree


def save_graph_tensors(
    out_path: str,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    sub_degree: torch.Tensor,
    user_to_id: Dict[str, int],
    sub_to_id: Dict[str, int],
    id_to_user: List[str],
    id_to_sub: List[str],
) -> None:
    """Persist graph tensors and id maps to a single .pt file.

    Args:
        out_path: Destination path for serialized graph.
        edge_index: [2, E] tensor of (user_id, subreddit_id) pairs.
        edge_weight: [E] weights for each edge.
        sub_degree: [#subreddits] degree-like weights for negatives.
        user_to_id: Mapping from username to id.
        sub_to_id: Mapping from subreddit name to id.
        id_to_user: Reverse map list for users.
        id_to_sub: Reverse map list for subreddits.
    """
    payload = {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "sub_degree": sub_degree,
        "num_users": len(id_to_user),
        "num_subs": len(id_to_sub),
        "user_to_id": user_to_id,
        "sub_to_id": sub_to_id,
        "id_to_user": id_to_user,
        "id_to_sub": id_to_sub,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    logger.info("Saved graph tensors to %s", out_path)


def save_id_maps_json(path: str, user_to_id: Dict[str, int], sub_to_id: Dict[str, int]) -> None:
    """Save id maps as JSON next to the graph for downstream alignment.

    Args:
        path: JSON file path to write.
        user_to_id: Mapping for users.
        sub_to_id: Mapping for subreddits.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"user_to_id": user_to_id, "sub_to_id": sub_to_id}, f)
    logger.info("Saved id maps to %s", path)


