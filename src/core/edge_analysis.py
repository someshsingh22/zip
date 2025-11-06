"""
Edge normalization analysis utilities.

This module provides helpers to:
- Load and harmonize post/comment parquet inputs into a combined edge dataframe
- Compute user/subreddit marginals and interaction statistics
- Derive several edge normalization schemes robust to popularity and activity skew
- Summarize distributions for quick inspection

Designed for research iteration: explicit, readable, minimal abstractions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import polars as pl


@dataclass(frozen=True)
class EdgeAnalysisInputs:
    """Container paths for input parquet files."""

    posts_file: str
    comments_file: str


@dataclass(frozen=True)
class EdgeStatistics:
    """Aggregate statistics derived from edges."""

    num_users: int
    num_subreddits: int
    num_edges: int
    total_interactions: int
    avg_user_total: float
    avg_subreddit_total: float


def build_id_maps(posts_df: pl.DataFrame, comments_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Create independent categorical encodings for users and subreddits.

    Args:
        posts_df: Polars DataFrame with at least 'author' and 'subreddit'.
        comments_df: Polars DataFrame with at least 'author' and 'subreddit'.

    Returns:
        user_map: DataFrame with columns ['author', 'user_id'].
        sub_map: DataFrame with columns ['subreddit', 'sub_id'].
    """
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
    return user_map, sub_map


def _map_edges(df: pl.DataFrame, user_map: pl.DataFrame, sub_map: pl.DataFrame, count_col: str) -> pl.DataFrame:
    """Map raw rows to integer user/subreddit ids and pick total_count as count_col.

    Args:
        df: Polars DataFrame with columns ['author', 'subreddit', 'total_count'].
        user_map: Mapping DataFrame ['author', 'user_id'].
        sub_map: Mapping DataFrame ['subreddit', 'sub_id'].
        count_col: Name to give the count column in the output.

    Returns:
        DataFrame with ['user_id', 'sub_id', count_col].
    """
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


def build_combined_edges(posts_df: pl.DataFrame, comments_df: pl.DataFrame) -> pl.DataFrame:
    """Combine post and comment counts to per-edge totals and simple features.

    The returned frame contains one row per (user_id, sub_id) with strictly positive total_count.

    Columns:
        user_id, sub_id, post_count, comment_count, total_count,
        comment_frac, post_frac, log_total_count

    Args:
        posts_df: Parquet-loaded Polars DataFrame of submissions with 'author','subreddit','total_count'.
        comments_df: Parquet-loaded Polars DataFrame of comments with 'author','subreddit','total_count'.

    Returns:
        Polars DataFrame with combined edge features.
    """
    user_map, sub_map = build_id_maps(posts_df, comments_df)
    post_edges = _map_edges(posts_df, user_map, sub_map, "post_count")
    comm_edges = _map_edges(comments_df, user_map, sub_map, "comment_count")

    combined = post_edges.join(comm_edges, on=["user_id", "sub_id"], how="outer").fill_null(0)
    combined = combined.with_columns(
        [
            (pl.col("post_count") + pl.col("comment_count")).alias("total_count"),
        ]
    )
    combined = combined.filter(pl.col("total_count") > 0)
    combined = combined.with_columns(
        [
            pl.when(pl.col("total_count") > 0)
            .then(pl.col("comment_count") / pl.col("total_count"))
            .otherwise(0.0)
            .alias("comment_frac"),
        ]
    )
    combined = combined.with_columns(
        [
            (1.0 - pl.col("comment_frac")).alias("post_frac"),
            (pl.col("total_count") + 1.0).log().alias("log_total_count"),
        ]
    )
    return combined


def compute_marginals(edges: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Compute user and subreddit marginals from edges.

    Args:
        edges: Combined edges frame with columns ['user_id','sub_id','total_count',...].

    Returns:
        user_stats: ['user_id','user_total','user_degree'].
        sub_stats: ['sub_id','sub_total','sub_degree'].
    """
    user_stats = (
        edges.group_by("user_id")
        .agg(
            pl.col("total_count").sum().alias("user_total"),
            pl.n_unique("sub_id").alias("user_degree"),
            pl.len().alias("user_edges"),  # equals degree when one row per pair
        )
        .sort("user_total", descending=True)
    )
    sub_stats = (
        edges.group_by("sub_id")
        .agg(
            pl.col("total_count").sum().alias("sub_total"),
            pl.n_unique("user_id").alias("sub_degree"),
            pl.len().alias("sub_edges"),
        )
        .sort("sub_total", descending=True)
    )
    return user_stats, sub_stats


def compute_edge_normalizations(
    edges: pl.DataFrame,
    user_stats: pl.DataFrame,
    sub_stats: pl.DataFrame,
    *,
    bm25_k1: float = 1.2,
    bm25_b: float = 0.75,
) -> pl.DataFrame:
    """Attach multiple normalization schemes as new columns on the edges frame.

    Schemes:
      - log1p: log(total_count + 1)
      - tf_user: total_count / user_total (row-normalized)
      - tf_sub: total_count / sub_total (col-normalized)
      - sym_norm: total_count / sqrt(user_total * sub_total) (symmetric)
      - pmi / ppmi: PMI/PPMI with global frequency baseline
      - bm25: Okapi BM25 per user-document
      - tfidf_user: TF-IDF per user-document

    Args:
        edges: Combined edges DataFrame.
        user_stats: User marginals from compute_marginals.
        sub_stats: Subreddit marginals from compute_marginals.
        bm25_k1: BM25 saturation parameter.
        bm25_b: BM25 length normalization parameter.

    Returns:
        New DataFrame with added columns for each scheme.
    """
    # Join marginals
    e = (
        edges.join(user_stats.select(["user_id", "user_total", "user_degree"]), on="user_id", how="left")
        .join(sub_stats.select(["sub_id", "sub_total", "sub_degree"]), on="sub_id", how="left")
        .with_columns([(pl.col("total_count") + 1.0).log().alias("w_log1p")])
    )

    # Global totals
    total_sum = e.select(pl.col("total_count").sum().alias("S")).item()
    num_users = user_stats.height
    num_subs = sub_stats.height
    avg_user_total = user_stats.select(pl.col("user_total").mean()).item()

    # Document frequency: users per subreddit (n_s)
    sub_df = sub_stats.select(["sub_id", pl.col("sub_degree").alias("df_sub")])

    e = e.join(sub_df, on="sub_id", how="left")

    # Core normalizations
    e = e.with_columns(
        [
            (pl.col("total_count") / pl.col("user_total")).alias("w_tf_user"),
            (pl.col("total_count") / pl.col("sub_total")).alias("w_tf_sub"),
            (
                pl.col("total_count")
                / (pl.col("user_total") * pl.col("sub_total")).sqrt()
            ).alias("w_sym_norm"),
        ]
    )

    # PMI / PPMI
    # PMI = log( (c_ij / S) / (r_i/S * c_j/S) ) = log( c_ij * S / (r_i * c_j) )
    e = e.with_columns(
        [
            (
                (pl.col("total_count") * total_sum) / (pl.col("user_total") * pl.col("sub_total"))
            )
            .clip(lower=1e-12)
            .log()
            .alias("w_pmi"),
        ]
    )
    e = e.with_columns([pl.when(pl.col("w_pmi") > 0).then(pl.col("w_pmi")).otherwise(0.0).alias("w_ppmi")])

    # BM25
    # idf_s = log( (N - n_s + 0.5) / (n_s + 0.5) + 1 )
    idf_sub = (
        ((num_users - pl.col("df_sub") + 0.5) / (pl.col("df_sub") + 0.5)).clip(lower=1e-12).log() + 1.0
    ).alias("idf_sub")
    K = bm25_k1 * (1.0 - bm25_b + bm25_b * (pl.col("user_total") / max(avg_user_total, 1e-9)))
    e = e.with_columns([idf_sub])
    e = e.with_columns(
        [
            (
                pl.col("idf_sub")
                * (pl.col("total_count") * (bm25_k1 + 1.0))
                / (pl.col("total_count") + K)
            ).alias("w_bm25"),
            # TF-IDF per user doc
            ((pl.col("total_count") / pl.col("user_total")) * (num_users / (pl.col("df_sub") + 1.0)).log()).alias(
                "w_tfidf_user"
            ),
        ]
    )

    return e


def summarize_quantiles(df: pl.DataFrame, cols: Iterable[str], quantiles: Iterable[float]) -> pd.DataFrame:
    """Compute quantile summary for given columns.

    Args:
        df: Polars DataFrame containing the columns.
        cols: Columns to summarize.
        quantiles: Quantiles in [0,1].

    Returns:
        Pandas DataFrame indexed by column with quantile values per column.
    """
    q_list = list(quantiles)
    records: List[Dict[str, float]] = []
    names: List[str] = []
    for c in cols:
        s = df.select(pl.col(c)).to_series()
        qs = np.quantile(s.to_numpy(), q_list)
        records.append({str(q): float(v) for q, v in zip(q_list, qs)})
        names.append(c)
    out = pd.DataFrame.from_records(records, index=names)
    out.index.name = "metric"
    return out


def compute_correlations(df: pl.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Compute Pearson correlation matrix among the given columns.

    Args:
        df: Polars DataFrame with numeric columns.
        cols: Columns to include.

    Returns:
        Pandas correlation matrix.
    """
    pdf = df.select([pl.col(c) for c in cols]).to_pandas()
    return pdf.corr(method="pearson")


def gather_edge_statistics(edges: pl.DataFrame, user_stats: pl.DataFrame, sub_stats: pl.DataFrame) -> EdgeStatistics:
    """Collect top-level statistics for reporting."""
    total_interactions = int(edges.select(pl.col("total_count").sum()).item())
    avg_user_total = float(user_stats.select(pl.col("user_total").mean()).item())
    avg_sub_total = float(sub_stats.select(pl.col("sub_total").mean()).item())
    return EdgeStatistics(
        num_users=user_stats.height,
        num_subreddits=sub_stats.height,
        num_edges=edges.height,
        total_interactions=total_interactions,
        avg_user_total=avg_user_total,
        avg_subreddit_total=avg_sub_total,
    )


