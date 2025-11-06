#!/usr/bin/env python3
"""
Analyze edge weights and propose normalizations.

Loads the same parquet inputs as prepare_data.py, builds the combined (user,subreddit)
edges, computes marginals, several normalization schemes, and produces:
  - Quantile summaries for raw and normalized weights
  - Correlation matrix among normalization schemes (CSV + heatmap PNG)
  - Histograms for key schemes (PNG)
  - Top users/subreddits by totals (CSV)
  - A recommendations.txt file with suggested normalizations to try

Usage:
  uv run python scripts/analyze_edges.py \
    --posts-file data/merged_submissions_filtered_gt1_dsp.parquet \
    --comments-file data/merged_comments_filtered_3x3_dsp.parquet \
    --out-dir reports/edge_analysis
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from src.core.edge_analysis import (
    compute_correlations,
    compute_edge_normalizations,
    compute_marginals,
    gather_edge_statistics,
    summarize_quantiles,
    build_combined_edges,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Edge normalization analysis")
    parser.add_argument(
        "--posts-file",
        type=str,
        default="data/merged_submissions_filtered_gt1_dsp.parquet",
        help="Path to submissions parquet",
    )
    parser.add_argument(
        "--comments-file",
        type=str,
        default="data/merged_comments_filtered_3x3_dsp.parquet",
        help="Path to comments parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports/edge_analysis",
        help="Directory to write outputs",
    )
    parser.add_argument(
        "--sample-edges",
        type=int,
        default=200_000,
        help="Number of edges to sample for CSV preview and hist plotting",
    )
    parser.add_argument(
        "--bm25-k1",
        type=float,
        default=1.2,
        help="BM25 k1 parameter",
    )
    parser.add_argument(
        "--bm25-b",
        type=float,
        default=0.75,
        help="BM25 b parameter",
    )
    return parser.parse_args()


def ensure_out_dir(path_str: str) -> Path:
    """Create output directory if not exists."""
    out = Path(path_str)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_histograms(sample_df: pd.DataFrame, cols: Sequence[str], out_dir: Path, bins: int = 200) -> None:
    """Save histogram PNGs for selected columns.

    Uses log-scaled x-axis where appropriate to highlight heavy tails.
    """
    for c in cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        series = sample_df[c].replace([np.inf, -np.inf], np.nan).dropna()
        # Auto decide log vs linear
        use_log = series.max() / (series.min() + 1e-12) > 1e3
        sns.histplot(series, bins=bins, ax=ax, stat="density")
        ax.set_title(c)
        ax.set_xlabel(c)
        if use_log:
            ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(out_dir / f"hist_{c}.png", dpi=150)
        plt.close(fig)


def save_corr_heatmap(corr: pd.DataFrame, out_path: Path) -> None:
    """Save a correlation heatmap PNG."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, vmin=-1, vmax=1, cmap="vlag", annot=False, square=True, cbar=True)
    plt.title("Correlation of edge normalization schemes")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    """Entry point for end-to-end analysis."""
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)

    print("📦 Loading Parquet files...")
    posts_df = pl.read_parquet(args.posts_file)
    comments_df = pl.read_parquet(args.comments_file)

    print("🔗 Building combined edges...")
    edges = build_combined_edges(posts_df, comments_df)
    user_stats, sub_stats = compute_marginals(edges)
    stats = gather_edge_statistics(edges, user_stats, sub_stats)
    print(
        f"✅ Users: {stats.num_users:,} | Subs: {stats.num_subreddits:,} | "
        f"Edges: {stats.num_edges:,} | Total interactions: {stats.total_interactions:,}"
    )

    print("🧮 Computing normalizations (log1p, tf, symmetric, PPMI, BM25, TF-IDF)...")
    enriched = compute_edge_normalizations(
        edges,
        user_stats,
        sub_stats,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
    )

    # Persist top-k lists for quick inspection
    print("💾 Saving top users/subreddits by totals...")
    user_stats.select(["user_id", "user_total", "user_degree"]).write_csv(out_dir / "top_users.csv")
    sub_stats.select(["sub_id", "sub_total", "sub_degree"]).write_csv(out_dir / "top_subreddits.csv")

    # Quantile summaries
    metrics = [
        "log_total_count",
        "w_log1p",
        "w_tf_user",
        "w_tf_sub",
        "w_sym_norm",
        "w_pmi",
        "w_ppmi",
        "w_bm25",
        "w_tfidf_user",
    ]
    print("📊 Summarizing quantiles...")
    quant_df = summarize_quantiles(enriched, metrics, quantiles=[0.5, 0.9, 0.99, 0.999])
    quant_df.to_csv(out_dir / "edge_weight_quantiles.csv")

    # Correlations among schemes
    print("🔗 Computing correlation matrix...")
    corr = compute_correlations(enriched, metrics)
    corr.to_csv(out_dir / "edge_weight_correlations.csv")
    save_corr_heatmap(corr, out_dir / "edge_weight_correlations.png")

    # Sample for plots and preview CSV
    print("🔍 Sampling edges for plots and preview...")
    sample_n = min(args.sample_edges, enriched.height)
    sample = enriched.sample(n=sample_n, with_replacement=False, shuffle=True).to_pandas()
    sample.to_csv(out_dir / "edge_sample.csv", index=False)

    # Histograms for key schemes
    print("📈 Writing histograms...")
    hist_cols: List[str] = ["log_total_count", "w_sym_norm", "w_ppmi", "w_bm25", "w_tfidf_user", "w_tf_user"]
    save_histograms(sample, hist_cols, out_dir)

    # Draft recommendations based on general properties
    rec_lines = [
        "Recommended normalization schemes to try:",
        "",
        "- Symmetric normalization (w_sym_norm = c_ij / sqrt(r_i * c_j))",
        "  Rationale: discounts both power users and popular subreddits; common in normalized adjacency.",
        "",
        "- Positive PMI (w_ppmi = max(0, log(c_ij * S / (r_i * c_j))))",
        "  Rationale: highlights associations stronger than popularity baseline; robust to hubs. ",
        "  Tip: optionally clip to a high percentile or take sqrt to reduce tail heaviness.",
        "",
        "- BM25 per-user (k1=1.2, b=0.75) (w_bm25)",
        "  Rationale: caps gains from repeated interactions and length-normalizes by user activity.",
        "",
        "- TF-IDF per-user (w_tfidf_user)",
        "  Rationale: simple baseline that downweights popular subreddits and normalizes per user length.",
        "",
        "Notes:",
        "- Keep log1p(total) around for ablations; it performs decently but favors hubs.",
        "- If using GNN message passing, sym-norm aligns with normalized adjacency practice.",
        "- For link prediction with dot-product, BM25/PPMI often help reduce popularity bias.",
    ]
    (out_dir / "recommendations.txt").write_text("\n".join(rec_lines))
    print("📝 Wrote recommendations.txt")

    print(f"✅ Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()


