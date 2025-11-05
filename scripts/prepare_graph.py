import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import polars as pl
import torch


# Make src/ importable without installation
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.data import (  # noqa: E402
    build_id_maps,
    build_id_maps_from_multiple,
    compute_power_tfidf_weights,
    load_counts_parquet,
    save_graph_tensors,
    save_id_maps_json,
    to_counts_df,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("prepare_graph")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare weighted bipartite graph tensors (comments + posts)")
    # Back-compat single counts parquet
    parser.add_argument("--counts-parquet", type=str, default=None,
                        help="Parquet with [author, subreddit, total_count] for single relation")

    # New: dual sources
    parser.add_argument("--comments-parquet", type=str, default=None,
                        help="Events parquet for comments; grouped into counts")
    parser.add_argument("--comments-user-col", type=str, default="author",
                        help="User column in comments parquet")
    parser.add_argument("--comments-sub-col", type=str, default="subreddit",
                        help="Subreddit column in comments parquet")
    parser.add_argument("--comments-count-col", type=str, default=None,
                        help="Optional count column to sum in comments parquet")

    parser.add_argument("--posts-parquet", type=str, default=None,
                        help="Events parquet for posts; grouped into counts")
    parser.add_argument("--posts-user-col", type=str, default="author",
                        help="User column in posts parquet")
    parser.add_argument("--posts-sub-col", type=str, default="subreddit",
                        help="Subreddit column in posts parquet")
    parser.add_argument("--posts-count-col", type=str, default=None,
                        help="Optional count column to sum in posts parquet")

    parser.add_argument("--out", type=str, required=True, help="Output .pt file for graph tensors")
    parser.add_argument("--alpha", type=float, default=0.75, help="Power-law exponent alpha")
    parser.add_argument("--tfidf-smooth", type=float, default=1.0, help="IDF smoothing constant")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine inputs mode
    if args.counts_parquet and not (args.comments_parquet or args.posts_parquet):
        # Back-compat single relation
        df = load_counts_parquet(args.counts_parquet)
        user_to_id, sub_to_id, id_to_user, id_to_sub = build_id_maps(df)

        edge_index, edge_weight, sub_degree = compute_power_tfidf_weights(
            df=df,
            user_to_id=user_to_id,
            sub_to_id=sub_to_id,
            alpha=float(args.alpha),
            tfidf_smooth=float(args.tfidf_smooth),
        )

        # Save legacy payload for compatibility
        save_graph_tensors(
            out_path=args.out,
            edge_index=edge_index,
            edge_weight=edge_weight,
            sub_degree=sub_degree,
            user_to_id=user_to_id,
            sub_to_id=sub_to_id,
            id_to_user=id_to_user,
            id_to_sub=id_to_sub,
        )

        id_map_path = str(Path(args.out).parent / "id_maps.json")
        save_id_maps_json(id_map_path, user_to_id=user_to_id, sub_to_id=sub_to_id)

        logger.info(
            "Prepared graph (single): %s users, %s subreddits, %s edges",
            len(id_to_user), len(id_to_sub), edge_index.shape[1],
        )
        return

    # Dual-source mode
    assert args.comments_parquet or args.posts_parquet, "Provide at least one of comments/posts parquet"

    df_comments = None
    if args.comments_parquet:
        df_raw = pl.read_parquet(args.comments_parquet)
        df_comments = to_counts_df(
            df_raw,
            user_col=args.comments_user_col,
            sub_col=args.comments_sub_col,
            count_col=args.comments_count_col,
        )

    df_posts = None
    if args.posts_parquet:
        df_raw = pl.read_parquet(args.posts_parquet)
        df_posts = to_counts_df(
            df_raw,
            user_col=args.posts_user_col,
            sub_col=args.posts_sub_col,
            count_col=args.posts_count_col,
        )

    # Build unified id maps from both
    dfs_for_ids = [d for d in [df_comments, df_posts] if d is not None]
    user_to_id, sub_to_id, id_to_user, id_to_sub = build_id_maps_from_multiple(
        dfs_for_ids, user_col="author", sub_col="subreddit"
    )

    # Compute weights per relation
    edge_index_comments = torch.empty(2, 0, dtype=torch.long)
    edge_weight_comments = None
    if df_comments is not None:
        eic, ewc, _deg = compute_power_tfidf_weights(
            df=df_comments,
            user_to_id=user_to_id,
            sub_to_id=sub_to_id,
            alpha=float(args.alpha),
            tfidf_smooth=float(args.tfidf_smooth),
        )
        edge_index_comments = eic
        edge_weight_comments = ewc

    edge_index_posts = torch.empty(2, 0, dtype=torch.long)
    edge_weight_posts = None
    if df_posts is not None:
        eip, ewp, _deg = compute_power_tfidf_weights(
            df=df_posts,
            user_to_id=user_to_id,
            sub_to_id=sub_to_id,
            alpha=float(args.alpha),
            tfidf_smooth=float(args.tfidf_smooth),
        )
        edge_index_posts = eip
        edge_weight_posts = ewp

    # Save consolidated payload
    payload = {
        "edge_index_comments": edge_index_comments,
        "edge_weight_comments": edge_weight_comments,
        "edge_index_posts": edge_index_posts,
        "edge_weight_posts": edge_weight_posts,
        "num_users": len(id_to_user),
        "num_subs": len(id_to_sub),
        "user_to_id": user_to_id,
        "sub_to_id": sub_to_id,
        "id_to_user": id_to_user,
        "id_to_sub": id_to_sub,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)

    id_map_path = str(Path(args.out).parent / "id_maps.json")
    save_id_maps_json(id_map_path, user_to_id=user_to_id, sub_to_id=sub_to_id)

    logger.info(
        "Prepared graph (multi): %s users, %s subreddits, %s comment edges, %s post edges",
        len(id_to_user), len(id_to_sub), edge_index_comments.shape[1], edge_index_posts.shape[1],
    )


if __name__ == "__main__":
    main()


