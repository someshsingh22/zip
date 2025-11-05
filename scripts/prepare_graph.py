import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Tuple


# Make src/ importable without installation
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.data import (  # noqa: E402
    build_id_maps,
    compute_power_tfidf_weights,
    load_counts_parquet,
    save_graph_tensors,
    save_id_maps_json,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("prepare_graph")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare weighted bipartite graph tensors")
    parser.add_argument("--counts-parquet", type=str, required=True,
                        help="Path to parquet with columns [author, subreddit, total_count]")
    parser.add_argument("--out", type=str, required=True, help="Output .pt file for graph tensors")
    parser.add_argument("--alpha", type=float, default=0.75, help="Power-law exponent alpha")
    parser.add_argument("--tfidf-smooth", type=float, default=1.0, help="IDF smoothing constant")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_counts_parquet(args.counts_parquet)
    user_to_id, sub_to_id, id_to_user, id_to_sub = build_id_maps(df)

    edge_index, edge_weight, sub_degree = compute_power_tfidf_weights(
        df=df,
        user_to_id=user_to_id,
        sub_to_id=sub_to_id,
        alpha=float(args.alpha),
        tfidf_smooth=float(args.tfidf_smooth),
    )

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
        "Prepared graph: %s users, %s subreddits, %s edges",
        len(id_to_user), len(id_to_sub), edge_index.shape[1],
    )


if __name__ == "__main__":
    main()


