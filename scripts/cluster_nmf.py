import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Make src/ importable without installation
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.clustering import build_bipartite_csr, cluster_nmf_bipartite  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("cluster_nmf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NMF baseline clustering on the raw bipartite graph")
    parser.add_argument("--graph-path", type=str, required=True, help="Path to graph.pt payload")
    parser.add_argument(
        "--axis",
        type=str,
        choices=["users", "subs"],
        default="users",
        help="Cluster users (rows of W) or subreddits (columns of H)",
    )
    parser.add_argument("--n-clusters", type=int, default=50)
    parser.add_argument(
        "--init",
        type=str,
        choices=["nndsvd", "nndsvda", "nndsvdar", "random"],
        default="nndsvda",
    )
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--l1-ratio", type=float, default=0.0)
    parser.add_argument("--alpha-W", type=float, default=0.0)
    parser.add_argument("--alpha-H", type=float, default=0.0)
    parser.add_argument("--normalize-W-rows", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to write labels; defaults to graph directory",
    )
    return parser.parse_args()


def _load_graph(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    import torch

    payload = torch.load(str(path), map_location="cpu")
    edge_index_t = payload["edge_index"].to(torch.long)
    edge_weight_t = payload["edge_weight"].to(torch.float32)
    num_users = int(payload["num_users"])  # rows
    num_subs = int(payload["num_subs"])    # cols

    edge_index = edge_index_t.detach().cpu().numpy()
    edge_weight = edge_weight_t.detach().cpu().numpy()
    return edge_index, edge_weight, num_users, num_subs


def main() -> None:
    args = parse_args()
    graph_path = Path(args.graph_path)
    out_dir = Path(args.out_dir) if args.out_dir is not None else graph_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading graph from %s", graph_path)
    edge_index, edge_weight, num_users, num_subs = _load_graph(graph_path)
    logger.info("Building CSR matrix [%d users × %d subs] with %d edges", num_users, num_subs, edge_weight.size)
    X = build_bipartite_csr(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_users=num_users,
        num_subs=num_subs,
    )

    logger.info("Running NMF: K=%d | axis=%s", int(args.n_clusters), args.axis)
    res = cluster_nmf_bipartite(
        X,
        n_clusters=int(args.n_clusters),
        axis=args.axis,
        init=args.init,
        max_iter=int(args.max_iter),
        random_state=int(args.random_state),
        l1_ratio=float(args.l1_ratio),
        alpha_W=float(args.alpha_W),
        alpha_H=float(args.alpha_H),
        normalize_W_rows=bool(args.normalize_W_rows),
    )

    name = f"nmf_{args.axis}_k{int(args.n_clusters)}"
    out_path = out_dir / f"{name}_labels.npy"
    np.save(str(out_path), res.labels)
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()

