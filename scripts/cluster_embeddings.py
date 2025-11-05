import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# Make src/ importable without installation
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.clustering import ClusterResult, cluster_hdbscan, cluster_spectral  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("cluster_embeddings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster embeddings with HDBSCAN and Spectral")
    parser.add_argument("--in-path", type=str, required=True, help="Path to embeddings (.pt or .npy)")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to write labels; defaults to the directory of --in-path",
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["both", "hdbscan", "spectral"],
        default="both",
        help="Which algorithm(s) to run",
    )

    # HDBSCAN params
    parser.add_argument("--min-cluster-size", type=int, default=15)
    parser.add_argument("--min-samples", type=int, default=-1, help="Set -1 to use default behavior")
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument("--cluster-selection-epsilon", type=float, default=0.0)
    parser.add_argument(
        "--cluster-selection-method",
        type=str,
        choices=["eom", "leaf"],
        default="eom",
    )
    parser.add_argument("--allow-single-cluster", action="store_true")
    parser.add_argument("--prediction-data", action="store_true")

    # Spectral params
    parser.add_argument("--spectral-n-clusters", type=int, default=50)
    parser.add_argument(
        "--spectral-affinity",
        type=str,
        choices=["nearest_neighbors", "rbf"],
        default="nearest_neighbors",
    )
    parser.add_argument("--spectral-n-neighbors", type=int, default=10)
    parser.add_argument(
        "--spectral-assign-labels",
        type=str,
        choices=["kmeans", "discretize"],
        default="kmeans",
    )
    parser.add_argument("--random-state", type=int, default=42)

    return parser.parse_args()


def _load_embeddings(path: Path) -> np.ndarray:
    """Load embeddings from .npy or .pt (tensor or dict with key 'user_emb'/'embeddings')."""
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(str(path))
        assert arr.ndim == 2, "Expected [num_points, dim] from .npy"
        return arr.astype(np.float32, copy=False)

    if ext == ".pt":
        import torch  # local import to avoid hard dep for non-PT users

        obj = torch.load(str(path), map_location="cpu")
        if isinstance(obj, torch.Tensor):
            emb = obj
        elif isinstance(obj, dict):
            if "user_emb" in obj:
                emb = obj["user_emb"]
            elif "embeddings" in obj:
                emb = obj["embeddings"]
            else:
                raise AssertionError("Unexpected dict keys in .pt; expected 'user_emb' or 'embeddings'")
        else:
            raise AssertionError(".pt must contain a Tensor or a dict with embeddings")
        assert emb.ndim == 2, "Expected [num_points, dim] Tensor"
        return emb.detach().cpu().numpy().astype(np.float32, copy=False)

    raise AssertionError("Unsupported embeddings file type; use .npy or .pt")


def _summarize(labels: np.ndarray) -> Tuple[int, float]:
    """Return (num_clusters, noise_ratio). Noise is label -1."""
    unique, counts = np.unique(labels, return_counts=True)
    mask = unique >= 0
    num_clusters = int(mask.sum())
    noise_count = int(counts[unique == -1].sum()) if (-1 in unique) else 0
    noise_ratio = float(noise_count) / float(labels.size)
    return num_clusters, noise_ratio


def _save_labels(out_dir: Path, name: str, labels: np.ndarray) -> Path:
    out_path = out_dir / f"{name}_labels.npy"
    np.save(str(out_path), labels)
    return out_path


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir) if args.out_dir is not None else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading embeddings from %s", in_path)
    X = _load_embeddings(in_path)
    logger.info("Embeddings shape: %s", tuple(X.shape))

    if args.algo in ("both", "hdbscan"):
        min_samples = None if int(args.min_samples) < 0 else int(args.min_samples)
        res_h: ClusterResult = cluster_hdbscan(
            X,
            min_cluster_size=int(args.min_cluster_size),
            min_samples=min_samples,
            metric=args.metric,
            cluster_selection_epsilon=float(args.cluster_selection_epsilon),
            cluster_selection_method=args.cluster_selection_method,
            allow_single_cluster=bool(args.allow_single_cluster),
            prediction_data=bool(args.prediction_data),
        )
        h_out = _save_labels(out_dir, "hdbscan", res_h.labels)
        ncl, nz = _summarize(res_h.labels)
        logger.info("HDBSCAN: clusters=%d | noise_ratio=%.3f | saved=%s", ncl, nz, h_out)
        if res_h.extras is not None:
            if "probabilities" in res_h.extras:
                np.save(str(out_dir / "hdbscan_probabilities.npy"), res_h.extras["probabilities"])
            if "outlier_scores" in res_h.extras:
                np.save(str(out_dir / "hdbscan_outlier_scores.npy"), res_h.extras["outlier_scores"])

    if args.algo in ("both", "spectral"):
        res_s: ClusterResult = cluster_spectral(
            X,
            n_clusters=int(args.spectral_n_clusters),
            affinity=args.spectral_affinity,
            n_neighbors=int(args.spectral_n_neighbors),
            assign_labels=args.spectral_assign_labels,
            random_state=int(args.random_state),
        )
        s_name = f"spectral_k{int(args.spectral_n_clusters)}"
        s_out = _save_labels(out_dir, s_name, res_s.labels)
        ncl, _ = _summarize(res_s.labels)
        logger.info("Spectral: clusters=%d | saved=%s", ncl, s_out)


if __name__ == "__main__":
    main()

