import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Make src/ importable without installation
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.clustering import (  # noqa: E402
    build_bipartite_csr,
    cluster_hdbscan,
    cluster_nmf_bipartite,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("viz_clusters")


# --------------------
# Utilities
# --------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize user clusters with labels from representative subreddits")
    parser.add_argument("--graph-path", type=str, required=True, help="Path to graph.pt payload")
    parser.add_argument("--emb-path", type=str, default=None, help="Optional user embeddings (.pt or .npy)")
    parser.add_argument("--labels-path", type=str, default=None, help="Optional precomputed labels (.npy)")

    # Clustering options
    parser.add_argument("--algo", type=str, choices=["auto", "hdbscan", "spectral", "nmf"], default="auto",
                        help="auto: use embeddings->HDBSCAN if emb provided else NMF on graph")
    parser.add_argument("--n-clusters", type=int, default=50, help="Used for spectral/NMF when needed")

    # Dimensionality reduction
    parser.add_argument("--dimred", type=str, choices=["pca", "svd"], default="pca",
                        help="pca for embeddings; svd for sparse graph")

    # Labeling
    parser.add_argument("--topk", type=int, default=5, help="Top-K subreddits per cluster label")
    parser.add_argument("--min-annotate", type=int, default=50, help="Min cluster size to annotate")

    # Plotting
    parser.add_argument("--max-points", type=int, default=150000, help="Subsample for plotting if larger")
    parser.add_argument("--figsize", type=float, nargs=2, default=(9.0, 7.0))
    parser.add_argument("--out", type=str, default=None, help="Output image path (.png). Defaults next to graph")

    return parser.parse_args()


def _load_graph(path: Path) -> Tuple[np.ndarray, np.ndarray, int, int, List[str]]:
    import torch

    payload = torch.load(str(path), map_location="cpu")
    edge_index = payload["edge_index"].to(torch.long).cpu().numpy()
    edge_weight = payload["edge_weight"].to(torch.float32).cpu().numpy()
    num_users = int(payload["num_users"])  # rows
    num_subs = int(payload["num_subs"])    # cols
    id_to_sub: List[str] = list(map(str, payload["id_to_sub"]))
    return edge_index, edge_weight, num_users, num_subs, id_to_sub


def _load_embeddings(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(str(path))
        assert arr.ndim == 2
        return arr.astype(np.float32, copy=False)
    if ext == ".pt":
        import torch

        obj = torch.load(str(path), map_location="cpu")
        if hasattr(obj, "detach"):
            arr = obj.detach().cpu().numpy()
        elif isinstance(obj, dict) and "user_emb" in obj:
            arr = obj["user_emb"].detach().cpu().numpy()
        else:
            raise AssertionError(".pt must be a Tensor or dict with 'user_emb'")
        assert arr.ndim == 2
        return arr.astype(np.float32, copy=False)
    raise AssertionError("Unsupported embeddings file type; use .npy or .pt")


def _compute_xy_from_embeddings(emb: np.ndarray, method: str = "pca") -> np.ndarray:
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=42).fit_transform(emb)
    raise AssertionError("Unknown dimred for embeddings")


def _compute_xy_from_sparse(X, method: str = "svd") -> np.ndarray:
    # X is CSR
    if method == "svd":
        from sklearn.decomposition import TruncatedSVD
        return TruncatedSVD(n_components=2, random_state=42).fit_transform(X)
    raise AssertionError("Unknown dimred for sparse graph")


def _subsample(xy: np.ndarray, labels: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = xy.shape[0]
    if n <= max_points:
        idx = np.arange(n, dtype=np.int64)
        return xy, labels, idx
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=max_points, replace=False)
    return xy[idx], labels[idx], idx


def _compute_cluster_labels(
    X,  # CSR
    labels: np.ndarray,
    id_to_sub: List[str],
    topk: int = 5,
    min_count: int = 10,
) -> Dict[int, List[str]]:
    # Global subreddit weights
    import numpy as _np

    global_deg = _np.asarray(X.sum(axis=0)).ravel() + 1e-8

    out: Dict[int, List[str]] = {}
    for c in sorted(set(int(v) for v in labels if v >= 0)):
        idx = _np.where(labels == c)[0]
        if idx.size < min_count:
            continue
        vec = _np.asarray(X[idx].sum(axis=0)).ravel()
        # Over-representation score (ratio vs global)
        score = (vec / global_deg)
        top = _np.argsort(-score)[:topk]
        out[c] = [id_to_sub[int(j)] for j in top]
    return out


def _place_annotations(ax, xy: np.ndarray, labels: np.ndarray, cluster_to_tags: Dict[int, List[str]], min_annotate: int) -> None:
    for c, tags in cluster_to_tags.items():
        pts = xy[labels == c]
        if pts.shape[0] < min_annotate:
            continue
        m = pts.mean(axis=0)
        text = " / ".join(tags)
        ax.text(m[0], m[1], text, ha="center", va="center", fontsize=9, weight="bold", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2.0))


# --------------------
# Main
# --------------------

def main() -> None:
    args = parse_args()
    graph_path = Path(args.graph_path)
    out_path = Path(args.out) if args.out is not None else (graph_path.parent / "viz_clusters.png")

    # Load graph and build CSR
    logger.info("Loading graph from %s", graph_path)
    edge_index, edge_weight, num_users, num_subs, id_to_sub = _load_graph(graph_path)
    X = build_bipartite_csr(edge_index=edge_index, edge_weight=edge_weight, num_users=num_users, num_subs=num_subs)

    # Labels
    emb = None
    labels = None

    if args.labels_path is not None:
        labels = np.load(str(Path(args.labels_path)))
        assert labels.shape[0] == num_users, "labels length must equal #users"
        logger.info("Loaded labels: %s clusters incl. noise", int(np.max(labels)) + 1)
    else:
        # Compute labels depending on availability
        if args.emb_path is not None and (args.algo in ("auto", "hdbscan", "spectral")):
            emb = _load_embeddings(Path(args.emb_path))
            assert emb.shape[0] == num_users, "embedding rows must equal #users"
            if args.algo in ("auto", "hdbscan"):
                res = cluster_hdbscan(emb, min_cluster_size=20, prediction_data=False)
                labels = res.labels
                logger.info("HDBSCAN produced %d clusters (+ noise)", int(np.sum(np.unique(labels) >= 0)))
            else:  # spectral
                from core.clustering import cluster_spectral  # local import
                res = cluster_spectral(emb, n_clusters=int(args.n_clusters))
                labels = res.labels
                logger.info("Spectral produced %d clusters", int(np.max(labels) + 1))
        else:
            # NMF baseline on raw graph
            res = cluster_nmf_bipartite(X, n_clusters=int(args.n_clusters), axis="users", init="nndsvda", max_iter=200)
            labels = res.labels
            logger.info("NMF produced %d user clusters", int(np.max(labels) + 1))

    # 2D coordinates
    if emb is None and args.emb_path is not None:
        emb = _load_embeddings(Path(args.emb_path))
    if emb is not None:
        logger.info("Computing 2D projection from embeddings via %s", args.dimred)
        xy = _compute_xy_from_embeddings(emb, method="pca")
    else:
        logger.info("Computing 2D projection from sparse graph via %s", args.dimred)
        xy = _compute_xy_from_sparse(X, method="svd")

    # Subsample for plotting
    xy_plot, lab_plot, idx = _subsample(xy, labels, max_points=int(args.max_points))

    # Compute cluster->top subreddits from ALL users (not subsampled)
    cluster_to_tags = _compute_cluster_labels(X, labels, id_to_sub, topk=int(args.topk), min_count=int(args.min_annotate))

    # Plot
    plt.figure(figsize=tuple(args.figsize))
    ax = plt.gca()
    # noise handling
    mask = lab_plot >= 0
    ax.scatter(xy_plot[~mask, 0], xy_plot[~mask, 1], s=3, c="#c7c7d0", alpha=0.35, linewidths=0, label="noise")

    # color clusters by id modulo a palette
    palette = plt.cm.tab20
    unique = sorted(set(int(v) for v in lab_plot if v >= 0))
    for c in unique:
        pts = xy_plot[lab_plot == c]
        color = palette(c % 20)
        ax.scatter(pts[:, 0], pts[:, 1], s=3, c=[color], alpha=0.8, linewidths=0)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_frame_on(False)

    _place_annotations(ax, xy, labels, cluster_to_tags, min_annotate=int(args.min_annotate))

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=220)
    logger.info("Saved figure to %s", out_path)


if __name__ == "__main__":
    main()
