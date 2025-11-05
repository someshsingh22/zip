from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np


@dataclass
class ClusterResult:
    """Container for clustering outputs.

    Attributes:
        algorithm: Name of the algorithm used (e.g., "hdbscan", "spectral", "nmf_users").
        labels: Array of cluster assignments of shape [num_points]. For HDBSCAN, -1 denotes noise.
        extras: Optional algorithm-specific fields (e.g., probabilities, outlier_scores).
    """

    algorithm: str
    labels: np.ndarray
    extras: Optional[Dict[str, np.ndarray]] = None


def cluster_hdbscan(
    embeddings: np.ndarray,
    *,
    min_cluster_size: int = 15,
    min_samples: Optional[int] = None,
    metric: str = "euclidean",
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: Literal["eom", "leaf"] = "eom",
    allow_single_cluster: bool = False,
    prediction_data: bool = False,
) -> ClusterResult:
    """Run HDBSCAN clustering on embeddings.

    Args:
        embeddings: Array of shape [num_points, dim], preferably float32.
        min_cluster_size: The minimum size of clusters; smaller clusters are labeled as noise.
        min_samples: The number of samples in a neighborhood for a point to be a core point.
            If None, defaults to ``min_cluster_size``.
        metric: Distance metric for clustering.
        cluster_selection_epsilon: Adds a constraint for approximate cluster stability.
        cluster_selection_method: Method for cluster selection: "eom" or "leaf".
        allow_single_cluster: If True, allows a single cluster to be selected.
        prediction_data: If True, enables soft clustering probabilities in extras.

    Returns:
        ClusterResult with labels and extras (probabilities, outlier_scores, num_clusters).
    """
    # Lazy import to avoid hard dependency during non-clustering workflows
    import hdbscan  # type: ignore

    assert embeddings.ndim == 2, "embeddings must be 2D [num_points, dim]"

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
        metric=metric,
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=bool(allow_single_cluster),
        prediction_data=bool(prediction_data),
    )
    labels = clusterer.fit_predict(embeddings)

    # Compute cluster count excluding noise (-1)
    unique = np.unique(labels)
    num_clusters = int(np.sum(unique >= 0))

    extras: Dict[str, np.ndarray] = {
        "probabilities": getattr(clusterer, "probabilities_", None),
        "outlier_scores": getattr(clusterer, "outlier_scores_", None),
        "cluster_persistence": getattr(clusterer, "cluster_persistence_", None),
        "condensed_tree": None,
        "num_clusters": np.array([num_clusters], dtype=np.int32),
    }
    # Remove keys with None to keep serialization simple
    extras = {k: v for k, v in extras.items() if v is not None}

    return ClusterResult(algorithm="hdbscan", labels=labels, extras=extras)


def cluster_spectral(
    embeddings: np.ndarray,
    *,
    n_clusters: int,
    affinity: Literal["nearest_neighbors", "rbf"] = "nearest_neighbors",
    n_neighbors: int = 10,
    assign_labels: Literal["kmeans", "discretize"] = "kmeans",
    random_state: int = 42,
) -> ClusterResult:
    """Run Spectral Clustering on embeddings.

    Notes:
        Spectral clustering requires the number of clusters ``n_clusters``. If unknown, set a
        reasonable estimate and adjust based on validation metrics.

    Args:
        embeddings: Array of shape [num_points, dim], preferably float32.
        n_clusters: Number of clusters to find.
        affinity: Affinity type (graph construction): "nearest_neighbors" or "rbf".
        n_neighbors: Neighborhood size when using nearest_neighbors affinity.
        assign_labels: Discretization method: "kmeans" or "discretize".
        random_state: Random seed for the label assignment stage.

    Returns:
        ClusterResult with integer labels in [0, n_clusters-1].
    """
    from sklearn.cluster import SpectralClustering  # type: ignore

    assert embeddings.ndim == 2, "embeddings must be 2D [num_points, dim]"

    model = SpectralClustering(
        n_clusters=int(n_clusters),
        affinity=affinity,
        n_neighbors=int(n_neighbors) if affinity == "nearest_neighbors" else None,
        assign_labels=assign_labels,
        random_state=int(random_state),
    )
    labels = model.fit_predict(embeddings)
    return ClusterResult(algorithm="spectral", labels=labels, extras=None)


def build_bipartite_csr(
    *,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    num_users: int,
    num_subs: int,
) -> Any:
    """Build a CSR user×subreddit matrix from edge list.

    Args:
        edge_index: [2, E] array of (user_id, sub_id) integer indices.
        edge_weight: [E] non-negative weights per edge.
        num_users: Total number of users (rows).
        num_subs: Total number of subreddits (cols).

    Returns:
        CSR sparse matrix shape [num_users, num_subs].
    """
    # Lazy import
    from scipy.sparse import coo_matrix  # type: ignore

    assert edge_index.shape[0] == 2, "edge_index must be [2, E]"
    rows = edge_index[0].astype(np.int64, copy=False)
    cols = edge_index[1].astype(np.int64, copy=False)
    data = np.asarray(edge_weight, dtype=np.float32)
    data = np.maximum(data, 0.0)
    coo = coo_matrix((data, (rows, cols)), shape=(int(num_users), int(num_subs)))
    return coo.tocsr()


def cluster_nmf_bipartite(
    X: Any,
    *,
    n_clusters: int,
    axis: Literal["users", "subs"] = "users",
    init: Literal["nndsvd", "nndsvda", "nndsvdar", "random"] = "nndsvda",
    max_iter: int = 200,
    random_state: int = 42,
    l1_ratio: float = 0.0,
    alpha_W: float = 0.0,
    alpha_H: float = 0.0,
    normalize_W_rows: bool = False,
) -> ClusterResult:
    """Cluster users or subreddits by NMF directly on the bipartite matrix.

    Strategy:
        Factorize X ≈ W @ H, with W∈R^{U×K}, H∈R^{K×S}. Assign clusters via argmax of
        row activations: users from rows of W; subreddits from columns of H.

    Args:
        X: CSR matrix of shape [num_users, num_subs], non-negative.
        n_clusters: Number of components/clusters K.
        axis: "users" to cluster users via W, or "subs" via H^T.
        init: NMF initialization scheme.
        max_iter: Max NMF iterations.
        random_state: Random seed.
        l1_ratio: The regularization mixing parameter.
        alpha_W: L1/L2 regularization parameter for W.
        alpha_H: L1/L2 regularization parameter for H.
        normalize_W_rows: If True, L2-normalize rows of W before argmax.

    Returns:
        ClusterResult with algorithm "nmf_users" or "nmf_subs" and integer labels in [0, K-1].
    """
    from sklearn.decomposition import NMF  # type: ignore

    assert X.ndim == 2, "X must be 2D"
    assert n_clusters >= 2, "n_clusters must be >= 2"

    # Configure NMF; 'mu' solver generally robust for sparse non-negative matrices
    model = NMF(
        n_components=int(n_clusters),
        init=init,
        max_iter=int(max_iter),
        random_state=int(random_state),
        l1_ratio=float(l1_ratio),
        alpha_W=float(alpha_W),
        alpha_H=float(alpha_H),
        solver="mu",
        beta_loss="frobenius",
    )

    W = model.fit_transform(X)  # [U, K]
    H = model.components_  # [K, S]

    if axis == "users":
        A = W
        if normalize_W_rows:
            # Safe L2 normalization of rows
            norms = np.linalg.norm(A, ord=2, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            A = A / norms
        labels = np.argmax(A, axis=1).astype(np.int32)
        return ClusterResult(algorithm="nmf_users", labels=labels, extras=None)

    # axis == "subs": use columns of H (i.e., rows of H.T)
    labels = np.argmax(H.T, axis=1).astype(np.int32)
    return ClusterResult(algorithm="nmf_subs", labels=labels, extras=None)
