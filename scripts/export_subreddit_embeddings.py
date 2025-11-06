import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("export_subreddit_embeddings")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for exporting subreddit embeddings.

    Returns:
        argparse.Namespace: Parsed arguments with paths.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Export subreddit embeddings as a .npy pickled dict "
            "{'subreddit_name': np.array(embedding_dim,)} suitable for "
            "np.load(path, allow_pickle=True).item()"
        )
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        default="/dev/shm/zip/data/processed/graph.pt",
        help="Path to graph payload with 'id_to_sub' and 'num_subs' (default: %(default)s)",
    )
    parser.add_argument(
        "--emb-path",
        type=str,
        default="/dev/shm/zip/data/processed/sub_text_emb.pt",
        help=(
            "Path to subreddit embeddings .pt (Tensor or dict with key 'embedding') "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/dev/shm/zip/data/processed/subreddit_embeddings.npy",
        help="Output .npy path for pickled dict (default: %(default)s)",
    )
    return parser.parse_args()


def _load_id_to_sub(graph_path: Path) -> List[str]:
    """Load subreddit index-to-name mapping from the graph payload.

    Args:
        graph_path (Path): Path to the graph .pt payload.

    Returns:
        List[str]: List mapping subreddit_id -> subreddit_name.
    """
    payload: Dict = torch.load(str(graph_path), map_location="cpu")
    assert "id_to_sub" in payload, "Graph payload missing 'id_to_sub'."
    id_to_sub: List[str] = list(map(str, payload["id_to_sub"]))
    assert "num_subs" in payload, "Graph payload missing 'num_subs'."
    num_subs = int(payload["num_subs"])
    assert len(id_to_sub) == num_subs, "len(id_to_sub) must equal payload['num_subs']."
    return id_to_sub


def _load_sub_embeddings(emb_path: Path) -> np.ndarray:
    """Load subreddit embeddings as a numpy matrix [num_subs, dim].

    Supports:
      - .pt torch.Tensor
      - .pt dict with key 'embedding'

    Args:
        emb_path (Path): Path to embeddings file.

    Returns:
        np.ndarray: Float32 array of shape [num_subs, dim].
    """
    obj = torch.load(str(emb_path), map_location="cpu")
    if isinstance(obj, dict) and "embedding" in obj:
        obj = obj["embedding"]
    if hasattr(obj, "detach"):
        arr = obj.detach().cpu().numpy()
    else:
        raise AssertionError(".pt must be a Tensor or dict with key 'embedding'.")
    assert arr.ndim == 2, "Embeddings must be 2D [num_subs, dim]."
    return arr.astype(np.float32, copy=False)


def build_name_to_vec(
    id_to_sub: List[str], embeddings: np.ndarray
) -> Dict[str, np.ndarray]:
    """Build mapping from subreddit name to its embedding vector.

    Args:
        id_to_sub (List[str]): Mapping subreddit_id -> subreddit_name.
        embeddings (np.ndarray): Embeddings matrix [num_subs, dim].

    Returns:
        Dict[str, np.ndarray]: Dict of {'subreddit_name': np.ndarray(dim,)}.
    """
    num_subs = len(id_to_sub)
    assert embeddings.shape[0] == num_subs, (
        f"Embeddings rows ({embeddings.shape[0]}) must match len(id_to_sub) ({num_subs})."
    )
    out: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(id_to_sub):
        vec = embeddings[idx]
        assert vec.ndim == 1, "Each embedding row should be 1D."
        out[name] = vec
    return out


def main() -> None:
    """Entry point: export subreddit embeddings to .npy pickled dict."""
    args = parse_args()
    graph_path = Path(args.graph_path)
    emb_path = Path(args.emb_path)
    out_path = Path(args.out)

    logger.info("Loading id_to_sub from %s", graph_path)
    id_to_sub = _load_id_to_sub(graph_path)

    logger.info("Loading subreddit embeddings from %s", emb_path)
    emb = _load_sub_embeddings(emb_path)
    logger.info("Embeddings shape: %s", tuple(emb.shape))

    # Optional: sanity check for 3072-d vectors (if using text-embedding-3-large)
    if emb.shape[1] != 3072:
        logger.warning(
            "Expected 3072-d embeddings; got %d. Proceeding anyway.", emb.shape[1]
        )

    logger.info("Building name->vector mapping")
    name_to_vec = build_name_to_vec(id_to_sub, emb)
    logger.info("Built mapping for %d subreddits", len(name_to_vec))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving pickled dict to %s", out_path)
    np.save(str(out_path), name_to_vec)
    logger.info(
        "Done. Load with: np.load('%s', allow_pickle=True).item()", out_path.as_posix()
    )


if __name__ == "__main__":
    main()


