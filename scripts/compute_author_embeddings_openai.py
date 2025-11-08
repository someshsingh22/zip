import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import polars as pl
from litellm import embedding as litellm_embedding
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


def _l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalize a matrix [N, D].

    Args:
        mat: Input matrix of shape [N, D].
        eps: Numerical stability epsilon.

    Returns:
        Row-normalized matrix of same shape [N, D].
    """
    norms = np.linalg.norm(mat, ord=2, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms


def _batch_embed_texts(
    texts: Sequence[str],
    model: str,
    batch_size: int,
    dimensions: int | None = None,
    max_retries: int = 5,
    retry_backoff: float = 2.0,
) -> np.ndarray:
    """Embed a list of texts via LiteLLM/OpenAI with batching and retries.

    Args:
        texts: Sequence of input texts.
        model: LiteLLM model identifier (e.g., 'openai/text-embedding-3-large').
        batch_size: Number of texts per API call.
        dimensions: Optional target dimensions if supported by the model.
        max_retries: Maximum number of retries per batch.
        retry_backoff: Exponential backoff base seconds.

    Returns:
        Array of shape [N, D] with L2-normalized embeddings (float32).
    """
    if len(texts) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    all_vecs: List[np.ndarray] = []
    pbar = tqdm(total=len(texts), desc="Embedding titles (OpenAI)", unit="titles")
    idx = 0
    while idx < len(texts):
        batch_texts = texts[idx : idx + batch_size]
        attempt = 0
        while True:
            try:
                kwargs = dict(
                    model=model,
                    input=batch_texts,
                    encoding_format="float",
                    timeout=600,
                )
                if dimensions is not None:
                    kwargs["dimensions"] = int(dimensions)
                resp = litellm_embedding(**kwargs)
                vecs = [
                    np.asarray(d["embedding"], dtype=np.float32) for d in resp["data"]
                ]
                mat = np.stack(vecs, axis=0)
                mat = _l2_normalize(mat)
                all_vecs.append(mat)
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    raise
                sleep_s = retry_backoff**attempt
                logger.warning("Batch embed failed (%s). Retrying in %.1fs", e, sleep_s)
                time.sleep(sleep_s)
        pbar.update(len(batch_texts))
        idx += batch_size
    pbar.close()

    out = np.concatenate(all_vecs, axis=0)
    return out


def _prepare_author_spans(
    csv_path: str,
    name_column: str,
    text_column: str,
    cap_per_user: int,
    seed: int,
) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:
    """Load CSV and construct flat title list plus per-author spans.

    Reads CSV using polars, groups by `name_column`, collects `text_column` titles,
    caps each author's titles deterministically, and returns a flattened list of
    titles with span indices for each author.

    Args:
        csv_path: Path to CSV with at least `name_column` and `text_column`.
        name_column: Author column name, e.g., 'author'.
        text_column: Title/text column name, e.g., 'title'.
        cap_per_user: Maximum number of titles per author to include.
        seed: Random seed for deterministic sampling.

    Returns:
        authors: List of author ids in the same order as spans.
        flat_texts: Flattened list of titles (concatenation across all authors).
        spans: List of (start_idx, end_idx) per author into flat_texts.
    """
    df = pl.read_csv(csv_path).select([name_column, text_column]).drop_nulls()
    grouped = df.group_by(name_column).agg(pl.col(text_column).alias("titles"))

    rng = np.random.default_rng(seed)
    authors: List[str] = []
    flat_texts: List[str] = []
    spans: List[Tuple[int, int]] = []

    for row in grouped.iter_rows(named=True):
        author = str(row[name_column])
        titles_list = list(row["titles"])
        titles_list = [
            str(t).strip()
            for t in titles_list
            if isinstance(t, str) and str(t).strip() != ""
        ]
        if not titles_list:
            continue
        if len(titles_list) > cap_per_user:
            idx = rng.permutation(len(titles_list))[:cap_per_user]
            titles_list = [titles_list[i] for i in idx]
        start = len(flat_texts)
        flat_texts.extend(titles_list)
        end = len(flat_texts)
        if end > start:
            authors.append(author)
            spans.append((start, end))

    return authors, flat_texts, spans


def embed_author_titles_and_save(
    input_csv: str,
    output_path: str,
    name_column: str = "author",
    text_column: str = "title",
    model: str = "openai/text-embedding-3-large",
    batch_size: int = 128,
    dimensions: int | None = None,
    cap_per_user: int = 50,
    seed: int = 1337,
    max_retries: int = 5,
    retry_backoff: float = 2.0,
) -> None:
    """Embed per-author titles with OpenAI and save author -> embedding dict.

    The per-author embedding is the mean of the author's title embeddings,
    followed by L2 normalization.

    Args:
        input_csv: Path to CSV containing `name_column` and `text_column`.
        output_path: Destination path for np.save(dict[author] = embedding).
        name_column: Column name for author/user id (default: 'author').
        text_column: Column name for text/title (default: 'title').
        model: LiteLLM model identifier for embeddings.
        batch_size: API batch size for embedding calls.
        dimensions: Optional target dimensions where supported.
        cap_per_user: Max titles used per author.
        seed: Random seed for deterministic sampling.
        max_retries: Max retries per batch on transient failures.
        retry_backoff: Exponential backoff base seconds.
    """
    authors, flat_texts, spans = _prepare_author_spans(
        csv_path=input_csv,
        name_column=name_column,
        text_column=text_column,
        cap_per_user=cap_per_user,
        seed=seed,
    )
    if len(authors) == 0:
        logger.warning(
            "No authors with non-empty '%s' found in %s", text_column, input_csv
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, {}, allow_pickle=True)
        return

    emb_mat = _batch_embed_texts(
        texts=flat_texts,
        model=model,
        batch_size=batch_size,
        dimensions=dimensions,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
    )
    assert emb_mat.shape[0] == len(
        flat_texts
    ), "Mismatch between embeddings and input texts"

    author_to_vec: Dict[str, np.ndarray] = {}
    for author, (start, end) in zip(authors, spans):
        vecs = emb_mat[start:end]  # [k, d]
        mean_vec = vecs.mean(axis=0, keepdims=True)  # [1, d]
        mean_vec = _l2_normalize(mean_vec).astype(np.float32)[0]
        author_to_vec[author] = mean_vec

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, author_to_vec, allow_pickle=True)
    logger.info(
        "Saved %d author embeddings to %s (dim=%d)",
        len(author_to_vec),
        output_path,
        next(iter(author_to_vec.values())).shape[0] if author_to_vec else -1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-author embeddings by averaging title embeddings via OpenAI (LiteLLM)."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/reddit/selected_users_v3_images_filtered_classified.csv",
        help="Input CSV with columns [author, title] by default.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/reddit/author_title_embeddings_openai.npy",
        help="Output .npy file mapping author -> embedding (float32).",
    )
    parser.add_argument(
        "--name-column",
        type=str,
        default="author",
        help="Author/user id column in CSV.",
    )
    parser.add_argument(
        "--text-column", type=str, default="title", help="Title/text column in CSV."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="azure/text-embedding-3-large",
        help="LiteLLM model id, e.g., openai/text-embedding-3-large",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for embedding API calls.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Optional target embedding dimension if the model supports it.",
    )
    parser.add_argument(
        "--cap-per-user", type=int, default=50, help="Max number of titles per author."
    )
    parser.add_argument(
        "--seed", type=int, default=1337, help="Random seed for per-author sampling."
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries per batch on transient failures.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=2.0,
        help="Exponential backoff base seconds.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    embed_author_titles_and_save(
        input_csv=args.input,
        output_path=args.output,
        name_column=args.name_column,
        text_column=args.text_column,
        model=args.model,
        batch_size=args.batch_size,
        dimensions=args.dimensions,
        cap_per_user=args.cap_per_user,
        seed=args.seed,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
    )
