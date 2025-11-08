import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from litellm import embedding as litellm_embedding
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


def _l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalize a matrix [N, D]."""
    norms = np.linalg.norm(mat, ord=2, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms


def _load_parquet(
    parquet_path: str,
    name_column: str,
    text_column: str,
) -> List[Tuple[str, str]]:
    """Load (display_name, description) pairs from parquet and clean rows."""
    df = pd.read_parquet(parquet_path)
    assert name_column in df.columns, f"Missing column: {name_column}"
    assert text_column in df.columns, f"Missing column: {text_column}"

    df = df[[name_column, text_column]].copy()
    df[name_column] = df[name_column].astype(str)
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column].str.len() > 0]
    df = df.drop_duplicates(subset=[name_column], keep="first")
    rows: List[Tuple[str, str]] = list(df.itertuples(index=False, name=None))
    return rows


def embed_and_save(
    parquet_path: str,
    output_path: str,
    name_column: str = "display_name",
    text_column: str = "description",
    model: str = "openai/text-embedding-3-large",
    batch_size: int = 128,
    dimensions: int | None = None,
    max_retries: int = 5,
    retry_backoff: float = 2.0,
) -> None:
    """Embed subreddit descriptions using LiteLLM/OpenAI and save name -> embedding dict.

    No manual truncation is applied to inputs.
    """
    pairs = _load_parquet(parquet_path, name_column, text_column)
    if len(pairs) == 0:
        logger.warning(
            "No rows with non-empty '%s' found in %s", text_column, parquet_path
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, {}, allow_pickle=True)
        return

    names = [n for n, _ in pairs]
    texts = [t for _, t in pairs]

    all_vecs: List[np.ndarray] = []
    pbar = tqdm(total=len(texts), desc="Embedding descriptions (OpenAI)", unit="desc")
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

    mat = np.concatenate(all_vecs, axis=0)
    assert mat.shape[0] == len(names)
    name_to_vec: Dict[str, np.ndarray] = {n: mat[i] for i, n in enumerate(names)}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, name_to_vec, allow_pickle=True)
    logger.info(
        "Saved %d subreddit embeddings to %s (dim=%d)",
        len(name_to_vec),
        output_path,
        mat.shape[1],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed subreddit descriptions with OpenAI via LiteLLM and save display_name -> embedding."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/reddit/subreddit_descriptions/filtered_subreddits_descriptions_v2.parquet",
        help="Path to input parquet with `display_name` and `description`.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/processed/subreddit_embeddings.npy",
        help="Output .npy file (np.save of dict[name] = embedding).",
    )
    parser.add_argument(
        "--name-column",
        type=str,
        default="display_name",
        help="Name column in parquet.",
    )
    parser.add_argument(
        "--text-column", type=str, default="description", help="Text column in parquet."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/text-embedding-3-large",
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
    embed_and_save(
        parquet_path=args.input,
        output_path=args.output,
        name_column=args.name_column,
        text_column=args.text_column,
        model=args.model,
        batch_size=args.batch_size,
        dimensions=args.dimensions,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
    )
