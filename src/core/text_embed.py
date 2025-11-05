import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
import torch
from litellm import embedding as litellm_embedding
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


def _load_id_maps(id_map_path: str) -> Dict[str, Dict[str, int]]:
    with open(id_map_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_descriptions(parquet_path: str, name_column: Optional[str] = "auto") -> Dict[str, str]:
    df = pl.read_parquet(parquet_path)
    cols = set(df.columns)

    # Resolve subreddit name column
    if name_column is None or name_column == "auto":
        name_col = None
        for cand in ("subreddit", "display_name", "name", "title"):
            if cand in cols:
                name_col = cand
                break
        assert name_col is not None, (
            f"No subreddit name column found. Available columns: {sorted(df.columns)}"
        )
    else:
        assert name_column in cols, (
            f"Requested name column '{name_column}' not in columns: {sorted(df.columns)}"
        )
        name_col = name_column

    # Resolve description column
    desc_col = "description" if "description" in cols else None
    if desc_col is None and "public_description" in cols:
        desc_col = "public_description"
    assert desc_col is not None, (
        f"No description column found. Need one of ['description', 'public_description']."
    )

    subs = df.get_column(name_col).to_list()
    descs = df.get_column(desc_col).to_list()
    out: Dict[str, str] = {}
    for s, d in zip(subs, descs):
        if d is None:
            continue
        d_str = str(d).strip()
        if not d_str:
            continue
        out[str(s)] = d_str
    return out


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))


def embed_subreddit_descriptions(
    desc_parquet: str,
    id_map_json: str,
    out_path: str,
    model: str = "azure/text-embedding-3-large",
    batch_size: int = 128,
    embedding_dim: Optional[int] = 3072,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    api_key: Optional[str] = None,
    name_column: Optional[str] = "auto",
    max_retries: int = 5,
    retry_backoff: float = 2.0,
) -> torch.Tensor:
    """Embed subreddit descriptions via LiteLLM (Azure OpenAI) and align to subreddit ids.

    Missing descriptions are initialized with N(0, 0.02).

    Args:
        desc_parquet: Path to parquet with columns [subreddit, description].
        id_map_json: Path to JSON from graph prep (`id_maps.json`).
        out_path: Destination .pt tensor file; shape [num_subs, dim].
        model: LiteLLM model id, e.g. "azure/<deployment-name>".
        batch_size: Batch size for API calls.
        embedding_dim: Target embedding dim; defaults 3072 for text-embedding-3-large.
        api_base: Azure endpoint; falls back to AZURE_API_BASE.
        api_version: Azure API version; falls back to AZURE_API_VERSION.
        api_key: Azure API key; falls back to AZURE_API_KEY.
        name_column: Column in descriptions parquet with subreddit names, or 'auto'.
        max_retries: Max retries per batch on transient failures.
        retry_backoff: Exponential backoff base seconds.

    Returns:
        Tensor of shape [num_subs, dim], float32, L2-normalized rows for provided descriptions.
    """
    id_maps = _load_id_maps(id_map_json)
    sub_to_id: Dict[str, int] = id_maps["sub_to_id"]
    num_subs = len(sub_to_id)

    descriptions = _load_descriptions(desc_parquet, name_column=name_column)
    if len(descriptions) == 0:
        logger.warning("No descriptions found; all subreddit vectors will be random init.")

    # Prepare items sorted by subreddit id for deterministic filling
    items: List[Tuple[int, str, str]] = []
    hits = 0
    for s, sid in sub_to_id.items():
        if s in descriptions:
            items.append((sid, s, descriptions[s]))
            hits += 1
    if hits == 0:
        logger.warning(
            "No subreddit names matched between id map and descriptions. Check name_column mapping."
        )
    else:
        logger.info("Matched %d/%d subreddits with descriptions (%.2f%%)", hits, len(sub_to_id), 100.0*hits/max(1,len(sub_to_id)))

    # Nothing to embed; return random
    if len(items) == 0:
        dim = int(embedding_dim or 3072)
        emb = torch.randn(num_subs, dim, dtype=torch.float32) * 0.02
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(emb, out_path)
        logger.info("Saved random subreddit embeddings to %s", out_path)
        return emb

    # Resolve Azure params from env if not provided
    api_base = api_base or os.getenv("AZURE_API_BASE")
    api_version = api_version or os.getenv("AZURE_API_VERSION")
    api_key = api_key or os.getenv("AZURE_API_KEY")
    assert api_base and api_version and api_key, (
        "Azure credentials not set. Provide api_base/api_version/api_key or set "
        "AZURE_API_BASE / AZURE_API_VERSION / AZURE_API_KEY"
    )

    dim = int(embedding_dim or 3072)
    emb = torch.randn(num_subs, dim, dtype=torch.float32) * 0.02  # default random init

    # Batch over items
    idx = 0
    pbar = tqdm(total=len(items), desc="Embedding subreddits", unit="sub")
    while idx < len(items):
        batch = items[idx : idx + batch_size]
        indices = [sid for sid, _, _ in batch]
        texts = [t for _, _, t in batch]
        attempt = 0
        while True:
            try:
                resp = litellm_embedding(
                    model=model,
                    input=texts,
                    api_base=api_base,
                    api_version=api_version,
                    api_key=api_key,
                    dimensions=dim,
                    encoding_format="float",
                    timeout=600,
                )
                vecs = [torch.tensor(d["embedding"], dtype=torch.float32) for d in resp["data"]]
                mat = torch.stack(vecs, dim=0)
                mat = _l2_normalize(mat, dim=1)
                emb[indices] = mat
                break
            except Exception as e:  # transient API errors
                attempt += 1
                if attempt > max_retries:
                    logger.error("Embedding failed after retries: %s", e)
                    raise
                sleep_s = retry_backoff ** attempt
                logger.warning("Embedding batch failed (%s). Retrying in %.1fs", e, sleep_s)
                time.sleep(sleep_s)
        idx += batch_size
        pbar.update(len(batch))
    pbar.close()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, out_path)
    logger.info("Saved subreddit text embeddings to %s (num_subs=%d, dim=%d)", out_path, num_subs, dim)
    return emb


