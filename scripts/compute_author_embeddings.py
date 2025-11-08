import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Sequence

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
from multiprocessing import Process
import os
import json


def mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean pool token embeddings using attention mask, then L2-normalize.

    Args:
        last_hidden_state: Tensor of shape [B, T, H].
        attention_mask: Attention mask of shape [B, T].

    Returns:
        Tensor of shape [B, H], L2-normalized.
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # [B,T,1]
    summed = (last_hidden_state * mask).sum(dim=1)  # [B,H]
    counts = mask.sum(dim=1).clamp(min=1e-6)  # [B,1]
    emb = summed / counts  # [B,H]
    return torch.nn.functional.normalize(emb, dim=-1)


def encode_texts(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Encode a list of texts into sentence embeddings with mean pooling.

    Args:
        texts: List of strings.
        model_name: HuggingFace model name.
        batch_size: Batch size for encoding.
        device: Torch device string, e.g. 'cuda:0' or 'cpu'.

    Returns:
        Float tensor of shape [N, D], L2-normalized.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    outputs: List[torch.Tensor] = []
    with torch.inference_mode():
        pbar = tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False)
        for i in pbar:
            chunk = texts[i : i + batch_size]
            toks = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            hidden = model(**toks).last_hidden_state
            pooled = mean_pool(hidden, toks["attention_mask"]).float()
            outputs.append(pooled)
    return torch.cat(outputs, dim=0)


def shard_list(items: Sequence[str], num_shards: int) -> List[List[str]]:
    """Split items into num_shards contiguous shards with roughly equal sizes."""
    n = len(items)
    base = n // num_shards
    rem = n % num_shards
    shards: List[List[str]] = []
    start = 0
    for s in range(num_shards):
        extra = 1 if s < rem else 0
        end = start + base + extra
        shards.append(list(items[start:end]))
        start = end
    return shards


def worker_process(
    rank: int,
    authors: List[str],
    author_to_titles: Dict[str, List[str]],
    gpu_id: int,
    model_name: str,
    batch_size: int,
    tmp_dir: Path,
) -> None:
    """Worker process: encode per-author titles into a single embedding and save a shard result.

    Each author's embedding is the mean over up to K (already capped) title embeddings, L2-normalized.
    A per-process JSON manifest is also written with counts for quick inspection.
    """
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    # Affinitize to a single GPU via environment to be safe with certain backends
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    results: Dict[str, np.ndarray] = {}
    manifest: Dict[str, int] = {}

    for author in tqdm(
        authors, desc=f"Rank {rank} (GPU {gpu_id})", position=rank, leave=True
    ):
        titles = author_to_titles.get(author, [])
        if not titles:
            continue
        # Encode in chunks to manage memory
        embs: List[torch.Tensor] = []
        with torch.inference_mode():
            for i in range(0, len(titles), batch_size):
                chunk = titles[i : i + batch_size]
                toks = tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt",
                )
                toks = {k: v.to(device) for k, v in toks.items()}
                hidden = model(**toks).last_hidden_state
                pooled = mean_pool(hidden, toks["attention_mask"]).float()  # [b, d]
                embs.append(pooled)
        all_embs = torch.cat(embs, dim=0)  # [n_titles, d]
        # Mean across titles, then L2-normalize
        author_emb = torch.nn.functional.normalize(
            all_embs.mean(dim=0, keepdim=True), dim=-1
        ).squeeze(0)
        results[author] = author_emb.detach().cpu().numpy().astype(np.float32)
        manifest[author] = len(titles)

    tmp_dir.mkdir(parents=True, exist_ok=True)
    shard_path = tmp_dir / f"shard_{rank}.npy"
    meta_path = tmp_dir / f"shard_{rank}.json"
    # Save as a numpy "pickle" (dict object in npy)
    np.save(str(shard_path), results, allow_pickle=True)
    with meta_path.open("w") as f:
        json.dump(manifest, f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-author sentence embeddings with multi-GPU multiprocessing."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/reddit/selected_users_v3_images_filtered_classified.csv",
        help="CSV file with at least columns: author,title",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/reddit/author_title_embeddings.npy",
        help="Output NumPy pickle (np.save with allow_pickle=True) mapping author -> embedding (float32).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model name for sentence embeddings.",
    )
    parser.add_argument(
        "--num-procs", type=int, default=8, help="Number of worker processes."
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=list(range(8)),
        help="GPU IDs to use; one per process (length must equal num-procs).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="Encoding batch size per process."
    )
    parser.add_argument(
        "--cap-per-user", type=int, default=50, help="Max number of titles per author."
    )
    parser.add_argument(
        "--seed", type=int, default=1337, help="Random seed for per-author sampling."
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default="data/reddit/author_emb_tmp",
        help="Temporary directory for shard outputs.",
    )
    args = parser.parse_args()

    assert args.num_procs > 0, "num-procs must be > 0"
    assert len(args.gpu_ids) == args.num_procs, "gpu-ids length must equal num-procs"

    # Load and group titles per author; cap per-user titles deterministically.
    df = pl.read_csv(args.input).select(["author", "title"]).drop_nulls()
    grouped = df.group_by("author").agg(pl.col("title").alias("titles"))

    rng = np.random.default_rng(args.seed)
    author_to_titles: Dict[str, List[str]] = {}
    authors: List[str] = []
    for row in grouped.iter_rows(named=True):
        author = str(row["author"])
        titles_list = list(row["titles"])
        if not titles_list:
            continue
        if len(titles_list) > args.cap_per_user:
            # Deterministic shuffle then take first K
            idx = rng.permutation(len(titles_list))[: args.cap_per_user]
            titles_list = [titles_list[i] for i in idx]
        authors.append(author)
        author_to_titles[author] = titles_list

    # Shard authors and launch processes
    shards = shard_list(authors, args.num_procs)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    procs: List[Process] = []
    for rank in range(args.num_procs):
        p = Process(
            target=worker_process,
            args=(
                rank,
                shards[rank],
                author_to_titles,
                args.gpu_ids[rank],
                args.model_name,
                args.batch_size,
                tmp_dir,
            ),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # Merge shards into a single dict and save
    merged: Dict[str, np.ndarray] = {}
    for rank in range(args.num_procs):
        shard_path = tmp_dir / f"shard_{rank}.npy"
        if shard_path.exists():
            part: Dict[str, np.ndarray] = np.load(
                str(shard_path), allow_pickle=True
            ).item()
            merged.update(part)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), merged, allow_pickle=True)
    print(f"✅ Saved {len(merged)} author embeddings to {out_path}")


if __name__ == "__main__":
    main()
