# Reddit BiNE Embeddings with Power–TFIDF and LiteLLM/Azure Init (8×A100)

## Overview
Train bipartite embeddings for users and subreddits from interaction counts. Edge
weights are power-law compressed then TF–IDF weighted. Subreddit embedding table is
initialized from Azure OpenAI via LiteLLM `text-embedding-3-large` (3072-dim). Training uses DDP on 8×A100.

## Environment
- Python >= 3.10
- Install deps (recommended: uv)

```bash
uv sync
uv run python -V
```

## Data
- Counts parquet: columns `author, subreddit, total_count`
- Subreddit descriptions parquet: `data/reddit/subreddit_descriptions/filtered_subreddits_descriptions_v2.parquet`

## Commands
Prepare weighted graph:
```bash
uv run python scripts/prepare_graph.py \
  --counts-parquet data/merged_submissions_filtered_gt1.parquet \
  --out data/processed/graph.pt \
  --alpha 0.75 --tfidf-smooth 1.0
```

Build subreddit text embeddings (LiteLLM + Azure OpenAI):
```bash
uv run python scripts/embed_subreddits.py \
  --desc-parquet data/reddit/subreddit_descriptions/filtered_subreddits_descriptions_v2.parquet \
  --id-map data/processed/id_maps.json \
  --out data/processed/sub_text_emb.pt \
  --model azure/text-embedding-3-large \
  --dimensions 3072 \
  --batch-size 128
```

Train with 8 GPUs:

GraphSAGE (heterogeneous link prediction with neighbor sampling):
```bash
uv run python scripts/train_sageconv.py --config configs/train_sageconv.yaml
```

MetaPath2Vec (unsupervised random-walk embeddings):
```bash
uv run python scripts/train_metapath2vec.py --config configs/train_metapath2vec.yaml
```

## Config
See `configs/train_bine.yaml` for BiNE and the new `configs/train_sageconv.yaml`, `configs/train_metapath2vec.yaml` for PyG baselines.

## Outputs
- Checkpoints under `data/processed/checkpoints/`
- Final embeddings: `user_emb.pt`, `sub_emb.pt`
- ID maps: `id_maps.json`

New baselines:
- GraphSAGE checkpoint: `data/processed/sageconv_latest.pt` (contains input embeddings and model state)
- MetaPath2Vec checkpoint: `data/processed/metapath2vec_latest.pt`

## Reproducibility
- Fixed seed, saved config, and deterministic mappings.
- WandB logging (set WANDB_PROJECT via config).

## Azure environment variables
- AZURE_API_BASE
- AZURE_API_VERSION
- AZURE_API_KEY

We use LiteLLM embeddings: see docs for supported embedding providers and Azure configuration.
- Supported embeddings: [docs.litellm.ai/docs/embedding/supported_embedding](https://docs.litellm.ai/docs/embedding/supported_embedding)
- Azure provider: [docs.litellm.ai/docs/providers/azure](https://docs.litellm.ai/docs/providers/azure)