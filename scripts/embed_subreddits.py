import argparse
import logging
import os
import sys
from pathlib import Path


# Make src/ importable without installation
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.text_embed import embed_subreddit_descriptions  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("embed_subreddits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed subreddit descriptions with LiteLLM (Azure OpenAI)")
    parser.add_argument("--desc-parquet", type=str, required=True,
                        help="Parquet with columns [subreddit, description]")
    parser.add_argument("--id-map", type=str, required=True, help="Path to id_maps.json")
    parser.add_argument("--out", type=str, required=True, help="Output .pt file for embeddings")
    parser.add_argument("--model", type=str, default="azure/text-embedding-3-large",
                        help="LiteLLM model id, e.g., azure/<deployment-name>")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding batch size")
    parser.add_argument("--dimensions", type=int, default=3072, help="Embedding dimension to request")
    parser.add_argument("--api-base", type=str, default=None, help="Azure API base; defaults to AZURE_API_BASE")
    parser.add_argument("--api-version", type=str, default=None, help="Azure API version; defaults to AZURE_API_VERSION")
    parser.add_argument("--api-key", type=str, default=None, help="Azure API key; defaults to AZURE_API_KEY")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embed_subreddit_descriptions(
        desc_parquet=args.desc_parquet,
        id_map_json=args.id_map,
        out_path=args.out,
        model=args.model,
        batch_size=int(args.batch_size),
        embedding_dim=int(args.dimensions),
        api_base=args.api_base,
        api_version=args.api_version,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()


