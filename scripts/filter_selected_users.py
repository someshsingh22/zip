import argparse
from pathlib import Path
from typing import Tuple

import polars as pl


def filter_images_by_users(
    users_csv_path: str,
    submissions_csv_glob: str,
    output_csv_path: str,
    min_title_words: int = 3,
) -> Tuple[int, str]:
    """Filter image submissions by selected users from CSVs.

    This function:
    - Reads selected users from a CSV (expects an 'author' column).
    - Reads all submission CSVs matching a glob pattern.
    - Keeps only rows whose 'author' is in the selected users.
    - Drops rows where 'over_18' is true.
    - Drops rows where the title has fewer than `min_title_words` words.

    Args:
        users_csv_path: Path to CSV file with 'author' column for selected users.
        submissions_csv_glob: Glob pattern to match submission CSVs.
        output_csv_path: Path to write the filtered submissions CSV.
        min_title_words: Minimum number of words required in the title.

    Returns:
        Tuple of (num_rows_written, output_path_str).
    """
    # Load users and standardize author dtype
    users_df = (
        pl.read_csv(users_csv_path, columns=["author"])
        .select(pl.col("author").cast(pl.Utf8))
        .unique()
    )
    users_lazy = users_df.lazy()

    # Lazily scan submissions with a glob; standardize dtypes for join/filter
    subs = pl.scan_csv(
        submissions_csv_glob,
        infer_schema_length=10000,
        ignore_errors=True,
        null_values=["", "null", "None", "NaN", "NA", "N/A"],
        dtypes={
            # Force ambiguous/mixed-type columns to strings to avoid parse errors
            "created_utc": pl.Utf8,
            "created_datetime": pl.Utf8,
        },
    )

    filtered_lazy = (
        subs.with_columns(
            [
                pl.col("author").cast(pl.Utf8),
                pl.col("title").cast(pl.Utf8),
                pl.col("over_18").cast(pl.Boolean, strict=False),
            ]
        )
        .join(users_lazy, on="author", how="inner")
        .with_columns(
            [
                pl.col("over_18").fill_null(False).alias("over_18_bool"),
                pl.col("title")
                .fill_null("")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
                .str.split(by=" ")
                .list.len()
                .alias("title_word_count"),
            ]
        )
        .filter(
            (pl.col("over_18_bool") == False)
            & (pl.col("title_word_count") >= min_title_words)
        )
        .drop(["over_18_bool", "title_word_count"])
    )

    # Materialize and write
    out_path = Path(output_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = filtered_lazy.collect(streaming=True)
    result.write_csv(str(out_path))
    return result.height, str(out_path)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Filter image submissions by selected users from visual subreddit CSVs."
    )
    parser.add_argument(
        "--users-csv",
        type=str,
        default="data/reddit/selected_users_v3.csv",
        help="Path to input users CSV with 'author' column.",
    )
    parser.add_argument(
        "--submissions-pattern",
        type=str,
        default="data/reddit/visual_subreddits24/*_submissions_detailed.csv",
        help="Glob pattern to CSVs of submissions with 'author','title','over_18' columns.",
    )
    parser.add_argument(
        "--min-title-words",
        type=int,
        default=3,
        help="Minimum number of words required in the title.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/reddit/selected_users_v3_images_filtered.csv",
        help="Path to write filtered submissions CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n_rows, out_path = filter_images_by_users(
        users_csv_path=args.users_csv,
        submissions_csv_glob=args.submissions_pattern,
        output_csv_path=args.output,
        min_title_words=args.min_title_words,
    )
    print(f"✅ Wrote {n_rows} rows to {out_path}")
