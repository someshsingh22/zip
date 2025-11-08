"""
dataset_creation.py
-------------------
Creates the HeteroData object with:
- 'user' and 'subreddit' node types
- One unified edge type: 'interacts'
- Edge features per (user, subreddit):
    [norm_log_total_count, comment_frac]
Saves to disk as dataset.pt for reuse.
"""

import polars as pl
import numpy as np
import torch
from torch_geometric.data import HeteroData

POSTS_FILE = "data/merged_submissions_filtered_gt1_dsp.parquet"
COMMENTS_FILE = "data/merged_comments_filtered_3x3_dsp.parquet"
SUBREDDIT_EMBED_FILE = "data/processed/subreddit_embeddings.npy"  # or .pkl/.json etc.
OUTPUT_FILE = "dataset.pt"

print("📦 Loading Parquet files...")
posts_df = pl.read_parquet(POSTS_FILE)
comments_df = pl.read_parquet(COMMENTS_FILE)

# ============================================================
# CREATE INDEPENDENT CATEGORICAL ENCODINGS
# ============================================================
# Users
print("🔢 Encoding users and subreddits...")
all_users = pl.concat([posts_df["author"], comments_df["author"]]).unique()
user_map = pl.DataFrame({
    "author": all_users,
    "user_id": pl.arange(0, all_users.len(), eager=True, dtype=pl.Int32)
})

# Subreddits
all_subs = pl.concat([posts_df["subreddit"], comments_df["subreddit"]]).unique()
sub_map = pl.DataFrame({
    "subreddit": all_subs,
    "sub_id": pl.arange(0, all_subs.len(), eager=True, dtype=pl.Int32)
})

n_users = all_users.len()
n_subs = all_subs.len()
print(f"✅ Users: {n_users:,}, Subreddits: {n_subs:,}")

# ============================================================
# MAP TO INTEGER INDICES AND COMBINE EDGE TYPES
# ============================================================
def _map_edges(df: pl.DataFrame, count_col: str) -> pl.DataFrame:
    return (
        df.join(user_map, on="author", how="inner")
          .join(sub_map, on="subreddit", how="inner")
          .select([
              pl.col("user_id").alias("user_id"),
              pl.col("sub_id").alias("sub_id"),
              pl.col("total_count").alias(count_col),
          ])
    )

print("🧩 Mapping post and comment edges...")
post_edges = _map_edges(posts_df, "post_count")
comm_edges = _map_edges(comments_df, "comment_count")

# Outer join to include pairs that appear in only one source
combined = post_edges.join(comm_edges, on=["user_id", "sub_id"], how="outer")
combined = combined.fill_null(0)

# Compute totals and fractions
combined = combined.with_columns([
    (pl.col("post_count") + pl.col("comment_count")).alias("total_count"),
])
combined = combined.with_columns([
    pl.when(pl.col("total_count") > 0)
      .then(pl.col("comment_count") / pl.col("total_count"))
      .otherwise(0.0)
      .alias("comment_frac"),
])

# Normalize log(1 + total_count) by sqrt user and subreddit activity
user_activity = (
    combined
    .group_by("user_id")
    .agg(pl.col("total_count").sum().alias("user_total"))
)
sub_activity = (
    combined
    .group_by("sub_id")
    .agg(pl.col("total_count").sum().alias("sub_total"))
)
combined = (
    combined
    .join(user_activity, on="user_id", how="left")
    .join(sub_activity, on="sub_id", how="left")
)
combined = combined.with_columns([
    ((pl.col("user_total").sqrt() * pl.col("sub_total").sqrt())).alias("denom"),
])
combined = combined.with_columns([
    (
        (pl.col("total_count") + 1.0).log()
        / pl.when(pl.col("denom") > 0.0).then(pl.col("denom")).otherwise(1e-12)
    ).alias("norm_log_total_count"),
])

# Filter out any zero-total edges (optional but keeps graph compact)
combined = combined.filter(pl.col("total_count") > 0)

u = torch.tensor(combined["user_id"].to_numpy(), dtype=torch.long)
s = torch.tensor(combined["sub_id"].to_numpy(), dtype=torch.long)
edge_attr_np = np.stack(
    [
        combined["norm_log_total_count"].to_numpy(),
        combined["comment_frac"].to_numpy(),
    ],
    axis=1,
).astype(np.float32)
edge_attr = torch.from_numpy(edge_attr_np)
print(f"🧩 interacts: {u.numel():,} edges (features: norm_log_total_count, comment_frac)")

# ============================================================
# LOAD SUBREDDIT EMBEDDINGS
# ============================================================
print("🧠 Loading subreddit embeddings...")
subreddit_embeddings = np.load(SUBREDDIT_EMBED_FILE, allow_pickle=True).item()

# Convert sub IDs → embeddings (fill missing with zeros)
sub_embs = torch.zeros((n_subs, 3072), dtype=torch.bfloat16)
missing = 0
for i, sub in enumerate(all_subs):
    emb = subreddit_embeddings.get(sub, None)
    if emb is not None:
        sub_embs[i] = torch.tensor(emb, dtype=torch.bfloat16)
    else:
        missing += 1

if missing:
    print(f"⚠️ {missing:,} subreddits missing embeddings (users, likely); filled with zeros.")

# ============================================================
# BUILD GRAPH OBJECT
# ============================================================
print("🔗 Building PyG HeteroData...")
data = HeteroData()
data["user"].num_nodes = n_users
data["subreddit"].x = sub_embs  # [n_subs, 3072]

data["user", "interacts", "subreddit"].edge_index = torch.stack([u, s])
data["user", "interacts", "subreddit"].edge_attr = edge_attr

data["subreddit", "rev_interacts", "user"].edge_index = data["user", "interacts", "subreddit"].edge_index.flip(0)
data["subreddit", "rev_interacts", "user"].edge_attr = data["user", "interacts", "subreddit"].edge_attr

# ============================================================
# SAVE
# ============================================================
torch.save(data, OUTPUT_FILE)
print(f"✅ Saved dataset: {OUTPUT_FILE}")