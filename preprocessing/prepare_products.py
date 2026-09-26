from datasets import load_dataset
from huggingface_hub import hf_hub_url
import pandas as pd
import json
import os


# ============================================================
# SHOPGRAPH - AMAZON ELECTRONICS DATA PREPARATION
# ============================================================

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"

FILE = (
    "raw_meta_Electronics/"
    "full-00000-of-00010.parquet"
)

MAX_PRODUCTS = 100_000

OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "products_100k.parquet"
)

STATS_FILE = os.path.join(
    OUTPUT_DIR,
    "dataset_stats.json"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CREATE REMOTE HUGGING FACE URL
# ============================================================

remote_url = hf_hub_url(
    repo_id=REPO_ID,
    filename=FILE,
    repo_type="dataset",
)

print("=" * 70)
print("SHOPGRAPH - AMAZON ELECTRONICS DATA PREPARATION")
print("=" * 70)

print("\nRemote dataset:")
print(remote_url)

print("\nTarget products:", MAX_PRODUCTS)

print("\nConnecting to remote Parquet file...")


# ============================================================
# STREAM DATASET
# ============================================================

dataset = load_dataset(
    "parquet",
    data_files=remote_url,
    split="train",
    streaming=True,
)

print("Connection successful!")

print("\nCollecting products...")


# ============================================================
# COLLECT FIRST 100,000 PRODUCTS
# ============================================================

products = []

for i, product in enumerate(dataset):

    products.append(product)

    if (i + 1) % 10_000 == 0:
        print(f"Collected {i + 1:,} products")

    if i + 1 >= MAX_PRODUCTS:
        break


print("\nCollection complete.")

print("Products collected:", len(products))


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

df = pd.DataFrame(products)

print("\nOriginal dataframe shape:")
print(df.shape)


# ============================================================
# REMOVE DUPLICATE PRODUCTS
# ============================================================

if "parent_asin" in df.columns:

    before = len(df)

    df = df.drop_duplicates(
        subset=["parent_asin"]
    ).reset_index(drop=True)

    after = len(df)

    print("\nDuplicate removal:")
    print("Before:", before)
    print("After :", after)
    print("Removed:", before - after)


# ============================================================
# SAVE DATASET
# ============================================================

df.to_parquet(
    OUTPUT_FILE,
    index=False
)

print("\nDataset saved to:")
print(OUTPUT_FILE)


# ============================================================
# DATASET STATISTICS
# ============================================================

stats = {
    "dataset": "Amazon Reviews 2023",
    "category": "Electronics",
    "source": "McAuley-Lab/Amazon-Reviews-2023",
    "source_file": FILE,
    "products_collected": int(len(products)),
    "products_after_deduplication": int(len(df)),
    "columns": list(df.columns),
}


# ============================================================
# SAVE STATISTICS
# ============================================================

with open(
    STATS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        stats,
        f,
        indent=4
    )


# ============================================================
# DISPLAY SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print("\nRows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")

for column in df.columns:
    print(" -", column)


print("\nFirst 3 products:")

print(
    df[
        [
            "title",
            "average_rating",
            "rating_number",
            "store",
            "parent_asin",
        ]
    ].head(3).to_string()
)


print("\nStatistics saved to:")
print(STATS_FILE)

print("\n" + "=" * 70)
print("PHASE 1 DATA COLLECTION COMPLETE")
print("=" * 70)