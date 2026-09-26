"""
ShopGraph - Fixed Amazon Electronics Metadata Extraction

Uses the current Parquet files from the Hugging Face repository
instead of the deprecated Amazon-Reviews-2023 dataset script.

Creates:
    data/processed/products_100k_fixed.parquet

Does NOT modify existing embeddings or FAISS.
"""

from pathlib import Path
import ast
import sys

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = (
    "https://huggingface.co/datasets/"
    "McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw_meta_Electronics/"
)

SHARD_COUNT = 10
MAX_PRODUCTS = 100_000

OUTPUT_DIR = Path("data") / "processed"
OUTPUT_FILE = OUTPUT_DIR / "products_100k_fixed.parquet"


# ============================================================
# HELPERS
# ============================================================

def is_missing(value):
    if value is None:
        return True

    try:
        result = pd.isna(value)

        if isinstance(result, bool):
            return result

    except (TypeError, ValueError):
        pass

    return False


def clean_string(value):
    if value is None or is_missing(value):
        return None

    if isinstance(value, str):
        value = value.strip()
        return value if value else None

    if isinstance(value, dict):
        parts = []

        for key, val in value.items():
            key_text = clean_string(key)
            val_text = clean_string(val)

            if key_text and val_text:
                parts.append(f"{key_text}: {val_text}")
            elif val_text:
                parts.append(val_text)

        return " | ".join(parts) if parts else None

    if isinstance(value, (list, tuple)):
        parts = []

        for item in value:
            text = clean_string(item)

            if text:
                parts.append(text)

        return " | ".join(parts) if parts else None

    if hasattr(value, "tolist"):
        try:
            return clean_string(value.tolist())
        except Exception:
            pass

    text = str(value).strip()

    return text if text else None


def normalize_list(value):
    if value is None or is_missing(value):
        return []

    if isinstance(value, (list, tuple)):
        result = []

        for item in value:
            text = clean_string(item)

            if text:
                result.append(text)

        return result

    if isinstance(value, str):

        text = value.strip()

        if not text:
            return []

        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)

                if isinstance(parsed, (list, tuple)):
                    return normalize_list(parsed)

            except Exception:
                pass

        return [text]

    return [str(value).strip()]


def normalize_number(value, integer=False):

    if value is None or is_missing(value):
        return None

    try:

        number = float(value)

        if integer:
            return int(number)

        return number

    except (TypeError, ValueError):

        text = str(value).strip()

        if not text:
            return None

        text = (
            text
            .replace("$", "")
            .replace(",", "")
            .strip()
        )

        try:

            number = float(text)

            if integer:
                return int(number)

            return number

        except ValueError:
            return None


# ============================================================
# NORMALIZE RAW DATAFRAME
# ============================================================

def normalize_dataframe(df):

    required_columns = [
        "parent_asin",
        "title",
        "store",
        "categories",
        "description",
        "features",
        "price",
        "average_rating",
        "rating_number",
    ]

    # Add missing columns if the shard does not contain one.
    for column in required_columns:

        if column not in df.columns:
            df[column] = None

    output = pd.DataFrame()

    output["parent_asin"] = df["parent_asin"].apply(clean_string)

    if "asin" in df.columns:
        output["asin"] = df["asin"].apply(clean_string)
    else:
        output["asin"] = None

    output["title"] = df["title"].apply(clean_string)

    output["store"] = df["store"].apply(clean_string)

    output["categories"] = df["categories"].apply(normalize_list)

    output["description"] = df["description"].apply(normalize_list)

    output["features"] = df["features"].apply(normalize_list)

    output["price"] = df["price"].apply(
        lambda x: normalize_number(x)
    )

    output["average_rating"] = df["average_rating"].apply(
        lambda x: normalize_number(x)
    )

    output["rating_number"] = df["rating_number"].apply(
        lambda x: normalize_number(x, integer=True)
    )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SHOPGRAPH - FIXED AMAZON ELECTRONICS EXTRACTION")
    print("=" * 80)

    print()
    print("Source:")
    print("McAuley-Lab/Amazon-Reviews-2023")
    print("raw_meta_Electronics")
    print()

    print(f"Target products: {MAX_PRODUCTS:,}")
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_parts = []

    total_products = 0

    seen_parent_asins = set()

    # ========================================================
    # READ PARQUET SHARDS
    # ========================================================

    for shard_number in range(SHARD_COUNT):

        if total_products >= MAX_PRODUCTS:
            break

        filename = (
            f"full-{shard_number:05d}-of-{SHARD_COUNT:05d}.parquet"
        )

        url = BASE_URL + filename

        print("=" * 80)
        print(f"Reading shard {shard_number + 1}/{SHARD_COUNT}")
        print(filename)
        print("=" * 80)
        print()

        try:

            # pandas + fastparquet can read remote Parquet URLs.
            shard = pd.read_parquet(
                url,
                engine="fastparquet",
            )

        except Exception as exc:

            print()
            print("ERROR while reading:")
            print(url)
            print()
            print(exc)
            print()

            print(
                "If remote Parquet reading is blocked on your system, "
                "we will switch to downloading the shards."
            )

            sys.exit(1)

        print(
            f"Raw rows in shard: {len(shard):,}"
        )

        print(
            f"Columns: {list(shard.columns)}"
        )

        # ====================================================
        # NORMALIZE
        # ====================================================

        normalized = normalize_dataframe(shard)

        # Remove records without parent ASIN.
        normalized = normalized[
            normalized["parent_asin"].notna()
        ].copy()

        # Remove duplicate parent ASINs across shards.
        normalized = normalized[
            ~normalized["parent_asin"].isin(
                seen_parent_asins
            )
        ].copy()

        if len(normalized) == 0:

            print("No new products in this shard.")
            continue

        # Only take what is needed.
        remaining = MAX_PRODUCTS - total_products

        if len(normalized) > remaining:

            normalized = normalized.iloc[
                :remaining
            ].copy()

        # Update seen IDs.
        seen_parent_asins.update(
            normalized["parent_asin"].tolist()
        )

        all_parts.append(normalized)

        total_products += len(normalized)

        print(
            f"New products added: {len(normalized):,}"
        )

        print(
            f"Total products: {total_products:,}"
        )

        print()

    # ========================================================
    # COMBINE
    # ========================================================

    if not all_parts:

        print("ERROR: No products were extracted.")

        sys.exit(1)

    df = pd.concat(
        all_parts,
        ignore_index=True
    )

    # Safety limit.
    df = df.iloc[
        :MAX_PRODUCTS
    ].copy()

    # ========================================================
    # QUALITY REPORT
    # ========================================================

    print()
    print("=" * 80)
    print("DATA QUALITY REPORT")
    print("=" * 80)

    print()

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Unique parent_asin: "
        f"{df['parent_asin'].nunique():,}"
    )

    print()

    fields = [
        "title",
        "store",
        "categories",
        "description",
        "features",
        "price",
        "average_rating",
        "rating_number",
    ]

    for column in fields:

        if column in [
            "categories",
            "description",
            "features",
        ]:

            available = df[column].apply(
                lambda x: (
                    isinstance(x, list)
                    and len(x) > 0
                )
            ).sum()

        else:

            available = df[column].notna().sum()

        missing = len(df) - available

        percentage = (
            available / len(df) * 100
        )

        print(
            f"{column:<20}"
            f"available: {available:>8,} "
            f"({percentage:6.2f}%)   "
            f"missing: {missing:>8,}"
        )

    # ========================================================
    # TEST DELL PRODUCTS
    # ========================================================

    test_ids = [
        "B09NN6SGV5",
        "B0C1MVXR8F",
        "B0BHY46CW1",
        "B0B6QBX3DT",
    ]

    print()
    print("=" * 80)
    print("DELL ASIN VERIFICATION")
    print("=" * 80)

    matches = df[
        df["parent_asin"].isin(test_ids)
    ]

    if len(matches) == 0:

        print()
        print("The four test ASINs were not found in the first 100,000 records.")

    else:

        print()

        for _, row in matches.iterrows():

            print(
                f"ASIN : {row['parent_asin']}"
            )

            print(
                f"Title: {row['title']}"
            )

            print(
                f"Store: {row['store']}"
            )

            print(
                f"Categories: {row['categories']}"
            )

            print()

    # ========================================================
    # SAVE
    # ========================================================

    print("=" * 80)
    print("SAVING")
    print("=" * 80)

    print()

    df.to_parquet(
        OUTPUT_FILE,
        engine="fastparquet",
        index=False,
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    # ========================================================
    # RELOAD VERIFICATION
    # ========================================================

    print()
    print("Reloading saved Parquet...")

    check = pd.read_parquet(
        OUTPUT_FILE,
        engine="fastparquet",
    )

    print(
        f"Reloaded rows: {len(check):,}"
    )

    print()

    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    title_available = check["title"].notna().sum()

    title_missing = len(check) - title_available

    print()

    print(
        f"Title available : "
        f"{title_available:,} / {len(check):,} "
        f"({title_available / len(check) * 100:.2f}%)"
    )

    print(
        f"Title missing   : "
        f"{title_missing:,} / {len(check):,} "
        f"({title_missing / len(check) * 100:.2f}%)"
    )

    print()

    print(
        "Existing embeddings and FAISS were NOT modified."
    )

    print()
    print("Extraction finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()