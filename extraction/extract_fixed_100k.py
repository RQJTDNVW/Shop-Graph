from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/raw/amazon_electronics/raw_meta_Electronics/"
    "full-00000-of-00010.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/products_100k_fixed.parquet"
)

TARGET_PRODUCTS = 100_000


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SHOPGRAPH — FIXED ELECTRONICS METADATA EXTRACTION")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE.resolve()}"
        )

    print(f"\nInput file:")
    print(INPUT_FILE.resolve())

    # --------------------------------------------------------
    # Load original shard
    # --------------------------------------------------------

    print("\nLoading original Electronics shard...")

    df = pd.read_parquet(
        INPUT_FILE,
        engine="fastparquet"
    )

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Remove rows without ASIN
    # --------------------------------------------------------

    df["parent_asin"] = df["parent_asin"].astype("string").str.strip()

    before = len(df)

    df = df[
        df["parent_asin"].notna()
        & (df["parent_asin"] != "")
    ].copy()

    print(f"Rows with valid parent_asin: {len(df):,}")
    print(f"Removed invalid ASIN rows: {before - len(df):,}")

    # --------------------------------------------------------
    # Keep one row per parent_asin
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=["parent_asin"],
        keep="first"
    ).reset_index(drop=True)

    print(f"Unique products: {len(df):,}")
    print(f"Duplicate rows removed: {before - len(df):,}")

    # --------------------------------------------------------
    # Verify that 100k products are available
    # --------------------------------------------------------

    if len(df) < TARGET_PRODUCTS:
        raise RuntimeError(
            f"Only {len(df):,} unique products available. "
            f"Need {TARGET_PRODUCTS:,}."
        )

    # --------------------------------------------------------
    # Select exactly 100k products
    # --------------------------------------------------------

    df = df.iloc[:TARGET_PRODUCTS].copy()

    print(f"\nSelected products: {len(df):,}")

    # --------------------------------------------------------
    # Verify important metadata
    # --------------------------------------------------------

    print("\nMetadata quality check:")

    title_count = df["title"].notna().sum()
    store_count = df["store"].notna().sum()
    category_count = df["categories"].notna().sum()
    feature_count = df["features"].notna().sum()
    description_count = df["description"].notna().sum()
    price_count = df["price"].notna().sum()

    print(f"  Title available:       {title_count:,} / {len(df):,}")
    print(f"  Store available:       {store_count:,} / {len(df):,}")
    print(f"  Categories available:  {category_count:,} / {len(df):,}")
    print(f"  Features available:    {feature_count:,} / {len(df):,}")
    print(f"  Description available: {description_count:,} / {len(df):,}")
    print(f"  Price available:       {price_count:,} / {len(df):,}")

    # --------------------------------------------------------
    # Show sample
    # --------------------------------------------------------

    print("\nSample products:")

    sample_columns = [
        "parent_asin",
        "title",
        "store",
        "price"
    ]

    print(
        df[sample_columns]
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        OUTPUT_FILE,
        engine="fastparquet",
        index=False
    )

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(f"\nSaved to:")
    print(OUTPUT_FILE.resolve())

    print(f"\nFinal shape: {df.shape}")

    print("\nOriginal files were NOT modified.")


if __name__ == "__main__":
    main()