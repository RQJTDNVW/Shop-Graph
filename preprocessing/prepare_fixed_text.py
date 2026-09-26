from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/processed/products_100k_fixed.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/products_text_fixed.parquet"
)


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    """
    Convert metadata values into clean text.
    Handles strings, lists, tuples and other values safely.
    """

    if value is None:
        return ""

    if isinstance(value, (list, tuple)):
        return " ".join(
            str(item)
            for item in value
            if item is not None
        )

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def build_embedding_text(row):
    """
    Build the text representation used by the embedding model.

    Priority:
        title
        store
        features
        description
        categories
    """

    title = clean_text(row["title"])
    store = clean_text(row["store"])
    features = clean_text(row["features"])
    description = clean_text(row["description"])
    categories = clean_text(row["categories"])

    # Prevent extremely long descriptions
    description = description[:3000]

    parts = []

    if title:
        parts.append(f"Title: {title}")

    if store:
        parts.append(f"Brand/Store: {store}")

    if categories:
        parts.append(f"Categories: {categories}")

    if features:
        parts.append(f"Features: {features}")

    if description:
        parts.append(f"Description: {description}")

    return " | ".join(parts)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SHOPGRAPH — FIXED PRODUCT TEXT PREPARATION")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE.resolve()}"
        )

    print("\nLoading fixed product metadata...")

    df = pd.read_parquet(
        INPUT_FILE,
        engine="fastparquet"
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    required_columns = [
        "parent_asin",
        "title",
        "store",
        "features",
        "description",
        "categories"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Clean core metadata
    # --------------------------------------------------------

    print("\nCleaning metadata...")

    df["clean_title"] = df["title"].apply(clean_text)
    df["clean_store"] = df["store"].apply(clean_text)
    df["clean_features"] = df["features"].apply(clean_text)
    df["clean_description"] = df["description"].apply(clean_text)
    df["category_text"] = df["categories"].apply(clean_text)

    # --------------------------------------------------------
    # Build embedding text
    # --------------------------------------------------------

    print("Building embedding text...")

    df["embedding_text"] = df.apply(
        build_embedding_text,
        axis=1
    )

    # --------------------------------------------------------
    # Length statistics
    # --------------------------------------------------------

    df["embedding_text_length"] = (
        df["embedding_text"]
        .str.len()
    )

    df["title_length"] = (
        df["clean_title"]
        .str.len()
    )

    df["description_length"] = (
        df["clean_description"]
        .str.len()
    )

    df["feature_length"] = (
        df["clean_features"]
        .str.len()
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\nText quality check:")

    title_available = (
        df["clean_title"].str.len() > 0
    ).sum()

    embedding_available = (
        df["embedding_text"].str.len() > 0
    ).sum()

    print(
        f"  Titles available: "
        f"{title_available:,} / {len(df):,}"
    )

    print(
        f"  Embedding text available: "
        f"{embedding_available:,} / {len(df):,}"
    )

    print(
        f"  Average embedding text length: "
        f"{df['embedding_text_length'].mean():.1f}"
    )

    print(
        f"  Maximum embedding text length: "
        f"{df['embedding_text_length'].max():,}"
    )

    # --------------------------------------------------------
    # Show samples
    # --------------------------------------------------------

    print("\nEmbedding text samples:")

    for i in range(min(5, len(df))):

        print("\n" + "-" * 70)

        print(
            f"ASIN: {df.iloc[i]['parent_asin']}"
        )

        print(
            f"Title: {df.iloc[i]['clean_title']}"
        )

        print(
            f"Embedding text:\n"
            f"{df.iloc[i]['embedding_text'][:1000]}"
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
    print("TEXT PREPARATION COMPLETE")
    print("=" * 70)

    print(
        f"\nSaved to:\n"
        f"{OUTPUT_FILE.resolve()}"
    )

    print(
        f"\nFinal shape: {df.shape}"
    )

    print("\nOriginal files were NOT modified.")


if __name__ == "__main__":
    main()