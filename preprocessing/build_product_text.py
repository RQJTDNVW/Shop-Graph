import json
import re
from pathlib import Path

import pandas as pd


# ============================================================
# SHOPGRAPH — PRODUCT TEXT PREPROCESSING
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "processed" / "products_100k.parquet"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "products_text.parquet"
STATS_FILE = BASE_DIR / "data" / "processed" / "text_preprocessing_stats.json"


# Maximum description characters used for embeddings.
# Title and features are preserved separately.
MAX_DESCRIPTION_LENGTH = 3000


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(value):
    """
    Convert a value into clean readable text.

    Handles:
    - None
    - NaN
    - strings
    - lists
    - tuples
    - numpy arrays
    - other objects
    """

    if value is None:
        return ""

    # Handle pandas missing values safely.
    if isinstance(value, (str, bytes)):
        text = value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else value
        text = str(text)

    elif isinstance(value, (list, tuple)):
        parts = []

        for item in value:
            cleaned = clean_text(item)

            if cleaned:
                parts.append(cleaned)

        text = " ".join(parts)

    else:
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass

        text = str(value)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    # Remove repeated separators.
    text = re.sub(r"\|+", " | ", text)

    # Remove excessive punctuation repetition.
    text = re.sub(r"\.{3,}", "...", text)

    return text.strip()


def clean_description(value):
    """
    Clean and limit description length.
    """

    text = clean_text(value)

    if not text:
        return ""

    if len(text) > MAX_DESCRIPTION_LENGTH:
        text = text[:MAX_DESCRIPTION_LENGTH].rsplit(" ", 1)[0]
        text += "..."

    return text


# ============================================================
# BUILD EMBEDDING TEXT
# ============================================================

def build_embedding_text(row):
    """
    Build structured text for the product embedding.

    Categories are intentionally NOT included here.
    They are retained separately for evaluation and analysis.
    """

    title = clean_text(row["title"])
    store = clean_text(row["store"])
    features = clean_text(row["features"])
    description = clean_description(row["description"])

    sections = []

    if title:
        sections.append(f"Title: {title}")

    if store:
        sections.append(f"Brand: {store}")

    if features:
        sections.append(f"Features: {features}")

    if description:
        sections.append(f"Description: {description}")

    return "\n".join(sections)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SHOPGRAPH — PRODUCT TEXT PREPROCESSING")
    print("=" * 80)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nInput dataset not found:\n{INPUT_FILE}\n\n"
            "Make sure Phase 1 was completed successfully."
        )

    print("\nLoading dataset:")
    print(INPUT_FILE)

    df = pd.read_parquet(INPUT_FILE)

    print(f"\nDataset shape: {df.shape}")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "parent_asin",
        "title",
        "store",
        "features",
        "description",
        "categories",
        "price",
        "average_rating",
        "rating_number",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"\nMissing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Clean core fields
    # --------------------------------------------------------

    print("\nCleaning product fields...")

    df["clean_title"] = df["title"].apply(clean_text)
    df["clean_store"] = df["store"].apply(clean_text)
    df["clean_features"] = df["features"].apply(clean_text)
    df["clean_description"] = df["description"].apply(clean_description)

    # --------------------------------------------------------
    # Preserve categories separately
    # --------------------------------------------------------

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
    # Remove products without usable embedding text
    # --------------------------------------------------------

    before_count = len(df)

    df = df[
        df["embedding_text"].str.strip().str.len() > 0
    ].copy()

    removed_count = before_count - len(df)

    # --------------------------------------------------------
    # Text statistics
    # --------------------------------------------------------

    df["embedding_text_length"] = (
        df["embedding_text"].str.len()
    )

    df["title_length"] = (
        df["clean_title"].str.len()
    )

    df["description_length"] = (
        df["clean_description"].str.len()
    )

    df["feature_length"] = (
        df["clean_features"].str.len()
    )

    # --------------------------------------------------------
    # Select final columns
    # --------------------------------------------------------

    final_columns = [
        # Product identity
        "parent_asin",

        # Original useful metadata
        "title",
        "store",
        "categories",
        "price",
        "average_rating",
        "rating_number",

        # Cleaned fields
        "clean_title",
        "clean_store",
        "clean_features",
        "clean_description",
        "category_text",

        # Main ML representation
        "embedding_text",

        # Statistics
        "embedding_text_length",
        "title_length",
        "description_length",
        "feature_length",
    ]

    df_final = df[final_columns].copy()

    # --------------------------------------------------------
    # Save processed dataset
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nSaving processed dataset...")

    df_final.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Generate statistics
    # --------------------------------------------------------

    stats = {
        "input_products": int(before_count),
        "output_products": int(len(df_final)),
        "removed_products": int(removed_count),

        "embedding_text": {
            "mean_length": float(
                df_final["embedding_text_length"].mean()
            ),
            "median_length": float(
                df_final["embedding_text_length"].median()
            ),
            "max_length": int(
                df_final["embedding_text_length"].max()
            ),
        },

        "title": {
            "available": int(
                (df_final["clean_title"].str.len() > 0).sum()
            ),
            "missing": int(
                (df_final["clean_title"].str.len() == 0).sum()
            ),
        },

        "description": {
            "available": int(
                (df_final["clean_description"].str.len() > 0).sum()
            ),
            "missing": int(
                (df_final["clean_description"].str.len() == 0).sum()
            ),
        },

        "features": {
            "available": int(
                (df_final["clean_features"].str.len() > 0).sum()
            ),
            "missing": int(
                (df_final["clean_features"].str.len() == 0).sum()
            ),
        },

        "categories": {
            "available": int(
                (df_final["category_text"].str.len() > 0).sum()
            ),
            "missing": int(
                (df_final["category_text"].str.len() == 0).sum()
            ),
        },

        "unique_parent_asin": int(
            df_final["parent_asin"].nunique()
        ),

        "max_description_characters": MAX_DESCRIPTION_LENGTH,
    }

    with open(
        STATS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            stats,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)

    print(f"\nInput products       : {before_count:,}")
    print(f"Output products      : {len(df_final):,}")
    print(f"Removed products     : {removed_count:,}")

    print("\nText availability:")

    print(
        f"Title                : "
        f"{stats['title']['available']:,}"
    )

    print(
        f"Description          : "
        f"{stats['description']['available']:,}"
    )

    print(
        f"Features             : "
        f"{stats['features']['available']:,}"
    )

    print(
        f"Categories           : "
        f"{stats['categories']['available']:,}"
    )

    print("\nEmbedding text length:")

    print(
        f"Mean                 : "
        f"{stats['embedding_text']['mean_length']:.2f}"
    )

    print(
        f"Median               : "
        f"{stats['embedding_text']['median_length']:.2f}"
    )

    print(
        f"Maximum              : "
        f"{stats['embedding_text']['max_length']:,}"
    )

    print("\nSaved files:")

    print(f"Products text        : {OUTPUT_FILE}")
    print(f"Statistics            : {STATS_FILE}")

    # --------------------------------------------------------
    # Show examples
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("SAMPLE EMBEDDING TEXT")
    print("=" * 80)

    for index, text in enumerate(
        df_final["embedding_text"].head(3),
        start=1
    ):

        print(f"\n--- Product {index} ---")
        print(text[:1000])


if __name__ == "__main__":
    main()