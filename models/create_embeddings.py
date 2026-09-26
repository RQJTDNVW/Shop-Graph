from pathlib import Path
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# SHOPGRAPH — PRODUCT EMBEDDING GENERATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "products_text.parquet"
)

OUTPUT_EMBEDDINGS = (
    BASE_DIR
    / "data"
    / "processed"
    / "product_embeddings.npy"
)

OUTPUT_IDS = (
    BASE_DIR
    / "data"
    / "processed"
    / "embedding_product_ids.parquet"
)


# Sentence-transformer model.
#
# 384-dimensional embeddings.
# Good balance between quality, speed and storage
# for the first 100K-product experiment.
MODEL_NAME = "all-MiniLM-L6-v2"

# Number of products processed at once.
BATCH_SIZE = 64


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_embedding_text(text):
    """
    Clean the already-created embedding text.

    Converts representations such as:

        ['Feature one' 'Feature two']

    into:

        Feature one Feature two
    """

    if text is None:
        return ""

    text = str(text)

    # Remove brackets.
    text = text.replace("[", " ")
    text = text.replace("]", " ")

    # Remove quote characters used by list representations.
    text = text.replace("'", " ")
    text = text.replace('"', " ")

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SHOPGRAPH — PRODUCT EMBEDDING GENERATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_FILE}"
        )

    print("\nLoading processed products:")
    print(INPUT_FILE)

    df = pd.read_parquet(INPUT_FILE)

    print(f"\nProducts loaded: {len(df):,}")

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "parent_asin",
        "embedding_text",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Prepare text
    # --------------------------------------------------------

    print("\nPreparing embedding text...")

    texts = (
        df["embedding_text"]
        .apply(normalize_embedding_text)
        .tolist()
    )

    print(f"Text samples prepared: {len(texts):,}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading embedding model:")
    print(MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Detect device
    # --------------------------------------------------------

    try:
        device = model.device
    except Exception:
        device = "unknown"

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    print("\nGenerating product embeddings...")
    print(f"Batch size: {BATCH_SIZE:,}")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # --------------------------------------------------------
    # Validate embeddings
    # --------------------------------------------------------

    print("\nEmbedding generation complete.")

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    print(
        f"Embedding dtype: {embeddings.dtype}"
    )

    print(
        f"Embedding dimension: {embeddings.shape[1]}"
    )

    # --------------------------------------------------------
    # Check for invalid values
    # --------------------------------------------------------

    nan_count = np.isnan(embeddings).sum()
    inf_count = np.isinf(embeddings).sum()

    print(
        f"\nNaN values: {nan_count:,}"
    )

    print(
        f"Infinite values: {inf_count:,}"
    )

    if nan_count > 0 or inf_count > 0:
        raise ValueError(
            "Invalid values detected in embeddings."
        )

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    OUTPUT_EMBEDDINGS.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nSaving embeddings...")

    np.save(
        OUTPUT_EMBEDDINGS,
        embeddings.astype(np.float32)
    )

    # --------------------------------------------------------
    # Save product ID mapping
    # --------------------------------------------------------

    id_mapping = pd.DataFrame({
        "embedding_index": np.arange(len(df)),
        "parent_asin": df["parent_asin"].values,
    })

    id_mapping.to_parquet(
        OUTPUT_IDS,
        index=False
    )

    # --------------------------------------------------------
    # Calculate storage
    # --------------------------------------------------------

    embedding_size_mb = (
        embeddings.astype(np.float32).nbytes
        / (1024 ** 2)
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 80)

    print(
        f"\nProducts embedded       : {len(embeddings):,}"
    )

    print(
        f"Embedding dimensions    : {embeddings.shape[1]}"
    )

    print(
        f"Embedding shape         : {embeddings.shape}"
    )

    print(
        f"Storage size            : {embedding_size_mb:.2f} MB"
    )

    print(
        f"NaN values              : {nan_count:,}"
    )

    print(
        f"Infinite values        : {inf_count:,}"
    )

    print("\nSaved files:")

    print(
        f"Embeddings              : "
        f"{OUTPUT_EMBEDDINGS}"
    )

    print(
        f"Product ID mapping      : "
        f"{OUTPUT_IDS}"
    )

    # --------------------------------------------------------
    # Show example vector
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("SAMPLE EMBEDDING")
    print("=" * 80)

    print(
        f"\nProduct: {df.iloc[0]['title']}"
    )

    print(
        "\nFirst 10 vector values:"
    )

    print(
        embeddings[0][:10]
    )

    print("\nEmbedding pipeline ready for clustering/search.")


if __name__ == "__main__":
    main()