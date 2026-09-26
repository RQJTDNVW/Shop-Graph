"""
ShopGraph — Similar Product Recommendation Engine
==================================================

Finds products similar to a given product using:
    1. MiniLM product embeddings
    2. FAISS cosine-similarity search
    3. Product metadata
    4. Duplicate/self-result filtering

Usage:
    python recommendation\recommend_similar.py --demo

    python recommendation\recommend_similar.py --asin B09NN6SGV5

    python recommendation\recommend_similar.py --asin B09NN6SGV5 --top-k 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
SEARCH_DIR = DATA_DIR / "search"

EMBEDDINGS_PATH = DATA_DIR / "product_embeddings.npy"
EMBEDDING_IDS_PATH = DATA_DIR / "embedding_product_ids.parquet"
METADATA_PATH = DATA_DIR / "products_text.parquet"
FAISS_INDEX_PATH = SEARCH_DIR / "faiss_products.index"

OUTPUT_PATH = SEARCH_DIR / "last_similar_recommendations.json"


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_TOP_K = 10
DEFAULT_CANDIDATE_MULTIPLIER = 10


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(value) -> str:
    """
    Safely convert metadata values into text.

    Handles:
        - strings
        - lists
        - tuples
        - numpy arrays
        - missing values
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        parts = []

        for item in value:
            text = clean_text(item)

            if text:
                parts.append(text)

        return " ".join(parts)

    return str(value).strip()


def safe_float(value):
    """Convert a value to float safely."""

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def display_price(value) -> str:
    """Format a price for terminal display."""

    price = safe_float(value)

    if price is None:
        return "N/A"

    return f"${price:,.2f}"


def get_first_existing_column(
    dataframe: pd.DataFrame,
    columns: list[str]
):
    """Return the first available column from a list."""

    for column in columns:
        if column in dataframe.columns:
            return column

    return None


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("SHOPGRAPH — SIMILAR PRODUCT RECOMMENDATION ENGINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    print("\n[1/5] Loading product embeddings...")

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings file not found:\n{EMBEDDINGS_PATH}"
        )

    embeddings = np.load(
        EMBEDDINGS_PATH,
        mmap_mode="r"
    )

    print(
        f"Embeddings shape: {embeddings.shape}"
    )

    # --------------------------------------------------------
    # Embedding IDs
    # --------------------------------------------------------

    print("\n[2/5] Loading embedding IDs...")

    if not EMBEDDING_IDS_PATH.exists():
        raise FileNotFoundError(
            f"Embedding IDs file not found:\n"
            f"{EMBEDDING_IDS_PATH}"
        )

    embedding_ids = pd.read_parquet(
        EMBEDDING_IDS_PATH,
        engine="fastparquet"
    )

    print(
        f"ID rows: {len(embedding_ids):,}"
    )

    print(
        f"Columns: {list(embedding_ids.columns)}"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    print("\n[3/5] Loading product metadata...")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n"
            f"{METADATA_PATH}"
        )

    metadata = pd.read_parquet(
        METADATA_PATH,
        engine="fastparquet"
    )

    print(
        f"Metadata rows: {len(metadata):,}"
    )

    print(
        f"Columns: {list(metadata.columns)}"
    )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    print("\n[4/5] Loading FAISS index...")

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found:\n"
            f"{FAISS_INDEX_PATH}"
        )

    index = faiss.read_index(
        str(FAISS_INDEX_PATH)
    )

    print(
        f"FAISS vectors: {index.ntotal:,}"
    )

    print(
        f"Vector dimension: {index.d}"
    )

    # --------------------------------------------------------
    # Alignment validation
    # --------------------------------------------------------

    print("\n[5/5] Validating alignment...")

    if len(embeddings) != len(embedding_ids):
        raise ValueError(
            "Embedding count does not match embedding ID count.\n"
            f"Embeddings: {len(embeddings):,}\n"
            f"IDs: {len(embedding_ids):,}"
        )

    if index.ntotal != len(embeddings):
        raise ValueError(
            "FAISS index count does not match embeddings.\n"
            f"FAISS: {index.ntotal:,}\n"
            f"Embeddings: {len(embeddings):,}"
        )

    embedding_id_column = get_first_existing_column(
        embedding_ids,
        [
            "parent_asin",
            "asin",
            "product_id"
        ]
    )

    metadata_id_column = get_first_existing_column(
        metadata,
        [
            "parent_asin",
            "asin",
            "product_id"
        ]
    )

    if embedding_id_column is None:
        raise ValueError(
            "No product ID column found in embedding IDs."
        )

    if metadata_id_column is None:
        raise ValueError(
            "No product ID column found in metadata."
        )

    print(
        f"Embedding ID column: {embedding_id_column}"
    )

    print(
        f"Metadata ID column: {metadata_id_column}"
    )

    print("Alignment: OK")

    return (
        embeddings,
        embedding_ids,
        metadata,
        index,
        embedding_id_column,
        metadata_id_column,
    )


# ============================================================
# LOAD MINILM
# ============================================================

def load_model():

    print("\nLoading MiniLM model...")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    elapsed = time.perf_counter() - start

    print(
        f"Model loaded in {elapsed:.2f} seconds"
    )

    return tokenizer, model, device


# ============================================================
# MEAN POOLING
# ============================================================

def mean_pooling(
    model_output,
    attention_mask
):
    """
    Mean pooling for MiniLM embeddings.
    """

    token_embeddings = model_output.last_hidden_state

    mask = (
        attention_mask
        .unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )

    summed = torch.sum(
        token_embeddings * mask,
        dim=1
    )

    counts = torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )

    return summed / counts


# ============================================================
# ENCODE TEXT
# ============================================================

def encode_text(
    text: str,
    tokenizer,
    model,
    device
):
    """
    Convert text into a normalized MiniLM vector.
    """

    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():

        output = model(
            **encoded
        )

    embedding = mean_pooling(
        output,
        encoded["attention_mask"]
    )

    embedding = torch.nn.functional.normalize(
        embedding,
        p=2,
        dim=1
    )

    return embedding.cpu().numpy().astype(
        np.float32
    )


# ============================================================
# PRODUCT LOOKUP
# ============================================================

def build_product_lookup(
    metadata: pd.DataFrame,
    metadata_id_column: str
):
    """
    Build:

        product_id -> metadata dataframe index

    If duplicate IDs exist, the first valid row is retained.
    """

    lookup = {}

    for dataframe_index, row in metadata.iterrows():

        product_id = clean_text(
            row.get(
                metadata_id_column
            )
        )

        if not product_id:
            continue

        if product_id not in lookup:
            lookup[product_id] = dataframe_index

    return lookup


# ============================================================
# FIND SOURCE PRODUCT
# ============================================================

def find_source_product(
    asin: str,
    embedding_ids: pd.DataFrame,
    metadata: pd.DataFrame,
    embedding_id_column: str,
    metadata_id_column: str
):
    """
    Find:

        1. embedding row
        2. metadata row
    """

    asin = str(asin).strip()

    embedding_matches = embedding_ids[
        embedding_ids[embedding_id_column]
        .astype(str)
        .str.strip()
        == asin
    ]

    if embedding_matches.empty:

        raise ValueError(
            f"ASIN not found in embedding IDs: {asin}"
        )

    embedding_row_index = int(
        embedding_matches.index[0]
    )

    metadata_matches = metadata[
        metadata[metadata_id_column]
        .astype(str)
        .str.strip()
        == asin
    ]

    if metadata_matches.empty:

        raise ValueError(
            f"ASIN not found in metadata: {asin}"
        )

    metadata_row = metadata_matches.iloc[0]

    return (
        embedding_row_index,
        metadata_row
    )


# ============================================================
# PRODUCT -> JSON
# ============================================================

def product_to_dict(
    row,
    similarity: float | None = None
):
    """
    Convert one metadata row into a clean dictionary.

    Important:
        Raw metadata fields can sometimes be empty.
        Therefore we use cleaned fields as fallbacks.
    """

    # --------------------------------------------------------
    # Product ID
    # --------------------------------------------------------

    product_id = clean_text(
        row.get(
            "parent_asin",
            ""
        )
    )

    if not product_id:
        product_id = clean_text(
            row.get(
                "asin",
                ""
            )
        )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    title = clean_text(
        row.get(
            "title",
            ""
        )
    )

    # Fallback to clean_title.
    if not title:

        title = clean_text(
            row.get(
                "clean_title",
                ""
            )
        )

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    store = clean_text(
        row.get(
            "store",
            ""
        )
    )

    # Fallback to clean_store.
    if not store:

        store = clean_text(
            row.get(
                "clean_store",
                ""
            )
        )

    # --------------------------------------------------------
    # CATEGORIES
    # --------------------------------------------------------

    categories = clean_text(
        row.get(
            "categories",
            ""
        )
    )

    # Fallback to category_text.
    if not categories:

        categories = clean_text(
            row.get(
                "category_text",
                ""
            )
        )

    # --------------------------------------------------------
    # Price
    # --------------------------------------------------------

    price = safe_float(
        row.get(
            "price"
        )
    )

    # --------------------------------------------------------
    # Rating
    # --------------------------------------------------------

    rating = safe_float(
        row.get(
            "average_rating"
        )
    )

    # --------------------------------------------------------
    # Rating count
    # --------------------------------------------------------

    rating_number = safe_float(
        row.get(
            "rating_number"
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    result = {
        "parent_asin": product_id,
        "title": title,
        "store": store,
        "categories": categories,
        "price": price,
        "price_display": display_price(price),
        "average_rating": rating,
        "rating_number": (
            int(rating_number)
            if rating_number is not None
            else None
        ),
    }

    if similarity is not None:

        result["similarity"] = round(
            float(similarity),
            4
        )

    return result


# ============================================================
# RECOMMEND SIMILAR PRODUCTS
# ============================================================

def recommend_similar(
    asin: str,
    top_k: int,
    embeddings,
    embedding_ids,
    metadata,
    index,
    embedding_id_column,
    metadata_id_column
):
    """
    Find products similar to the source product.
    """

    print("\n" + "=" * 70)
    print("GENERATING RECOMMENDATIONS")
    print("=" * 70)

    print(
        f"\nSource ASIN: {asin}"
    )

    print(
        f"Requested recommendations: {top_k}"
    )

    # --------------------------------------------------------
    # Find source
    # --------------------------------------------------------

    (
        source_embedding_row,
        source_metadata
    ) = find_source_product(
        asin=asin,
        embedding_ids=embedding_ids,
        metadata=metadata,
        embedding_id_column=embedding_id_column,
        metadata_id_column=metadata_id_column
    )

    print(
        f"Embedding row: {source_embedding_row}"
    )

    source_product = product_to_dict(
        source_metadata
    )

    print(
        f"Product: {source_product['title']}"
    )

    # --------------------------------------------------------
    # Get source vector
    # --------------------------------------------------------

    query_vector = np.array(
        embeddings[source_embedding_row],
        dtype=np.float32,
        copy=True
    )

    query_vector = query_vector.reshape(
        1,
        -1
    )

    # Ensure normalized vector.
    faiss.normalize_L2(
        query_vector
    )

    # --------------------------------------------------------
    # Candidate pool
    # --------------------------------------------------------

    candidate_count = max(
        top_k * DEFAULT_CANDIDATE_MULTIPLIER,
        100
    )

    candidate_count = min(
        candidate_count,
        index.ntotal
    )

    print(
        f"\nFAISS candidate search: "
        f"{candidate_count}"
    )

    start = time.perf_counter()

    similarities, indices = index.search(
        query_vector,
        candidate_count
    )

    elapsed = time.perf_counter() - start

    print(
        f"FAISS search time: "
        f"{elapsed * 1000:.2f} ms"
    )

    # --------------------------------------------------------
    # Metadata lookup
    # --------------------------------------------------------

    metadata_lookup = build_product_lookup(
        metadata,
        metadata_id_column
    )

    recommendations = []

    seen_ids = set()

    source_asin = str(
        source_product["parent_asin"]
    ).strip()

    skipped_self = 0
    skipped_duplicate = 0
    skipped_missing_metadata = 0

    # --------------------------------------------------------
    # Process candidates
    # --------------------------------------------------------

    for similarity, embedding_index in zip(
        similarities[0],
        indices[0]
    ):

        embedding_index = int(
            embedding_index
        )

        if embedding_index < 0:
            continue

        # ----------------------------------------------------
        # Get product ID from embedding row.
        # ----------------------------------------------------

        embedding_row = embedding_ids.iloc[
            embedding_index
        ]

        product_id = clean_text(
            embedding_row.get(
                embedding_id_column
            )
        )

        if not product_id:
            continue

        # ----------------------------------------------------
        # Remove source product.
        # ----------------------------------------------------

        if product_id == source_asin:

            skipped_self += 1

            continue

        # ----------------------------------------------------
        # Remove duplicate products.
        # ----------------------------------------------------

        if product_id in seen_ids:

            skipped_duplicate += 1

            continue

        # ----------------------------------------------------
        # Find metadata.
        # ----------------------------------------------------

        metadata_index = metadata_lookup.get(
            product_id
        )

        if metadata_index is None:

            skipped_missing_metadata += 1

            continue

        product_row = metadata.loc[
            metadata_index
        ]

        # ----------------------------------------------------
        # Create recommendation.
        # ----------------------------------------------------

        recommendation = product_to_dict(
            product_row,
            similarity=float(similarity)
        )

        recommendations.append(
            recommendation
        )

        seen_ids.add(
            product_id
        )

        # Stop after requested number.
        if len(recommendations) >= top_k:
            break

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = {
        "candidate_count": int(
            candidate_count
        ),
        "results_returned": len(
            recommendations
        ),
        "self_results_removed": int(
            skipped_self
        ),
        "duplicate_results_removed": int(
            skipped_duplicate
        ),
        "missing_metadata_results": int(
            skipped_missing_metadata
        ),
        "faiss_search_ms": round(
            elapsed * 1000,
            3
        ),
    }

    return (
        source_product,
        recommendations,
        stats
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    source_product,
    recommendations,
    stats
):

    print("\n" + "=" * 70)
    print("SOURCE PRODUCT")
    print("=" * 70)

    print(
        f"\nASIN: {source_product['parent_asin']}"
    )

    print(
        f"Title: {source_product['title']}"
    )

    print(
        f"Store: "
        f"{source_product['store'] or 'N/A'}"
    )

    print(
        f"Price: "
        f"{source_product['price_display']}"
    )

    if source_product["average_rating"] is not None:

        print(
            f"Rating: "
            f"{source_product['average_rating']:.2f}"
        )

    print("\n" + "=" * 70)
    print("RECOMMENDED PRODUCTS")
    print("=" * 70)

    if not recommendations:

        print(
            "\nNo recommendations found."
        )

        return

    for position, product in enumerate(
        recommendations,
        start=1
    ):

        print(
            f"\n#{position}"
        )

        print(
            f"Title: "
            f"{product['title'] or 'N/A'}"
        )

        print(
            f"ASIN: "
            f"{product['parent_asin']}"
        )

        print(
            f"Store: "
            f"{product['store'] or 'N/A'}"
        )

        print(
            f"Price: "
            f"{product['price_display']}"
        )

        if product["average_rating"] is not None:

            print(
                f"Rating: "
                f"{product['average_rating']:.2f}"
            )

        print(
            f"Similarity: "
            f"{product['similarity']:.4f}"
        )

    print("\n" + "=" * 70)
    print("SEARCH STATISTICS")
    print("=" * 70)

    for key, value in stats.items():

        print(
            f"{key}: {value}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    source_product,
    recommendations,
    stats,
    asin,
    top_k
):

    SEARCH_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "project": "ShopGraph",
        "task": "similar_product_recommendation",
        "source_asin": asin,
        "top_k": top_k,
        "source_product": source_product,
        "recommendations": recommendations,
        "statistics": stats,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_PATH}"
    )


# ============================================================
# DEMO
# ============================================================

def run_demo(
    embeddings,
    embedding_ids,
    metadata,
    index,
    embedding_id_column,
    metadata_id_column,
    top_k
):
    """
    Select the known demo ASIN.

    Falls back to the first valid ASIN if the demo
    product is unavailable.
    """

    demo_asin = "B09NN6SGV5"

    available_ids = (
        embedding_ids[
            embedding_id_column
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )

    available_set = set(
        available_ids
    )

    if demo_asin not in available_set:

        demo_asin = None

        for product_id in available_ids:

            if product_id:

                demo_asin = product_id

                break

    if not demo_asin:

        raise RuntimeError(
            "Could not find a valid demo product."
        )

    print(
        f"\nDemo product selected: "
        f"{demo_asin}"
    )

    return demo_asin


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "ShopGraph similar product "
            "recommendation engine"
        )
    )

    parser.add_argument(
        "--asin",
        type=str,
        default=None,
        help=(
            "ASIN of the product used "
            "for recommendations"
        )
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "Number of recommendations "
            "to return"
        )
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run using the built-in demo ASIN"
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    if args.top_k < 1:

        print(
            "ERROR: --top-k must be at least 1."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Load existing ShopGraph data.
    # --------------------------------------------------------

    (
        embeddings,
        embedding_ids,
        metadata,
        index,
        embedding_id_column,
        metadata_id_column,
    ) = load_data()

    # --------------------------------------------------------
    # Select product.
    # --------------------------------------------------------

    if args.demo:

        asin = run_demo(
            embeddings,
            embedding_ids,
            metadata,
            index,
            embedding_id_column,
            metadata_id_column,
            args.top_k
        )

    elif args.asin:

        asin = args.asin.strip()

    else:

        print(
            "\nERROR: Please provide either:"
        )

        print(
            "--demo"
        )

        print(
            "or"
        )

        print(
            "--asin YOUR_ASIN"
        )

        print(
            "\nExample:"
        )

        print(
            "python recommendation\\recommend_similar.py --demo"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Recommendation.
    # --------------------------------------------------------

    total_start = time.perf_counter()

    (
        source_product,
        recommendations,
        stats
    ) = recommend_similar(
        asin=asin,
        top_k=args.top_k,
        embeddings=embeddings,
        embedding_ids=embedding_ids,
        metadata=metadata,
        index=index,
        embedding_id_column=embedding_id_column,
        metadata_id_column=metadata_id_column
    )

    total_time = (
        time.perf_counter()
        - total_start
    )

    stats[
        "total_recommendation_time_ms"
    ] = round(
        total_time * 1000,
        3
    )

    # --------------------------------------------------------
    # Print.
    # --------------------------------------------------------

    print_results(
        source_product,
        recommendations,
        stats
    )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    save_results(
        source_product=source_product,
        recommendations=recommendations,
        stats=stats,
        asin=asin,
        top_k=args.top_k
    )

    print(
        "\nRecommendation engine completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()