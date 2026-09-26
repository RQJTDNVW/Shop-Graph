"""
ShopGraph — Natural Language Product Search
============================================

Phase 4.3

Pipeline:
User query
    ↓
all-MiniLM-L6-v2
    ↓
384-D normalized query embedding
    ↓
FAISS
    ↓
Top-K semantically similar products

Examples:
    python search\natural_language_search.py --query "wireless earbuds for commuting"
    python search\natural_language_search.py --query "laptop with 16GB RAM for university"
    python search\natural_language_search.py --query "gaming laptop with RTX graphics"
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
from sentence_transformers import SentenceTransformer


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "search"
    / "faiss_products.index"
)

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "product_embeddings.npy"
)

PRODUCT_IDS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embedding_product_ids.parquet"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "products_text.parquet"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_TOP_K = 10
SEARCH_BUFFER = 10


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def print_separator():
    print("\n" + "=" * 80)


def is_missing(value) -> bool:
    """Safely determine whether a scalar value is missing."""
    if value is None:
        return True

    if isinstance(value, (list, tuple, np.ndarray, dict)):
        return False

    try:
        result = pd.isna(value)

        if isinstance(result, (bool, np.bool_)):
            return bool(result)

    except Exception:
        pass

    return False


def format_value(value, max_length: int = 500) -> str:
    """Convert metadata values into readable text."""

    if is_missing(value):
        return "N/A"

    if isinstance(value, dict):
        text = " | ".join(
            f"{key}: {val}"
            for key, val in value.items()
        )

    elif isinstance(value, (list, tuple, np.ndarray)):
        parts = []

        for item in value:
            if isinstance(item, (list, tuple, np.ndarray)):
                parts.append(
                    " / ".join(str(x) for x in item)
                )
            elif isinstance(item, dict):
                parts.append(
                    " | ".join(
                        f"{k}: {v}"
                        for k, v in item.items()
                    )
                )
            else:
                parts.append(str(item))

        text = " | ".join(parts)

    else:
        text = str(value)

    text = text.replace("\n", " ").replace("\r", " ")

    if len(text) > max_length:
        text = text[:max_length - 3] + "..."

    return text


# =============================================================================
# FILE CHECKING
# =============================================================================

def check_files():
    print_separator()
    print("CHECKING FILES")
    print("=" * 14)

    required_files = [
        INDEX_PATH,
        EMBEDDINGS_PATH,
        PRODUCT_IDS_PATH,
        METADATA_PATH,
    ]

    for path in required_files:
        if path.exists():
            print(f"OK: {path}")
        else:
            print(f"ERROR: Missing file:")
            print(path)
            sys.exit(1)


# =============================================================================
# LOAD DATA
# =============================================================================

def load_search_data():

    print_separator()
    print("LOADING SEARCH DATA")
    print("=" * 20)

    # -------------------------------------------------------------------------
    # FAISS
    # -------------------------------------------------------------------------

    print("\nLoading FAISS index...")

    index = faiss.read_index(str(INDEX_PATH))

    print(f"FAISS vectors : {index.ntotal:,}")
    print(f"Vector dimension: {index.d}")

    # -------------------------------------------------------------------------
    # Product IDs
    # -------------------------------------------------------------------------

    print("\nLoading product IDs...")

    product_ids = pd.read_parquet(PRODUCT_IDS_PATH)

    print(f"Product ID columns: {list(product_ids.columns)}")
    print(f"Products: {len(product_ids):,}")

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    print("\nLoading product metadata...")

    metadata = pd.read_parquet(METADATA_PATH)

    print(f"Metadata shape: {metadata.shape}")

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------

    print("\nLoading embeddings...")

    embeddings = np.load(
        EMBEDDINGS_PATH,
        mmap_mode="r"
    )

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype : {embeddings.dtype}")

    # -------------------------------------------------------------------------
    # Alignment
    # -------------------------------------------------------------------------

    print("\nValidating alignment...")

    if index.ntotal != len(product_ids):
        raise ValueError(
            f"FAISS/product ID mismatch: "
            f"{index.ntotal} vs {len(product_ids)}"
        )

    if len(product_ids) != len(metadata):
        raise ValueError(
            f"Product ID/metadata mismatch: "
            f"{len(product_ids)} vs {len(metadata)}"
        )

    if embeddings.shape[0] != len(product_ids):
        raise ValueError(
            f"Embedding/product ID mismatch: "
            f"{embeddings.shape[0]} vs {len(product_ids)}"
        )

    if embeddings.shape[1] != index.d:
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"{embeddings.shape[1]} vs {index.d}"
        )

    print("Dataset alignment: OK")

    return index, product_ids, metadata


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_embedding_model():

    print_separator()
    print("LOADING EMBEDDING MODEL")
    print("=" * 24)

    print(f"Model: {MODEL_NAME}")

    start = time.perf_counter()

    model = SentenceTransformer(
        MODEL_NAME
    )

    elapsed = time.perf_counter() - start

    print(f"Model loaded in: {elapsed:.2f} seconds")

    # Show device being used.
    try:
        device = model.device
        print(f"Device: {device}")
    except Exception:
        print("Device: automatic")

    return model


# =============================================================================
# CREATE QUERY EMBEDDING
# =============================================================================

def create_query_embedding(model, query: str):

    print_separator()
    print("CREATING QUERY EMBEDDING")
    print("=" * 25)

    print(f"Query: {query}")

    start = time.perf_counter()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    elapsed = time.perf_counter() - start

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    print(f"Embedding shape: {query_embedding.shape}")
    print(f"Embedding time : {elapsed * 1000:.2f} ms")

    # Safety check.
    norm = np.linalg.norm(query_embedding[0])

    print(f"Query vector norm: {norm:.6f}")

    return query_embedding


# =============================================================================
# PRODUCT INFORMATION
# =============================================================================

def get_product_info(
    product_index: int,
    product_ids: pd.DataFrame,
    metadata: pd.DataFrame,
):
    """Safely retrieve product information."""

    product_id_row = product_ids.iloc[product_index]
    metadata_row = metadata.iloc[product_index]

    asin = product_id_row.get(
        "parent_asin",
        "N/A"
    )

    title = metadata_row.get(
        "title",
        "N/A"
    )

    store = metadata_row.get(
        "store",
        "N/A"
    )

    rating = metadata_row.get(
        "average_rating",
        metadata_row.get("rating", "N/A")
    )

    price = metadata_row.get(
        "price",
        "N/A"
    )

    categories = metadata_row.get(
        "categories",
        "N/A"
    )

    return {
        "asin": format_value(asin),
        "title": format_value(title, 600),
        "store": format_value(store, 150),
        "rating": format_value(rating, 50),
        "price": format_value(price, 50),
        "categories": format_value(categories, 500),
    }


# =============================================================================
# SEARCH
# =============================================================================

def search_products(
    index,
    query_embedding,
    product_ids,
    metadata,
    top_k: int,
):

    print_separator()
    print("SEARCHING FAISS")
    print("=" * 16)

    # Search extra vectors so we can safely remove duplicates
    # or any exact query match if one happens to exist.
    search_k = min(
        top_k + SEARCH_BUFFER,
        index.ntotal
    )

    print(f"FAISS search candidates: {search_k}")

    start = time.perf_counter()

    similarities, indices = index.search(
        query_embedding,
        search_k
    )

    elapsed = time.perf_counter() - start

    print(
        f"FAISS search time: "
        f"{elapsed * 1000:.4f} ms"
    )

    results = []

    seen_indices = set()

    for similarity, product_index in zip(
        similarities[0],
        indices[0]
    ):

        product_index = int(product_index)

        if product_index < 0:
            continue

        if product_index in seen_indices:
            continue

        seen_indices.add(product_index)

        product = get_product_info(
            product_index,
            product_ids,
            metadata
        )

        product["embedding_index"] = product_index
        product["similarity"] = float(similarity)
        product["similarity_percent"] = float(
            similarity * 100
        )

        results.append(product)

        if len(results) >= top_k:
            break

    return results, elapsed


# =============================================================================
# DISPLAY RESULTS
# =============================================================================

def display_results(
    query: str,
    results,
):

    print_separator()
    print("NATURAL LANGUAGE SEARCH RESULTS")
    print("=" * 34)

    print(f"\nQuery:")
    print(query)

    print(f"\nResults returned: {len(results)}")

    for rank, product in enumerate(results, start=1):

        print_separator()
        print(f"#{rank}")
        print()

        print(
            f"Similarity      : "
            f"{product['similarity']:.4f}"
        )

        print(
            f"Similarity (%)  : "
            f"{product['similarity_percent']:.2f}%"
        )

        print(
            f"Embedding index : "
            f"{product['embedding_index']}"
        )

        print(
            f"ASIN            : "
            f"{product['asin']}"
        )

        print(
            f"Title           : "
            f"{product['title']}"
        )

        print(
            f"Store           : "
            f"{product['store']}"
        )

        print(
            f"Rating          : "
            f"{product['rating']}"
        )

        print(
            f"Price           : "
            f"{product['price']}"
        )

        print(
            f"Categories      : "
            f"{product['categories']}"
        )


# =============================================================================
# SAVE SEARCH RESULTS
# =============================================================================

def save_results(
    query: str,
    results,
    search_time,
):

    output_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "search"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / "last_natural_language_search.json"
    )

    output = {
        "query": query,
        "model": MODEL_NAME,
        "search_method": "FAISS IndexFlatIP",
        "similarity_metric": "cosine_similarity",
        "search_time_ms": search_time * 1000,
        "results": results,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print_separator()
    print("SEARCH OUTPUT")
    print("=" * 14)

    print(f"Results saved to:")
    print(output_path)


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "ShopGraph natural language "
            "semantic product search using FAISS."
        )
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help=(
            "Natural language product search query."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            f"Number of products to return "
            f"(default: {DEFAULT_TOP_K})."
        ),
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_arguments()

    if not args.query.strip():
        print("ERROR: Query cannot be empty.")
        sys.exit(1)

    if args.top_k < 1:
        print("ERROR: --top-k must be at least 1.")
        sys.exit(1)

    print_separator()
    print("SHOPGRAPH — NATURAL LANGUAGE PRODUCT SEARCH")
    print("=" * 48)

    print("\nSemantic search using:")
    print("Sentence Transformer + FAISS")

    print(f"\nQuery : {args.query}")
    print(f"Top-K : {args.top_k}")

    # -------------------------------------------------------------------------
    # Files
    # -------------------------------------------------------------------------

    check_files()

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    index, product_ids, metadata = load_search_data()

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------

    model = load_embedding_model()

    # -------------------------------------------------------------------------
    # Query embedding
    # -------------------------------------------------------------------------

    query_embedding = create_query_embedding(
        model,
        args.query
    )

    # -------------------------------------------------------------------------
    # FAISS search
    # -------------------------------------------------------------------------

    results, search_time = search_products(
        index,
        query_embedding,
        product_ids,
        metadata,
        args.top_k,
    )

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    display_results(
        args.query,
        results
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    save_results(
        args.query,
        results,
        search_time
    )

    # -------------------------------------------------------------------------
    # Complete
    # -------------------------------------------------------------------------

    print_separator()
    print("SEARCH COMPLETE")
    print("=" * 16)


if __name__ == "__main__":
    main()