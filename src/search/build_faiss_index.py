"""
SHOPGRAPH — FAISS INDEX BUILDER
Phase 4.2 — Fast Semantic Product Search

Builds a FAISS vector index from the existing
384-dimensional MiniLM product embeddings.

Input:
    data/processed/product_embeddings.npy

Output:
    data/processed/search/faiss_products.index
    data/processed/search/faiss_index_stats.json

Method:
    L2-normalized embeddings + Inner Product

Because the embeddings are normalized:
    Inner Product == Cosine Similarity
"""

from pathlib import Path
import json
import time

import numpy as np
import faiss


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "product_embeddings.npy"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "search"
)

INDEX_FILE = (
    OUTPUT_DIR
    / "faiss_products.index"
)

STATS_FILE = (
    OUTPUT_DIR
    / "faiss_index_stats.json"
)


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_DIMENSION = 384


# ============================================================
# UTILITY
# ============================================================

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# CHECK INPUT
# ============================================================

def check_input():

    print_section("CHECKING INPUT")

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"\nEmbeddings file not found:\n"
            f"{EMBEDDINGS_FILE}"
        )

    print(f"OK: {EMBEDDINGS_FILE}")


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings():

    print_section("LOADING EMBEDDINGS")

    print("Loading embeddings...")

    embeddings = np.load(
        EMBEDDINGS_FILE,
        mmap_mode="r"
    )

    print(f"Shape: {embeddings.shape}")
    print(f"Dtype: {embeddings.dtype}")

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if embeddings.ndim != 2:

        raise ValueError(
            "Embeddings must be a 2-dimensional array."
        )

    number_of_products = embeddings.shape[0]
    dimension = embeddings.shape[1]

    if dimension != EXPECTED_DIMENSION:

        raise ValueError(
            f"Unexpected embedding dimension: {dimension}\n"
            f"Expected: {EXPECTED_DIMENSION}"
        )

    if number_of_products == 0:

        raise ValueError(
            "Embedding file contains zero products."
        )

    print(
        f"Products: {number_of_products:,}"
    )

    print(
        f"Dimensions: {dimension}"
    )

    return embeddings


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_index(embeddings):

    print_section("BUILDING FAISS INDEX")

    dimension = embeddings.shape[1]

    print(
        "Creating exact inner-product FAISS index..."
    )

    # --------------------------------------------------------
    # IndexFlatIP
    #
    # Inner product is equivalent to cosine similarity
    # when vectors are L2 normalized.
    # --------------------------------------------------------

    index = faiss.IndexFlatIP(
        dimension
    )

    print(
        f"Vector dimension: {dimension}"
    )

    print(
        f"Products: {len(embeddings):,}"
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # The .npy file is loaded using mmap_mode='r',
    # which produces read-only memory.
    #
    # FAISS normalization modifies vectors in-place.
    #
    # Therefore we explicitly create a writable copy.
    # --------------------------------------------------------

    print(
        "\nCopying embeddings into writable memory..."
    )

    start_copy = time.time()

    vectors = np.array(
        embeddings,
        dtype=np.float32,
        copy=True
    )

    copy_time = (
        time.time() - start_copy
    )

    print(
        f"Writable copy created in "
        f"{copy_time:.2f} seconds."
    )

    # --------------------------------------------------------
    # Normalize vectors
    # --------------------------------------------------------

    print(
        "\nNormalizing embeddings..."
    )

    start_normalization = time.time()

    faiss.normalize_L2(
        vectors
    )

    normalization_time = (
        time.time() - start_normalization
    )

    print(
        f"Normalization complete in "
        f"{normalization_time:.2f} seconds."
    )

    # --------------------------------------------------------
    # Verify normalization
    # --------------------------------------------------------

    print(
        "\nVerifying vector normalization..."
    )

    sample_count = min(
        100,
        len(vectors)
    )

    sample_norms = np.linalg.norm(
        vectors[:sample_count],
        axis=1
    )

    max_norm_error = float(
        np.max(
            np.abs(
                sample_norms - 1.0
            )
        )
    )

    print(
        f"Maximum sample norm error: "
        f"{max_norm_error:.8f}"
    )

    if max_norm_error > 1e-4:

        raise ValueError(
            "Embedding normalization verification failed."
        )

    print(
        "Normalization verification: OK"
    )

    # --------------------------------------------------------
    # Add vectors to FAISS
    # --------------------------------------------------------

    print(
        "\nAdding vectors to FAISS..."
    )

    start_add = time.time()

    index.add(
        vectors
    )

    add_time = (
        time.time() - start_add
    )

    print(
        f"Vectors added: "
        f"{index.ntotal:,}"
    )

    print(
        f"FAISS add time: "
        f"{add_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Verify index count
    # --------------------------------------------------------

    if index.ntotal != len(embeddings):

        raise ValueError(
            "\nFAISS index count does not match "
            "the embedding count.\n"
            f"Embeddings: {len(embeddings):,}\n"
            f"FAISS index: {index.ntotal:,}"
        )

    print(
        "Index count verification: OK"
    )

    total_time = (
        copy_time
        + normalization_time
        + add_time
    )

    return index, total_time


# ============================================================
# SAVE INDEX
# ============================================================

def save_index(
    index,
    build_time
):

    print_section("SAVING INDEX")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save FAISS index
    # --------------------------------------------------------

    print(
        "Writing FAISS index..."
    )

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    print(
        f"FAISS index saved:\n"
        f"{INDEX_FILE}"
    )

    # --------------------------------------------------------
    # Collect statistics
    # --------------------------------------------------------

    index_size_mb = (
        INDEX_FILE.stat().st_size
        / (1024 * 1024)
    )

    stats = {
        "project": "ShopGraph",
        "phase": "4.2",
        "index_type": "IndexFlatIP",
        "search_method": "exact_inner_product",
        "similarity_metric": "cosine_similarity",
        "embedding_dimension": int(
            index.d
        ),
        "number_of_products": int(
            index.ntotal
        ),
        "normalized_embeddings": True,
        "build_time_seconds": round(
            build_time,
            4
        ),
        "index_size_mb": round(
            index_size_mb,
            4
        ),
        "source_embeddings": str(
            EMBEDDINGS_FILE
        ),
        "index_file": str(
            INDEX_FILE
        )
    }

    # --------------------------------------------------------
    # Save statistics JSON
    # --------------------------------------------------------

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

    print(
        f"\nStatistics saved:\n"
        f"{STATS_FILE}"
    )

    print(
        f"\nIndex size: "
        f"{index_size_mb:.2f} MB"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_section(
        "SHOPGRAPH — FAISS INDEX BUILDER"
    )

    print(
        "Building a fast semantic-search index "
        "from the existing MiniLM embeddings."
    )

    print(
        "\nFAISS index type: IndexFlatIP"
    )

    print(
        "Similarity: Cosine similarity"
    )

    print(
        "Embeddings: 384-dimensional MiniLM"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    check_input()

    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------

    embeddings = load_embeddings()

    # --------------------------------------------------------
    # Build index
    # --------------------------------------------------------

    index, build_time = build_index(
        embeddings
    )

    # --------------------------------------------------------
    # Save index
    # --------------------------------------------------------

    save_index(
        index,
        build_time
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print_section("FAISS BUILD COMPLETE")

    print(
        "FAISS index successfully created."
    )

    print(
        f"\nProducts indexed : "
        f"{index.ntotal:,}"
    )

    print(
        f"Dimensions       : "
        f"{index.d}"
    )

    print(
        "Similarity        : "
        "Cosine similarity"
    )

    print(
        "Index type        : "
        "IndexFlatIP"
    )

    print(
        f"Build time        : "
        f"{build_time:.2f} seconds"
    )

    print(
        "\nOutput:"
    )

    print(
        f"  {INDEX_FILE}"
    )

    print(
        f"  {STATS_FILE}"
    )

    print(
        "\nNext step: "
        "connect this FAISS index to the "
        "ShopGraph similarity-search system."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()