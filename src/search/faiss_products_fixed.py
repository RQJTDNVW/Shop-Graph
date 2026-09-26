from pathlib import Path
import time
import json

import faiss
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDINGS_FILE = Path(
    "data/processed/product_embeddings_fixed.npy"
)

INDEX_FILE = Path(
    "data/processed/search/faiss_products_fixed.index"
)

STATS_FILE = Path(
    "data/processed/search/faiss_products_fixed_stats.json"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SHOPGRAPH — FIXED FAISS INDEX BUILDER")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embeddings not found:\n"
            f"{EMBEDDINGS_FILE.resolve()}"
        )

    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------

    print("\nLoading fixed embeddings...")

    embeddings = np.load(
        EMBEDDINGS_FILE,
        mmap_mode="r"
    )

    print(
        f"Shape: {embeddings.shape}"
    )

    print(
        f"Dtype: {embeddings.dtype}"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if embeddings.ndim != 2:
        raise RuntimeError(
            "Embeddings must be a 2D matrix."
        )

    if embeddings.shape != (100_000, 384):
        raise RuntimeError(
            f"Unexpected embedding shape: "
            f"{embeddings.shape}"
        )

    if np.isnan(embeddings).any():
        raise RuntimeError(
            "Embeddings contain NaN values."
        )

    if np.isinf(embeddings).any():
        raise RuntimeError(
            "Embeddings contain Inf values."
        )

    # --------------------------------------------------------
    # Make writable float32 copy
    # --------------------------------------------------------

    print("\nPreparing vectors for FAISS...")

    vectors = np.array(
        embeddings,
        dtype=np.float32,
        copy=True
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    print("Normalizing vectors...")

    start = time.time()

    faiss.normalize_L2(
        vectors
    )

    normalization_time = (
        time.time() - start
    )

    norms = np.linalg.norm(
        vectors,
        axis=1
    )

    max_norm_error = np.max(
        np.abs(norms - 1.0)
    )

    print(
        f"Normalization time: "
        f"{normalization_time:.3f}s"
    )

    print(
        f"Maximum norm error: "
        f"{max_norm_error:.8f}"
    )

    # --------------------------------------------------------
    # Create FAISS index
    # --------------------------------------------------------

    dimension = vectors.shape[1]

    print(
        f"\nCreating FAISS IndexFlatIP..."
    )

    index = faiss.IndexFlatIP(
        dimension
    )

    # --------------------------------------------------------
    # Add vectors
    # --------------------------------------------------------

    print(
        f"Adding {len(vectors):,} vectors..."
    )

    start = time.time()

    index.add(
        vectors
    )

    add_time = (
        time.time() - start
    )

    print(
        f"Vectors added: "
        f"{index.ntotal:,}"
    )

    print(
        f"Add time: "
        f"{add_time:.3f}s"
    )

    # --------------------------------------------------------
    # Validate index
    # --------------------------------------------------------

    if index.ntotal != len(vectors):
        raise RuntimeError(
            "FAISS index size does not match "
            "embedding count."
        )

    print(
        "\nFAISS validation: OK"
    )

    # --------------------------------------------------------
    # Save index
    # --------------------------------------------------------

    INDEX_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nSaving FAISS index...")

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    # --------------------------------------------------------
    # Test search
    # --------------------------------------------------------

    print("\nRunning test search...")

    query = vectors[0:1]

    start = time.time()

    distances, indices = index.search(
        query,
        10
    )

    search_time = (
        time.time() - start
    )

    print(
        f"Search time: "
        f"{search_time * 1000:.3f} ms"
    )

    print("\nTop 10 nearest embedding indices:")

    for rank, (idx, score) in enumerate(
        zip(indices[0], distances[0]),
        start=1
    ):

        print(
            f"  #{rank:02d} "
            f"index={idx:,} "
            f"similarity={score:.6f}"
        )

    # --------------------------------------------------------
    # File information
    # --------------------------------------------------------

    index_size_mb = (
        INDEX_FILE.stat().st_size
        / (1024 * 1024)
    )

    print(
        f"\nIndex size: "
        f"{index_size_mb:.2f} MB"
    )

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    stats = {
        "embedding_file": str(
            EMBEDDINGS_FILE
        ),
        "index_file": str(
            INDEX_FILE
        ),
        "num_vectors": int(
            index.ntotal
        ),
        "dimension": int(
            dimension
        ),
        "index_type": "IndexFlatIP",
        "similarity": "cosine_similarity",
        "max_norm_error": float(
            max_norm_error
        ),
        "normalization_time_seconds": float(
            normalization_time
        ),
        "add_time_seconds": float(
            add_time
        ),
        "test_search_time_ms": float(
            search_time * 1000
        ),
        "index_size_mb": float(
            index_size_mb
        )
    }

    with open(
        STATS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            stats,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FIXED FAISS INDEX BUILD COMPLETE")
    print("=" * 70)

    print(
        f"\nIndex saved to:\n"
        f"{INDEX_FILE.resolve()}"
    )

    print(
        f"\nStatistics saved to:\n"
        f"{STATS_FILE.resolve()}"
    )

    print(
        f"\nVectors: {index.ntotal:,}"
    )

    print(
        f"Dimension: {dimension}"
    )

    print(
        f"Index size: {index_size_mb:.2f} MB"
    )


if __name__ == "__main__":
    main()