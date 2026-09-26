from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ============================================================
# SHOPGRAPH — EMBEDDING ANALYSIS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

EMBEDDINGS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "product_embeddings.npy"
)

PRODUCT_IDS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "embedding_product_ids.parquet"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "pca"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# CONFIGURATION
# ============================================================

# Components used for the main PCA representation.
PCA_COMPONENTS = 50

# Components used for 2D visualization.
PCA_2D_COMPONENTS = 2


# ============================================================
# HELPER FUNCTION
# ============================================================

def components_for_variance(explained_variance, target):
    """
    Return the minimum number of PCA components required
    to reach the requested cumulative explained variance.
    """

    cumulative = np.cumsum(explained_variance)

    indices = np.where(cumulative >= target)[0]

    if len(indices) == 0:
        return len(explained_variance)

    return int(indices[0] + 1)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SHOPGRAPH — EMBEDDING ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"\nEmbeddings file not found:\n{EMBEDDINGS_FILE}"
        )

    if not PRODUCT_IDS_FILE.exists():
        raise FileNotFoundError(
            f"\nProduct ID mapping not found:\n{PRODUCT_IDS_FILE}"
        )

    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------

    print("\nLoading embeddings:")
    print(EMBEDDINGS_FILE)

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    print(
        f"\nEmbedding shape: {embeddings.shape}"
    )

    print(
        f"Embedding dtype: {embeddings.dtype}"
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2-dimensional matrix."
        )

    if np.isnan(embeddings).any():
        raise ValueError(
            "Embeddings contain NaN values."
        )

    if np.isinf(embeddings).any():
        raise ValueError(
            "Embeddings contain infinite values."
        )

    n_products, embedding_dimension = embeddings.shape

    print(
        f"Products: {n_products:,}"
    )

    print(
        f"Dimensions: {embedding_dimension}"
    )

    # --------------------------------------------------------
    # Load product ID mapping
    # --------------------------------------------------------

    print("\nLoading product ID mapping...")

    product_ids = pd.read_parquet(
        PRODUCT_IDS_FILE
    )

    print(
        f"Product IDs loaded: {len(product_ids):,}"
    )

    if len(product_ids) != n_products:
        raise ValueError(
            "Number of product IDs does not match "
            "number of embedding vectors."
        )

    # --------------------------------------------------------
    # PCA explained variance analysis
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("PCA EXPLAINED VARIANCE ANALYSIS")
    print("=" * 80)

    print(
        "\nFitting PCA with all available components..."
    )

    # PCA with all components allows us to determine
    # how many dimensions are needed to preserve
    # different amounts of information.
    pca_full = PCA(
        n_components=None,
        svd_solver="full"
    )

    pca_full.fit(embeddings)

    explained_variance = (
        pca_full.explained_variance_ratio_
    )

    cumulative_variance = np.cumsum(
        explained_variance
    )

    # --------------------------------------------------------
    # Variance thresholds
    # --------------------------------------------------------

    components_80 = components_for_variance(
        explained_variance,
        0.80
    )

    components_90 = components_for_variance(
        explained_variance,
        0.90
    )

    components_95 = components_for_variance(
        explained_variance,
        0.95
    )

    components_99 = components_for_variance(
        explained_variance,
        0.99
    )

    print("\nComponents required:")

    print(
        f"80% variance : {components_80}"
    )

    print(
        f"90% variance : {components_90}"
    )

    print(
        f"95% variance : {components_95}"
    )

    print(
        f"99% variance : {components_99}"
    )

    # --------------------------------------------------------
    # First 20 components
    # --------------------------------------------------------

    print("\nExplained variance by first 20 components:")

    for i in range(
        min(20, len(explained_variance))
    ):

        print(
            f"PC{i + 1:02d}: "
            f"{explained_variance[i] * 100:.4f}% "
            f"| cumulative: "
            f"{cumulative_variance[i] * 100:.4f}%"
        )

    # --------------------------------------------------------
    # Save variance information
    # --------------------------------------------------------

    variance_df = pd.DataFrame({
        "component": np.arange(
            1,
            len(explained_variance) + 1
        ),
        "explained_variance_ratio": explained_variance,
        "explained_variance_percent":
            explained_variance * 100,
        "cumulative_variance_ratio":
            cumulative_variance,
        "cumulative_variance_percent":
            cumulative_variance * 100,
    })

    variance_file = (
        OUTPUT_DIR
        / "pca_explained_variance.parquet"
    )

    variance_df.to_parquet(
        variance_file,
        index=False
    )

    # --------------------------------------------------------
    # PCA 50-dimensional representation
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("CREATING PCA REPRESENTATION")
    print("=" * 80)

    actual_components = min(
        PCA_COMPONENTS,
        embedding_dimension
    )

    print(
        f"\nReducing {embedding_dimension} dimensions "
        f"to {actual_components} dimensions..."
    )

    pca_50 = PCA(
        n_components=actual_components,
        svd_solver="randomized",
        random_state=42
    )

    embeddings_pca = pca_50.fit_transform(
        embeddings
    )

    pca_50_variance = (
        pca_50.explained_variance_ratio_.sum()
    )

    print(
        f"PCA representation shape: "
        f"{embeddings_pca.shape}"
    )

    print(
        f"Variance retained: "
        f"{pca_50_variance * 100:.2f}%"
    )

    pca_50_file = (
        OUTPUT_DIR
        / "product_embeddings_pca50.npy"
    )

    np.save(
        pca_50_file,
        embeddings_pca.astype(np.float32)
    )

    # --------------------------------------------------------
    # PCA 2D representation
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("CREATING 2D PCA REPRESENTATION")
    print("=" * 80)

    pca_2d = PCA(
        n_components=PCA_2D_COMPONENTS,
        random_state=42
    )

    embeddings_2d = pca_2d.fit_transform(
        embeddings
    )

    print(
        f"\n2D PCA shape: "
        f"{embeddings_2d.shape}"
    )

    pca_2d_variance = (
        pca_2d.explained_variance_ratio_.sum()
    )

    print(
        f"2D variance retained: "
        f"{pca_2d_variance * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Create 2D dataframe
    # --------------------------------------------------------

    pca_2d_df = pd.DataFrame({
        "embedding_index":
            product_ids["embedding_index"].values,

        "parent_asin":
            product_ids["parent_asin"].values,

        "pc1":
            embeddings_2d[:, 0],

        "pc2":
            embeddings_2d[:, 1],
    })

    pca_2d_file = (
        OUTPUT_DIR
        / "product_embeddings_pca2.parquet"
    )

    pca_2d_df.to_parquet(
        pca_2d_file,
        index=False
    )

    # --------------------------------------------------------
    # Save PCA statistics
    # --------------------------------------------------------

    stats = {
        "original_products": int(n_products),

        "original_dimensions": int(
            embedding_dimension
        ),

        "components_for_80_percent": int(
            components_80
        ),

        "components_for_90_percent": int(
            components_90
        ),

        "components_for_95_percent": int(
            components_95
        ),

        "components_for_99_percent": int(
            components_99
        ),

        "pca_50_dimensions": int(
            actual_components
        ),

        "pca_50_variance_retained": float(
            pca_50_variance
        ),

        "pca_2_dimensions": 2,

        "pca_2_variance_retained": float(
            pca_2d_variance
        ),

        "embedding_model": "all-MiniLM-L6-v2",

        "random_state": 42,
    }

    stats_file = (
        OUTPUT_DIR
        / "pca_stats.json"
    )

    with open(
        stats_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            stats,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("PCA ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        f"\nOriginal representation : "
        f"{n_products:,} × {embedding_dimension}"
    )

    print(
        f"PCA representation     : "
        f"{embeddings_pca.shape}"
    )

    print(
        f"PCA-50 variance        : "
        f"{pca_50_variance * 100:.2f}%"
    )

    print(
        f"PCA-2 variance         : "
        f"{pca_2d_variance * 100:.2f}%"
    )

    print("\nSaved files:")

    print(
        f"Variance data           : "
        f"{variance_file}"
    )

    print(
        f"PCA-50 embeddings       : "
        f"{pca_50_file}"
    )

    print(
        f"PCA-2 embeddings        : "
        f"{pca_2d_file}"
    )

    print(
        f"PCA statistics          : "
        f"{stats_file}"
    )

    print("\nNext step: visualize the product embedding space.")


if __name__ == "__main__":
    main()