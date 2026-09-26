from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
)


# ============================================================
# SHOPGRAPH — K-MEANS CLUSTER ANALYSIS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PCA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "pca"
    / "product_embeddings_pca50.npy"
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
    / "clustering"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

K_VALUES = [
    5,
    10,
    15,
    20,
    25,
    30,
]

RANDOM_STATE = 42

N_INIT = 10

# Silhouette calculation on the full 100K dataset can be
# expensive. A representative sample gives us a practical
# comparison during model selection.
SILHOUETTE_SAMPLE_SIZE = 20_000


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SHOPGRAPH — K-MEANS CLUSTER ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not PCA_FILE.exists():
        raise FileNotFoundError(
            f"\nPCA-50 file not found:\n{PCA_FILE}"
        )

    if not PRODUCT_IDS_FILE.exists():
        raise FileNotFoundError(
            f"\nProduct ID mapping not found:\n{PRODUCT_IDS_FILE}"
        )

    # --------------------------------------------------------
    # Load PCA data
    # --------------------------------------------------------

    print("\nLoading PCA-50 embeddings:")
    print(PCA_FILE)

    X = np.load(
        PCA_FILE
    )

    print(
        f"\nEmbedding shape: {X.shape}"
    )

    if X.ndim != 2:
        raise ValueError(
            "PCA embeddings must be 2-dimensional."
        )

    if np.isnan(X).any():
        raise ValueError(
            "PCA embeddings contain NaN values."
        )

    if np.isinf(X).any():
        raise ValueError(
            "PCA embeddings contain infinite values."
        )

    # --------------------------------------------------------
    # Load product IDs
    # --------------------------------------------------------

    product_ids = pd.read_parquet(
        PRODUCT_IDS_FILE
    )

    if len(product_ids) != len(X):
        raise ValueError(
            "Product ID count does not match "
            "embedding count."
        )

    print(
        f"Products: {len(X):,}"
    )

    print(
        f"Dimensions: {X.shape[1]}"
    )

    # --------------------------------------------------------
    # Prepare silhouette sample
    # --------------------------------------------------------

    sample_size = min(
        SILHOUETTE_SAMPLE_SIZE,
        len(X)
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    sample_indices = rng.choice(
        len(X),
        size=sample_size,
        replace=False
    )

    X_sample = X[sample_indices]

    print(
        f"\nSilhouette sample size: "
        f"{sample_size:,}"
    )

    # --------------------------------------------------------
    # Run K-Means experiments
    # --------------------------------------------------------

    results = []

    print("\n" + "=" * 80)
    print("K-MEANS EXPERIMENTS")
    print("=" * 80)

    for k in K_VALUES:

        print(
            f"\n{'-' * 70}"
        )

        print(
            f"Testing K = {k}"
        )

        start_time = time.time()

        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=N_INIT,
            random_state=RANDOM_STATE,
            max_iter=300,
            verbose=0,
        )

        labels = model.fit_predict(
            X
        )

        training_time = (
            time.time() - start_time
        )

        # ----------------------------------------------------
        # Inertia
        # ----------------------------------------------------

        inertia = float(
            model.inertia_
        )

        # ----------------------------------------------------
        # Silhouette
        # ----------------------------------------------------

        sample_labels = labels[
            sample_indices
        ]

        silhouette = silhouette_score(
            X_sample,
            sample_labels,
            metric="euclidean"
        )

        # ----------------------------------------------------
        # Calinski-Harabasz
        # ----------------------------------------------------

        ch_score = calinski_harabasz_score(
            X_sample,
            sample_labels
        )

        # ----------------------------------------------------
        # Cluster sizes
        # ----------------------------------------------------

        unique_labels, counts = np.unique(
            labels,
            return_counts=True
        )

        smallest_cluster = int(
            counts.min()
        )

        largest_cluster = int(
            counts.max()
        )

        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette_score": float(
                silhouette
            ),
            "calinski_harabasz_score": float(
                ch_score
            ),
            "smallest_cluster": smallest_cluster,
            "largest_cluster": largest_cluster,
            "training_time_seconds": float(
                training_time
            ),
        })

        print(
            f"Inertia              : {inertia:,.2f}"
        )

        print(
            f"Silhouette score     : "
            f"{silhouette:.4f}"
        )

        print(
            f"Calinski-Harabasz   : "
            f"{ch_score:,.2f}"
        )

        print(
            f"Smallest cluster    : "
            f"{smallest_cluster:,}"
        )

        print(
            f"Largest cluster     : "
            f"{largest_cluster:,}"
        )

        print(
            f"Training time       : "
            f"{training_time:.2f} sec"
        )

    # --------------------------------------------------------
    # Results dataframe
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_file = (
        OUTPUT_DIR
        / "kmeans_comparison.parquet"
    )

    results_df.to_parquet(
        results_file,
        index=False
    )

    # Also save CSV for easy inspection.
    csv_file = (
        OUTPUT_DIR
        / "kmeans_comparison.csv"
    )

    results_df.to_csv(
        csv_file,
        index=False
    )

    # --------------------------------------------------------
    # Print comparison
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("K-MEANS COMPARISON")
    print("=" * 80)

    display_columns = [
        "k",
        "inertia",
        "silhouette_score",
        "calinski_harabasz_score",
        "smallest_cluster",
        "largest_cluster",
        "training_time_seconds",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Identify highest silhouette
    # --------------------------------------------------------

    best_silhouette_row = results_df.loc[
        results_df[
            "silhouette_score"
        ].idxmax()
    ]

    best_silhouette_k = int(
        best_silhouette_row["k"]
    )

    best_silhouette = float(
        best_silhouette_row[
            "silhouette_score"
        ]
    )

    # --------------------------------------------------------
    # Save experiment statistics
    # --------------------------------------------------------

    stats = {
        "products": int(len(X)),
        "dimensions": int(X.shape[1]),
        "k_values_tested": K_VALUES,
        "random_state": RANDOM_STATE,
        "n_init": N_INIT,
        "silhouette_sample_size": int(
            sample_size
        ),
        "highest_silhouette_k": (
            best_silhouette_k
        ),
        "highest_silhouette_score": (
            best_silhouette
        ),
    }

    stats_file = (
        OUTPUT_DIR
        / "kmeans_analysis_stats.json"
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
    print("K-MEANS ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        f"\nHighest silhouette score: "
        f"{best_silhouette:.4f}"
    )

    print(
        f"Corresponding K: "
        f"{best_silhouette_k}"
    )

    print("\nSaved files:")

    print(
        f"K-Means comparison: "
        f"{results_file}"
    )

    print(
        f"CSV comparison: "
        f"{csv_file}"
    )

    print(
        f"Analysis statistics: "
        f"{stats_file}"
    )

    print(
        "\nNext step: evaluate the resulting "
        "clusters and inspect what products "
        "belong to each cluster."
    )


if __name__ == "__main__":
    main()