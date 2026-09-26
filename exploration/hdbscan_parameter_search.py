"""
ShopGraph — HDBSCAN Parameter Search

Tests multiple HDBSCAN parameter combinations on PCA-50
product embeddings and compares the resulting clustering
structures.

Input:
    data/processed/pca/product_embeddings_pca50.npy

Output:
    data/processed/clustering/hdbscan_parameter_search.csv
"""


from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PCA_EMBEDDINGS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "pca"
    / "product_embeddings_pca50.npy"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "clustering"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

SILHOUETTE_SAMPLE_SIZE = 20_000


# ============================================================
# PARAMETER COMBINATIONS
# ============================================================

PARAMETER_COMBINATIONS = [
    {
        "min_cluster_size": 50,
        "min_samples": 10,
    },
    {
        "min_cluster_size": 100,
        "min_samples": 10,
    },
    {
        "min_cluster_size": 100,
        "min_samples": 20,
    },
    {
        "min_cluster_size": 250,
        "min_samples": 20,
    },
    {
        "min_cluster_size": 500,
        "min_samples": 30,
    },
]


# ============================================================
# START
# ============================================================

print("=" * 80)
print("SHOPGRAPH — HDBSCAN PARAMETER SEARCH")
print("=" * 80)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading PCA-50 embeddings...")

if not PCA_EMBEDDINGS_PATH.exists():

    raise FileNotFoundError(
        f"\nPCA embeddings not found:\n"
        f"{PCA_EMBEDDINGS_PATH}"
    )

X = np.load(
    PCA_EMBEDDINGS_PATH
)

print(
    f"Embedding shape: {X.shape}"
)

print(
    f"Embedding dtype: {X.dtype}"
)


# ============================================================
# VALIDATION
# ============================================================

if np.isnan(X).any():

    raise ValueError(
        "Embeddings contain NaN values."
    )

if np.isinf(X).any():

    raise ValueError(
        "Embeddings contain infinite values."
    )


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = []


for experiment_number, params in enumerate(
    PARAMETER_COMBINATIONS,
    start=1,
):

    min_cluster_size = params[
        "min_cluster_size"
    ]

    min_samples = params[
        "min_samples"
    ]

    print("\n" + "=" * 80)

    print(
        f"EXPERIMENT {experiment_number}/"
        f"{len(PARAMETER_COMBINATIONS)}"
    )

    print("=" * 80)

    print(
        f"\nmin_cluster_size: "
        f"{min_cluster_size}"
    )

    print(
        f"min_samples: "
        f"{min_samples}"
    )

    print(
        "\nRunning HDBSCAN..."
    )

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )

    labels = clusterer.fit_predict(
        X
    )

    # --------------------------------------------------------
    # CLUSTER COUNTS
    # --------------------------------------------------------

    unique_labels = np.unique(
        labels
    )

    real_clusters = unique_labels[
        unique_labels != -1
    ]

    number_of_clusters = len(
        real_clusters
    )

    noise_count = int(
        np.sum(
            labels == -1
        )
    )

    noise_percentage = (
        noise_count
        / len(labels)
        * 100
    )

    # --------------------------------------------------------
    # CLUSTER SIZES
    # --------------------------------------------------------

    cluster_sizes = (
        pd.Series(labels)
        .value_counts()
    )

    real_cluster_sizes = (
        cluster_sizes[
            cluster_sizes.index != -1
        ]
    )

    if len(real_cluster_sizes) > 0:

        smallest_cluster = int(
            real_cluster_sizes.min()
        )

        largest_cluster = int(
            real_cluster_sizes.max()
        )

    else:

        smallest_cluster = 0
        largest_cluster = 0

    # --------------------------------------------------------
    # SILHOUETTE
    # --------------------------------------------------------

    silhouette = None

    valid_mask = (
        labels != -1
    )

    X_valid = X[
        valid_mask
    ]

    labels_valid = labels[
        valid_mask
    ]

    if (
        len(X_valid) >= 2
        and len(
            np.unique(labels_valid)
        ) >= 2
    ):

        sample_size = min(
            SILHOUETTE_SAMPLE_SIZE,
            len(X_valid),
        )

        rng = np.random.default_rng(
            RANDOM_STATE
        )

        sample_indices = rng.choice(
            len(X_valid),
            size=sample_size,
            replace=False,
        )

        X_sample = X_valid[
            sample_indices
        ]

        labels_sample = labels_valid[
            sample_indices
        ]

        if len(
            np.unique(labels_sample)
        ) >= 2:

            silhouette = silhouette_score(
                X_sample,
                labels_sample,
            )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"\nClusters: "
        f"{number_of_clusters}"
    )

    print(
        f"Noise: "
        f"{noise_count:,}"
    )

    print(
        f"Noise %: "
        f"{noise_percentage:.2f}%"
    )

    print(
        f"Smallest cluster: "
        f"{smallest_cluster:,}"
    )

    print(
        f"Largest cluster: "
        f"{largest_cluster:,}"
    )

    if silhouette is not None:

        print(
            f"Silhouette: "
            f"{silhouette:.4f}"
        )

    else:

        print(
            "Silhouette: N/A"
        )

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append(
        {
            "experiment": experiment_number,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "number_of_clusters": (
                number_of_clusters
            ),
            "noise_count": noise_count,
            "noise_percentage": (
                noise_percentage
            ),
            "smallest_cluster": (
                smallest_cluster
            ),
            "largest_cluster": (
                largest_cluster
            ),
            "silhouette_score": (
                silhouette
                if silhouette is not None
                else None
            ),
        }
    )


# ============================================================
# CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("HDBSCAN PARAMETER COMPARISON")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = (
    OUTPUT_DIR
    / "hdbscan_parameter_search.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("PARAMETER SEARCH COMPLETE")
print("=" * 80)

print(
    f"\nResults saved to:"
)

print(
    output_path
)

print("\nDone.")