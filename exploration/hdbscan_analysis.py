"""
SHOPGRAPH — HDBSCAN CLUSTER ANALYSIS
Phase 3.4 — Density-Based Product Discovery

Uses scikit-learn's HDBSCAN implementation so the project
remains compatible with Python 3.12 and avoids the blocked
external hdbscan native extension.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
import sklearn


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PCA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pca"
    / "product_embeddings_pca50.npy"
)

PRODUCT_IDS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embedding_product_ids.parquet"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "products_text.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clustering"
)

OUTPUT_CLUSTERS = OUTPUT_DIR / "hdbscan_clusters.parquet"
OUTPUT_SUMMARY = OUTPUT_DIR / "hdbscan_cluster_summary.parquet"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "hdbscan_cluster_summary.csv"
OUTPUT_STATS = OUTPUT_DIR / "hdbscan_analysis_stats.json"


# HDBSCAN configuration selected from parameter search
MIN_CLUSTER_SIZE = 50
MIN_SAMPLES = 10

METRIC = "euclidean"
CLUSTER_SELECTION_METHOD = "eom"

# Maximum points used for silhouette calculation
SILHOUETTE_SAMPLE_SIZE = 20_000

# Number of representative products per cluster
REPRESENTATIVE_PRODUCTS = 10

RANDOM_STATE = 42


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# =============================================================================
# FILE VALIDATION
# =============================================================================

def check_input_files():
    print_section("CHECKING INPUT FILES")

    files = {
        "PCA embeddings": PCA_FILE,
        "Product IDs": PRODUCT_IDS_FILE,
        "Product metadata": METADATA_FILE,
    }

    for name, path in files.items():
        if path.exists():
            print(f"OK: {path}")
        else:
            raise FileNotFoundError(
                f"\nMissing {name}:\n{path}"
            )


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():

    print_section("LOADING DATA")

    print("Loading PCA-50 embeddings...")

    embeddings = np.load(PCA_FILE)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")

    print("\nLoading product IDs...")

    product_ids = pd.read_parquet(PRODUCT_IDS_FILE)

    print(f"Product ID columns: {list(product_ids.columns)}")

    if "parent_asin" not in product_ids.columns:
        raise ValueError(
            "Expected 'parent_asin' column in product ID file."
        )

    print(f"Product IDs: {len(product_ids):,}")

    print("\nLoading product metadata...")

    metadata = pd.read_parquet(METADATA_FILE)

    print(f"Metadata shape: {metadata.shape}")

    return embeddings, product_ids, metadata


# =============================================================================
# DATA ALIGNMENT
# =============================================================================

def validate_alignment(embeddings, product_ids, metadata):

    print_section("VALIDATING DATA ALIGNMENT")

    n_embeddings = len(embeddings)
    n_ids = len(product_ids)
    n_metadata = len(metadata)

    print(f"Embeddings : {n_embeddings:,}")
    print(f"Product IDs: {n_ids:,}")
    print(f"Metadata   : {n_metadata:,}")

    if not (
        n_embeddings == n_ids == n_metadata
    ):
        raise ValueError(
            "Dataset alignment failed. "
            "Embeddings, IDs and metadata have different lengths."
        )

    if product_ids["parent_asin"].isna().any():
        raise ValueError(
            "Product ID column contains missing parent_asin values."
        )

    if product_ids["parent_asin"].duplicated().any():
        duplicates = product_ids["parent_asin"].duplicated().sum()

        raise ValueError(
            f"Found {duplicates:,} duplicate product IDs."
        )

    print("Alignment successful.")


# =============================================================================
# HDBSCAN
# =============================================================================

def run_hdbscan(embeddings):

    print_section("RUNNING HDBSCAN")

    print("Configuration:")
    print(f"min_cluster_size       : {MIN_CLUSTER_SIZE}")
    print(f"min_samples             : {MIN_SAMPLES}")
    print(f"metric                  : {METRIC}")
    print(f"cluster_selection       : {CLUSTER_SELECTION_METHOD}")
    print(f"random_state            : {RANDOM_STATE}")

    print(f"\nClustering {len(embeddings):,} products...")
    print("This may take several minutes.")

    # Suppress the known sklearn future warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The default value of `copy` will change",
            category=FutureWarning,
        )

        clusterer = HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=MIN_SAMPLES,
            metric=METRIC,
            cluster_selection_method=CLUSTER_SELECTION_METHOD,
            copy=False,
        )

        labels = clusterer.fit_predict(embeddings)

    return labels, clusterer


# =============================================================================
# CLUSTER STATISTICS
# =============================================================================

def calculate_cluster_statistics(labels):

    print_section("HDBSCAN RESULTS")

    unique_labels, counts = np.unique(
        labels,
        return_counts=True
    )

    noise_count = int(np.sum(labels == -1))

    cluster_labels = unique_labels[
        unique_labels != -1
    ]

    cluster_counts = counts[
        unique_labels != -1
    ]

    number_of_clusters = len(cluster_labels)

    total_products = len(labels)

    noise_percentage = (
        noise_count / total_products * 100
    )

    print(f"Clusters found: {number_of_clusters}")
    print(f"Noise products: {noise_count:,}")
    print(f"Noise percentage: {noise_percentage:.2f}%")

    if number_of_clusters > 0:

        print("\nCluster sizes:")

        for label, count in zip(
            cluster_labels,
            cluster_counts
        ):
            print(
                f"Cluster {label}: {count:,}"
            )

        print(
            f"\nSmallest cluster: "
            f"{cluster_counts.min():,}"
        )

        print(
            f"Largest cluster: "
            f"{cluster_counts.max():,}"
        )

    return {
        "number_of_clusters": int(number_of_clusters),
        "noise_count": noise_count,
        "noise_percentage": float(noise_percentage),
        "smallest_cluster": (
            int(cluster_counts.min())
            if len(cluster_counts) > 0
            else 0
        ),
        "largest_cluster": (
            int(cluster_counts.max())
            if len(cluster_counts) > 0
            else 0
        ),
    }


# =============================================================================
# SILHOUETTE
# =============================================================================

def calculate_silhouette(embeddings, labels):

    print_section("SILHOUETTE EVALUATION")

    # Remove HDBSCAN noise.
    valid_indices = np.where(labels != -1)[0]

    if len(valid_indices) < 2:
        print(
            "Not enough non-noise samples for silhouette."
        )
        return None

    valid_labels = labels[valid_indices]

    unique_clusters = np.unique(valid_labels)

    if len(unique_clusters) < 2:
        print(
            "Fewer than two clusters remain after "
            "removing noise."
        )
        return None

    rng = np.random.default_rng(RANDOM_STATE)

    if len(valid_indices) > SILHOUETTE_SAMPLE_SIZE:

        selected = rng.choice(
            valid_indices,
            size=SILHOUETTE_SAMPLE_SIZE,
            replace=False,
        )

    else:
        selected = valid_indices

    selected_labels = labels[selected]

    print(
        f"Calculating silhouette on "
        f"{len(selected):,} non-noise samples..."
    )

    score = silhouette_score(
        embeddings[selected],
        selected_labels,
        metric=METRIC,
    )

    print(
        f"Silhouette score: {score:.4f}"
    )

    return float(score)


# =============================================================================
# METADATA HELPERS
# =============================================================================

def find_column(df, candidates):

    for column in candidates:
        if column in df.columns:
            return column

    return None


def get_top_value(series):

    series = series.dropna()

    if len(series) == 0:
        return None

    values = series.astype(str).str.strip()

    values = values[
        values != ""
    ]

    if len(values) == 0:
        return None

    return values.value_counts().index[0]


# =============================================================================
# CLUSTER SUMMARY
# =============================================================================

def build_cluster_summary(
    labels,
    product_ids,
    metadata,
):

    print_section("BUILDING CLUSTER SUMMARY")

    df = product_ids.copy()

    df["cluster"] = labels

    # -------------------------------------------------------------------------
    # Add metadata
    # -------------------------------------------------------------------------

    metadata_reset = metadata.reset_index(drop=True)

    for column in metadata_reset.columns:

        if column == "parent_asin":
            continue

        if column not in df.columns:
            df[column] = metadata_reset[column].values

    total_products = len(df)

    rows = []

    store_column = find_column(
        df,
        [
            "store",
            "brand",
            "brand_name",
        ],
    )

    category_column = find_column(
        df,
        [
            "categories",
            "category",
            "category_name",
        ],
    )

    rating_column = find_column(
        df,
        [
            "average_rating",
            "rating",
            "ratings",
        ],
    )

    title_column = find_column(
        df,
        [
            "title",
            "product_title",
        ],
    )

    # -------------------------------------------------------------------------
    # Build summaries
    # -------------------------------------------------------------------------

    for cluster_id, group in df.groupby(
        "cluster",
        sort=True
    ):

        cluster_size = len(group)

        cluster_type = (
            "noise"
            if cluster_id == -1
            else "cluster"
        )

        percentage = (
            cluster_size
            / total_products
            * 100
        )

        average_rating = None

        if rating_column:

            numeric_ratings = pd.to_numeric(
                group[rating_column],
                errors="coerce",
            )

            if numeric_ratings.notna().any():
                average_rating = float(
                    numeric_ratings.mean()
                )

        top_store = None

        if store_column:
            top_store = get_top_value(
                group[store_column]
            )

        top_category = None

        if category_column:
            top_category = get_top_value(
                group[category_column]
            )

        # ---------------------------------------------------------------------
        # Representative products
        #
        # We use the first products for now. Later we can replace this with
        # medoid / centroid-nearest representatives.
        # ---------------------------------------------------------------------

        representative_ids = (
            group["parent_asin"]
            .astype(str)
            .head(REPRESENTATIVE_PRODUCTS)
            .tolist()
        )

        representative_titles = []

        if title_column:

            representative_titles = (
                group[title_column]
                .dropna()
                .astype(str)
                .head(REPRESENTATIVE_PRODUCTS)
                .tolist()
            )

        rows.append(
            {
                "cluster": int(cluster_id),
                "cluster_size": int(cluster_size),
                "cluster_type": cluster_type,
                "cluster_percentage": round(
                    percentage,
                    4,
                ),
                "top_store": top_store,
                "top_category": top_category,
                "average_rating": average_rating,
                "representative_product_ids":
                    representative_ids,
                "representative_product_titles":
                    representative_titles,
            }
        )

    summary = pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Sort: noise last, then largest clusters first
    # -------------------------------------------------------------------------

    summary["_sort_noise"] = (
        summary["cluster"]
        == -1
    ).astype(int)

    summary = (
        summary
        .sort_values(
            [
                "_sort_noise",
                "cluster_size",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop(columns="_sort_noise")
        .reset_index(drop=True)
    )

    return summary, df


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_outputs(
    cluster_df,
    summary,
    stats,
):

    print_section("SAVING OUTPUTS")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cluster_df.to_parquet(
        OUTPUT_CLUSTERS,
        index=False,
    )

    summary.to_parquet(
        OUTPUT_SUMMARY,
        index=False,
    )

    # CSV cannot store Python lists cleanly,
    # so convert representative lists to JSON strings.

    csv_summary = summary.copy()

    for column in [
        "representative_product_ids",
        "representative_product_titles",
    ]:

        if column in csv_summary.columns:

            csv_summary[column] = (
                csv_summary[column]
                .apply(
                    lambda x: json.dumps(x)
                    if isinstance(x, list)
                    else x
                )
            )

    csv_summary.to_csv(
        OUTPUT_SUMMARY_CSV,
        index=False,
    )

    with open(
        OUTPUT_STATS,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            stats,
            f,
            indent=4,
        )

    print(f"Clusters saved:")
    print(OUTPUT_CLUSTERS)

    print(f"\nSummary saved:")
    print(OUTPUT_SUMMARY)

    print(f"\nCSV saved:")
    print(OUTPUT_SUMMARY_CSV)

    print(f"\nStatistics saved:")
    print(OUTPUT_STATS)


# =============================================================================
# PRINT SUMMARY
# =============================================================================

def print_cluster_summary(summary):

    print_section(
        "TOP HDBSCAN CLUSTERS"
    )

    display_columns = [
        "cluster",
        "cluster_size",
        "cluster_percentage",
        "top_store",
        "top_category",
        "average_rating",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in summary.columns
    ]

    print(
        summary[available_columns]
        .head(20)
        .to_string(index=False)
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_section(
        "SHOPGRAPH — HDBSCAN CLUSTER ANALYSIS"
    )

    print(
        f"scikit-learn version: "
        f"{sklearn.__version__}"
    )

    print(
        "Using scikit-learn HDBSCAN."
    )

    print(
        "Configuration selected from "
        "the parameter search:"
    )

    print(
        f"min_cluster_size={MIN_CLUSTER_SIZE}, "
        f"min_samples={MIN_SAMPLES}"
    )

    # -------------------------------------------------------------------------
    # Check files
    # -------------------------------------------------------------------------

    check_input_files()

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    embeddings, product_ids, metadata = (
        load_data()
    )

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    validate_alignment(
        embeddings,
        product_ids,
        metadata,
    )

    # -------------------------------------------------------------------------
    # Cluster
    # -------------------------------------------------------------------------

    labels, clusterer = run_hdbscan(
        embeddings
    )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    cluster_stats = (
        calculate_cluster_statistics(
            labels
        )
    )

    # -------------------------------------------------------------------------
    # Silhouette
    # -------------------------------------------------------------------------

    silhouette = calculate_silhouette(
        embeddings,
        labels,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary, cluster_df = (
        build_cluster_summary(
            labels,
            product_ids,
            metadata,
        )
    )

    print_cluster_summary(summary)

    # -------------------------------------------------------------------------
    # Full statistics
    # -------------------------------------------------------------------------

    stats = {
        "algorithm": "HDBSCAN",
        "implementation": (
            "sklearn.cluster.HDBSCAN"
        ),
        "sklearn_version": sklearn.__version__,
        "dataset_size": int(len(labels)),
        "embedding_dimensions": int(
            embeddings.shape[1]
        ),
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "min_samples": MIN_SAMPLES,
        "metric": METRIC,
        "cluster_selection_method":
            CLUSTER_SELECTION_METHOD,
        "random_state": RANDOM_STATE,
        "number_of_clusters":
            cluster_stats[
                "number_of_clusters"
            ],
        "noise_count":
            cluster_stats[
                "noise_count"
            ],
        "noise_percentage":
            cluster_stats[
                "noise_percentage"
            ],
        "smallest_cluster":
            cluster_stats[
                "smallest_cluster"
            ],
        "largest_cluster":
            cluster_stats[
                "largest_cluster"
            ],
        "silhouette_score":
            silhouette,
    }

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    save_outputs(
        cluster_df,
        summary,
        stats,
    )

    # -------------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------------

    print_section(
        "HDBSCAN ANALYSIS COMPLETE"
    )

    print(
        f"Clusters: "
        f"{stats['number_of_clusters']}"
    )

    print(
        f"Noise: "
        f"{stats['noise_count']:,} "
        f"({stats['noise_percentage']:.2f}%)"
    )

    if silhouette is not None:

        print(
            f"Silhouette: "
            f"{silhouette:.4f}"
        )

    print(
        "\nHDBSCAN 50/10 clustering "
        "has been saved successfully."
    )


if __name__ == "__main__":
    main()