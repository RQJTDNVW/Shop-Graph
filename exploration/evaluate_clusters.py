"""
SHOPGRAPH — CLUSTER EVALUATION

Purpose:
    Evaluate the final K-Means clustering and inspect what products
    belong to each cluster.

Input:
    data/processed/pca/product_embeddings_pca50.npy
    data/processed/embedding_product_ids.parquet
    data/processed/products_text.parquet

Output:
    data/processed/clustering/kmeans_clusters.parquet
    data/processed/clustering/kmeans_cluster_summary.parquet
    data/processed/clustering/kmeans_cluster_summary.csv
    data/processed/clustering/kmeans_cluster_evaluation.json
"""

from pathlib import Path
import json
import ast
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


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

ID_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embedding_product_ids.parquet"
)

PRODUCT_FILE = (
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

CLUSTERS_FILE = OUTPUT_DIR / "kmeans_clusters.parquet"
SUMMARY_FILE = OUTPUT_DIR / "kmeans_cluster_summary.parquet"
SUMMARY_CSV = OUTPUT_DIR / "kmeans_cluster_summary.csv"
EVALUATION_FILE = OUTPUT_DIR / "kmeans_cluster_evaluation.json"

K = 30
RANDOM_STATE = 42
N_INIT = 10
MAX_ITER = 300

# Number of representative products printed per cluster
REPRESENTATIVE_PRODUCTS = 5


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_value(value):
    """
    Convert list-like / string-like values into readable text.
    """

    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    if isinstance(value, (list, tuple, np.ndarray)):
        values = []

        for item in value:
            if item is None:
                continue

            if isinstance(item, float) and np.isnan(item):
                continue

            text = str(item).strip()

            if text:
                values.append(text)

        return " | ".join(values)

    text = str(value).strip()

    if not text:
        return ""

    # Handle strings that contain Python-style lists
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, (list, tuple)):
                values = []

                for item in parsed:
                    item_text = str(item).strip()

                    if item_text:
                        values.append(item_text)

                return " | ".join(values)

        except Exception:
            pass

    return text


def get_top_values(series, top_n=5):
    """
    Return the most common non-empty values.
    """

    cleaned = series.apply(clean_value)

    cleaned = cleaned[
        cleaned.astype(str).str.strip() != ""
    ]

    if cleaned.empty:
        return []

    counts = cleaned.value_counts().head(top_n)

    return [
        {
            "value": str(index),
            "count": int(count),
        }
        for index, count in counts.items()
    ]


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("SHOPGRAPH — K-MEANS CLUSTER EVALUATION")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------------------------------
    # LOAD PCA EMBEDDINGS
    # -------------------------------------------------------------------------

    print("\nLoading PCA-50 embeddings:")

    print(PCA_FILE)

    embeddings = np.load(PCA_FILE)

    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Products: {len(embeddings):,}")
    print(f"Dimensions: {embeddings.shape[1]}")

    # -------------------------------------------------------------------------
    # LOAD PRODUCT IDS
    # -------------------------------------------------------------------------

    print("\nLoading product IDs...")

    id_df = pd.read_parquet(ID_FILE)

    print(f"Product ID rows: {len(id_df):,}")

    # Try to automatically identify the ID column
    possible_id_columns = [
        "parent_asin",
        "product_id",
        "asin",
    ]

    id_column = None

    for column in possible_id_columns:
        if column in id_df.columns:
            id_column = column
            break

    if id_column is None:
        if len(id_df.columns) == 1:
            id_column = id_df.columns[0]
        else:
            raise ValueError(
                "Could not identify the product ID column."
            )

    product_ids = id_df[id_column].astype(str).reset_index(drop=True)

    if len(product_ids) != len(embeddings):
        raise ValueError(
            "Number of product IDs does not match number of embeddings."
        )

    # -------------------------------------------------------------------------
    # LOAD PRODUCT METADATA
    # -------------------------------------------------------------------------

    print("\nLoading product metadata...")

    products = pd.read_parquet(PRODUCT_FILE)

    print(f"Metadata rows: {len(products):,}")

    # -------------------------------------------------------------------------
    # MERGE PRODUCT IDS
    # -------------------------------------------------------------------------

    print("\nPreparing product records...")

    products[id_column] = products[id_column].astype(str)

    # Keep one row per product ID
    products = products.drop_duplicates(
        subset=[id_column]
    ).copy()

    cluster_base = pd.DataFrame({
        id_column: product_ids
    })

    cluster_base = cluster_base.merge(
        products,
        on=id_column,
        how="left",
        suffixes=("", "_metadata")
    )

    # -------------------------------------------------------------------------
    # TRAIN FINAL K-MEANS
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print(f"TRAINING FINAL K-MEANS — K = {K}")
    print("=" * 80)

    print("\nParameters:")
    print(f"Clusters       : {K}")
    print(f"Random state   : {RANDOM_STATE}")
    print(f"N init         : {N_INIT}")
    print(f"Max iterations : {MAX_ITER}")

    model = KMeans(
        n_clusters=K,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
        max_iter=MAX_ITER,
    )

    print("\nTraining...")

    labels = model.fit_predict(embeddings)

    print("Training complete!")

    # -------------------------------------------------------------------------
    # BASIC MODEL INFORMATION
    # -------------------------------------------------------------------------

    print("\nModel inertia:")
    print(f"{model.inertia_:,.2f}")

    print("\nIterations:")
    print(model.n_iter_)

    # -------------------------------------------------------------------------
    # ADD CLUSTER LABELS
    # -------------------------------------------------------------------------

    cluster_base["cluster_id"] = labels

    # -------------------------------------------------------------------------
    # CLUSTER SIZE ANALYSIS
    # -------------------------------------------------------------------------

    cluster_counts = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
    )

    print("\n" + "=" * 80)
    print("CLUSTER SIZE ANALYSIS")
    print("=" * 80)

    print(
        f"\nNumber of clusters: {len(cluster_counts)}"
    )

    print(
        f"Largest cluster : {cluster_counts.max():,}"
    )

    print(
        f"Smallest cluster: {cluster_counts.min():,}"
    )

    print(
        f"Average cluster : {cluster_counts.mean():,.2f}"
    )

    print(
        f"Median cluster  : {cluster_counts.median():,.2f}"
    )

    print("\nCluster sizes:")

    for cluster_id, count in cluster_counts.items():

        percentage = (
            count / len(labels)
        ) * 100

        print(
            f"Cluster {cluster_id:02d}: "
            f"{count:6,} products "
            f"({percentage:5.2f}%)"
        )

    # -------------------------------------------------------------------------
    # CLUSTER SUMMARY
    # -------------------------------------------------------------------------

    summary_rows = []

    print("\n" + "=" * 80)
    print("CLUSTER CONTENT INSPECTION")
    print("=" * 80)

    for cluster_id in range(K):

        cluster_products = cluster_base[
            cluster_base["cluster_id"] == cluster_id
        ].copy()

        cluster_size = len(cluster_products)

        print("\n" + "-" * 80)
        print(
            f"CLUSTER {cluster_id} "
            f"({cluster_size:,} products)"
        )
        print("-" * 80)

        # ---------------------------------------------------------------------
        # REPRESENTATIVE PRODUCTS
        # ---------------------------------------------------------------------

        representative = cluster_products.head(
            REPRESENTATIVE_PRODUCTS
        )

        print("\nRepresentative products:")

        for _, row in representative.iterrows():

            title = clean_value(
                row.get("title", "")
            )

            store = clean_value(
                row.get("store", "")
            )

            if not title:
                title = "Unknown title"

            if not store:
                store = "Unknown store"

            print(
                f"  • {title[:140]}"
            )

            print(
                f"    Store: {store[:80]}"
            )

        # ---------------------------------------------------------------------
        # TOP STORES
        # ---------------------------------------------------------------------

        top_stores = get_top_values(
            cluster_products["store"],
            top_n=5
        )

        if top_stores:

            print("\nTop stores/brands:")

            for item in top_stores:

                print(
                    f"  • {item['value'][:80]} "
                    f"({item['count']:,})"
                )

        # ---------------------------------------------------------------------
        # TOP CATEGORIES
        # ---------------------------------------------------------------------

        top_categories = get_top_values(
            cluster_products["category_text"],
            top_n=5
        )

        if top_categories:

            print("\nTop categories:")

            for item in top_categories:

                print(
                    f"  • {item['value'][:100]} "
                    f"({item['count']:,})"
                )

        # ---------------------------------------------------------------------
        # RATINGS
        # ---------------------------------------------------------------------

        rating_mean = None

        if "average_rating" in cluster_products.columns:

            rating_values = pd.to_numeric(
                cluster_products["average_rating"],
                errors="coerce"
            )

            if rating_values.notna().any():

                rating_mean = float(
                    rating_values.mean()
                )

                print(
                    f"\nAverage rating: "
                    f"{rating_mean:.2f}"
                )

        # ---------------------------------------------------------------------
        # SUMMARY RECORD
        # ---------------------------------------------------------------------

        summary_rows.append({
            "cluster_id": int(cluster_id),
            "product_count": int(cluster_size),
            "percentage": float(
                cluster_size / len(labels) * 100
            ),
            "average_rating": rating_mean,
            "top_stores": top_stores,
            "top_categories": top_categories,
        })

    # -------------------------------------------------------------------------
    # SAVE CLUSTER ASSIGNMENTS
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("SAVING CLUSTER ASSIGNMENTS")
    print("=" * 80)

    # Save only useful columns plus metadata
    cluster_base.to_parquet(
        CLUSTERS_FILE,
        index=False
    )

    print(
        f"\nSaved:\n{CLUSTERS_FILE}"
    )

    # -------------------------------------------------------------------------
    # SAVE SUMMARY
    # -------------------------------------------------------------------------

    summary_df = pd.DataFrame(summary_rows)

    # Convert nested structures to JSON strings for Parquet/CSV compatibility
    summary_df["top_stores"] = summary_df[
        "top_stores"
    ].apply(json.dumps)

    summary_df["top_categories"] = summary_df[
        "top_categories"
    ].apply(json.dumps)

    summary_df.to_parquet(
        SUMMARY_FILE,
        index=False
    )

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False
    )

    print(
        f"Saved:\n{SUMMARY_FILE}"
    )

    print(
        f"Saved:\n{SUMMARY_CSV}"
    )

    # -------------------------------------------------------------------------
    # SAVE JSON EVALUATION
    # -------------------------------------------------------------------------

    evaluation = {
        "model": "KMeans",
        "k": K,
        "random_state": RANDOM_STATE,
        "n_init": N_INIT,
        "max_iter": MAX_ITER,
        "embedding_shape": list(
            embeddings.shape
        ),
        "product_count": int(
            len(embeddings)
        ),
        "dimensions": int(
            embeddings.shape[1]
        ),
        "inertia": float(
            model.inertia_
        ),
        "iterations": int(
            model.n_iter_
        ),
        "cluster_count": int(K),
        "smallest_cluster": int(
            cluster_counts.min()
        ),
        "largest_cluster": int(
            cluster_counts.max()
        ),
        "average_cluster_size": float(
            cluster_counts.mean()
        ),
        "median_cluster_size": float(
            cluster_counts.median()
        ),
        "clusters": summary_rows,
    }

    with open(
        EVALUATION_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            evaluation,
            f,
            indent=4
        )

    print(
        f"Saved:\n{EVALUATION_FILE}"
    )

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("K-MEANS CLUSTER EVALUATION COMPLETE")
    print("=" * 80)

    print(
        f"\nFinal K: {K}"
    )

    print(
        f"Products assigned: {len(labels):,}"
    )

    print(
        f"Clusters created: {K}"
    )

    print(
        f"Smallest cluster: {cluster_counts.min():,}"
    )

    print(
        f"Largest cluster: {cluster_counts.max():,}"
    )

    print("\nOutput files:")

    print(
        f"  Cluster assignments:\n  {CLUSTERS_FILE}"
    )

    print(
        f"  Cluster summary:\n  {SUMMARY_FILE}"
    )

    print(
        f"  CSV summary:\n  {SUMMARY_CSV}"
    )

    print(
        f"  Evaluation JSON:\n  {EVALUATION_FILE}"
    )

    print("\nNext step:")

    print(
        "Inspect the cluster themes and determine whether "
        "K=30 produces meaningful product groups."
    )


if __name__ == "__main__":
    main()