from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SHOPGRAPH — PRODUCT EMBEDDING VISUALIZATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "pca"
    / "product_embeddings_pca2.parquet"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "pca"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "pca_product_space.png"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SHOPGRAPH — PRODUCT EMBEDDING VISUALIZATION")
    print("=" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nPCA file not found:\n{INPUT_FILE}"
        )

    print("\nLoading PCA representation...")
    print(INPUT_FILE)

    df = pd.read_parquet(INPUT_FILE)

    print(
        f"\nProducts loaded: {len(df):,}"
    )

    required_columns = [
        "pc1",
        "pc2",
        "parent_asin",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Create plot
    # --------------------------------------------------------

    print("\nCreating 2D PCA visualization...")

    plt.figure(
        figsize=(14, 10)
    )

    plt.scatter(
        df["pc1"],
        df["pc2"],
        s=2,
        alpha=0.25,
    )

    plt.xlabel(
        "Principal Component 1"
    )

    plt.ylabel(
        "Principal Component 2"
    )

    plt.title(
        "ShopGraph — Amazon Electronics Product Embedding Space"
    )

    plt.grid(
        alpha=0.15
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    plt.savefig(
        OUTPUT_FILE,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("\nVisualization saved:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()