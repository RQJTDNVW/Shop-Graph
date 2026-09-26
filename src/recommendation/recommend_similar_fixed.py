from pathlib import Path
import time

import faiss
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDINGS_FILE = Path(
    "data/processed/product_embeddings_fixed.npy"
)

IDS_FILE = Path(
    "data/processed/embedding_product_ids_fixed.parquet"
)

METADATA_FILE = Path(
    "data/processed/products_text_fixed.parquet"
)

FAISS_INDEX_FILE = Path(
    "data/processed/search/faiss_products_fixed.index"
)

OUTPUT_FILE = Path(
    "data/processed/search/last_similar_recommendations_fixed.json"
)

DEFAULT_DEMO_ASIN = "B00MCW7G9M"


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    if value is None:
        return ""

    if isinstance(value, (list, tuple)):
        return " ".join(
            str(x)
            for x in value
            if x is not None
        )

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def safe_float(value):
    try:
        if value is None:
            return None

        if pd.isna(value):
            return None

        return float(value)

    except (TypeError, ValueError):
        return None


def display_price(value):
    price = safe_float(value)

    if price is None:
        return "N/A"

    return f"${price:,.2f}"


def product_to_dict(row):
    title = clean_text(
        row.get("title", "")
    )

    if not title:
        title = clean_text(
            row.get("clean_title", "")
        )

    store = clean_text(
        row.get("store", "")
    )

    if not store:
        store = clean_text(
            row.get("clean_store", "")
        )

    categories = clean_text(
        row.get("categories", "")
    )

    if not categories:
        categories = clean_text(
            row.get("category_text", "")
        )

    return {
        "parent_asin": clean_text(
            row.get("parent_asin", "")
        ),
        "title": title if title else "N/A",
        "store": store if store else "N/A",
        "categories": (
            categories
            if categories
            else "N/A"
        ),
        "price": safe_float(
            row.get("price")
        ),
        "price_display": display_price(
            row.get("price")
        ),
        "average_rating": safe_float(
            row.get("average_rating")
        ),
        "rating_number": safe_float(
            row.get("rating_number")
        )
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading fixed recommendation data...")

    embeddings = np.load(
        EMBEDDINGS_FILE,
        mmap_mode="r"
    )

    ids_df = pd.read_parquet(
        IDS_FILE,
        engine="fastparquet"
    )

    metadata = pd.read_parquet(
        METADATA_FILE,
        engine="fastparquet"
    )

    index = faiss.read_index(
        str(FAISS_INDEX_FILE)
    )

    print(
        f"Embeddings: {embeddings.shape}"
    )

    print(
        f"IDs: {len(ids_df):,}"
    )

    print(
        f"Metadata: {len(metadata):,}"
    )

    print(
        f"FAISS vectors: {index.ntotal:,}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if len(embeddings) != len(ids_df):
        raise RuntimeError(
            "Embedding / ID count mismatch."
        )

    if len(embeddings) != len(metadata):
        raise RuntimeError(
            "Embedding / metadata count mismatch."
        )

    if index.ntotal != len(embeddings):
        raise RuntimeError(
            "FAISS / embedding count mismatch."
        )

    if (
        "embedding_index" not in ids_df.columns
        or "parent_asin" not in ids_df.columns
    ):
        raise RuntimeError(
            "Invalid ID mapping columns."
        )

    if "parent_asin" not in metadata.columns:
        raise RuntimeError(
            "Metadata missing parent_asin."
        )

    print(
        "\nAlignment: OK"
    )

    return (
        embeddings,
        ids_df,
        metadata,
        index
    )


# ============================================================
# BUILD METADATA LOOKUP
# ============================================================

def build_metadata_lookup(metadata):

    lookup = {}

    for _, row in metadata.iterrows():

        asin = clean_text(
            row["parent_asin"]
        )

        if asin and asin not in lookup:
            lookup[asin] = row

    return lookup


# ============================================================
# FIND EMBEDDING ROW
# ============================================================

def find_embedding_index(
    asin,
    ids_df
):

    matches = ids_df.index[
        ids_df["parent_asin"].astype(str)
        == str(asin)
    ].tolist()

    if not matches:
        return None

    row = matches[0]

    return int(
        ids_df.iloc[row][
            "embedding_index"
        ]
    )


# ============================================================
# RECOMMEND
# ============================================================

def recommend(
    asin,
    top_k,
    embeddings,
    ids_df,
    metadata,
    index
):

    metadata_lookup = (
        build_metadata_lookup(metadata)
    )

    embedding_index = (
        find_embedding_index(
            asin,
            ids_df
        )
    )

    if embedding_index is None:
        raise ValueError(
            f"ASIN not found: {asin}"
        )

    if embedding_index >= len(embeddings):
        raise RuntimeError(
            "Embedding index outside embedding matrix."
        )

    # --------------------------------------------------------
    # Source product
    # --------------------------------------------------------

    source_asin = clean_text(asin)

    if source_asin not in metadata_lookup:
        raise RuntimeError(
            "Source ASIN exists in embedding "
            "mapping but not metadata."
        )

    source_row = metadata_lookup[
        source_asin
    ]

    source_product = product_to_dict(
        source_row
    )

    # --------------------------------------------------------
    # Query vector
    # --------------------------------------------------------

    query_vector = np.array(
        embeddings[
            embedding_index
        ],
        dtype=np.float32,
        copy=True
    ).reshape(1, -1)

    faiss.normalize_L2(
        query_vector
    )

    # Search extra candidates because
    # the source itself must be removed.
    candidate_k = max(
        top_k * 10,
        100
    )

    print(
        f"\nCandidate search: "
        f"{candidate_k}"
    )

    start = time.time()

    scores, indices = index.search(
        query_vector,
        candidate_k
    )

    search_time = (
        time.time() - start
    )

    print(
        f"FAISS search: "
        f"{search_time * 1000:.3f} ms"
    )

    # --------------------------------------------------------
    # Build recommendations
    # --------------------------------------------------------

    recommendations = []

    seen_asins = set()

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        idx = int(idx)

        if idx < 0:
            continue

        if idx == embedding_index:
            continue

        if idx >= len(ids_df):
            continue

        candidate_asin = clean_text(
            ids_df.iloc[idx][
                "parent_asin"
            ]
        )

        if not candidate_asin:
            continue

        if candidate_asin == source_asin:
            continue

        if candidate_asin in seen_asins:
            continue

        if candidate_asin not in metadata_lookup:
            continue

        seen_asins.add(
            candidate_asin
        )

        candidate_row = metadata_lookup[
            candidate_asin
        ]

        product = product_to_dict(
            candidate_row
        )

        product["similarity"] = float(
            score
        )

        product["rank"] = (
            len(recommendations) + 1
        )

        recommendations.append(
            product
        )

        if len(recommendations) >= top_k:
            break

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        "source_product": source_product,
        "source_embedding_index": (
            embedding_index
        ),
        "recommendations": recommendations,
        "search_time_ms": (
            search_time * 1000
        ),
        "candidate_count": candidate_k
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(result):

    source = result[
        "source_product"
    ]

    print("\n" + "=" * 70)
    print("SOURCE PRODUCT")
    print("=" * 70)

    print(
        f"ASIN: {source['parent_asin']}"
    )

    print(
        f"Title: {source['title']}"
    )

    print(
        f"Store: {source['store']}"
    )

    print(
        f"Price: {source['price_display']}"
    )

    print(
        f"Rating: {source['average_rating']}"
    )

    print("\n" + "=" * 70)
    print("SIMILAR PRODUCT RECOMMENDATIONS")
    print("=" * 70)

    for product in result[
        "recommendations"
    ]:

        print(
            f"\n#{product['rank']}"
        )

        print(
            f"ASIN: "
            f"{product['parent_asin']}"
        )

        print(
            f"Title: "
            f"{product['title']}"
        )

        print(
            f"Store: "
            f"{product['store']}"
        )

        print(
            f"Price: "
            f"{product['price_display']}"
        )

        print(
            f"Similarity: "
            f"{product['similarity']:.4f}"
        )

    print("\n" + "=" * 70)

    print(
        f"Results returned: "
        f"{len(result['recommendations'])}"
    )

    print(
        f"FAISS search time: "
        f"{result['search_time_ms']:.3f} ms"
    )

    print("=" * 70)


# ============================================================
# SAVE
# ============================================================

def save_result(result):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_to_save = {
        "source_product": result[
            "source_product"
        ],
        "source_embedding_index": result[
            "source_embedding_index"
        ],
        "recommendations": result[
            "recommendations"
        ],
        "search_time_ms": result[
            "search_time_ms"
        ],
        "candidate_count": result[
            "candidate_count"
        ]
    }

    import json

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result_to_save,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_FILE.resolve()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "ShopGraph fixed similar-product "
            "recommendation engine"
        )
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--demo",
        action="store_true",
        help="Run demo recommendation"
    )

    group.add_argument(
        "--asin",
        type=str,
        help="Recommend products similar to an ASIN"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of recommendations"
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError(
            "top-k must be greater than zero."
        )

    asin = (
        DEFAULT_DEMO_ASIN
        if args.demo
        else args.asin.strip()
    )

    print("=" * 70)
    print("SHOPGRAPH — FIXED SIMILAR PRODUCT ENGINE")
    print("=" * 70)

    print(
        f"\nQuery ASIN: {asin}"
    )

    (
        embeddings,
        ids_df,
        metadata,
        index
    ) = load_data()

    result = recommend(
        asin=asin,
        top_k=args.top_k,
        embeddings=embeddings,
        ids_df=ids_df,
        metadata=metadata,
        index=index
    )

    print_results(
        result
    )

    save_result(
        result
    )


if __name__ == "__main__":
    main()