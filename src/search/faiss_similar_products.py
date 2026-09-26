"""
SHOPGRAPH — FAISS SIMILAR PRODUCT SEARCH
Phase 4.2 — Fast Semantic Product Search

Uses the pre-built FAISS index to retrieve
semantically similar products.

Input:
    data/processed/search/faiss_products.index
    data/processed/embedding_product_ids.parquet
    data/processed/products_text.parquet

Query:
    Amazon parent_asin

Method:
    L2-normalized MiniLM embeddings
    + FAISS IndexFlatIP

Because the vectors are normalized:

    Inner Product == Cosine Similarity
"""


from pathlib import Path
import argparse
import time
import warnings

import numpy as np
import pandas as pd
import faiss


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "search"
    / "faiss_products.index"
)

EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "product_embeddings.npy"
)

PRODUCT_IDS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embedding_product_ids.parquet"
)

PRODUCT_TEXT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "products_text.parquet"
)


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_TOP_K = 10
DEFAULT_ASIN = "B09NN6SGV5"


# ============================================================
# DISPLAY
# ============================================================

def print_section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# CHECK FILES
# ============================================================

def check_files():

    print_section("CHECKING FILES")

    required_files = {
        "FAISS index": INDEX_FILE,
        "Embeddings": EMBEDDINGS_FILE,
        "Product IDs": PRODUCT_IDS_FILE,
        "Product metadata": PRODUCT_TEXT_FILE,
    }

    for name, path in required_files.items():

        if path.exists():

            print(f"OK: {path}")

        else:

            raise FileNotFoundError(
                f"\nMissing {name} file:\n{path}"
            )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print_section("LOADING SEARCH DATA")

    # --------------------------------------------------------
    # FAISS INDEX
    # --------------------------------------------------------

    print("Loading FAISS index...")

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    print(
        f"FAISS vectors: {index.ntotal:,}"
    )

    print(
        f"Vector dimension: {index.d}"
    )

    # --------------------------------------------------------
    # PRODUCT IDs
    # --------------------------------------------------------

    print("\nLoading product IDs...")

    product_ids = pd.read_parquet(
        PRODUCT_IDS_FILE
    )

    print(
        f"Product ID columns: "
        f"{list(product_ids.columns)}"
    )

    print(
        f"Products: {len(product_ids):,}"
    )

    if "parent_asin" not in product_ids.columns:

        raise ValueError(
            "Product ID file does not contain "
            "'parent_asin'."
        )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    print("\nLoading product metadata...")

    metadata = pd.read_parquet(
        PRODUCT_TEXT_FILE
    )

    print(
        f"Metadata shape: {metadata.shape}"
    )

    # --------------------------------------------------------
    # EMBEDDINGS
    #
    # Loaded only when needed for query vectors.
    # --------------------------------------------------------

    print("\nLoading embeddings...")

    embeddings = np.load(
        EMBEDDINGS_FILE,
        mmap_mode="r"
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # --------------------------------------------------------
    # ALIGNMENT VALIDATION
    # --------------------------------------------------------

    print("\nValidating alignment...")

    if index.ntotal != len(product_ids):

        raise ValueError(
            "FAISS index count does not match "
            "product ID count."
        )

    if len(embeddings) != len(product_ids):

        raise ValueError(
            "Embedding count does not match "
            "product ID count."
        )

    if len(metadata) != len(product_ids):

        raise ValueError(
            "Metadata count does not match "
            "product ID count."
        )

    if embeddings.shape[1] != index.d:

        raise ValueError(
            "Embedding dimension does not match "
            "FAISS index dimension."
        )

    print(
        "Dataset alignment: OK"
    )

    return (
        index,
        embeddings,
        product_ids,
        metadata,
    )


# ============================================================
# MISSING VALUE HANDLING
# ============================================================

def is_missing_scalar(value):

    if value is None:

        return True

    if isinstance(
        value,
        (np.ndarray, list, tuple, dict)
    ):

        return False

    try:

        result = pd.isna(value)

        if isinstance(
            result,
            (bool, np.bool_)
        ):

            return bool(result)

        return False

    except (
        TypeError,
        ValueError
    ):

        return False


# ============================================================
# FORMAT VALUES
# ============================================================

def format_value(
    value,
    max_length=500
):

    if is_missing_scalar(value):

        return "N/A"

    if isinstance(
        value,
        np.ndarray
    ):

        if value.size == 0:

            return "N/A"

        value = value.tolist()

    if isinstance(
        value,
        (list, tuple)
    ):

        if len(value) == 0:

            return "N/A"

        parts = []

        for item in value:

            if isinstance(
                item,
                np.ndarray
            ):

                item = item.tolist()

            if isinstance(
                item,
                (list, tuple)
            ):

                parts.append(
                    " > ".join(
                        str(x)
                        for x in item
                    )
                )

            else:

                parts.append(
                    str(item)
                )

        text = " | ".join(parts)

    elif isinstance(
        value,
        dict
    ):

        text = str(value)

    else:

        text = str(value)

    text = " ".join(
        text.split()
    )

    if len(text) > max_length:

        text = (
            text[:max_length - 3]
            + "..."
        )

    return text


# ============================================================
# FIND METADATA COLUMN
# ============================================================

def find_column(
    dataframe,
    candidates
):

    for column in candidates:

        if column in dataframe.columns:

            return column

    return None


# ============================================================
# GET PRODUCT INFO
# ============================================================

def get_product_info(
    index,
    product_ids,
    metadata
):

    asin = product_ids.iloc[index][
        "parent_asin"
    ]

    info = {
        "index": index,
        "parent_asin": asin,
        "title": None,
        "store": None,
        "rating": None,
        "price": None,
        "categories": None,
    }

    metadata_columns = {

        "title": [
            "title",
            "product_title",
        ],

        "store": [
            "store",
            "brand",
            "brand_name",
        ],

        "rating": [
            "average_rating",
            "rating",
            "ratings",
        ],

        "price": [
            "price",
        ],

        "categories": [
            "categories",
            "category",
            "category_name",
        ],
    }

    for output_name, candidates in (
        metadata_columns.items()
    ):

        column = find_column(
            metadata,
            candidates
        )

        if column is None:

            continue

        value = metadata.iloc[index][
            column
        ]

        if is_missing_scalar(value):

            info[output_name] = None

        else:

            info[output_name] = value

    return info


# ============================================================
# FIND ASIN
# ============================================================

def find_product_index(
    asin,
    product_ids
):

    asin = str(
        asin
    ).strip()

    matches = np.where(
        product_ids[
            "parent_asin"
        ]
        .astype(str)
        .values
        == asin
    )[0]

    if len(matches) == 0:

        return None

    return int(
        matches[0]
    )


# ============================================================
# CREATE QUERY VECTOR
# ============================================================

def get_query_vector(
    index,
    embeddings
):

    print(
        "\nPreparing query vector..."
    )

    query_vector = np.array(
        embeddings[index],
        dtype=np.float32,
        copy=True
    ).reshape(
        1,
        -1
    )

    # --------------------------------------------------------
    # Normalize query vector.
    #
    # This must match the normalization used
    # when the FAISS index was built.
    # --------------------------------------------------------

    faiss.normalize_L2(
        query_vector
    )

    return query_vector


# ============================================================
# FAISS SEARCH
# ============================================================

def search_faiss(
    query_vector,
    index,
    top_k
):

    print(
        "\nSearching FAISS index..."
    )

    start_time = time.perf_counter()

    # Ask for one additional result because
    # the query product itself may appear first.
    search_k = min(
    top_k + 10,
    index.ntotal
    )

    similarities, indices = index.search(
        query_vector,
        search_k
    )

    elapsed_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    print(
        f"FAISS search time: "
        f"{elapsed_ms:.4f} ms"
    )

    results = []

    for similarity, index_id in zip(
        similarities[0],
        indices[0]
    ):

        index_id = int(
            index_id
        )

        if index_id < 0:

            continue

        results.append(
            (
                index_id,
                float(similarity)
            )
        )

        if len(results) >= top_k:

            break

    return results, elapsed_ms


# ============================================================
# DISPLAY QUERY
# ============================================================

def display_query_product(
    product_info
):

    print_section("QUERY PRODUCT")

    print(
        f"Embedding index : "
        f"{product_info['index']}"
    )

    print(
        f"ASIN            : "
        f"{format_value(product_info['parent_asin'])}"
    )

    print(
        f"Title           : "
        f"{format_value(product_info['title'])}"
    )

    print(
        f"Store           : "
        f"{format_value(product_info['store'])}"
    )

    print(
        f"Rating          : "
        f"{format_value(product_info['rating'])}"
    )

    print(
        f"Price           : "
        f"{format_value(product_info['price'])}"
    )

    print(
        f"Categories      : "
        f"{format_value(product_info['categories'])}"
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    results,
    product_ids,
    metadata
):

    print_section(
        "FAISS SIMILAR PRODUCTS"
    )

    if not results:

        print(
            "No similar products found."
        )

        return

    for rank, (
        index,
        similarity
    ) in enumerate(
        results,
        start=1
    ):

        info = get_product_info(
            index,
            product_ids,
            metadata
        )

        print(
            f"\n#{rank}"
        )

        print(
            f"Similarity      : "
            f"{similarity:.4f}"
        )

        print(
            f"Similarity (%)  : "
            f"{similarity * 100:.2f}%"
        )

        print(
            f"Embedding index : "
            f"{index}"
        )

        print(
            f"ASIN            : "
            f"{format_value(info['parent_asin'])}"
        )

        print(
            f"Title           : "
            f"{format_value(info['title'])}"
        )

        print(
            f"Store           : "
            f"{format_value(info['store'])}"
        )

        print(
            f"Rating          : "
            f"{format_value(info['rating'])}"
        )

        print(
            f"Price           : "
            f"{format_value(info['price'])}"
        )

        print(
            f"Categories      : "
            f"{format_value(info['categories'])}"
        )


# ============================================================
# SEARCH BY ASIN
# ============================================================

def search_by_asin(
    asin,
    index,
    embeddings,
    product_ids,
    metadata,
    top_k
):

    product_index = find_product_index(
        asin,
        product_ids
    )

    if product_index is None:

        print(
            f"\nProduct not found: {asin}"
        )

        print(
            "\nPlease provide an actual "
            "parent_asin from the dataset."
        )

        return False

    print(
        f"\nFound product at embedding "
        f"index {product_index}."
    )

    # --------------------------------------------------------
    # Query information
    # --------------------------------------------------------

    query_info = get_product_info(
        product_index,
        product_ids,
        metadata
    )

    display_query_product(
        query_info
    )

    # --------------------------------------------------------
    # Query vector
    # --------------------------------------------------------

    query_vector = get_query_vector(
        product_index,
        embeddings
    )

    # --------------------------------------------------------
    # FAISS search
    # --------------------------------------------------------

    results, search_time = search_faiss(
        query_vector,
        index,
        top_k
    )

    # Remove query product if returned.
    results = [
        (
            result_index,
            similarity
        )
        for result_index, similarity
        in results
        if result_index != product_index
    ]

    # Make sure we still have top_k after removing query.
    results = results[:top_k]

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    display_results(
        results,
        product_ids,
        metadata
    )

    # --------------------------------------------------------
    # Performance information
    # --------------------------------------------------------

    print_section(
        "SEARCH PERFORMANCE"
    )

    print(
        f"FAISS search time : "
        f"{search_time:.4f} ms"
    )

    print(
        f"Products searched : "
        f"{index.ntotal:,}"
    )

    print(
        f"Results returned  : "
        f"{len(results)}"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "ShopGraph FAISS "
            "similar-product search"
        )
    )

    parser.add_argument(
        "--asin",
        type=str,
        default=DEFAULT_ASIN,
        help=(
            "Amazon parent ASIN "
            "of the query product"
        )
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "Number of similar products "
            "to return. Default: 10"
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if args.top_k <= 0:

        raise ValueError(
            "--top-k must be greater than 0."
        )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print_section(
        "SHOPGRAPH — FAISS SIMILAR PRODUCT SEARCH"
    )

    print(
        "Fast semantic product retrieval "
        "using FAISS."
    )

    print(
        f"\nQuery ASIN : {args.asin}"
    )

    print(
        f"Top-K      : {args.top_k}"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    check_files()

    # --------------------------------------------------------
    # Load everything
    # --------------------------------------------------------

    (
        index,
        embeddings,
        product_ids,
        metadata
    ) = load_data()

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    success = search_by_asin(
        args.asin,
        index,
        embeddings,
        product_ids,
        metadata,
        args.top_k
    )

    if success:

        print_section(
            "SEARCH COMPLETE"
        )

        print(
            "FAISS semantic search completed successfully."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        main()