"""
SHOPGRAPH — SIMILAR PRODUCT SEARCH
Phase 4.1 — Product Intelligence

Semantic product similarity using the existing
384-dimensional MiniLM embeddings.

Input:
    data/processed/product_embeddings.npy
    data/processed/embedding_product_ids.parquet
    data/processed/products_text.parquet

Method:
    Cosine similarity

Examples:
    python search\similar_products.py --demo

    python search\similar_products.py --asin B09NN6SGV5

    python search\similar_products.py --asin B09NN6SGV5 --top-k 20
"""

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

DEFAULT_TOP_K = 10

RANDOM_STATE = 42


# =============================================================================
# DISPLAY
# =============================================================================

def print_section(title):
    """Print a formatted section heading."""

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# =============================================================================
# FILE CHECK
# =============================================================================

def check_files():
    """Verify that all required input files exist."""

    print_section("CHECKING FILES")

    required_files = {
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


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():
    """Load embeddings, product IDs and metadata."""

    print_section("LOADING DATA")

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------

    print("Loading embeddings...")

    embeddings = np.load(
        EMBEDDINGS_FILE,
        mmap_mode="r",
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    print(
        f"Embedding dtype: {embeddings.dtype}"
    )

    if len(embeddings.shape) != 2:
        raise ValueError(
            "Embeddings must be a 2-dimensional array."
        )

    # -------------------------------------------------------------------------
    # Product IDs
    # -------------------------------------------------------------------------

    print("\nLoading product IDs...")

    product_ids = pd.read_parquet(
        PRODUCT_IDS_FILE
    )

    if "parent_asin" not in product_ids.columns:

        raise ValueError(
            "Product ID file does not contain "
            "'parent_asin'."
        )

    print(
        f"Product ID columns: "
        f"{list(product_ids.columns)}"
    )

    print(
        f"Products: {len(product_ids):,}"
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    print("\nLoading product metadata...")

    metadata = pd.read_parquet(
        PRODUCT_TEXT_FILE
    )

    print(
        f"Metadata shape: {metadata.shape}"
    )

    # -------------------------------------------------------------------------
    # Alignment validation
    # -------------------------------------------------------------------------

    if len(embeddings) != len(product_ids):

        raise ValueError(
            "Embedding count does not match "
            "product ID count."
        )

    if len(embeddings) != len(metadata):

        raise ValueError(
            "Embedding count does not match "
            "metadata count."
        )

    print("\nDataset alignment: OK")

    return embeddings, product_ids, metadata


# =============================================================================
# COLUMN FINDER
# =============================================================================

def find_column(df, candidates):
    """
    Find the first available column from a list of candidates.
    """

    for column in candidates:

        if column in df.columns:
            return column

    return None


# =============================================================================
# SAFE VALUE HANDLING
# =============================================================================

def is_missing_scalar(value):
    """
    Safely determine whether a scalar value is missing.

    This avoids the error:

        ValueError:
        The truth value of an array with more than one
        element is ambiguous.
    """

    if value is None:
        return True

    # Arrays/lists are not treated as scalar missing values.
    if isinstance(
        value,
        (np.ndarray, list, tuple, dict),
    ):
        return False

    try:

        result = pd.isna(value)

        if isinstance(result, (bool, np.bool_)):
            return bool(result)

        return False

    except (
        TypeError,
        ValueError,
    ):
        return False


# =============================================================================
# VALUE FORMATTER
# =============================================================================

def format_value(value, max_length=500):
    """
    Convert metadata values into readable terminal text.

    Handles:
        strings
        numbers
        NumPy arrays
        lists
        tuples
        dictionaries
        nested arrays
        missing values
    """

    if is_missing_scalar(value):
        return "N/A"

    # -------------------------------------------------------------------------
    # NumPy arrays
    # -------------------------------------------------------------------------

    if isinstance(value, np.ndarray):

        if value.size == 0:
            return "N/A"

        value = value.tolist()

    # -------------------------------------------------------------------------
    # Lists / tuples
    # -------------------------------------------------------------------------

    if isinstance(
        value,
        (list, tuple),
    ):

        if len(value) == 0:
            return "N/A"

        parts = []

        for item in value:

            if isinstance(
                item,
                np.ndarray,
            ):

                item = item.tolist()

            if isinstance(
                item,
                (list, tuple),
            ):

                parts.append(
                    " > ".join(
                        str(x)
                        for x in item
                    )
                )

            elif isinstance(
                item,
                dict,
            ):

                parts.append(
                    str(item)
                )

            else:

                parts.append(
                    str(item)
                )

        text = " | ".join(parts)

    # -------------------------------------------------------------------------
    # Dictionaries
    # -------------------------------------------------------------------------

    elif isinstance(value, dict):

        text = str(value)

    # -------------------------------------------------------------------------
    # Normal scalar
    # -------------------------------------------------------------------------

    else:

        text = str(value)

    # -------------------------------------------------------------------------
    # Clean whitespace
    # -------------------------------------------------------------------------

    text = " ".join(
        text.split()
    )

    # -------------------------------------------------------------------------
    # Limit terminal output
    # -------------------------------------------------------------------------

    if len(text) > max_length:

        text = (
            text[:max_length - 3]
            + "..."
        )

    return text


# =============================================================================
# PRODUCT INFORMATION
# =============================================================================

def get_product_info(
    index,
    product_ids,
    metadata,
):
    """
    Retrieve readable information for a product.
    """

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

    # -------------------------------------------------------------------------
    # Candidate metadata columns
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Extract values
    # -------------------------------------------------------------------------

    for output_name, candidates in (
        metadata_columns.items()
    ):

        column = find_column(
            metadata,
            candidates,
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


# =============================================================================
# FIND PRODUCT INDEX
# =============================================================================

def find_product_index(
    asin,
    product_ids,
):
    """
    Find embedding index for an ASIN.
    """

    asin = str(asin).strip()

    matches = np.where(
        product_ids["parent_asin"]
        .astype(str)
        .values
        == asin
    )[0]

    if len(matches) == 0:
        return None

    return int(matches[0])


# =============================================================================
# SIMILAR PRODUCT SEARCH
# =============================================================================

def find_similar_products(
    query_index,
    embeddings,
    top_k=10,
):
    """
    Find the top-k products using cosine similarity.
    """

    print(
        "\nCalculating cosine similarity..."
    )

    # -------------------------------------------------------------------------
    # Query embedding
    # -------------------------------------------------------------------------

    query_embedding = np.asarray(
        embeddings[query_index],
        dtype=np.float32,
    ).reshape(
        1,
        -1,
    )

    # -------------------------------------------------------------------------
    # Calculate similarity
    # -------------------------------------------------------------------------

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        similarities = cosine_similarity(
            query_embedding,
            embeddings,
        )[0]

    # -------------------------------------------------------------------------
    # Remove the query product itself
    # -------------------------------------------------------------------------

    similarities[query_index] = -1.0

    # -------------------------------------------------------------------------
    # Determine number of results
    # -------------------------------------------------------------------------

    available_products = (
        len(similarities) - 1
    )

    candidate_count = min(
        top_k,
        available_products,
    )

    if candidate_count <= 0:
        return []

    # -------------------------------------------------------------------------
    # Get highest similarities
    # -------------------------------------------------------------------------

    top_indices = np.argpartition(
        similarities,
        -candidate_count,
    )[-candidate_count:]

    # -------------------------------------------------------------------------
    # Sort from highest to lowest
    # -------------------------------------------------------------------------

    top_indices = top_indices[
        np.argsort(
            similarities[top_indices]
        )[::-1]
    ]

    results = []

    for index in top_indices:

        results.append(
            (
                int(index),
                float(
                    similarities[index]
                ),
            )
        )

    return results


# =============================================================================
# DISPLAY QUERY PRODUCT
# =============================================================================

def display_query_product(
    query_info,
):
    """Display the selected product."""

    print_section("QUERY PRODUCT")

    print(
        f"Embedding index : "
        f"{query_info['index']}"
    )

    print(
        f"ASIN            : "
        f"{format_value(query_info['parent_asin'])}"
    )

    print(
        f"Title           : "
        f"{format_value(query_info['title'])}"
    )

    print(
        f"Store           : "
        f"{format_value(query_info['store'])}"
    )

    print(
        f"Rating          : "
        f"{format_value(query_info['rating'])}"
    )

    print(
        f"Price           : "
        f"{format_value(query_info['price'])}"
    )

    print(
        f"Categories      : "
        f"{format_value(query_info['categories'])}"
    )


# =============================================================================
# DISPLAY RESULTS
# =============================================================================

def display_results(
    results,
    product_ids,
    metadata,
):
    """Display similar products."""

    print_section(
        "SIMILAR PRODUCTS"
    )

    if not results:

        print(
            "No similar products found."
        )

        return

    for rank, (
        index,
        similarity,
    ) in enumerate(
        results,
        start=1,
    ):

        info = get_product_info(
            index,
            product_ids,
            metadata,
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


# =============================================================================
# SEARCH BY ASIN
# =============================================================================

def search_by_asin(
    asin,
    embeddings,
    product_ids,
    metadata,
    top_k,
):
    """Search for products similar to a specified ASIN."""

    index = find_product_index(
        asin,
        product_ids,
    )

    if index is None:

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
        f"index {index}."
    )

    # -------------------------------------------------------------------------
    # Query information
    # -------------------------------------------------------------------------

    query_info = get_product_info(
        index,
        product_ids,
        metadata,
    )

    display_query_product(
        query_info
    )

    # -------------------------------------------------------------------------
    # Similar products
    # -------------------------------------------------------------------------

    results = find_similar_products(
        index,
        embeddings,
        top_k,
    )

    display_results(
        results,
        product_ids,
        metadata,
    )

    return True


# =============================================================================
# RANDOM DEMO
# =============================================================================

def random_demo(
    embeddings,
    product_ids,
    metadata,
    top_k,
):
    """Run similarity search for a reproducible random product."""

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    index = int(
        rng.integers(
            0,
            len(embeddings),
        )
    )

    asin = product_ids.iloc[index][
        "parent_asin"
    ]

    print(
        f"\nRandom demo product selected: "
        f"{asin}"
    )

    return search_by_asin(
        asin,
        embeddings,
        product_ids,
        metadata,
        top_k,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "ShopGraph semantic "
            "similar-product search"
        )
    )

    parser.add_argument(
        "--asin",
        type=str,
        default=None,
        help=(
            "Amazon parent ASIN of "
            "the query product"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "Number of similar products "
            "to return. Default: 10"
        ),
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Run a reproducible random "
            "product similarity demo"
        ),
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Validate arguments
    # -------------------------------------------------------------------------

    if args.top_k <= 0:

        raise ValueError(
            "--top-k must be greater than 0."
        )

    if args.asin and args.demo:

        raise ValueError(
            "Use either --asin or --demo, "
            "not both."
        )

    # -------------------------------------------------------------------------
    # Check files
    # -------------------------------------------------------------------------

    check_files()

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    (
        embeddings,
        product_ids,
        metadata,
    ) = load_data()

    # -------------------------------------------------------------------------
    # Search mode
    # -------------------------------------------------------------------------

    if args.asin:

        search_by_asin(
            args.asin,
            embeddings,
            product_ids,
            metadata,
            args.top_k,
        )

    elif args.demo:

        random_demo(
            embeddings,
            product_ids,
            metadata,
            args.top_k,
        )

    else:

        print_section(
            "NO SEARCH MODE SELECTED"
        )

        print(
            "Use one of the following:"
        )

        print(
            "\n1. Random demo:"
        )

        print(
            "   python "
            "search\\similar_products.py "
            "--demo"
        )

        print(
            "\n2. Search by ASIN:"
        )

        print(
            "   python "
            "search\\similar_products.py "
            "--asin B09NN6SGV5"
        )

        print(
            "\n3. Search with custom result count:"
        )

        print(
            "   python "
            "search\\similar_products.py "
            "--asin B09NN6SGV5 "
            "--top-k 20"
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()