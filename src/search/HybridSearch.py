"""
ShopGraph - Hybrid Product Search

Combines:
1. Semantic search using SentenceTransformer
2. FAISS vector similarity
3. Keyword matching
4. Product-type matching
5. Specification matching
6. Price filtering

Usage:

python search\HybridSearch.py --query "gaming laptop with RTX 3070 and 16GB RAM under $1500"

python search\HybridSearch.py --query "wireless earbuds for commuting"

python search\HybridSearch.py --query "laptop for university students" --top-k 10
"""

from pathlib import Path
import argparse
import json
import re
import time

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
SEARCH_DIR = DATA_DIR / "search"

EMBEDDINGS_PATH = DATA_DIR / "product_embeddings.npy"
IDS_PATH = DATA_DIR / "embedding_product_ids.parquet"
METADATA_PATH = DATA_DIR / "products_100k.parquet"
FAISS_INDEX_PATH = SEARCH_DIR / "faiss_products.index"

OUTPUT_PATH = SEARCH_DIR / "last_hybrid_search.json"


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# QUERY PARSER
# ============================================================

def parse_query(query):
    """
    Extract useful structured constraints from the natural-language query.
    """

    query_lower = query.lower()

    result = {
        "product_type": None,
        "ram_gb": None,
        "gpu": None,
        "storage_gb": None,
        "price_max": None,
        "price_min": None,
        "keywords": [],
    }

    # --------------------------------------------------------
    # RAM
    # --------------------------------------------------------

    ram_patterns = [
        r"(\d+)\s*gb\s*ram",
        r"ram\s*(?:of|with)?\s*(\d+)\s*gb",
        r"(\d+)\s*gb\s*memory",
    ]

    for pattern in ram_patterns:
        match = re.search(pattern, query_lower)

        if match:
            result["ram_gb"] = int(match.group(1))
            break

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    gpu_patterns = [
        r"(rtx\s*\d{3,4}(?:\s*ti|\s*super)?)",
        r"(gtx\s*\d{3,4}(?:\s*ti|\s*super)?)",
        r"(rx\s*\d{3,4}(?:\s*xt)?)",
    ]

    for pattern in gpu_patterns:
        match = re.search(pattern, query_lower)

        if match:
            result["gpu"] = re.sub(r"\s+", " ", match.group(1)).strip()
            break

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    storage_patterns = [
        r"(\d+)\s*tb\s*(?:ssd|storage|hard drive|hdd)?",
        r"(\d+)\s*gb\s*(?:ssd|storage|hard drive|hdd)",
    ]

    for pattern in storage_patterns:
        match = re.search(pattern, query_lower)

        if match:
            value = int(match.group(1))

            if "tb" in match.group(0):
                value *= 1000

            result["storage_gb"] = value
            break

    # --------------------------------------------------------
    # MAX PRICE
    # --------------------------------------------------------

    price_patterns = [
        r"under\s*\$?\s*([\d,]+)",
        r"below\s*\$?\s*([\d,]+)",
        r"less\s+than\s*\$?\s*([\d,]+)",
        r"up\s*to\s*\$?\s*([\d,]+)",
        r"maximum\s*\$?\s*([\d,]+)",
        r"max\s*\$?\s*([\d,]+)",
    ]

    for pattern in price_patterns:
        match = re.search(pattern, query_lower)

        if match:
            result["price_max"] = float(
                match.group(1).replace(",", "")
            )
            break

    # --------------------------------------------------------
    # MIN PRICE
    # --------------------------------------------------------

    price_min_patterns = [
        r"over\s*\$?\s*([\d,]+)",
        r"above\s*\$?\s*([\d,]+)",
        r"more\s+than\s*\$?\s*([\d,]+)",
        r"at\s+least\s*\$?\s*([\d,]+)",
    ]

    for pattern in price_min_patterns:
        match = re.search(pattern, query_lower)

        if match:
            result["price_min"] = float(
                match.group(1).replace(",", "")
            )
            break

    # --------------------------------------------------------
    # PRODUCT TYPE
    # --------------------------------------------------------

    product_types = {
        "laptop": [
            "laptop",
            "notebook",
            "gaming laptop",
            "business laptop",
        ],
        "headphones": [
            "headphone",
            "headphones",
            "headset",
            "earbuds",
            "earphones",
        ],
        "phone": [
            "phone",
            "smartphone",
            "mobile",
        ],
        "tablet": [
            "tablet",
            "ipad",
        ],
        "monitor": [
            "monitor",
            "display",
        ],
        "keyboard": [
            "keyboard",
        ],
        "mouse": [
            "mouse",
            "gaming mouse",
        ],
        "camera": [
            "camera",
            "dslr",
            "mirrorless",
        ],
        "printer": [
            "printer",
        ],
        "router": [
            "router",
            "wifi router",
            "wireless router",
        ],
        "charger": [
            "charger",
            "power adapter",
            "adapter",
        ],
        "cable": [
            "cable",
            "usb cable",
            "hdmi cable",
        ],
        "storage": [
            "ssd",
            "hard drive",
            "hdd",
            "storage drive",
            "usb drive",
        ],
        "graphics_card": [
            "graphics card",
            "gpu",
            "video card",
        ],
    }

    for product_type, terms in product_types.items():

        if any(term in query_lower for term in terms):
            result["product_type"] = product_type
            break

    # --------------------------------------------------------
    # IMPORTANT KEYWORDS
    # --------------------------------------------------------

    keyword_terms = [
        "gaming",
        "wireless",
        "bluetooth",
        "portable",
        "student",
        "university",
        "business",
        "professional",
        "office",
        "work",
        "school",
        "travel",
        "commuting",
        "mechanical",
        "touchscreen",
        "4k",
        "5g",
        "usb-c",
        "usb c",
        "wifi",
        "noise cancelling",
        "noise cancellation",
        "rgb",
    ]

    for keyword in keyword_terms:

        if keyword in query_lower:
            result["keywords"].append(keyword)

    return result


# ============================================================
# TEXT HELPERS
# ============================================================

def safe_text(value):
    """
    Convert metadata values into safe searchable text.
    """

    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(
            safe_text(item)
            for item in value
        )

    if isinstance(value, dict):
        return " ".join(
            f"{safe_text(k)} {safe_text(v)}"
            for k, v in value.items()
        )

    return str(value)


def row_text(row):
    """
    Build searchable text from product metadata.
    """

    fields = [
        "title",
        "store",
        "description",
        "features",
        "categories",
        "brand",
    ]

    parts = []

    for field in fields:

        if field in row.index:
            value = safe_text(row[field])

            if value:
                parts.append(value)

    return " ".join(parts).lower()


# ============================================================
# PRICE EXTRACTION
# ============================================================

def extract_price(value):
    """
    Attempt to extract a numeric product price.
    """

    if value is None:
        return None

    if isinstance(value, (int, float, np.integer, np.floating)):

        if np.isnan(value):
            return None

        return float(value)

    text = safe_text(value)

    match = re.search(
        r"(\d+(?:\.\d+)?)",
        text.replace(",", "")
    )

    if not match:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


# ============================================================
# SPEC MATCHING
# ============================================================

def specification_score(
    text,
    parsed_query
):
    """
    Calculate specification match score.

    Score range: 0 - 1
    """

    if not text:
        return 0.0

    score = 0.0
    checks = 0

    # --------------------------------------------------------
    # RAM
    # --------------------------------------------------------

    ram = parsed_query["ram_gb"]

    if ram is not None:

        checks += 1

        ram_pattern = rf"\b{ram}\s*gb\b"

        if re.search(ram_pattern, text):
            score += 1.0

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    gpu = parsed_query["gpu"]

    if gpu is not None:

        checks += 1

        normalized_gpu = re.sub(
            r"\s+",
            "",
            gpu.lower()
        )

        normalized_text = re.sub(
            r"\s+",
            "",
            text.lower()
        )

        if normalized_gpu in normalized_text:
            score += 1.0

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    storage = parsed_query["storage_gb"]

    if storage is not None:

        checks += 1

        if f"{storage}gb" in text.replace(" ", ""):
            score += 1.0

        elif storage >= 1000:

            tb = storage // 1000

            if f"{tb}tb" in text.replace(" ", ""):
                score += 1.0

    if checks == 0:
        return 0.0

    return score / checks


# ============================================================
# PRODUCT TYPE SCORE
# ============================================================

def product_type_score(
    text,
    product_type
):
    """
    Determine whether a product matches the requested type.
    """

    if not product_type:
        return 0.0

    type_terms = {

        "laptop": [
            "laptop",
            "notebook",
        ],

        "headphones": [
            "headphone",
            "headset",
            "earbuds",
            "earphones",
        ],

        "phone": [
            "smartphone",
            "cell phone",
            "mobile phone",
            "iphone",
            "android",
        ],

        "tablet": [
            "tablet",
            "ipad",
        ],

        "monitor": [
            "monitor",
            "display",
        ],

        "keyboard": [
            "keyboard",
        ],

        "mouse": [
            "mouse",
        ],

        "camera": [
            "camera",
            "dslr",
            "mirrorless",
        ],

        "printer": [
            "printer",
        ],

        "router": [
            "router",
        ],

        "charger": [
            "charger",
            "power adapter",
            "adapter",
        ],

        "cable": [
            "cable",
        ],

        "storage": [
            "ssd",
            "hard drive",
            "hdd",
            "usb drive",
            "flash drive",
        ],

        "graphics_card": [
            "graphics card",
            "video card",
            "gpu",
        ],
    }

    terms = type_terms.get(
        product_type,
        []
    )

    if any(term in text for term in terms):
        return 1.0

    return 0.0


# ============================================================
# KEYWORD SCORE
# ============================================================

def keyword_score(
    text,
    keywords
):
    """
    Calculate keyword matching score.
    """

    if not keywords:
        return 0.0

    matches = 0

    for keyword in keywords:

        normalized_keyword = keyword.lower()

        if normalized_keyword in text:
            matches += 1

    return matches / len(keywords)


# ============================================================
# PRICE SCORE
# ============================================================

def price_score(
    price,
    parsed_query
):
    """
    Score price constraint.

    1.0 = satisfies constraint
    0.0 = violates constraint
    """

    if price is None:
        return None

    minimum = parsed_query["price_min"]
    maximum = parsed_query["price_max"]

    if minimum is not None and price < minimum:
        return 0.0

    if maximum is not None and price > maximum:
        return 0.0

    return 1.0


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 80)
    print("SHOPGRAPH HYBRID PRODUCT SEARCH")
    print("=" * 80)

    print("\nLoading FAISS index...")

    index = faiss.read_index(
        str(FAISS_INDEX_PATH)
    )

    print(
        f"FAISS vectors: {index.ntotal:,}"
    )

    print("\nLoading embeddings...")

    embeddings = np.load(
        EMBEDDINGS_PATH,
        mmap_mode="r"
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    print("\nLoading product IDs...")

    ids_df = pd.read_parquet(
        IDS_PATH
    )

    print(
        f"Product IDs: {len(ids_df):,}"
    )

    print("\nLoading metadata...")

    metadata = pd.read_parquet(
        METADATA_PATH
    )

    print(
        f"Metadata rows: {len(metadata):,}"
    )

    return (
        index,
        embeddings,
        ids_df,
        metadata,
    )


# ============================================================
# MAIN SEARCH
# ============================================================

def hybrid_search(
    query,
    top_k=10,
):

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    (
        index,
        embeddings,
        ids_df,
        metadata,
    ) = load_data()

    # --------------------------------------------------------
    # PARSE QUERY
    # --------------------------------------------------------

    parsed = parse_query(query)

    print("\n" + "=" * 80)
    print("QUERY")
    print("=" * 80)

    print(query)

    print("\nParsed constraints:")

    for key, value in parsed.items():
        print(f"  {key}: {value}")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading SentenceTransformer...")

    model_start = time.time()

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        f"Model device: {model.device}"
    )

    print(
        f"Model load time: "
        f"{time.time() - model_start:.2f}s"
    )

    # --------------------------------------------------------
    # EMBED QUERY
    # --------------------------------------------------------

    print("\nGenerating query embedding...")

    embedding_start = time.time()

    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    embedding_time = (
        time.time() - embedding_start
    )

    print(
        f"Query embedding time: "
        f"{embedding_time:.4f}s"
    )

    # --------------------------------------------------------
    # FAISS SEARCH
    # --------------------------------------------------------

    search_buffer = max(
        top_k * 10,
        100
    )

    search_buffer = min(
        search_buffer,
        index.ntotal
    )

    print(
        f"\nFAISS candidates: "
        f"{search_buffer}"
    )

    faiss_start = time.time()

    similarities, indices = index.search(
        query_vector,
        search_buffer
    )

    faiss_time = (
        time.time() - faiss_start
    )

    print(
        f"FAISS search time: "
        f"{faiss_time * 1000:.2f} ms"
    )

    # --------------------------------------------------------
    # ALIGN METADATA
    # --------------------------------------------------------

    if "parent_asin" in ids_df.columns:
        id_column = "parent_asin"
    else:
        id_column = ids_df.columns[0]

    product_ids = ids_df[
        id_column
    ].astype(str).tolist()

    metadata_index = {}

    if "parent_asin" in metadata.columns:

        for idx, asin in enumerate(
            metadata["parent_asin"].astype(str)
        ):
            metadata_index[asin] = idx

    # --------------------------------------------------------
    # SCORE CANDIDATES
    # --------------------------------------------------------

    candidates = []

    for rank, (similarity, idx) in enumerate(
        zip(
            similarities[0],
            indices[0]
        ),
        start=1
    ):

        if idx < 0:
            continue

        if idx >= len(product_ids):
            continue

        asin = product_ids[idx]

        metadata_row_idx = metadata_index.get(
            asin
        )

        if metadata_row_idx is None:
            continue

        row = metadata.iloc[
            metadata_row_idx
        ]

        text = row_text(row)

        semantic = float(similarity)

        type_score = product_type_score(
            text,
            parsed["product_type"]
        )

        spec_score = specification_score(
            text,
            parsed
        )

        kw_score = keyword_score(
            text,
            parsed["keywords"]
        )

        price = None

        if "price" in row.index:
            price = extract_price(
                row["price"]
            )

        p_score = price_score(
            price,
            parsed
        )

        # ----------------------------------------------------
        # HARD PRICE FILTER
        # ----------------------------------------------------

        price_violation = False

        if p_score == 0.0:
            price_violation = True

        # ----------------------------------------------------
        # HYBRID SCORE
        # ----------------------------------------------------

        hybrid_score = (
            0.55 * semantic
            + 0.20 * type_score
            + 0.15 * spec_score
            + 0.10 * kw_score
        )

        # ----------------------------------------------------
        # PENALIZE PRICE VIOLATIONS
        # ----------------------------------------------------

        if price_violation:
            hybrid_score *= 0.25

        # ----------------------------------------------------
        # PRODUCT RESULT
        # ----------------------------------------------------

        title = safe_text(
            row.get(
                "title",
                ""
            )
        )

        store = safe_text(
            row.get(
                "store",
                ""
            )
        )

        brand = safe_text(
            row.get(
                "brand",
                ""
            )
        )

        candidates.append({

            "rank": rank,

            "parent_asin": asin,

            "title": title,

            "brand": brand,

            "store": store,

            "price": price,

            "semantic_score": round(
                semantic,
                4
            ),

            "product_type_score": round(
                type_score,
                4
            ),

            "specification_score": round(
                spec_score,
                4
            ),

            "keyword_score": round(
                kw_score,
                4
            ),

            "hybrid_score": round(
                hybrid_score,
                4
            ),

            "price_constraint_satisfied": (
                not price_violation
            ),
        })

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    results = candidates[:top_k]

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print(
        f"TOP {len(results)} HYBRID RESULTS"
    )
    print("=" * 80)

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n#{i} "
            f"Hybrid: {result['hybrid_score']:.4f}"
        )

        print(
            f"Title: {result['title']}"
        )

        if result["brand"]:
            print(
                f"Brand: {result['brand']}"
            )

        if result["store"]:
            print(
                f"Store: {result['store']}"
            )

        if result["price"] is not None:
            print(
                f"Price: ${result['price']:.2f}"
            )

        print(
            f"Semantic: "
            f"{result['semantic_score']:.4f}"
        )

        print(
            f"Type: "
            f"{result['product_type_score']:.4f}"
        )

        print(
            f"Specs: "
            f"{result['specification_score']:.4f}"
        )

        print(
            f"Keywords: "
            f"{result['keyword_score']:.4f}"
        )

        print(
            f"Price constraint: "
            f"{result['price_constraint_satisfied']}"
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    output = {

        "query": query,

        "parsed_query": parsed,

        "search_configuration": {

            "model": MODEL_NAME,

            "semantic_weight": 0.55,

            "product_type_weight": 0.20,

            "specification_weight": 0.15,

            "keyword_weight": 0.10,

            "candidate_pool": search_buffer,
        },

        "timing": {

            "embedding_seconds": round(
                embedding_time,
                4
            ),

            "faiss_seconds": round(
                faiss_time,
                4
            ),
        },

        "results": results,
    }

    SEARCH_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_PATH
    )

    return results


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="ShopGraph Hybrid Product Search"
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Natural-language product query",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of products to return",
    )

    args = parser.parse_args()

    hybrid_search(
        query=args.query,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()