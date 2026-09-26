"""
ShopGraph - Hybrid Product Search v2

Combines:
1. SentenceTransformer semantic search
2. FAISS cosine similarity
3. Product-type matching
4. Exact GPU matching
5. Minimum RAM matching
6. Storage constraints
7. Price constraints
8. Keyword matching

v2 improvements over HybridSearch.py:
- True hard price filtering
- Exact GPU model matching
- Minimum RAM semantics
- Stronger product-type filtering
- Better constraint scoring
- Cleaner Windows paths
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR_CANDIDATES = (
    BASE_DIR / "data" / "processed",
    BASE_DIR / "data" / "raw" / "processed",
)
DATA_DIR = next(
    (
        directory
        for directory in DATA_DIR_CANDIDATES
        if (directory / "products_text.parquet").exists()
    ),
    DATA_DIR_CANDIDATES[0],
)

EMBEDDINGS_PATH = DATA_DIR / "product_embeddings.npy"
IDS_PATH = DATA_DIR / "embedding_product_ids.parquet"
METADATA_PATH = DATA_DIR / "products_text.parquet"

FAISS_PATH = DATA_DIR / "search" / "faiss_products.index"

OUTPUT_PATH = DATA_DIR / "search" / "last_hybrid_search_v2.json"


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# PRODUCT TYPE DEFINITIONS
# ============================================================

PRODUCT_TYPE_TERMS = {
    "laptop": [
        "laptop",
        "notebook",
        "notebook computer",
        "gaming laptop",
        "ultrabook",
        "chromebook",
        "macbook",
        "portable computer",
    ],

    "desktop": [
        "desktop",
        "desktop computer",
        "tower pc",
        "gaming pc",
        "desktop pc",
        "computer tower",
    ],

    "headphones": [
        "headphones",
        "headphone",
        "earbuds",
        "earbud",
        "wireless earbuds",
        "bluetooth earbuds",
        "in-ear",
        "over-ear",
        "on-ear",
        "headset",
    ],

    "tablet": [
        "tablet",
        "ipad",
        "android tablet",
        "tablet computer",
    ],

    "monitor": [
        "monitor",
        "computer monitor",
        "display",
        "lcd monitor",
        "led monitor",
    ],

    "keyboard": [
        "keyboard",
        "mechanical keyboard",
        "wireless keyboard",
        "gaming keyboard",
    ],

    "mouse": [
        "mouse",
        "gaming mouse",
        "wireless mouse",
        "computer mouse",
    ],

    "graphics_card": [
        "graphics card",
        "graphic card",
        "video card",
        "gpu",
        "geforce",
        "radeon",
        "graphics processor",
    ],

    "storage": [
        "ssd",
        "hard drive",
        "hdd",
        "solid state drive",
        "flash drive",
        "usb drive",
        "memory card",
    ],

    "charger": [
        "charger",
        "power adapter",
        "ac adapter",
        "laptop charger",
        "power supply",
    ],

    "camera": [
        "camera",
        "digital camera",
        "dslr",
        "mirrorless",
        "camcorder",
    ],

    "networking": [
        "router",
        "network adapter",
        "wifi adapter",
        "wireless router",
        "ethernet",
        "network switch",
    ],
}


# ============================================================
# GENERIC KEYWORDS
# ============================================================

STOPWORDS = {
    "a",
    "an",
    "the",
    "for",
    "with",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "is",
    "are",
    "under",
    "below",
    "less",
    "than",
    "up",
    "minimum",
    "at",
    "least",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("=" * 80)
    print("SHOPGRAPH HYBRID SEARCH v2")
    print("=" * 80)

    print("\nLoading FAISS index...")
    index = faiss.read_index(str(FAISS_PATH))

    print(f"FAISS vectors: {index.ntotal:,}")

    print("\nLoading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")

    print(f"Embedding shape: {embeddings.shape}")

    print("\nLoading product IDs...")
    ids_df = pd.read_parquet(IDS_PATH)

    print(f"Product IDs: {len(ids_df):,}")

    print("\nLoading metadata...")
    metadata = pd.read_parquet(METADATA_PATH)

    print(f"Metadata rows: {len(metadata):,}")

    # --------------------------------------------------------
    # Alignment checks
    # --------------------------------------------------------

    if len(ids_df) != len(embeddings):
        raise ValueError(
            f"ID/embedding mismatch: {len(ids_df)} vs {len(embeddings)}"
        )

    if len(metadata) != len(embeddings):
        raise ValueError(
            f"Metadata/embedding mismatch: {len(metadata)} vs {len(embeddings)}"
        )

    print("\nData alignment verified.")

    return index, embeddings, ids_df, metadata


# ============================================================
# SAFE VALUE HELPERS
# ============================================================

def safe_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(safe_text(x) for x in value)

    if isinstance(value, dict):
        return " ".join(
            f"{safe_text(k)} {safe_text(v)}"
            for k, v in value.items()
        )

    return str(value)


def get_image_url(row: pd.Series) -> str:
    """Return the first usable Amazon image URL from product metadata."""
    for column in ("images.hi_res", "images.large", "images.thumb"):
        if column not in row.index:
            continue

        value = row[column]
        if isinstance(value, (list, tuple, np.ndarray)):
            for item in value:
                url = safe_text(item).strip()
                if url.startswith("http"):
                    return url

        text = safe_text(value).strip()
        match = re.search(r"https?://[^\s'\"]+", text)
        if match:
            return match.group(0)

    return ""


def normalize_text(text: str) -> str:
    text = safe_text(text).lower()
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("×", "x")

    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# METADATA TEXT
# ============================================================

def get_product_text(row: pd.Series) -> str:
    parts = []

    for column in [
        "title",
        "brand",
        "store",
        "features",
        "description",
        "categories",
    ]:
        if column in row.index:
            value = safe_text(row[column])

            if value:
                parts.append(value)

    return normalize_text(" ".join(parts))


# ============================================================
# PRODUCT TYPE DETECTION
# ============================================================

def detect_query_product_type(query: str) -> Optional[str]:
    q = normalize_text(query)

    # More specific types first.
    ordered_types = [
        "graphics_card",
        "headphones",
        "networking",
        "keyboard",
        "mouse",
        "storage",
        "charger",
        "monitor",
        "tablet",
        "camera",
        "laptop",
        "desktop",
    ]

    for product_type in ordered_types:
        terms = PRODUCT_TYPE_TERMS[product_type]

        for term in terms:
            if term in q:
                return product_type

    return None


def detect_product_type(text: str) -> Optional[str]:
    text = normalize_text(text)

    scores = {}

    for product_type, terms in PRODUCT_TYPE_TERMS.items():
        score = 0

        for term in terms:
            if term in text:
                # Longer terms get more weight.
                score += max(1, len(term.split()))

        if score > 0:
            scores[product_type] = score

    if not scores:
        return None

    return max(scores, key=scores.get)


# ============================================================
# GPU PARSING
# ============================================================

GPU_PATTERN = re.compile(
    r"\b("
    r"rtx\s*\d{3,4}"
    r"|gtx\s*\d{3,4}"
    r"|rx\s*\d{3,4}"
    r"|radeon\s*[a-z]*\s*\d{3,4}"
    r"|geforce\s*(?:rtx|gtx)?\s*\d{3,4}"
    r")\b",
    re.IGNORECASE,
)


def normalize_gpu_name(gpu: str) -> str:
    gpu = normalize_text(gpu)

    gpu = re.sub(r"\s+", " ", gpu)

    gpu = gpu.replace("geforce ", "")

    return gpu.strip()


def extract_gpu(query: str) -> Optional[str]:
    match = GPU_PATTERN.search(normalize_text(query))

    if not match:
        return None

    return normalize_gpu_name(match.group(1))


def extract_gpu_from_text(text: str) -> Optional[str]:
    matches = GPU_PATTERN.findall(normalize_text(text))

    if not matches:
        return None

    return normalize_gpu_name(matches[0])


def gpu_exact_match(query_gpu: str, product_text: str) -> bool:
    """
    Exact GPU model matching.

    Example:
        RTX 3070 -> RTX 3070 = True
        RTX 3070 -> RTX 3070 Ti = False
        RTX 3070 -> RTX 3080 = False
    """

    query_gpu = normalize_gpu_name(query_gpu)
    text = normalize_text(product_text)

    # Build exact token boundary pattern.
    escaped = re.escape(query_gpu)

    pattern = rf"\b{escaped}\b"

    if re.search(pattern, text):
        return True

    return False


# ============================================================
# RAM PARSING
# ============================================================

def extract_ram_gb(query: str) -> Optional[float]:
    q = normalize_text(query)

    patterns = [
        r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory)",
        r"(\d+(?:\.\d+)?)\s*gb\s*system\s*memory",
        r"(\d+(?:\.\d+)?)\s*gb",
    ]

    for pattern in patterns:
        match = re.search(pattern, q)

        if match:
            return float(match.group(1))

    return None


def extract_ram_from_text(text: str) -> Optional[float]:
    text = normalize_text(text)

    patterns = [
        r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory)",
        r"(\d+(?:\.\d+)?)\s*gb\s*ddr",
        r"(\d+(?:\.\d+)?)\s*gb\s*system\s*memory",
    ]

    values = []

    for pattern in patterns:
        matches = re.findall(pattern, text)

        for value in matches:
            try:
                values.append(float(value))
            except ValueError:
                pass

    if not values:
        return None

    # Usually the largest RAM figure is the installed RAM.
    return max(values)


# ============================================================
# STORAGE PARSING
# ============================================================

def extract_storage_gb(query: str) -> Optional[float]:
    q = normalize_text(query)

    patterns = [
        r"(\d+(?:\.\d+)?)\s*tb",
        r"(\d+(?:\.\d+)?)\s*gb\s*(?:ssd|storage|hdd)",
        r"(\d+(?:\.\d+)?)\s*gb\s*(?:solid state|drive)",
    ]

    for pattern in patterns:
        match = re.search(pattern, q)

        if match:
            value = float(match.group(1))

            if "tb" in match.group(0):
                value *= 1000

            return value

    return None


def extract_storage_from_text(text: str) -> Optional[float]:
    text = normalize_text(text)

    values = []

    tb_matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*tb",
        text
    )

    for value in tb_matches:
        values.append(float(value) * 1000)

    gb_matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*gb\s*(?:ssd|storage|hdd|drive)",
        text
    )

    for value in gb_matches:
        values.append(float(value))

    if not values:
        return None

    return max(values)


# ============================================================
# PRICE PARSING
# ============================================================

def parse_price(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, float) and np.isnan(value):
        return None

    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)

        if value <= 0:
            return None

        return value

    text = safe_text(value)

    if not text:
        return None

    # Remove currency symbols and commas.
    text = text.replace(",", "")

    matches = re.findall(
        r"\d+(?:\.\d+)?",
        text
    )

    if not matches:
        return None

    try:
        price = float(matches[0])

        if price <= 0:
            return None

        return price

    except ValueError:
        return None


def get_product_price(row: pd.Series) -> Optional[float]:
    price_columns = [
        "price",
        "Price",
    ]

    for column in price_columns:
        if column in row.index:
            price = parse_price(row[column])

            if price is not None:
                return price

    return None


# ============================================================
# PRICE CONSTRAINTS
# ============================================================

def extract_price_constraints(
    query: str,
) -> tuple[Optional[float], Optional[float]]:

    q = normalize_text(query)

    price_min = None
    price_max = None

    # --------------------------------------------------------
    # Maximum price
    # --------------------------------------------------------

    max_patterns = [
        r"(?:under|below|less than|up to|max(?:imum)?(?: price)?(?: of)?)\s*\$?\s*(\d+(?:\.\d+)?)",
        r"\$?\s*(\d+(?:\.\d+)?)\s*(?:or less|and under)",
    ]

    for pattern in max_patterns:
        match = re.search(pattern, q)

        if match:
            price_max = float(match.group(1))
            break

    # --------------------------------------------------------
    # Minimum price
    # --------------------------------------------------------

    min_patterns = [
        r"(?:over|above|more than|at least|min(?:imum)?(?: price)?(?: of)?)\s*\$?\s*(\d+(?:\.\d+)?)",
        r"\$?\s*(\d+(?:\.\d+)?)\s*(?:or more|and above)",
    ]

    for pattern in min_patterns:
        match = re.search(pattern, q)

        if match:
            price_min = float(match.group(1))
            break

    return price_min, price_max


# ============================================================
# KEYWORDS
# ============================================================

def extract_keywords(query: str) -> list[str]:
    q = normalize_text(query)

    # Remove structured specifications.
    q = GPU_PATTERN.sub(" ", q)

    q = re.sub(
        r"\b\d+(?:\.\d+)?\s*(?:gb|tb)\b",
        " ",
        q,
    )

    q = re.sub(
        r"\$?\d+(?:\.\d+)?",
        " ",
        q,
    )

    words = re.findall(r"[a-z0-9]+", q)

    keywords = []

    for word in words:
        if word in STOPWORDS:
            continue

        if len(word) < 3:
            continue

        keywords.append(word)

    return list(dict.fromkeys(keywords))


# ============================================================
# QUERY PARSER
# ============================================================

def parse_query(query: str) -> dict[str, Any]:

    product_type = detect_query_product_type(query)

    gpu = extract_gpu(query)

    ram_gb = extract_ram_gb(query)

    storage_gb = extract_storage_gb(query)

    price_min, price_max = extract_price_constraints(query)

    keywords = extract_keywords(query)

    return {
        "product_type": product_type,
        "gpu": gpu,
        "ram_gb": ram_gb,
        "storage_gb": storage_gb,
        "price_min": price_min,
        "price_max": price_max,
        "keywords": keywords,
    }


# ============================================================
# PRODUCT TYPE MATCH
# ============================================================

def product_type_match(
    query_type: Optional[str],
    product_type: Optional[str],
    product_text: str,
) -> float:

    if query_type is None:
        return 0.0

    text = normalize_text(product_text)

    # Direct type match.
    if product_type == query_type:
        return 1.0

    # Strong explicit detection.
    query_terms = PRODUCT_TYPE_TERMS.get(query_type, [])

    matched_terms = sum(
        1 for term in query_terms
        if term in text
    )

    if matched_terms > 0:
        return 1.0

    return 0.0


# ============================================================
# HARD PRODUCT TYPE FILTER
# ============================================================

def passes_product_type_filter(
    query_type: Optional[str],
    product_type: Optional[str],
    product_text: str,
) -> bool:

    if query_type is None:
        return True

    text = normalize_text(product_text)

    # --------------------------------------------------------
    # Laptop
    # --------------------------------------------------------

    if query_type == "laptop":

        laptop_terms = PRODUCT_TYPE_TERMS["laptop"]

        has_laptop_term = any(
            term in text
            for term in laptop_terms
        )

        # Explicit desktop-only products should not match.
        desktop_terms = PRODUCT_TYPE_TERMS["desktop"]

        has_desktop_term = any(
            term in text
            for term in desktop_terms
        )

        # Standalone graphics cards are definitely excluded.
        graphics_terms = PRODUCT_TYPE_TERMS["graphics_card"]

        graphics_hits = sum(
            1 for term in graphics_terms
            if term in text
        )

        if graphics_hits >= 2 and not has_laptop_term:
            return False

        if has_laptop_term:
            return True

        if product_type == "desktop":
            return False

        return False

    # --------------------------------------------------------
    # Headphones / earbuds
    # --------------------------------------------------------

    if query_type == "headphones":

        headphone_terms = PRODUCT_TYPE_TERMS["headphones"]

        return any(
            term in text
            for term in headphone_terms
        )

    # --------------------------------------------------------
    # Graphics cards
    # --------------------------------------------------------

    if query_type == "graphics_card":

        graphics_terms = PRODUCT_TYPE_TERMS["graphics_card"]

        return any(
            term in text
            for term in graphics_terms
        )

    # --------------------------------------------------------
    # Other product types
    # --------------------------------------------------------

    if product_type == query_type:
        return True

    query_terms = PRODUCT_TYPE_TERMS.get(query_type, [])

    return any(
        term in text
        for term in query_terms
    )


# ============================================================
# SPEC MATCH
# ============================================================

def calculate_spec_score(
    parsed: dict[str, Any],
    product_text: str,
) -> tuple[float, dict[str, Any]]:

    score_parts = []

    details = {
        "gpu_match": None,
        "ram_match": None,
        "storage_match": None,
    }

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    if parsed["gpu"] is not None:

        query_gpu = parsed["gpu"]

        exact_gpu = gpu_exact_match(
            query_gpu,
            product_text,
        )

        details["gpu_match"] = exact_gpu

        score_parts.append(
            1.0 if exact_gpu else 0.0
        )

    # --------------------------------------------------------
    # RAM
    # --------------------------------------------------------

    if parsed["ram_gb"] is not None:

        product_ram = extract_ram_from_text(
            product_text
        )

        query_ram = parsed["ram_gb"]

        if product_ram is not None:
            ram_match = product_ram >= query_ram
        else:
            ram_match = False

        details["ram_match"] = ram_match

        score_parts.append(
            1.0 if ram_match else 0.0
        )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    if parsed["storage_gb"] is not None:

        product_storage = extract_storage_from_text(
            product_text
        )

        query_storage = parsed["storage_gb"]

        if product_storage is not None:
            storage_match = product_storage >= query_storage
        else:
            storage_match = False

        details["storage_match"] = storage_match

        score_parts.append(
            1.0 if storage_match else 0.0
        )

    if not score_parts:
        return 0.0, details

    return float(np.mean(score_parts)), details


# ============================================================
# HARD SPEC FILTER
# ============================================================

def passes_hard_spec_filter(
    parsed: dict[str, Any],
    product_text: str,
) -> bool:

    # --------------------------------------------------------
    # GPU exact constraint
    # --------------------------------------------------------

    if parsed["gpu"] is not None:

        if not gpu_exact_match(
            parsed["gpu"],
            product_text,
        ):
            return False

    # --------------------------------------------------------
    # RAM minimum
    # --------------------------------------------------------

    if parsed["ram_gb"] is not None:

        product_ram = extract_ram_from_text(
            product_text
        )

        if product_ram is None:
            return False

        if product_ram < parsed["ram_gb"]:
            return False

    # --------------------------------------------------------
    # Storage minimum
    # --------------------------------------------------------

    if parsed["storage_gb"] is not None:

        product_storage = extract_storage_from_text(
            product_text
        )

        if product_storage is None:
            return False

        if product_storage < parsed["storage_gb"]:
            return False

    return True


# ============================================================
# KEYWORD SCORE
# ============================================================

def keyword_score(
    keywords: list[str],
    product_text: str,
) -> float:

    if not keywords:
        return 0.0

    text = normalize_text(product_text)

    matches = sum(
        1 for keyword in keywords
        if keyword in text
    )

    return matches / len(keywords)


# ============================================================
# PRICE FILTER
# ============================================================

def passes_price_filter(
    price: Optional[float],
    price_min: Optional[float],
    price_max: Optional[float],
) -> tuple[bool, str]:

    # No price constraint.
    if price_min is None and price_max is None:
        return True, "no_constraint"

    # Unknown product price.
    if price is None:
        return False, "unknown_price"

    if price_min is not None and price < price_min:
        return False, "below_minimum"

    if price_max is not None and price > price_max:
        return False, "above_maximum"

    return True, "matched"


# ============================================================
# MAIN SEARCH
# ============================================================

def search_products(
    query: str,
    top_k: int,
    index,
    embeddings,
    ids_df,
    metadata,
    model,
):

    start_time = time.perf_counter()

    parsed = parse_query(query)

    print("\n" + "=" * 80)
    print("QUERY")
    print("=" * 80)

    print(query)

    print("\nParsed constraints:")
    print(json.dumps(parsed, indent=2))

    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    embedding_start = time.perf_counter()

    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    embedding_time = (
        time.perf_counter() - embedding_start
    )

    # --------------------------------------------------------
    # FAISS candidate retrieval
    # --------------------------------------------------------

    candidate_k = min(
        max(top_k * 20, 500),
        index.ntotal,
    )

    search_start = time.perf_counter()

    distances, indices = index.search(
        query_vector,
        candidate_k,
    )

    faiss_time = (
        time.perf_counter() - search_start
    )

    # --------------------------------------------------------
    # Candidate evaluation
    # --------------------------------------------------------

    candidates = []

    filtered_counts = {
        "product_type": 0,
        "specification": 0,
        "price": 0,
        "unknown_price": 0,
    }

    for similarity, idx in zip(
        distances[0],
        indices[0],
    ):

        if idx < 0:
            continue

        row = metadata.iloc[int(idx)]

        product_text = get_product_text(row)

        product_type = detect_product_type(
            product_text
        )

        # ----------------------------------------------------
        # Product type hard filter
        # ----------------------------------------------------

        if not passes_product_type_filter(
            parsed["product_type"],
            product_type,
            product_text,
        ):
            filtered_counts["product_type"] += 1
            continue

        # ----------------------------------------------------
        # Specification hard filter
        # ----------------------------------------------------

        if not passes_hard_spec_filter(
            parsed,
            product_text,
        ):
            filtered_counts["specification"] += 1
            continue

        # ----------------------------------------------------
        # Price hard filter
        # ----------------------------------------------------

        price = get_product_price(row)

        price_pass, price_status = passes_price_filter(
            price,
            parsed["price_min"],
            parsed["price_max"],
        )

        if not price_pass:

            if price_status == "unknown_price":
                filtered_counts["unknown_price"] += 1
            else:
                filtered_counts["price"] += 1

            continue

        # ----------------------------------------------------
        # Scores
        # ----------------------------------------------------

        semantic_score = float(similarity)

        type_score = product_type_match(
            parsed["product_type"],
            product_type,
            product_text,
        )

        spec_score, spec_details = calculate_spec_score(
            parsed,
            product_text,
        )

        kw_score = keyword_score(
            parsed["keywords"],
            product_text,
        )

        # ----------------------------------------------------
        # Hybrid score
        # ----------------------------------------------------

        hybrid_score = (
            0.60 * semantic_score
            + 0.15 * type_score
            + 0.20 * spec_score
            + 0.05 * kw_score
        )

        product_id = safe_text(
            ids_df.iloc[int(idx)].iloc[0]
        )

        title = ""

        if "title" in row.index:
            title = safe_text(row["title"])

        store = ""

        if "store" in row.index:
            store = safe_text(row["store"])

        brand = ""

        if "brand" in row.index:
            brand = safe_text(row["brand"])

        candidates.append(
            {
                "rank": 0,
                "product_id": product_id,
                "parent_asin": safe_text(row["parent_asin"])
                if "parent_asin" in row.index else product_id,
                "title": title,
                "brand": brand,
                "store": store,
                "price": price,
                "image_url": get_image_url(row),
                "product_type": product_type,
                "semantic_score": round(
                    semantic_score,
                    4,
                ),
                "type_score": round(
                    type_score,
                    4,
                ),
                "spec_score": round(
                    spec_score,
                    4,
                ),
                "keyword_score": round(
                    kw_score,
                    4,
                ),
                "hybrid_score": round(
                    hybrid_score,
                    4,
                ),
                "gpu_match": spec_details[
                    "gpu_match"
                ],
                "ram_match": spec_details[
                    "ram_match"
                ],
                "storage_match": spec_details[
                    "storage_match"
                ],
                "price_constraint": True,
            }
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True,
    )

    results = candidates[:top_k]

    for rank, result in enumerate(
        results,
        start=1,
    ):
        result["rank"] = rank

    total_time = (
        time.perf_counter() - start_time
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = {
        "query": query,
        "parsed_query": parsed,
        "search_config": {
            "semantic_weight": 0.60,
            "type_weight": 0.15,
            "spec_weight": 0.20,
            "keyword_weight": 0.05,
            "candidate_pool": candidate_k,
            "hard_price_filter": True,
            "hard_gpu_filter": True,
            "hard_ram_filter": True,
            "hard_storage_filter": True,
            "hard_product_type_filter": True,
        },
        "performance": {
            "embedding_ms": round(
                embedding_time * 1000,
                2,
            ),
            "faiss_ms": round(
                faiss_time * 1000,
                2,
            ),
            "total_ms": round(
                total_time * 1000,
                2,
            ),
        },
        "filtered_counts": filtered_counts,
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("HYBRID SEARCH v2 RESULTS")
    print("=" * 80)

    print(
        f"\nCandidates retrieved: {candidate_k:,}"
    )

    print(
        f"Filtered by product type: "
        f"{filtered_counts['product_type']:,}"
    )

    print(
        f"Filtered by specification: "
        f"{filtered_counts['specification']:,}"
    )

    print(
        f"Filtered by price: "
        f"{filtered_counts['price']:,}"
    )

    print(
        f"Filtered because price unknown: "
        f"{filtered_counts['unknown_price']:,}"
    )

    if not results:
        print("\nNo products satisfied all hard constraints.")

    else:

        print(
            f"\nShowing top {len(results)} results:"
        )

        print("-" * 80)

        for result in results:

            print(
                f"\n#{result['rank']} "
                f"{result['title'][:100]}"
            )

            print(
                f"   Product ID: "
                f"{result['product_id']}"
            )

            print(
                f"   Brand: "
                f"{result['brand']}"
            )

            print(
                f"   Product type: "
                f"{result['product_type']}"
            )

            if result["price"] is not None:
                print(
                    f"   Price: "
                    f"${result['price']:.2f}"
                )
            else:
                print(
                    "   Price: Unknown"
                )

            print(
                f"   Semantic: "
                f"{result['semantic_score']:.4f}"
            )

            print(
                f"   Type: "
                f"{result['type_score']:.4f}"
            )

            print(
                f"   Specs: "
                f"{result['spec_score']:.4f}"
            )

            print(
                f"   Keywords: "
                f"{result['keyword_score']:.4f}"
            )

            print(
                f"   Hybrid: "
                f"{result['hybrid_score']:.4f}"
            )

            print(
                f"   GPU match: "
                f"{result['gpu_match']}"
            )

            print(
                f"   RAM match: "
                f"{result['ram_match']}"
            )

            print(
                f"   Storage match: "
                f"{result['storage_match']}"
            )

            print(
                f"   Price constraint: "
                f"{result['price_constraint']}"
            )

    print("\n" + "-" * 80)

    print(
        f"Embedding time: "
        f"{embedding_time * 1000:.2f} ms"
    )

    print(
        f"FAISS time: "
        f"{faiss_time * 1000:.2f} ms"
    )

    print(
        f"Total search time: "
        f"{total_time * 1000:.2f} ms"
    )

    print(
        f"\nSaved results to:\n"
        f"{OUTPUT_PATH}"
    )

    return output


# ============================================================
# API ADAPTER
# ============================================================

class HybridSearch:
    """Reusable, in-memory adapter used by the FastAPI search endpoint."""

    def __init__(self):
        self.index, self.embeddings, self.ids_df, self.metadata = load_data()
        print("\nLoading sentence-transformer model...")
        self.model = SentenceTransformer(MODEL_NAME)

    def search(self, query: str, top_k: int = 10):
        output = search_products(
            query=query,
            top_k=top_k,
            index=self.index,
            embeddings=self.embeddings,
            ids_df=self.ids_df,
            metadata=self.metadata,
            model=self.model,
        )
        return output["results"]


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="ShopGraph Hybrid Search v2"
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Natural language product query",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load everything
    # --------------------------------------------------------

    index, embeddings, ids_df, metadata = load_data()

    print("\nLoading SentenceTransformer...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Model loaded.")

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    search_products(
        query=args.query,
        top_k=args.top_k,
        index=index,
        embeddings=embeddings,
        ids_df=ids_df,
        metadata=metadata,
        model=model,
    )


if __name__ == "__main__":
    main()
