"""
ShopGraph - Hybrid Product Search v2.9

Combines:
1. Semantic search using all-MiniLM-L6-v2
2. FAISS cosine similarity
3. Product-type filtering
4. Strict RAM matching
5. Strict storage matching
6. Strict GPU matching
7. Hard price constraints
8. Keyword scoring
9. Robust metadata/embedding alignment

Important:
- Uses Transformers + PyTorch directly.
- Does NOT require sentence-transformers.
- Uses fastparquet explicitly because pyarrow may be unavailable.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SEARCH_DIR = PROCESSED_DIR / "search"

EMBEDDINGS_PATH = PROCESSED_DIR / "product_embeddings.npy"
EMBEDDING_IDS_PATH = PROCESSED_DIR / "embedding_product_ids.parquet"
METADATA_PATH = PROCESSED_DIR / "products_text.parquet"
FAISS_INDEX_PATH = SEARCH_DIR / "faiss_products.index"

OUTPUT_PATH = SEARCH_DIR / "last_hybrid_search_v2.json"


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# SEARCH SETTINGS
# ============================================================

DEFAULT_TOP_K = 10

SEMANTIC_WEIGHT = 0.60
TYPE_WEIGHT = 0.15
SPEC_WEIGHT = 0.20
KEYWORD_WEIGHT = 0.05

CANDIDATE_MULTIPLIER = 50
MIN_CANDIDATES = 2000


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(value) -> str:
    """
    Convert metadata values safely into text.

    Handles:
    - strings
    - lists
    - tuples
    - numpy arrays
    - NaN
    - None
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple, set)):
        return " ".join(clean_text(v) for v in value)

    return str(value).strip()


def normalize_for_matching(text: str) -> str:
    """Normalize text for matching."""
    text = clean_text(text).lower()
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("×", "x")
    return text


# ============================================================
# QUERY PARSING
# ============================================================

def extract_product_type(query: str) -> Optional[str]:
    """
    Detect broad product type.
    """

    q = normalize_for_matching(query)

    product_types = {
        "laptop": [
            "laptop",
            "notebook",
            "chromebook",
            "macbook",
        ],
        "desktop": [
            "desktop",
            "desktop computer",
            "pc",
        ],
        "monitor": [
            "monitor",
            "computer monitor",
            "display",
        ],
        "tablet": [
            "tablet",
            "ipad",
        ],
        "headphones": [
            "headphones",
            "headphone",
            "earbuds",
            "earbud",
            "wireless earbuds",
            "headset",
        ],
        "keyboard": [
            "keyboard",
            "keyboards",
        ],
        "mouse": [
            "mouse",
            "gaming mouse",
        ],
        "camera": [
            "camera",
            "webcam",
            "digital camera",
        ],
        "printer": [
            "printer",
            "printers",
        ],
        "router": [
            "router",
            "wifi router",
            "wireless router",
        ],
        "ssd": [
            "ssd",
            "solid state drive",
        ],
        "hard_drive": [
            "hard drive",
            "hard disk",
            "hdd",
        ],
        "graphics_card": [
            "graphics card",
            "gpu",
            "video card",
        ],
        "phone": [
            "smartphone",
            "phone",
            "cell phone",
        ],
        "charger": [
            "charger",
            "power adapter",
            "ac adapter",
        ],
    }

    # More specific types first
    ordered_types = sorted(
        product_types.items(),
        key=lambda x: max(len(v) for v in x[1]),
        reverse=True,
    )

    for product_type, phrases in ordered_types:
        for phrase in phrases:
            if re.search(
                rf"\b{re.escape(phrase)}\b",
                q,
                flags=re.IGNORECASE,
            ):
                return product_type

    return None


def extract_ram_constraint(query: str) -> Optional[float]:
    """
    Extract minimum RAM requirement.

    Examples:
        16GB RAM -> 16
        32 GB RAM -> 32
        at least 32GB RAM -> 32

    Important:
        Does NOT confuse storage with RAM.
        Does NOT interpret arbitrary numbers as RAM.
    """

    q = normalize_for_matching(query)

    patterns = [
        # at least 32GB RAM
        r"\bat\s+least\s+(\d+(?:\.\d+)?)\s*(gb|tb)\s*(?:of\s*)?(?:ram|memory)\b",

        # minimum 32GB RAM
        r"\bminimum\s+(\d+(?:\.\d+)?)\s*(gb|tb)\s*(?:of\s*)?(?:ram|memory)\b",

        # 32GB RAM
        r"\b(\d+(?:\.\d+)?)\s*(gb|tb)\s*(?:ddr\d[\w-]*\s*)?(?:ram|memory)\b",

        # RAM 32GB
        r"\b(?:ram|memory)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(gb|tb)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)

        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()

            if unit == "tb":
                value *= 1024

            return value

    return None


def extract_storage_constraint(query: str) -> Optional[float]:
    """
    Extract minimum storage requirement.

    Examples:
        1TB SSD -> 1024 GB
        512GB SSD -> 512 GB
        at least 1TB storage -> 1024 GB

    Storage must be associated with a storage term.
    """

    q = normalize_for_matching(query)

    patterns = [
        # at least 1TB storage
        r"\bat\s+least\s+(\d+(?:\.\d+)?)\s*(gb|tb)\s*(?:of\s*)?(?:storage|ssd|hdd|hard\s*drive|hard\s*disk)\b",

        # minimum 1TB storage
        r"\bminimum\s+(\d+(?:\.\d+)?)\s*(gb|tb)\s*(?:of\s*)?(?:storage|ssd|hdd|hard\s*drive|hard\s*disk)\b",

        # 1TB SSD / 512GB HDD
        r"\b(\d+(?:\.\d+)?)\s*(gb|tb)\s*(?:ssd|hdd|storage|hard\s*drive|hard\s*disk)\b",

        # SSD 1TB
        r"\b(?:ssd|hdd|storage|hard\s*drive|hard\s*disk)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(gb|tb)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)

        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()

            if unit == "tb":
                value *= 1024

            return value

    return None


def extract_gpu_constraint(query: str) -> Optional[str]:
    """
    Extract NVIDIA/AMD GPU requirement.
    """

    q = normalize_for_matching(query)

    gpu_patterns = [
        r"\brtx\s*(\d{3,4})\b",
        r"\bgtx\s*(\d{3,4})\b",
        r"\bradeon\s*(rx\s*)?(\d{3,4})\b",
    ]

    for pattern in gpu_patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)

        if match:
            value = " ".join(
                group for group in match.groups() if group
            ).strip()

            value = re.sub(r"\s+", " ", value)

            if value.startswith("rx "):
                return f"radeon {value}"

            if value.isdigit():
                # This branch handles the simple RTX/GTX pattern.
                if "rtx" in q[match.start():match.end()].lower():
                    return f"rtx {value}"

                if "gtx" in q[match.start():match.end()].lower():
                    return f"gtx {value}"

            return value

    # Fallback for generic NVIDIA/AMD terms
    generic_patterns = [
        r"\b(nvidia\s+rtx\s*\d{3,4})\b",
        r"\b(nvidia\s+gtx\s*\d{3,4})\b",
        r"\b(amd\s+radeon\s+rx\s*\d{3,4})\b",
    ]

    for pattern in generic_patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)

        if match:
            return re.sub(
                r"\s+",
                " ",
                match.group(1),
            ).strip()

    return None


def extract_price_constraints(
    query: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract ONLY explicit dollar price constraints.

    Supported:

        under $1500
        below $1500
        less than $1500
        up to $1500
        over $500
        above $500
        more than $500
        at least $500

    IMPORTANT:

    A bare number is NOT treated as a price.

    Therefore:

        "laptop with at least 32GB RAM"

    will produce:

        price_min = None
        price_max = None

    while:

        "laptop with at least 32GB RAM under $1500"

    produces:

        price_max = 1500
    """

    q = normalize_for_matching(query)

    # Explicit dollar sign is REQUIRED.
    price_number = (
        r"\$\s*"
        r"([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)"
    )

    price_min = None
    price_max = None

    # --------------------------------------------------------
    # MAX PRICE
    # --------------------------------------------------------

    max_patterns = [
        rf"\bunder\s+{price_number}",
        rf"\bbelow\s+{price_number}",
        rf"\bless\s+than\s+{price_number}",
        rf"\bup\s+to\s+{price_number}",
        rf"\bmaximum\s+{price_number}",
        rf"\bmax(?:imum)?\s+price\s+(?:of\s+)?{price_number}",
    ]

    for pattern in max_patterns:
        match = re.search(
            pattern,
            q,
            flags=re.IGNORECASE,
        )

        if match:
            value = match.group(1).replace(",", "")
            price_max = float(value)
            break

    # --------------------------------------------------------
    # MIN PRICE
    # --------------------------------------------------------

    min_patterns = [
        rf"\bover\s+{price_number}",
        rf"\babove\s+{price_number}",
        rf"\bmore\s+than\s+{price_number}",
        rf"\bat\s+least\s+{price_number}",
        rf"\bminimum\s+{price_number}",
        rf"\bmin(?:imum)?\s+price\s+(?:of\s+)?{price_number}",
    ]

    for pattern in min_patterns:
        match = re.search(
            pattern,
            q,
            flags=re.IGNORECASE,
        )

        if match:
            value = match.group(1).replace(",", "")
            price_min = float(value)
            break

    return price_min, price_max


def extract_keywords(
    query: str,
    product_type: Optional[str],
    ram_gb: Optional[float],
    storage_gb: Optional[float],
    gpu: Optional[str],
    price_min: Optional[float],
    price_max: Optional[float],
) -> List[str]:
    """
    Extract meaningful keywords for lightweight keyword scoring.

    Technical constraints such as:
        32GB RAM
        RTX 3070
        $1500

    are handled separately.
    """

    q = normalize_for_matching(query)

    # Remove technical constraint expressions
    q = re.sub(
        r"\b(?:at\s+least|minimum)\s+\d+(?:\.\d+)?\s*(?:gb|tb)\s*(?:of\s*)?(?:ram|memory)\b",
        " ",
        q,
        flags=re.IGNORECASE,
    )

    q = re.sub(
        r"\b\d+(?:\.\d+)?\s*(?:gb|tb)\s*(?:ddr\d[\w-]*\s*)?(?:ram|memory)\b",
        " ",
        q,
        flags=re.IGNORECASE,
    )

    q = re.sub(
        r"\b(?:at\s+least|minimum)\s+\d+(?:\.\d+)?\s*(?:gb|tb)\s*(?:of\s*)?(?:storage|ssd|hdd|hard\s*drive|hard\s*disk)\b",
        " ",
        q,
        flags=re.IGNORECASE,
    )

    q = re.sub(
        r"\b\d+(?:\.\d+)?\s*(?:gb|tb)\s*(?:ssd|hdd|storage|hard\s*drive|hard\s*disk)\b",
        " ",
        q,
        flags=re.IGNORECASE,
    )

    q = re.sub(
        r"\b(?:rtx|gtx)\s*\d{3,4}\b",
        " ",
        q,
        flags=re.IGNORECASE,
    )

    q = re.sub(
        r"\$\s*[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?",
        " ",
        q,
    )

    # Remove common constraint words
    stop_words = {
        "with",
        "at",
        "least",
        "minimum",
        "under",
        "below",
        "less",
        "than",
        "up",
        "to",
        "over",
        "above",
        "more",
        "maximum",
        "max",
        "price",
        "of",
        "for",
        "the",
        "a",
        "an",
        "and",
        "or",
        "that",
        "has",
        "have",
        "having",
        "gb",
        "tb",
        "ram",
        "memory",
        "storage",
        "ssd",
        "hdd",
    }

    # Remove detected product type words
    if product_type:
        product_type_words = {
            "laptop": {"laptop", "notebook", "chromebook", "macbook"},
            "desktop": {"desktop", "pc", "computer"},
            "monitor": {"monitor", "display"},
            "tablet": {"tablet", "ipad"},
            "headphones": {
                "headphones",
                "headphone",
                "earbuds",
                "earbud",
                "headset",
            },
            "keyboard": {"keyboard", "keyboards"},
            "mouse": {"mouse"},
            "camera": {"camera", "webcam"},
            "printer": {"printer", "printers"},
            "router": {"router"},
            "ssd": {"ssd"},
            "hard_drive": {
                "hard",
                "drive",
                "disk",
                "hdd",
            },
            "graphics_card": {
                "graphics",
                "card",
                "gpu",
            },
            "phone": {
                "phone",
                "smartphone",
            },
            "charger": {
                "charger",
            },
        }

        stop_words.update(
            product_type_words.get(
                product_type,
                set(),
            )
        )

    tokens = re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9-]*\b",
        q,
    )

    keywords = []

    for token in tokens:
        token = token.lower().strip()

        if len(token) < 2:
            continue

        if token in stop_words:
            continue

        if token not in keywords:
            keywords.append(token)

    return keywords


def parse_query(query: str) -> Dict:
    """
    Parse natural-language query.
    """

    product_type = extract_product_type(query)
    ram_gb = extract_ram_constraint(query)
    storage_gb = extract_storage_constraint(query)
    gpu = extract_gpu_constraint(query)

    price_min, price_max = extract_price_constraints(query)

    keywords = extract_keywords(
        query=query,
        product_type=product_type,
        ram_gb=ram_gb,
        storage_gb=storage_gb,
        gpu=gpu,
        price_min=price_min,
        price_max=price_max,
    )

    return {
        "product_type": product_type,
        "ram_gb": ram_gb,
        "storage_gb": storage_gb,
        "gpu": gpu,
        "price_min": price_min,
        "price_max": price_max,
        "keywords": keywords,
    }


# ============================================================
# PRODUCT TEXT
# ============================================================

def build_product_text(row: pd.Series) -> str:
    """
    Build searchable product text from metadata.
    """

    fields = [
        row.get("title", ""),
        row.get("store", ""),
        row.get("categories", ""),
        row.get("clean_title", ""),
        row.get("clean_store", ""),
        row.get("clean_features", ""),
        row.get("clean_description", ""),
        row.get("category_text", ""),
        row.get("embedding_text", ""),
    ]

    parts = []

    for value in fields:
        text = clean_text(value)

        if text:
            parts.append(text)

    return " ".join(parts)


# ============================================================
# PRODUCT TYPE MATCHING
# ============================================================

PRODUCT_TYPE_TERMS = {
    "laptop": [
        "laptop",
        "notebook",
        "chromebook",
        "macbook",
        "ultrabook",
    ],
    "desktop": [
        "desktop",
        "desktop computer",
        "tower pc",
        "desktop pc",
    ],
    "monitor": [
        "monitor",
        "computer monitor",
        "display",
    ],
    "tablet": [
        "tablet",
        "ipad",
    ],
    "headphones": [
        "headphones",
        "headphone",
        "earbuds",
        "earbud",
        "headset",
    ],
    "keyboard": [
        "keyboard",
    ],
    "mouse": [
        "computer mouse",
        "gaming mouse",
        "wireless mouse",
        "mouse",
    ],
    "camera": [
        "camera",
        "webcam",
        "digital camera",
    ],
    "printer": [
        "printer",
    ],
    "router": [
        "router",
        "wifi router",
        "wireless router",
    ],
    "ssd": [
        "solid state drive",
        "ssd",
    ],
    "hard_drive": [
        "hard drive",
        "hard disk",
        "hdd",
    ],
    "graphics_card": [
        "graphics card",
        "video card",
        "gpu",
    ],
    "phone": [
        "smartphone",
        "cell phone",
        "mobile phone",
        "phone",
    ],
    "charger": [
        "charger",
        "power adapter",
        "ac adapter",
    ],
}


def product_type_match(
    text: str,
    product_type: Optional[str],
) -> bool:
    """
    Strict product-type match.
    """

    if not product_type:
        return True

    text = normalize_for_matching(text)

    terms = PRODUCT_TYPE_TERMS.get(
        product_type,
        [],
    )

    for term in terms:
        if re.search(
            rf"\b{re.escape(term)}\b",
            text,
            flags=re.IGNORECASE,
        ):
            return True

    return False


# ============================================================
# RAM PARSING
# ============================================================

def extract_product_ram(text: str) -> Optional[float]:
    """
    Strict RAM extraction.

    This function is intentionally conservative.

    Correct:
        "32GB RAM" -> 32
        "32 GB DDR4 RAM" -> 32
        "RAM: 32GB" -> 32
        "4GB LPDDR4 RAM, 32GB eMMC" -> 4
        "512MB RAM, 100GB Hard Drive" -> 0.5

    Incorrect values such as:
        32GB eMMC
        500GB HDD
        1TB SSD

    are NOT treated as RAM.
    """

    text = normalize_for_matching(text)

    # --------------------------------------------------------
    # RAM VALUE BEFORE RAM/MEMORY
    # --------------------------------------------------------

    patterns_before = [
        # 32GB DDR4 RAM
        r"\b(\d+(?:\.\d+)?)\s*(gb|tb|mb)\s*(?:ddr\d[\w-]*|lpddr\d[\w-]*|sdram|dram|dimm|sodimm)?\s*(?:ram|memory)\b",

        # 32GB of RAM
        r"\b(\d+(?:\.\d+)?)\s*(gb|tb|mb)\s+of\s+(?:ram|memory)\b",

        # 32GB RAM
        r"\b(\d+(?:\.\d+)?)\s*(gb|tb|mb)\s+(?:ram|memory)\b",
    ]

    for pattern in patterns_before:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()

            if unit == "tb":
                value *= 1024

            elif unit == "mb":
                value /= 1024

            return value

    # --------------------------------------------------------
    # RAM/MEMORY BEFORE VALUE
    # --------------------------------------------------------

    patterns_after = [
        r"\b(?:ram|memory)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(gb|tb|mb)\b",
    ]

    for pattern in patterns_after:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()

            if unit == "tb":
                value *= 1024

            elif unit == "mb":
                value /= 1024

            return value

    return None


# ============================================================
# STORAGE PARSING
# ============================================================

def extract_product_storage(text: str) -> Optional[float]:
    """
    Extract storage capacity.

    Requires storage-related terminology so RAM
    values are not mistaken for storage.

    Examples:
        1TB SSD -> 1024
        512GB SSD -> 512
        500GB HDD -> 500
        32GB eMMC -> 32
    """

    text = normalize_for_matching(text)

    storage_patterns = [
        # 1TB SSD
        r"\b(\d+(?:\.\d+)?)\s*(tb|gb|mb)\s*(?:ssd|solid\s+state\s+drive)\b",

        # 500GB HDD
        r"\b(\d+(?:\.\d+)?)\s*(tb|gb|mb)\s*(?:hdd|hard\s+drive|hard\s+disk)\b",

        # 32GB eMMC
        r"\b(\d+(?:\.\d+)?)\s*(tb|gb|mb)\s*emmc\b",

        # 128GB storage
        r"\b(\d+(?:\.\d+)?)\s*(tb|gb|mb)\s*(?:storage|flash\s+storage)\b",

        # SSD 1TB
        r"\b(?:ssd|solid\s+state\s+drive)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(tb|gb|mb)\b",

        # HDD 500GB
        r"\b(?:hdd|hard\s+drive|hard\s+disk)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(tb|gb|mb)\b",

        # eMMC 32GB
        r"\bemmc\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(tb|gb|mb)\b",
    ]

    capacities = []

    for pattern in storage_patterns:
        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            value = float(match.group(1))
            unit = match.group(2).lower()

            if unit == "tb":
                value *= 1024

            elif unit == "mb":
                value /= 1024

            capacities.append(value)

    if not capacities:
        return None

    # Return largest storage capacity mentioned.
    return max(capacities)


# ============================================================
# GPU PARSING
# ============================================================

def extract_product_gpu(text: str) -> Optional[str]:
    """
    Extract GPU model.
    """

    text = normalize_for_matching(text)

    patterns = [
        r"\brtx\s*\d{3,4}\b",
        r"\bgtx\s*\d{3,4}\b",
        r"\bradeon\s+rx\s*\d{3,4}\b",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return re.sub(
                r"\s+",
                " ",
                match.group(0),
            ).lower().strip()

    return None


# ============================================================
# PRICE PARSING
# ============================================================

def parse_price(value) -> Optional[float]:
    """
    Safely parse product price.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)

        if np.isfinite(value):
            return value

        return None

    text = clean_text(value)

    if not text:
        return None

    match = re.search(
        r"([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)",
        text,
    )

    if not match:
        return None

    try:
        price = float(
            match.group(1).replace(",", "")
        )

        if np.isfinite(price):
            return price

    except Exception:
        pass

    return None


# ============================================================
# SPECIFICATION MATCHING
# ============================================================

def specification_matches(
    text: str,
    query_info: Dict,
) -> Tuple[bool, Dict]:
    """
    Check hard technical requirements.

    Returns:
        (matches, diagnostics)
    """

    ram_required = query_info["ram_gb"]
    storage_required = query_info["storage_gb"]
    gpu_required = query_info["gpu"]

    detected_ram = extract_product_ram(text)
    detected_storage = extract_product_storage(text)
    detected_gpu = extract_product_gpu(text)

    # --------------------------------------------------------
    # RAM
    # --------------------------------------------------------

    if ram_required is not None:

        if detected_ram is None:
            return False, {
                "reason": "ram_unknown",
                "detected_ram_gb": None,
                "detected_storage_gb": detected_storage,
                "detected_gpu": detected_gpu,
            }

        if detected_ram < ram_required:
            return False, {
                "reason": "ram_below_requirement",
                "detected_ram_gb": detected_ram,
                "detected_storage_gb": detected_storage,
                "detected_gpu": detected_gpu,
            }

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    if storage_required is not None:

        if detected_storage is None:
            return False, {
                "reason": "storage_unknown",
                "detected_ram_gb": detected_ram,
                "detected_storage_gb": None,
                "detected_gpu": detected_gpu,
            }

        if detected_storage < storage_required:
            return False, {
                "reason": "storage_below_requirement",
                "detected_ram_gb": detected_ram,
                "detected_storage_gb": detected_storage,
                "detected_gpu": detected_gpu,
            }

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    if gpu_required is not None:

        if detected_gpu is None:
            return False, {
                "reason": "gpu_unknown",
                "detected_ram_gb": detected_ram,
                "detected_storage_gb": detected_storage,
                "detected_gpu": None,
            }

        if normalize_for_matching(
            detected_gpu
        ) != normalize_for_matching(
            gpu_required
        ):
            return False, {
                "reason": "gpu_mismatch",
                "detected_ram_gb": detected_ram,
                "detected_storage_gb": detected_storage,
                "detected_gpu": detected_gpu,
            }

    return True, {
        "reason": "match",
        "detected_ram_gb": detected_ram,
        "detected_storage_gb": detected_storage,
        "detected_gpu": detected_gpu,
    }


# ============================================================
# KEYWORD MATCHING
# ============================================================

def keyword_score(
    text: str,
    keywords: List[str],
) -> float:
    """
    Calculate keyword coverage.
    """

    if not keywords:
        return 1.0

    text = normalize_for_matching(text)

    matched = 0

    for keyword in keywords:
        if re.search(
            rf"\b{re.escape(keyword)}\b",
            text,
            flags=re.IGNORECASE,
        ):
            matched += 1

    return matched / len(keywords)


# ============================================================
# MODEL LOADING
# ============================================================

def load_embedding_model():
    """
    Load MiniLM using Transformers directly.
    """

    print()
    print("=" * 80)
    print("LOADING EMBEDDING MODEL")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME
    )

    model.to(DEVICE)
    model.eval()

    elapsed = time.time() - start

    print(
        f"Model loaded in {elapsed:.2f} seconds"
    )

    return tokenizer, model


def mean_pooling(
    model_output,
    attention_mask,
):
    """
    Mean pooling with attention mask.
    """

    token_embeddings = model_output.last_hidden_state

    input_mask_expanded = (
        attention_mask
        .unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )

    return torch.sum(
        token_embeddings * input_mask_expanded,
        dim=1,
    ) / torch.clamp(
        input_mask_expanded.sum(dim=1),
        min=1e-9,
    )


def embed_query(
    query: str,
    tokenizer,
    model,
) -> np.ndarray:
    """
    Generate normalized MiniLM query embedding.
    """

    start = time.time()

    encoded = tokenizer(
        query,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        model_output = model(**encoded)

        embedding = mean_pooling(
            model_output,
            encoded["attention_mask"],
        )

        embedding = torch.nn.functional.normalize(
            embedding,
            p=2,
            dim=1,
        )

    vector = embedding.cpu().numpy().astype(
        np.float32
    )

    elapsed = time.time() - start

    print(
        f"Query embedding time: {elapsed * 1000:.2f} ms"
    )

    return vector


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """
    Load embeddings, IDs, metadata and FAISS index.

    embedding_index is treated as the authoritative
    embedding-row mapping.
    """

    print()
    print("=" * 80)
    print("LOADING SEARCH DATA")
    print("=" * 80)

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    embeddings = np.load(
        EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    print(
        f"Embeddings shape: {embeddings.shape}"
    )

    # --------------------------------------------------------
    # Embedding IDs
    # --------------------------------------------------------

    embedding_ids = pd.read_parquet(
        EMBEDDING_IDS_PATH,
        engine="fastparquet",
    )

    print(
        f"Embedding IDs shape: {embedding_ids.shape}"
    )

    print(
        f"Embedding ID columns: {list(embedding_ids.columns)}"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = pd.read_parquet(
        METADATA_PATH,
        engine="fastparquet",
    )

    print(
        f"Metadata shape: {metadata.shape}"
    )

    print(
        f"Metadata columns: {list(metadata.columns)}"
    )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    index = faiss.read_index(
        str(FAISS_INDEX_PATH)
    )

    print(
        f"FAISS vectors: {index.ntotal}"
    )

    # --------------------------------------------------------
    # Ensure embedding_index exists
    # --------------------------------------------------------

    if "embedding_index" not in embedding_ids.columns:

        print(
            "embedding_index missing; "
            "creating sequential mapping."
        )

        embedding_ids = embedding_ids.copy()

        embedding_ids.insert(
            0,
            "embedding_index",
            np.arange(
                len(embedding_ids),
                dtype=np.int64,
            ),
        )

    embedding_ids["embedding_index"] = (
        pd.to_numeric(
            embedding_ids["embedding_index"],
            errors="coerce",
        )
    )

    # --------------------------------------------------------
    # ASIN columns
    # --------------------------------------------------------

    embedding_asin_column = None

    for column in [
        "parent_asin",
        "asin",
        "product_id",
    ]:
        if column in embedding_ids.columns:
            embedding_asin_column = column
            break

    metadata_asin_column = None

    for column in [
        "parent_asin",
        "asin",
        "product_id",
    ]:
        if column in metadata.columns:
            metadata_asin_column = column
            break

    print(
        f"Embedding ID column: {embedding_asin_column}"
    )

    print(
        f"Metadata ID column: {metadata_asin_column}"
    )

    # --------------------------------------------------------
    # Build metadata lookup by ASIN
    # --------------------------------------------------------

    metadata_lookup = {}

    if metadata_asin_column is not None:

        for _, row in metadata.iterrows():

            asin = clean_text(
                row[metadata_asin_column]
            )

            # Missing ASINs are NOT duplicates.
            if not asin:
                continue

            if asin not in metadata_lookup:
                metadata_lookup[asin] = row

    # --------------------------------------------------------
    # Build searchable row mapping
    # --------------------------------------------------------

    searchable_rows = []

    missing_embedding_asins = 0
    missing_metadata_asins = 0

    for _, id_row in embedding_ids.iterrows():

        embedding_index = id_row[
            "embedding_index"
        ]

        if pd.isna(embedding_index):
            continue

        embedding_index = int(
            embedding_index
        )

        if embedding_index < 0:
            continue

        if embedding_index >= len(embeddings):
            continue

        # --------------------------------------------
        # Find ASIN
        # --------------------------------------------

        asin = ""

        if embedding_asin_column is not None:

            asin = clean_text(
                id_row[
                    embedding_asin_column
                ]
            )

        if not asin:

            missing_embedding_asins += 1
            continue

        # --------------------------------------------
        # Find metadata
        # --------------------------------------------

        metadata_row = metadata_lookup.get(
            asin
        )

        if metadata_row is None:

            missing_metadata_asins += 1
            continue

        searchable_rows.append(
            {
                "embedding_index": embedding_index,
                "asin": asin,
                "metadata_row": metadata_row,
            }
        )

    print(
        f"Missing embedding ASINs: "
        f"{missing_embedding_asins}"
    )

    print(
        f"Missing metadata ASINs: "
        f"{missing_metadata_asins}"
    )

    print(
        f"Searchable rows: "
        f"{len(searchable_rows)}"
    )

    return (
        embeddings,
        embedding_ids,
        metadata,
        index,
        searchable_rows,
    )


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
):
    """
    Execute hybrid product search.
    """

    print()
    print("=" * 80)
    print("HYBRID SEARCH v2.9")
    print("=" * 80)

    print(f"Query: {query}")
    print(f"Top K: {top_k}")

    # --------------------------------------------------------
    # Parse query
    # --------------------------------------------------------

    query_info = parse_query(query)

    print()
    print("Parsed query:")
    print(
        f"  Product type: {query_info['product_type']}"
    )
    print(
        f"  RAM: {query_info['ram_gb']}"
    )
    print(
        f"  Storage: {query_info['storage_gb']}"
    )
    print(
        f"  GPU: {query_info['gpu']}"
    )
    print(
        f"  Price min: {query_info['price_min']}"
    )
    print(
        f"  Price max: {query_info['price_max']}"
    )
    print(
        f"  Keywords: {query_info['keywords']}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        embeddings,
        embedding_ids,
        metadata,
        index,
        searchable_rows,
    ) = load_data()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    tokenizer, model = load_embedding_model()

    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_vector = embed_query(
        query,
        tokenizer,
        model,
    )

    # --------------------------------------------------------
    # Candidate pool
    # --------------------------------------------------------

    candidate_count = max(
        top_k * CANDIDATE_MULTIPLIER,
        MIN_CANDIDATES,
    )

    candidate_count = min(
        candidate_count,
        index.ntotal,
    )

    print()
    print(
        f"FAISS candidate pool: {candidate_count}"
    )

    start = time.time()

    similarities, indices = index.search(
        query_vector,
        candidate_count,
    )

    faiss_time = time.time() - start

    print(
        f"FAISS search time: "
        f"{faiss_time * 1000:.2f} ms"
    )

    # --------------------------------------------------------
    # Searchable mapping
    # --------------------------------------------------------

    searchable_by_embedding_index = {
        item["embedding_index"]: item
        for item in searchable_rows
    }

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    type_match_count = 0
    specification_match_count = 0
    price_match_count = 0

    unknown_ram_count = 0
    unknown_storage_count = 0
    unknown_gpu_count = 0
    unknown_price_count = 0

    rejected_type_count = 0
    rejected_spec_count = 0
    rejected_price_count = 0

    results = []

    # --------------------------------------------------------
    # Candidate processing
    # --------------------------------------------------------

    for rank, faiss_index in enumerate(
        indices[0]
    ):

        if faiss_index < 0:
            continue

        similarity = float(
            similarities[0][rank]
        )

        item = searchable_by_embedding_index.get(
            int(faiss_index)
        )

        if item is None:
            continue

        metadata_row = item["metadata_row"]

        title = clean_text(
            metadata_row.get(
                "title",
                "",
            )
        )

        store = clean_text(
            metadata_row.get(
                "store",
                "",
            )
        )

        product_text = build_product_text(
            metadata_row
        )

        # ----------------------------------------------------
        # Product type
        # ----------------------------------------------------

        type_matches = product_type_match(
            product_text,
            query_info["product_type"],
        )

        if not type_matches:

            rejected_type_count += 1
            continue

        type_score = 1.0

        if query_info["product_type"] is not None:
            type_match_count += 1

        # ----------------------------------------------------
        # Specifications
        # ----------------------------------------------------

        spec_matches, spec_info = (
            specification_matches(
                product_text,
                query_info,
            )
        )

        if query_info["ram_gb"] is not None:

            if (
                spec_info[
                    "detected_ram_gb"
                ]
                is None
            ):
                unknown_ram_count += 1

        if query_info["storage_gb"] is not None:

            if (
                spec_info[
                    "detected_storage_gb"
                ]
                is None
            ):
                unknown_storage_count += 1

        if query_info["gpu"] is not None:

            if (
                spec_info[
                    "detected_gpu"
                ]
                is None
            ):
                unknown_gpu_count += 1

        if not spec_matches:

            rejected_spec_count += 1
            continue

        specification_match_count += 1

        spec_score = 1.0

        # ----------------------------------------------------
        # Price
        # ----------------------------------------------------

        price = parse_price(
            metadata_row.get(
                "price",
                None,
            )
        )

        price_constraint_exists = (
            query_info["price_min"] is not None
            or query_info["price_max"] is not None
        )

        price_valid = True

        if price_constraint_exists:

            if price is None:

                unknown_price_count += 1
                price_valid = False

            else:

                if (
                    query_info["price_min"]
                    is not None
                    and price
                    < query_info["price_min"]
                ):
                    price_valid = False

                if (
                    query_info["price_max"]
                    is not None
                    and price
                    > query_info["price_max"]
                ):
                    price_valid = False

            if not price_valid:

                rejected_price_count += 1
                continue

            price_match_count += 1

        # ----------------------------------------------------
        # Keyword score
        # ----------------------------------------------------

        keyword_match_score = keyword_score(
            product_text,
            query_info["keywords"],
        )

        # ----------------------------------------------------
        # Hybrid score
        # ----------------------------------------------------

        hybrid_score = (
            SEMANTIC_WEIGHT * similarity
            + TYPE_WEIGHT * type_score
            + SPEC_WEIGHT * spec_score
            + KEYWORD_WEIGHT * keyword_match_score
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        results.append(
            {
                "rank": len(results) + 1,
                "parent_asin": item["asin"],
                "title": title,
                "store": store,
                "price": price,
                "average_rating": metadata_row.get(
                    "average_rating",
                    None,
                ),
                "rating_number": metadata_row.get(
                    "rating_number",
                    None,
                ),
                "semantic_similarity": round(
                    similarity,
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
                    keyword_match_score,
                    4,
                ),
                "hybrid_score": round(
                    hybrid_score,
                    4,
                ),
                "detected_ram_gb": spec_info[
                    "detected_ram_gb"
                ],
                "detected_storage_gb": spec_info[
                    "detected_storage_gb"
                ],
                "detected_gpu": spec_info[
                    "detected_gpu"
                ],
                "price_constraint": (
                    "hard_filter"
                    if price_constraint_exists
                    else "no_constraint"
                ),
            }
        )

        # Stop once enough valid results exist.
        if len(results) >= top_k:
            break

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True,
    )

    # Re-number after sorting
    for i, result in enumerate(
        results[:top_k],
        start=1,
    ):
        result["rank"] = i

    results = results[:top_k]

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = {
        "candidate_count": candidate_count,
        "results_returned": len(results),
        "type_matches": type_match_count,
        "specification_matches": specification_match_count,
        "price_matches": price_match_count,
        "unknown_ram": unknown_ram_count,
        "unknown_storage": unknown_storage_count,
        "unknown_gpu": unknown_gpu_count,
        "unknown_prices": unknown_price_count,
        "rejected_type": rejected_type_count,
        "rejected_specification": rejected_spec_count,
        "rejected_price": rejected_price_count,
    }

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = {
        "query": query,
        "parsed_query": query_info,
        "search_method": "hybrid_v2.9",
        "model": MODEL_NAME,
        "device": DEVICE,
        "weights": {
            "semantic": SEMANTIC_WEIGHT,
            "type": TYPE_WEIGHT,
            "specification": SPEC_WEIGHT,
            "keyword": KEYWORD_WEIGHT,
        },
        "statistics": stats,
        "results": results,
    }

    SEARCH_DIR.mkdir(
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
            default=str,
        )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("SEARCH RESULTS")
    print("=" * 80)

    if not results:

        print()
        print(
            "No products satisfied all hard constraints."
        )

        print()
        print("Search statistics:")

        for key, value in stats.items():
            print(f"  {key}: {value}")

        print()
        print(
            f"Output saved to: {OUTPUT_PATH}"
        )

        return output

    for result in results:

        print()
        print(
            f"#{result['rank']} "
            f"{result['title']}"
        )

        print(
            f"    ASIN: "
            f"{result['parent_asin']}"
        )

        print(
            f"    Store: "
            f"{result['store']}"
        )

        print(
            f"    Price: "
            f"{result['price']}"
        )

        print(
            f"    Semantic: "
            f"{result['semantic_similarity']:.4f}"
        )

        print(
            f"    Hybrid: "
            f"{result['hybrid_score']:.4f}"
        )

        print(
            f"    RAM detected: "
            f"{result['detected_ram_gb']}"
        )

        print(
            f"    Storage detected: "
            f"{result['detected_storage_gb']}"
        )

        print(
            f"    GPU detected: "
            f"{result['detected_gpu']}"
        )

        print(
            f"    Price filter: "
            f"{result['price_constraint']}"
        )

    print()
    print("=" * 80)
    print("SEARCH STATISTICS")
    print("=" * 80)

    for key, value in stats.items():
        print(
            f"{key}: {value}"
        )

    print()
    print(
        f"Results saved to: {OUTPUT_PATH}"
    )

    return output


# ============================================================
# PARSER TESTS
# ============================================================

def run_parser_tests():
    """
    Test query and product specification parsers.
    """

    print()
    print("=" * 80)
    print("PARSER TESTS")
    print("=" * 80)

    # --------------------------------------------------------
    # RAM
    # --------------------------------------------------------

    ram_tests = [
        (
            "Lenovo Slim 7i 32GB RAM 1TB SSD",
            32.0,
        ),
        (
            "HP Chromebook 4GB LPDDR4 RAM, 32GB eMMC",
            4.0,
        ),
        (
            "Acer Chromebook 4GB LPDDR4 RAM, 128GB eMMC",
            4.0,
        ),
        (
            "ASUS laptop 4GB DDR4 RAM, 500GB HDD",
            4.0,
        ),
        (
            "Dell XPS 17 1TB SSD - 32GB RAM",
            32.0,
        ),
        (
            "HP Pavilion 32GB DDR4 RAM 1TB SSD",
            32.0,
        ),
        (
            "Compaq Presario 512MB RAM, 100GB Hard Drive",
            0.5,
        ),
        (
            "HP Pavilion 4GB/8GB/16GB/32GB RAM",
            None,
        ),
    ]

    print()
    print("RAM parser:")

    for text, expected in ram_tests:

        result = extract_product_ram(text)

        status = (
            "PASS"
            if result == expected
            else "FAIL"
        )

        print(
            f"[{status}] "
            f"{text}"
        )

        print(
            f"       Expected: {expected}"
        )

        print(
            f"       Got:      {result}"
        )

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    storage_tests = [
        (
            "1TB SSD",
            1024.0,
        ),
        (
            "512GB SSD",
            512.0,
        ),
        (
            "500GB HDD",
            500.0,
        ),
        (
            "32GB eMMC",
            32.0,
        ),
        (
            "4GB RAM, 500GB HDD",
            500.0,
        ),
        (
            "32GB RAM, 1TB SSD",
            1024.0,
        ),
    ]

    print()
    print("Storage parser:")

    for text, expected in storage_tests:

        result = extract_product_storage(text)

        status = (
            "PASS"
            if result == expected
            else "FAIL"
        )

        print(
            f"[{status}] "
            f"{text}"
        )

        print(
            f"       Expected: {expected}"
        )

        print(
            f"       Got:      {result}"
        )

    # --------------------------------------------------------
    # QUERY TESTS
    # --------------------------------------------------------

    query_tests = [
        "laptop with at least 32GB RAM",
        "gaming laptop with RTX 3070 and 16GB RAM under $1500",
        "RTX 3070 laptop under $1500",
        "laptop with 1TB SSD",
        "wireless earbuds for commuting",
    ]

    print()
    print("Query parser:")

    for query in query_tests:

        result = parse_query(query)

        print()
        print(
            f"Query: {query}"
        )

        print(
            f"  Product type: "
            f"{result['product_type']}"
        )

        print(
            f"  RAM: "
            f"{result['ram_gb']}"
        )

        print(
            f"  Storage: "
            f"{result['storage_gb']}"
        )

        print(
            f"  GPU: "
            f"{result['gpu']}"
        )

        print(
            f"  Price min: "
            f"{result['price_min']}"
        )

        print(
            f"  Price max: "
            f"{result['price_max']}"
        )

        print(
            f"  Keywords: "
            f"{result['keywords']}"
        )

    print()
    print("=" * 80)
    print("PARSER TESTS COMPLETE")
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "ShopGraph Hybrid Product Search v2.9"
        )
    )

    parser.add_argument(
        "--query",
        type=str,
        help="Natural-language product query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of results to return.",
    )

    parser.add_argument(
        "--test-parser",
        action="store_true",
        help="Run parser tests only.",
    )

    args = parser.parse_args()

    if args.test_parser:

        run_parser_tests()
        return

    if not args.query:

        print(
            "Please provide a query using:"
        )

        print()

        print(
            'python "search\\HybridSearch v2.1.py" '
            '--query "laptop with at least 32GB RAM"'
        )

        print()

        print(
            "Or run parser tests:"
        )

        print()

        print(
            'python "search\\HybridSearch v2.1.py" '
            "--test-parser"
        )

        return

    hybrid_search(
        query=args.query,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()