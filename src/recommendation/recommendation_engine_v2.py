"""
ShopGraph - Recommendation Engine V2

Content-based recommendation engine using:
    FAISS semantic retrieval
        ↓
    Product-type compatibility
        ↓
    Category compatibility
        ↓
    Semantic similarity
        ↓
    Brand diversity
        ↓
    Duplicate removal
        ↓
    Final recommendations

Input:
    data/processed/products_text_fixed.parquet
    data/processed/product_embeddings_fixed.npy
    data/processed/embedding_product_ids_fixed.parquet
    data/processed/search/faiss_products_fixed.index

Output:
    data/processed/search/last_recommendations_v2.json

Usage:
    python recommendation/recommendation_engine_v2.py --demo

    python recommendation/recommendation_engine_v2.py --asin B00MCW7G9M

    python recommendation/recommendation_engine_v2.py \
        --asin B00MCW7G9M \
        --top-k 10
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

METADATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "products_text_fixed.parquet"
)

EMBEDDINGS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "product_embeddings_fixed.npy"
)

ID_MAPPING_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "embedding_product_ids_fixed.parquet"
)

FAISS_INDEX_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "search"
    / "faiss_products_fixed.index"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "search"
    / "last_recommendations_v2.json"
)


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_ASIN = "B00MCW7G9M"
DEFAULT_TOP_K = 10

# Large candidate pool allows the reranker to find
# compatible products instead of simply taking the
# nearest semantic neighbors.
CANDIDATE_MULTIPLIER = 50
MIN_CANDIDATES = 500

# Scoring weights
SEMANTIC_WEIGHT = 0.60
TYPE_WEIGHT = 0.20
CATEGORY_WEIGHT = 0.10
QUALITY_WEIGHT = 0.10

# Maximum number of products from the same brand
MAX_SAME_BRAND = 2


# ============================================================
# PRODUCT TYPE RULES
# ============================================================

# Ordered intentionally.
# More specific types must be detected before broad types.

PRODUCT_TYPE_RULES: List[Tuple[str, List[str]]] = [
    (
        "gaming_laptop",
        [
            "gaming laptop",
            "gaming notebook",
            "gaming computer",
        ],
    ),
    (
        "laptop",
        [
            "laptop",
            "notebook computer",
            "notebook pc",
            "notebook",
            "chromebook",
            "macbook",
            "ultrabook",
        ],
    ),
    (
        "desktop",
        [
            "desktop computer",
            "desktop pc",
            "desktop",
            "tower pc",
            "all in one pc",
            "all-in-one pc",
        ],
    ),
    (
        "tablet",
        [
            "tablet",
            "ipad",
            "android tablet",
        ],
    ),
    (
        "smartphone",
        [
            "smartphone",
            "cell phone",
            "mobile phone",
            "android phone",
            "iphone",
        ],
    ),
    (
        "monitor",
        [
            "computer monitor",
            "gaming monitor",
            "lcd monitor",
            "led monitor",
            "display monitor",
            "monitor",
        ],
    ),
    (
        "television",
        [
            "smart tv",
            "television",
            "tv",
        ],
    ),
    (
        "headphones",
        [
            "headphones",
            "headphone",
            "over ear",
            "over-ear",
            "on ear",
            "on-ear",
        ],
    ),
    (
        "earbuds",
        [
            "earbuds",
            "earbud",
            "true wireless",
            "tws",
        ],
    ),
    (
        "speaker",
        [
            "bluetooth speaker",
            "portable speaker",
            "smart speaker",
            "speaker",
        ],
    ),
    (
        "microphone",
        [
            "microphone",
            "mic",
        ],
    ),
    (
        "webcam",
        [
            "webcam",
            "web camera",
            "pc camera",
        ],
    ),
    (
        "camera",
        [
            "digital camera",
            "mirrorless camera",
            "dslr camera",
            "action camera",
            "camera",
        ],
    ),
    (
        "drone",
        [
            "drone",
            "quadcopter",
            "multicopter",
            "uav",
        ],
    ),
    (
        "keyboard",
        [
            "keyboard",
            "keypad",
        ],
    ),
    (
        "mouse",
        [
            "computer mouse",
            "gaming mouse",
            "wireless mouse",
            "mouse",
        ],
    ),
    (
        "controller",
        [
            "game controller",
            "gaming controller",
            "controller",
            "gamepad",
            "joystick",
        ],
    ),
    (
        "graphics_card",
        [
            "graphics card",
            "graphic card",
            "video card",
            "gpu",
        ],
    ),
    (
        "processor",
        [
            "processor",
            "cpu",
            "core i3",
            "core i5",
            "core i7",
            "core i9",
            "ryzen",
        ],
    ),
    (
        "ram",
        [
            "ram memory",
            "desktop ram",
            "laptop ram",
            "memory module",
            "ddr4",
            "ddr5",
        ],
    ),
    (
        "ssd",
        [
            "solid state drive",
            "ssd",
            "nvme drive",
            "nvme ssd",
        ],
    ),
    (
        "hard_drive",
        [
            "hard drive",
            "hard disk drive",
            "hdd",
        ],
    ),
    (
        "usb_drive",
        [
            "usb flash drive",
            "flash drive",
            "thumb drive",
            "usb drive",
        ],
    ),
    (
        "charger",
        [
            "phone charger",
            "usb charger",
            "wall charger",
            "power adapter",
            "charger",
        ],
    ),
    (
        "power_bank",
        [
            "power bank",
            "powerbank",
            "portable charger",
        ],
    ),
    (
        "cable",
        [
            "usb cable",
            "hdmi cable",
            "displayport cable",
            "ethernet cable",
            "charging cable",
            "cable",
        ],
    ),
    (
        "case",
        [
            "phone case",
            "tablet case",
            "laptop case",
            "protective case",
            "cover case",
            "case",
        ],
    ),
    (
        "router",
        [
            "wireless router",
            "wifi router",
            "wi-fi router",
            "router",
        ],
    ),
    (
        "network_adapter",
        [
            "wifi adapter",
            "wi-fi adapter",
            "wireless adapter",
            "network adapter",
        ],
    ),
    (
        "printer",
        [
            "printer",
            "laser printer",
            "inkjet printer",
        ],
    ),
    (
        "scanner",
        [
            "scanner",
            "document scanner",
        ],
    ),
    (
        "projector",
        [
            "projector",
            "video projector",
        ],
    ),
]


# ============================================================
# RELATED PRODUCT TYPES
# ============================================================

# Some products are naturally related even when they are
# technically different product types.

RELATED_TYPES: Dict[str, set] = {
    "laptop": {
        "gaming_laptop",
        "laptop",
        "charger",
        "case",
        "ram",
        "ssd",
        "mouse",
        "keyboard",
        "monitor",
    },
    "gaming_laptop": {
        "gaming_laptop",
        "laptop",
        "mouse",
        "keyboard",
        "monitor",
        "ram",
        "ssd",
        "charger",
        "case",
    },
    "desktop": {
        "desktop",
        "gaming_laptop",
        "monitor",
        "keyboard",
        "mouse",
        "graphics_card",
        "processor",
        "ram",
        "ssd",
        "hard_drive",
    },
    "smartphone": {
        "smartphone",
        "case",
        "charger",
        "cable",
        "power_bank",
        "earbuds",
        "headphones",
    },
    "tablet": {
        "tablet",
        "case",
        "charger",
        "cable",
        "keyboard",
        "stylus",
    },
    "headphones": {
        "headphones",
        "earbuds",
        "microphone",
        "speaker",
        "cable",
    },
    "earbuds": {
        "earbuds",
        "headphones",
        "charger",
        "cable",
    },
    "camera": {
        "camera",
        "webcam",
        "drone",
        "microphone",
        "memory_card",
        "cable",
        "case",
    },
    "webcam": {
        "webcam",
        "camera",
        "microphone",
        "cable",
        "monitor",
    },
    "drone": {
        "drone",
        "camera",
        "battery",
        "charger",
        "cable",
        "controller",
    },
    "keyboard": {
        "keyboard",
        "mouse",
        "controller",
        "desktop",
        "laptop",
    },
    "mouse": {
        "mouse",
        "keyboard",
        "controller",
        "desktop",
        "laptop",
    },
    "monitor": {
        "monitor",
        "desktop",
        "laptop",
        "keyboard",
        "mouse",
        "cable",
    },
    "ssd": {
        "ssd",
        "hard_drive",
        "desktop",
        "laptop",
        "ram",
    },
    "hard_drive": {
        "hard_drive",
        "ssd",
        "desktop",
        "laptop",
    },
    "charger": {
        "charger",
        "power_bank",
        "cable",
        "smartphone",
        "tablet",
        "laptop",
    },
    "power_bank": {
        "power_bank",
        "charger",
        "cable",
        "smartphone",
        "tablet",
    },
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clean_text(value: Any) -> str:
    """Convert arbitrary metadata values to searchable text."""

    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    if isinstance(value, (list, tuple, set)):
        return " ".join(clean_text(item) for item in value)

    if isinstance(value, dict):
        return " ".join(
            f"{clean_text(k)} {clean_text(v)}"
            for k, v in value.items()
        )

    return str(value)


def normalize_text(text: str) -> str:
    """Normalize text for matching."""

    text = clean_text(text).lower()

    text = re.sub(r"[_/\\\-]+", " ", text)
    text = re.sub(r"[^a-z0-9.+ ]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def safe_float(value: Any) -> Optional[float]:
    """Safely convert a value to float."""

    try:
        if value is None:
            return None

        if isinstance(value, float) and np.isnan(value):
            return None

        return float(value)

    except (TypeError, ValueError):
        return None


def split_categories(value: Any) -> List[str]:
    """Extract category strings from Amazon metadata."""

    text = clean_text(value)

    if not text:
        return []

    # Common separators found in metadata.
    parts = re.split(r"\s*(?:>|>>|\||;)\s*", text)

    cleaned = []

    for part in parts:
        part = normalize_text(part)

        if part and part not in cleaned:
            cleaned.append(part)

    return cleaned


# ============================================================
# PRODUCT TYPE DETECTION
# ============================================================

def detect_product_type(row: pd.Series) -> str:
    """
    Infer product type using title + category + embedding text.

    The rules are intentionally deterministic so the same
    product receives the same type every time.
    """

    title = normalize_text(row.get("title", ""))
    categories = normalize_text(row.get("categories", ""))
    embedding_text = normalize_text(row.get("embedding_text", ""))

    # Title gets the highest importance.
    title_text = title

    # Categories are useful when title is ambiguous.
    category_text = categories

    # Full embedding text is only a fallback.
    combined_text = f"{title_text} {category_text} {embedding_text}"

    for product_type, keywords in PRODUCT_TYPE_RULES:

        for keyword in keywords:

            keyword_normalized = normalize_text(keyword)

            if keyword_normalized in title_text:
                return product_type

        for keyword in keywords:

            keyword_normalized = normalize_text(keyword)

            if keyword_normalized in category_text:
                return product_type

    # Last fallback.
    for product_type, keywords in PRODUCT_TYPE_RULES:

        for keyword in keywords:

            keyword_normalized = normalize_text(keyword)

            if keyword_normalized in combined_text:
                return product_type

    return "other"


# ============================================================
# BRAND EXTRACTION
# ============================================================

def get_brand(row: pd.Series) -> str:
    """
    Extract brand/store information.

    Amazon metadata can have Store, Brand, or similar fields.
    """

    for column in [
        "brand",
        "Brand",
        "store",
        "Store",
    ]:

        if column in row.index:

            value = clean_text(row[column]).strip()

            if value:
                return value

    return "Unknown"


# ============================================================
# CATEGORY COMPATIBILITY
# ============================================================

def category_tokens(row: pd.Series) -> set:
    """Return normalized category tokens."""

    raw = clean_text(row.get("categories", ""))

    if not raw:
        return set()

    tokens = set()

    for part in split_categories(raw):

        words = normalize_text(part).split()

        for word in words:

            if len(word) >= 3:
                tokens.add(word)

    return tokens


def category_similarity(
    source_row: pd.Series,
    candidate_row: pd.Series,
) -> float:
    """
    Compute category overlap.

    Returns:
        1.0  = strong overlap
        0.5  = partial/weak overlap
        0.0  = no useful overlap
    """

    source_tokens = category_tokens(source_row)
    candidate_tokens = category_tokens(candidate_row)

    if not source_tokens or not candidate_tokens:
        return 0.5

    intersection = source_tokens.intersection(candidate_tokens)

    if not intersection:
        return 0.0

    union = source_tokens.union(candidate_tokens)

    jaccard = len(intersection) / max(len(union), 1)

    if jaccard >= 0.30:
        return 1.0

    if jaccard >= 0.10:
        return 0.7

    return 0.4


# ============================================================
# PRODUCT TYPE COMPATIBILITY
# ============================================================

def type_compatibility(
    source_type: str,
    candidate_type: str,
) -> float:
    """
    Score compatibility between product types.

    1.0 = same product type
    0.6 = related product type
    0.0 = unrelated product type
    """

    if source_type == candidate_type:
        return 1.0

    related = RELATED_TYPES.get(source_type, set())

    if candidate_type in related:
        return 0.6

    return 0.0


# ============================================================
# QUALITY SCORE
# ============================================================

def quality_score(row: pd.Series) -> float:
    """
    Lightweight product quality score.

    Uses rating/review information when available.

    This is deliberately normalized and capped so that
    popularity/ratings do not dominate semantic similarity.
    """

    rating = None
    review_count = None

    for column in [
        "rating",
        "average_rating",
        "averageRating",
    ]:

        if column in row.index:

            rating = safe_float(row[column])

            if rating is not None:
                break

    for column in [
        "rating_number",
        "review_count",
        "rating_count",
        "num_reviews",
    ]:

        if column in row.index:

            review_count = safe_float(row[column])

            if review_count is not None:
                break

    rating_component = 0.5

    if rating is not None:

        if rating > 5:
            rating = 5.0

        rating_component = max(0.0, min(rating / 5.0, 1.0))

    review_component = 0.5

    if review_count is not None and review_count > 0:

        # Log scaling prevents huge review counts from
        # overwhelming everything else.
        review_component = min(
            np.log1p(review_count) / np.log1p(10000),
            1.0,
        )

    return (
        0.7 * rating_component
        + 0.3 * review_component
    )


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

class RecommendationEngineV2:

    def __init__(self) -> None:

        print("=" * 80)
        print("SHOPGRAPH - RECOMMENDATION ENGINE V2")
        print("=" * 80)

        self._load_data()

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    def _load_data(self) -> None:

        start = time.perf_counter()

        print("\n[1/4] Loading metadata...")

        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Metadata file not found:\n{METADATA_PATH}"
            )

        self.metadata = pd.read_parquet(
            METADATA_PATH,
            engine="fastparquet",
        )

        if "parent_asin" not in self.metadata.columns:
            raise ValueError(
                "Metadata does not contain 'parent_asin'."
            )

        self.metadata["parent_asin"] = (
            self.metadata["parent_asin"]
            .astype(str)
            .str.strip()
        )

        self.metadata_by_asin = (
            self.metadata
            .drop_duplicates("parent_asin")
            .set_index("parent_asin", drop=False)
        )

        print(
            f"    Products loaded: {len(self.metadata):,}"
        )

        print("\n[2/4] Loading embeddings...")

        if not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                f"Embeddings file not found:\n{EMBEDDINGS_PATH}"
            )

        self.embeddings = np.load(
            EMBEDDINGS_PATH,
            mmap_mode="r",
        )

        print(
            f"    Embedding shape: {self.embeddings.shape}"
        )

        if self.embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D matrix."
            )

        print("\n[3/4] Loading ID mapping...")

        if not ID_MAPPING_PATH.exists():
            raise FileNotFoundError(
                f"ID mapping not found:\n{ID_MAPPING_PATH}"
            )

        self.id_mapping = pd.read_parquet(
            ID_MAPPING_PATH,
            engine="fastparquet",
        )

        required_columns = {
            "embedding_index",
            "parent_asin",
        }

        missing = (
            required_columns
            - set(self.id_mapping.columns)
        )

        if missing:
            raise ValueError(
                f"ID mapping missing columns: {missing}"
            )

        self.id_mapping["parent_asin"] = (
            self.id_mapping["parent_asin"]
            .astype(str)
            .str.strip()
        )

        self.asin_to_embedding = dict(
            zip(
                self.id_mapping["parent_asin"],
                self.id_mapping["embedding_index"],
            )
        )

        print(
            f"    Mappings loaded: {len(self.id_mapping):,}"
        )

        print("\n[4/4] Loading FAISS index...")

        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found:\n{FAISS_INDEX_PATH}"
            )

        self.index = faiss.read_index(
            str(FAISS_INDEX_PATH)
        )

        print(
            f"    FAISS vectors: {self.index.ntotal:,}"
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if self.index.ntotal != len(self.embeddings):
            raise ValueError(
                "FAISS index and embeddings have different "
                "numbers of vectors."
            )

        if self.index.ntotal != len(self.id_mapping):
            raise ValueError(
                "FAISS index and ID mapping are misaligned."
            )

        print(
            f"\nData loaded successfully in "
            f"{time.perf_counter() - start:.2f}s"
        )

    # --------------------------------------------------------
    # SOURCE PRODUCT
    # --------------------------------------------------------

    def get_source_product(
        self,
        asin: str,
    ) -> pd.Series:

        asin = str(asin).strip()

        if asin not in self.metadata_by_asin.index:
            raise ValueError(
                f"ASIN not found in metadata: {asin}"
            )

        if asin not in self.asin_to_embedding:
            raise ValueError(
                f"ASIN not found in embedding mapping: {asin}"
            )

        return self.metadata_by_asin.loc[asin]

    # --------------------------------------------------------
    # FAISS RETRIEVAL
    # --------------------------------------------------------

    def retrieve_candidates(
        self,
        source_asin: str,
        top_k: int,
    ) -> List[Tuple[str, float, int]]:

        embedding_index = int(
            self.asin_to_embedding[source_asin]
        )

        query_vector = np.asarray(
            self.embeddings[embedding_index],
            dtype=np.float32,
        ).reshape(1, -1)

        # Number of candidates before reranking.
        candidate_count = max(
            top_k * CANDIDATE_MULTIPLIER,
            MIN_CANDIDATES,
        )

        candidate_count = min(
            candidate_count,
            self.index.ntotal,
        )

        search_start = time.perf_counter()

        similarities, indices = self.index.search(
            query_vector,
            candidate_count,
        )

        search_time = (
            time.perf_counter() - search_start
        ) * 1000

        print(
            f"\nFAISS candidate search: "
            f"{search_time:.3f} ms"
        )

        candidates = []

        for similarity, index_position in zip(
            similarities[0],
            indices[0],
        ):

            if index_position < 0:
                continue

            candidate_index = int(index_position)

            candidate_asin = str(
                self.id_mapping.iloc[
                    candidate_index
                ]["parent_asin"]
            )

            # Never recommend the source product.
            if candidate_asin == source_asin:
                continue

            candidates.append(
                (
                    candidate_asin,
                    float(similarity),
                    candidate_index,
                )
            )

        return candidates

    # --------------------------------------------------------
    # RERANK
    # --------------------------------------------------------

    def rerank(
        self,
        source_row: pd.Series,
        candidates: List[Tuple[str, float, int]],
        top_k: int,
    ) -> List[Dict[str, Any]]:

        source_type = detect_product_type(
            source_row
        )

        source_brand = get_brand(source_row)

        print(
            f"\nSource product type: {source_type}"
        )

        print(
            f"Source brand: {source_brand}"
        )

        scored_candidates = []

        seen_asins = set()

        for candidate_asin, semantic_similarity, embedding_index in candidates:

            if candidate_asin in seen_asins:
                continue

            seen_asins.add(candidate_asin)

            if candidate_asin not in self.metadata_by_asin.index:
                continue

            candidate_row = (
                self.metadata_by_asin
                .loc[candidate_asin]
            )

            candidate_type = detect_product_type(
                candidate_row
            )

            candidate_brand = get_brand(
                candidate_row
            )

            type_score = type_compatibility(
                source_type,
                candidate_type,
            )

            category_score = category_similarity(
                source_row,
                candidate_row,
            )

            quality = quality_score(
                candidate_row
            )

            # FAISS uses cosine similarity because
            # embeddings were normalized.
            semantic_score = max(
                0.0,
                min(
                    float(semantic_similarity),
                    1.0,
                ),
            )

            final_score = (
                SEMANTIC_WEIGHT
                * semantic_score
                +
                TYPE_WEIGHT
                * type_score
                +
                CATEGORY_WEIGHT
                * category_score
                +
                QUALITY_WEIGHT
                * quality
            )

            scored_candidates.append(
                {
                    "asin": candidate_asin,
                    "embedding_index": int(
                        embedding_index
                    ),
                    "semantic_score": semantic_score,
                    "type_score": type_score,
                    "category_score": category_score,
                    "quality_score": quality,
                    "final_score": final_score,
                    "product_type": candidate_type,
                    "brand": candidate_brand,
                    "row": candidate_row,
                }
            )

        # Highest scoring products first.
        scored_candidates.sort(
            key=lambda item: item["final_score"],
            reverse=True,
        )

        # ----------------------------------------------------
        # BRAND DIVERSITY
        # ----------------------------------------------------

        final_candidates = []

        brand_counts: Dict[str, int] = {}

        # First pass:
        # enforce same-brand limit.
        for item in scored_candidates:

            brand = (
                item["brand"]
                .strip()
                .lower()
            )

            count = brand_counts.get(
                brand,
                0,
            )

            if count >= MAX_SAME_BRAND:
                continue

            brand_counts[brand] = count + 1

            final_candidates.append(item)

            if len(final_candidates) >= top_k:
                break

        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        # If the dataset does not contain enough distinct
        # brands, fill remaining positions without the
        # brand restriction.
        if len(final_candidates) < top_k:

            selected_asins = {
                item["asin"]
                for item in final_candidates
            }

            for item in scored_candidates:

                if item["asin"] in selected_asins:
                    continue

                final_candidates.append(item)

                if len(final_candidates) >= top_k:
                    break

        return final_candidates

    # --------------------------------------------------------
    # FORMAT RESULT
    # --------------------------------------------------------

    def format_product(
        self,
        item: Dict[str, Any],
        rank: int,
    ) -> Dict[str, Any]:

        row = item["row"]

        title = clean_text(
            row.get("title", "")
        ).strip()

        if not title:
            title = "Untitled Product"

        store = get_brand(row)

        categories = clean_text(
            row.get("categories", "")
        )

        price = row.get(
            "price",
            None,
        )

        rating = row.get(
            "rating",
            row.get(
                "average_rating",
                None,
            ),
        )

        result = {
            "rank": rank,
            "asin": item["asin"],
            "title": title,
            "store": store,
            "brand": store,
            "price": (
                clean_text(price)
                if price is not None
                else "N/A"
            ),
            "rating": (
                safe_float(rating)
                if rating is not None
                else None
            ),
            "categories": categories,
            "product_type": item["product_type"],
            "embedding_index": item[
                "embedding_index"
            ],
            "semantic_similarity": round(
                item["semantic_score"],
                6,
            ),
            "type_compatibility": round(
                item["type_score"],
                6,
            ),
            "category_compatibility": round(
                item["category_score"],
                6,
            ),
            "quality_score": round(
                item["quality_score"],
                6,
            ),
            "recommendation_score": round(
                item["final_score"],
                6,
            ),
        }

        return result

    # --------------------------------------------------------
    # MAIN RECOMMENDATION METHOD
    # --------------------------------------------------------

    def recommend(
        self,
        asin: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, Any]:

        total_start = time.perf_counter()

        asin = str(asin).strip()

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        print("\n" + "=" * 80)
        print("GENERATING RECOMMENDATIONS")
        print("=" * 80)

        source_row = self.get_source_product(
            asin
        )

        source_type = detect_product_type(
            source_row
        )

        source_brand = get_brand(
            source_row
        )

        print("\nSOURCE PRODUCT")
        print("-" * 80)

        print(f"ASIN       : {asin}")

        print(
            f"Title      : "
            f"{clean_text(source_row.get('title', ''))}"
        )

        print(
            f"Brand      : {source_brand}"
        )

        print(
            f"Type       : {source_type}"
        )

        candidates = self.retrieve_candidates(
            asin,
            top_k,
        )

        print(
            f"Candidates retrieved: "
            f"{len(candidates):,}"
        )

        reranked = self.rerank(
            source_row,
            candidates,
            top_k,
        )

        recommendations = []

        for rank, item in enumerate(
            reranked,
            start=1,
        ):

            recommendations.append(
                self.format_product(
                    item,
                    rank,
                )
            )

        total_time = (
            time.perf_counter()
            - total_start
        )

        result = {
            "engine": "ShopGraph Recommendation Engine V2",
            "version": "2.0",
            "source": {
                "asin": asin,
                "title": clean_text(
                    source_row.get(
                        "title",
                        "",
                    )
                ),
                "brand": source_brand,
                "product_type": source_type,
            },
            "configuration": {
                "top_k": top_k,
                "candidate_multiplier": CANDIDATE_MULTIPLIER,
                "min_candidates": MIN_CANDIDATES,
                "semantic_weight": SEMANTIC_WEIGHT,
                "type_weight": TYPE_WEIGHT,
                "category_weight": CATEGORY_WEIGHT,
                "quality_weight": QUALITY_WEIGHT,
                "max_same_brand": MAX_SAME_BRAND,
            },
            "statistics": {
                "candidates_retrieved": len(
                    candidates
                ),
                "recommendations_returned": len(
                    recommendations
                ),
                "total_time_ms": round(
                    total_time * 1000,
                    3,
                ),
            },
            "recommendations": recommendations,
        }

        return result

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    def save_result(
        self,
        result: Dict[str, Any],
    ) -> None:

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            OUTPUT_PATH,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                result,
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        print(
            f"\nSaved recommendation result:"
        )

        print(
            f"    {OUTPUT_PATH}"
        )


# ============================================================
# DISPLAY
# ============================================================

def print_results(
    result: Dict[str, Any],
) -> None:

    print("\n" + "=" * 80)
    print("RECOMMENDATION RESULTS")
    print("=" * 80)

    source = result["source"]

    print(
        f"\nSource: {source['title']}"
    )

    print(
        f"ASIN: {source['asin']}"
    )

    print(
        f"Type: {source['product_type']}"
    )

    print("\nAI RECOMMENDATIONS")
    print("-" * 80)

    for item in result["recommendations"]:

        print(
            f"\n#{item['rank']} "
            f"{item['title']}"
        )

        print(
            f"   ASIN       : "
            f"{item['asin']}"
        )

        print(
            f"   Brand      : "
            f"{item['brand']}"
        )

        print(
            f"   Type       : "
            f"{item['product_type']}"
        )

        print(
            f"   Price      : "
            f"{item['price']}"
        )

        print(
            f"   Rating     : "
            f"{item['rating']}"
        )

        print(
            f"   Semantic   : "
            f"{item['semantic_similarity']:.4f}"
        )

        print(
            f"   Type Match : "
            f"{item['type_compatibility']:.2f}"
        )

        print(
            f"   Category   : "
            f"{item['category_compatibility']:.2f}"
        )

        print(
            f"   Final Score: "
            f"{item['recommendation_score']:.4f}"
        )

    print("\n" + "-" * 80)

    stats = result["statistics"]

    print(
        f"Candidates retrieved : "
        f"{stats['candidates_retrieved']:,}"
    )

    print(
        f"Recommendations       : "
        f"{stats['recommendations_returned']}"
    )

    print(
        f"Total time            : "
        f"{stats['total_time_ms']:.3f} ms"
    )

    print("=" * 80)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "ShopGraph Recommendation Engine V2"
        )
    )

    parser.add_argument(
        "--asin",
        type=str,
        default=DEFAULT_ASIN,
        help=(
            "Source product ASIN "
            f"(default: {DEFAULT_ASIN})"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "Number of recommendations "
            f"(default: {DEFAULT_TOP_K})"
        ),
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Run the default ShopGraph demo."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    asin = (
        DEFAULT_ASIN
        if args.demo
        else args.asin
    )

    engine = RecommendationEngineV2()

    result = engine.recommend(
        asin=asin,
        top_k=args.top_k,
    )

    print_results(result)

    engine.save_result(result)

    print(
        "\nRecommendation Engine V2 completed successfully."
    )


if __name__ == "__main__":
    main()