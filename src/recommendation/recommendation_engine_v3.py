"""
ShopGraph - Recommendation Engine V3.1

Personalized E-Commerce Recommendation Engine

Combines:
    - MiniLM embeddings
    - FAISS vector search
    - User preference profiles
    - User interaction history
    - Search interests
    - Preference transfer
    - Product compatibility
    - Negative filtering
    - Explainable scoring

V3.1 fixes the V3 preference-transfer problem:
    A user's preference score is NOT copied equally to every
    candidate. Preference is transferred according to the
    candidate's actual similarity to the preferred product.

Existing V1/V2 files are NOT modified.

Output:
    data/processed/search/last_recommendations_v3.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


# ============================================================
# PROJECT PATHS
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
        if (directory / "products_text_fixed.parquet").exists()
    ),
    DATA_DIR_CANDIDATES[0],
)

PRODUCTS_PATH = (
    DATA_DIR
    / "products_text_fixed.parquet"
)

EMBEDDINGS_PATH = (
    DATA_DIR
    / "product_embeddings_fixed.npy"
)

MAPPING_PATH = (
    DATA_DIR
    / "embedding_product_ids_fixed.parquet"
)

FAISS_PATH = (
    DATA_DIR
    / "search"
    / "faiss_products_fixed.index"
)

USER_PROFILES_PATH = (
    DATA_DIR
    / "user_profiles.parquet"
)

INTERACTIONS_PATH = (
    DATA_DIR
    / "user_interactions.parquet"
)

OUTPUT_PATH = (
    DATA_DIR
    / "search"
    / "last_recommendations_v3.json"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# V3.1 CONFIGURATION
# ============================================================

DEFAULT_TOP_K = 10

# Retrieve more candidates than required so that compatibility
# filtering and personalization can happen before final ranking.
CANDIDATE_MULTIPLIER = 50

MIN_CANDIDATES = 1000

# Final scoring weights.
#
# Semantic:
# General embedding similarity.
#
# Preference:
# Similarity-weighted transfer of the user's preferences.
#
# Search:
# Similarity to the user's search interests.
#
# Compatibility:
# Whether the candidate belongs to a compatible product family.
#
# Interaction:
# Existing behavioral signal if the candidate has interaction
# evidence that survives filtering.
SEMANTIC_WEIGHT = 0.35
PREFERENCE_WEIGHT = 0.25
SEARCH_WEIGHT = 0.20
COMPATIBILITY_WEIGHT = 0.15
INTERACTION_WEIGHT = 0.05


# ============================================================
# PRODUCT FAMILY DEFINITIONS
# ============================================================

# These are intentionally conservative.
#
# The purpose is NOT to build a perfect product taxonomy.
# The purpose is to prevent obviously unrelated products from
# dominating recommendations simply because they are
# semantically close in embedding space.

PRODUCT_FAMILIES = {

    "headphones": {
        "keywords": [
            "headphone",
            "headphones",
            "headset",
            "earbud",
            "earbuds",
            "earphone",
            "earphones",
            "wireless earbuds",
            "wireless headphones",
            "bluetooth headphones",
            "bluetooth earbuds",
            "noise cancelling",
            "noise canceling",
            "noise-cancelling",
            "in-ear",
            "over-ear",
            "on-ear",
        ],
    },

    "speakers": {
        "keywords": [
            "speaker",
            "speakers",
            "soundbar",
            "sound bar",
            "bluetooth speaker",
            "wireless speaker",
            "subwoofer",
        ],
    },

    "camera": {
        "keywords": [
            "camera",
            "webcam",
            "camcorder",
            "dash cam",
            "dashcam",
            "security camera",
            "backup camera",
            "action camera",
            "digital camera",
        ],
    },

    "display": {
        "keywords": [
            "monitor",
            "display",
            "screen",
            "television",
            "smart tv",
            "tv",
            "projector",
        ],
    },

    "computer": {
        "keywords": [
            "laptop",
            "notebook",
            "desktop computer",
            "desktop pc",
            "computer",
            "chromebook",
            "workstation",
        ],
    },

    "phone": {
        "keywords": [
            "smartphone",
            "smart phone",
            "cell phone",
            "mobile phone",
            "android phone",
            "iphone",
        ],
    },

    "tablet": {
        "keywords": [
            "tablet",
            "ipad",
        ],
    },

    "gaming": {
        "keywords": [
            "gaming",
            "game controller",
            "controller",
            "gaming mouse",
            "gaming keyboard",
            "gaming headset",
            "playstation",
            "xbox",
            "nintendo",
        ],
    },

    "networking": {
        "keywords": [
            "router",
            "wifi router",
            "wi-fi router",
            "network switch",
            "ethernet switch",
            "access point",
            "wireless access point",
            "modem",
        ],
    },

    "storage": {
        "keywords": [
            "ssd",
            "hard drive",
            "hard disk",
            "hdd",
            "flash drive",
            "usb drive",
            "memory card",
            "sd card",
        ],
    },

    "cables": {
        "keywords": [
            "hdmi cable",
            "usb cable",
            "ethernet cable",
            "audio cable",
            "charging cable",
            "displayport cable",
            "adapter cable",
        ],
    },

    "drone": {
        "keywords": [
            "drone",
            "quadcopter",
            "multicopter",
            "fpv drone",
            "racing drone",
        ],
    },

    "vr": {
        "keywords": [
            "vr headset",
            "virtual reality headset",
            "vr goggles",
            "virtual reality",
        ],
    },
}


# Products that can reasonably be considered alternatives or
# adjacent products for a family.
#
# This prevents a headphones recommendation from turning into
# a dash-camera recommendation just because both contain words
# such as "wireless", "camera", "video", "device", etc.
RELATED_FAMILIES = {

    "headphones": {
        "headphones",
        "speakers",
        "gaming",
        "vr",
    },

    "speakers": {
        "speakers",
        "headphones",
        "gaming",
    },

    "camera": {
        "camera",
        "drone",
    },

    "display": {
        "display",
        "vr",
    },

    "computer": {
        "computer",
        "gaming",
        "storage",
        "networking",
    },

    "phone": {
        "phone",
        "tablet",
        "cables",
    },

    "tablet": {
        "tablet",
        "phone",
        "cables",
    },

    "gaming": {
        "gaming",
        "headphones",
        "computer",
    },

    "networking": {
        "networking",
        "computer",
        "cables",
    },

    "storage": {
        "storage",
        "computer",
    },

    "cables": {
        "cables",
        "computer",
        "phone",
        "tablet",
        "networking",
    },

    "drone": {
        "drone",
        "camera",
    },

    "vr": {
        "vr",
        "headphones",
        "display",
        "gaming",
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value) -> str:
    """Safely convert arbitrary metadata into text."""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    if isinstance(value, list):
        return " ".join(
            clean_text(item)
            for item in value
        )

    if isinstance(value, tuple):
        return " ".join(
            clean_text(item)
            for item in value
        )

    return str(value)


def safe_float(value, default=0.0) -> float:
    """Safely convert a value to float."""

    try:
        number = float(value)

        if not math.isfinite(number):
            return default

        return number

    except Exception:
        return default


def l2_normalize(
    vectors: np.ndarray
) -> np.ndarray:
    """L2-normalize vectors."""

    vectors = np.asarray(
        vectors,
        dtype=np.float32
    )

    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1.0

    return vectors / norms


def tokenize(text: str) -> Set[str]:
    """Basic word tokenizer."""

    text = clean_text(text).lower()

    return set(
        re.findall(
            r"[a-z0-9]+",
            text
        )
    )


def contains_keyword(
    text: str,
    keyword: str
) -> bool:
    """
    Match keywords safely.

    Important:
    Short terms such as 'tv' must not match arbitrary strings
    like 'tview'.
    """

    text = clean_text(text).lower()
    keyword = clean_text(keyword).lower().strip()

    if not keyword:
        return False

    # Multi-word phrases can be matched normally.
    if " " in keyword:
        return keyword in text

    # Single-word keywords use boundaries.
    return re.search(
        rf"\b{re.escape(keyword)}\b",
        text
    ) is not None


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

class RecommendationEngineV3:

    def __init__(self):

        print("=" * 80)
        print("SHOPGRAPH - RECOMMENDATION ENGINE V3.1")
        print("=" * 80)

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Device: {self.device}"
        )

        self._validate_files()
        self._load_products()
        self._load_embeddings()
        self._load_mapping()
        self._load_faiss()
        self._load_user_data()
        self._load_embedding_model()

        print("\nEngine initialization complete.")
        print("=" * 80)

    # ========================================================
    # FILE VALIDATION
    # ========================================================

    def _validate_files(self):

        print("\nChecking required files...")

        required_files = {
            "Products": PRODUCTS_PATH,
            "Embeddings": EMBEDDINGS_PATH,
            "Embedding mapping": MAPPING_PATH,
            "FAISS index": FAISS_PATH,
            "User profiles": USER_PROFILES_PATH,
            "User interactions": INTERACTIONS_PATH,
        }

        missing = []

        for name, path in required_files.items():

            if path.exists():

                print(
                    f"[OK] {name}: {path}"
                )

            else:

                print(
                    f"[MISSING] {name}: {path}"
                )

                missing.append(name)

        if missing:

            raise FileNotFoundError(
                "Missing required files: "
                + ", ".join(missing)
            )

    # ========================================================
    # LOAD PRODUCTS
    # ========================================================

    def _load_products(self):

        start = time.perf_counter()

        print(
            "\nLoading product metadata..."
        )

        self.products = pd.read_parquet(
            PRODUCTS_PATH,
            engine="fastparquet"
        )

        if "parent_asin" not in self.products.columns:

            raise ValueError(
                "products_text_fixed.parquet must contain "
                "'parent_asin'."
            )

        self.products["parent_asin"] = (
            self.products["parent_asin"]
            .astype(str)
            .str.strip()
        )

        self.product_lookup = {}

        for _, row in self.products.iterrows():

            asin = str(
                row["parent_asin"]
            ).strip()

            if not asin:
                continue

            if asin not in self.product_lookup:

                self.product_lookup[asin] = (
                    row.to_dict()
                )

        print(
            f"Products loaded: "
            f"{len(self.products):,}"
        )

        print(
            "Metadata load time: "
            f"{time.perf_counter() - start:.3f}s"
        )

    # ========================================================
    # LOAD EMBEDDINGS
    # ========================================================

    def _load_embeddings(self):

        print(
            "\nLoading product embeddings..."
        )

        self.embeddings = np.load(
            EMBEDDINGS_PATH,
            mmap_mode="r"
        )

        if self.embeddings.ndim != 2:

            raise ValueError(
                "Embeddings must be a 2D matrix."
            )

        self.embedding_count = (
            self.embeddings.shape[0]
        )

        self.embedding_dimension = (
            self.embeddings.shape[1]
        )

        print(
            f"Embeddings: "
            f"{self.embeddings.shape}"
        )

    # ========================================================
    # LOAD MAPPING
    # ========================================================

    def _load_mapping(self):

        print(
            "\nLoading embedding → ASIN mapping..."
        )

        mapping = pd.read_parquet(
            MAPPING_PATH,
            engine="fastparquet"
        )

        required_columns = {
            "embedding_index",
            "parent_asin",
        }

        missing = (
            required_columns
            - set(mapping.columns)
        )

        if missing:

            raise ValueError(
                f"Mapping missing columns: {missing}"
            )

        mapping["embedding_index"] = pd.to_numeric(
            mapping["embedding_index"],
            errors="coerce"
        )

        mapping["parent_asin"] = (
            mapping["parent_asin"]
            .astype(str)
            .str.strip()
        )

        mapping = mapping.dropna(
            subset=["embedding_index"]
        )

        mapping["embedding_index"] = (
            mapping["embedding_index"]
            .astype(int)
        )

        self.mapping = mapping

        self.index_to_asin = dict(
            zip(
                mapping["embedding_index"],
                mapping["parent_asin"]
            )
        )

        self.asin_to_index = dict(
            zip(
                mapping["parent_asin"],
                mapping["embedding_index"]
            )
        )

        print(
            f"Mapping entries: "
            f"{len(mapping):,}"
        )

    # ========================================================
    # LOAD FAISS
    # ========================================================

    def _load_faiss(self):

        print(
            "\nLoading FAISS index..."
        )

        self.faiss_index = faiss.read_index(
            str(FAISS_PATH)
        )

        if (
            self.faiss_index.ntotal
            != self.embedding_count
        ):

            raise ValueError(
                "FAISS vector count does not match "
                "embedding count."
            )

        print(
            f"FAISS vectors: "
            f"{self.faiss_index.ntotal:,}"
        )

    # ========================================================
    # LOAD USERS
    # ========================================================

    def _load_user_data(self):

        print(
            "\nLoading user profiles..."
        )

        self.user_profiles = pd.read_parquet(
            USER_PROFILES_PATH,
            engine="fastparquet"
        )

        self.interactions = pd.read_parquet(
            INTERACTIONS_PATH,
            engine="fastparquet"
        )

        if "user_id" not in self.user_profiles.columns:

            raise ValueError(
                "user_profiles.parquet must contain user_id."
            )

        if "user_id" not in self.interactions.columns:

            raise ValueError(
                "user_interactions.parquet must contain user_id."
            )

        self.user_profiles["user_id"] = (
            self.user_profiles["user_id"]
            .astype(str)
        )

        self.interactions["user_id"] = (
            self.interactions["user_id"]
            .astype(str)
        )

        print(
            f"User profiles: "
            f"{len(self.user_profiles):,}"
        )

        print(
            f"Interactions: "
            f"{len(self.interactions):,}"
        )

    # ========================================================
    # LOAD MINILM
    # ========================================================

    def _load_embedding_model(self):

        print(
            "\nLoading MiniLM model..."
        )

        start = time.perf_counter()

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                MODEL_NAME
            )
        )

        self.embedding_model = (
            AutoModel.from_pretrained(
                MODEL_NAME
            )
        )

        self.embedding_model.to(
            self.device
        )

        self.embedding_model.eval()

        print(
            "MiniLM loaded in "
            f"{time.perf_counter() - start:.3f}s"
        )

    # ========================================================
    # MINILM ENCODING
    # ========================================================

    @torch.no_grad()
    def encode_text(
        self,
        texts: List[str]
    ) -> np.ndarray:
        """
        Encode text using MiniLM mean pooling.
        """

        if not texts:

            return np.empty(
                (
                    0,
                    self.embedding_dimension
                ),
                dtype=np.float32
            )

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        outputs = self.embedding_model(
            **encoded
        )

        token_embeddings = (
            outputs.last_hidden_state
        )

        attention_mask = (
            encoded["attention_mask"]
        )

        mask = (
            attention_mask
            .unsqueeze(-1)
            .expand(
                token_embeddings.size()
            )
            .float()
        )

        summed = torch.sum(
            token_embeddings * mask,
            dim=1
        )

        counts = torch.clamp(
            mask.sum(dim=1),
            min=1e-9
        )

        embeddings = summed / counts

        embeddings = (
            torch.nn.functional.normalize(
                embeddings,
                p=2,
                dim=1
            )
        )

        return (
            embeddings
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    # ========================================================
    # SAFE LIST
    # ========================================================

    def _safe_list(
        self,
        value
    ) -> List:

        if value is None:
            return []

        if isinstance(value, float):

            if np.isnan(value):
                return []

        if isinstance(
            value,
            np.ndarray
        ):

            return value.tolist()

        if isinstance(
            value,
            (list, tuple, set)
        ):

            return list(value)

        if isinstance(value, str):

            value = value.strip()

            if not value:
                return []

            try:

                parsed = json.loads(
                    value
                )

                if isinstance(
                    parsed,
                    list
                ):

                    return parsed

            except Exception:
                pass

            return [value]

        return [value]

    # ========================================================
    # USER PROFILE
    # ========================================================

    def get_user_profile(
        self,
        user_id: str
    ) -> Dict:

        user_id = str(user_id)

        rows = self.user_profiles[
            self.user_profiles["user_id"]
            == user_id
        ]

        if rows.empty:

            return {
                "user_id": user_id,
                "total_interactions": 0,
                "positive_products": [],
                "negative_products": [],
                "search_interests": [],
                "product_preferences": [],
            }

        row = rows.iloc[0]

        return {
            column: row[column]
            for column in self.user_profiles.columns
        }

    # ========================================================
    # USER INTERACTIONS
    # ========================================================

    def get_user_interactions(
        self,
        user_id: str
    ) -> pd.DataFrame:

        return self.interactions[
            self.interactions["user_id"]
            == str(user_id)
        ].copy()

    # ========================================================
    # POSITIVE PRODUCTS
    # ========================================================

    def get_positive_products(
        self,
        profile: Dict
    ) -> Set[str]:

        values = self._safe_list(
            profile.get(
                "positive_products",
                []
            )
        )

        return {
            str(value).strip()
            for value in values
            if str(value).strip()
        }

    # ========================================================
    # NEGATIVE PRODUCTS
    # ========================================================

    def get_negative_products(
        self,
        profile: Dict
    ) -> Set[str]:

        values = self._safe_list(
            profile.get(
                "negative_products",
                []
            )
        )

        return {
            str(value).strip()
            for value in values
            if str(value).strip()
        }

    # ========================================================
    # SEARCH INTERESTS
    # ========================================================

    def get_search_interests(
        self,
        profile: Dict
    ) -> List[str]:

        values = self._safe_list(
            profile.get(
                "search_interests",
                []
            )
        )

        return [
            str(value).strip()
            for value in values
            if str(value).strip()
        ]

    # ========================================================
    # PREFERENCE MAP
    # ========================================================

    def get_preference_map(
        self,
        profile: Dict
    ) -> Dict[str, float]:

        preferences = {}

        values = self._safe_list(
            profile.get(
                "product_preferences",
                []
            )
        )

        for item in values:

            if not isinstance(
                item,
                dict
            ):
                continue

            asin = str(
                item.get(
                    "product_asin",
                    ""
                )
            ).strip()

            if not asin:
                continue

            score = safe_float(
                item.get(
                    "preference_score",
                    0.0
                )
            )

            preferences[asin] = score

        return preferences

    # ========================================================
    # PRODUCT TEXT
    # ========================================================

    def get_product_text(
        self,
        asin: str
    ) -> str:

        metadata = self.product_lookup.get(
            asin,
            {}
        )

        title = clean_text(
            metadata.get(
                "title",
                ""
            )
        )

        store = clean_text(
            metadata.get(
                "store",
                ""
            )
        )

        categories = clean_text(
            metadata.get(
                "categories",
                ""
            )
        )

        features = clean_text(
            metadata.get(
                "features",
                ""
            )
        )

        description = clean_text(
            metadata.get(
                "description",
                ""
            )
        )

        return " ".join(
            part
            for part in [
                title,
                store,
                categories,
                features,
                description,
            ]
            if part
        )

    # ========================================================
    # PRODUCT FAMILY
    # ========================================================

    def classify_product(
        self,
        asin: str
    ) -> Set[str]:

        metadata = self.product_lookup.get(
            asin,
            {}
        )

        # Title is intentionally weighted more strongly because
        # it normally describes the actual product.
        title = clean_text(
            metadata.get(
                "title",
                ""
            )
        ).lower()

        categories = clean_text(
            metadata.get(
                "categories",
                ""
            )
        ).lower()

        features = clean_text(
            metadata.get(
                "features",
                ""
            )
        ).lower()

        combined = (
            title
            + " "
            + categories
            + " "
            + features
        )

        families = set()

        # ----------------------------------------------------
        # Strong title matching
        # ----------------------------------------------------

        for family, data in PRODUCT_FAMILIES.items():

            keywords = data["keywords"]

            for keyword in keywords:

                if contains_keyword(
                    title,
                    keyword
                ):

                    families.add(
                        family
                    )

                    break

        # ----------------------------------------------------
        # Category / feature matching
        # ----------------------------------------------------

        if not families:

            for family, data in PRODUCT_FAMILIES.items():

                for keyword in data["keywords"]:

                    if contains_keyword(
                        categories,
                        keyword
                    ) or contains_keyword(
                        features,
                        keyword
                    ):

                        families.add(
                            family
                        )

                        break

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        if not families:

            # Avoid assigning a misleading family.
            families.add(
                "other"
            )

        return families

    # ========================================================
    # QUERY FAMILY
    # ========================================================

    def classify_query(
        self,
        query: str
    ) -> Set[str]:

        query = clean_text(
            query
        ).lower()

        families = set()

        for family, data in PRODUCT_FAMILIES.items():

            for keyword in data["keywords"]:

                if contains_keyword(
                    query,
                    keyword
                ):

                    families.add(
                        family
                    )

                    break

        if not families:

            families.add(
                "other"
            )

        return families

    # ========================================================
    # COMPATIBILITY
    # ========================================================

    def compatibility_score(
        self,
        source_families: Set[str],
        candidate_families: Set[str]
    ) -> float:
        """
        Estimate product compatibility.

        1.0 = same family
        0.75 = related family
        0.35 = unknown/other
        0.10 = clearly unrelated
        """

        if not source_families:

            return 0.35

        if not candidate_families:

            return 0.35

        if "other" in source_families:

            return 0.50

        if "other" in candidate_families:

            return 0.35

        if source_families & candidate_families:

            return 1.0

        for source_family in source_families:

            related = RELATED_FAMILIES.get(
                source_family,
                set()
            )

            if related & candidate_families:

                return 0.75

        return 0.10

    # ========================================================
    # FAISS SEARCH
    # ========================================================

    def search_faiss(
        self,
        query_vector: np.ndarray,
        top_n: int
    ) -> List[Tuple[int, float]]:

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32
        )

        query_vector = l2_normalize(
            query_vector
        )

        k = min(
            int(top_n),
            int(self.faiss_index.ntotal)
        )

        scores, indices = (
            self.faiss_index.search(
                query_vector,
                k
            )
        )

        results = []

        for index, score in zip(
            indices[0],
            scores[0]
        ):

            if index < 0:
                continue

            results.append(
                (
                    int(index),
                    float(score)
                )
            )

        return results

    # ========================================================
    # PRODUCT CANDIDATES
    # ========================================================

    def generate_product_candidates(
        self,
        positive_products: Set[str],
        preference_map: Dict[str, float],
        top_n: int
    ) -> Dict[str, Dict]:

        candidates = {}

        for source_asin in positive_products:

            source_index = (
                self.asin_to_index.get(
                    source_asin
                )
            )

            if source_index is None:
                continue

            source_vector = np.asarray(
                self.embeddings[
                    source_index
                ],
                dtype=np.float32
            ).reshape(
                1,
                -1
            )

            source_preference = (
                preference_map.get(
                    source_asin,
                    0.0
                )
            )

            results = self.search_faiss(
                source_vector,
                top_n
            )

            for candidate_index, similarity in results:

                candidate_asin = (
                    self.index_to_asin.get(
                        candidate_index
                    )
                )

                if not candidate_asin:
                    continue

                if (
                    candidate_asin
                    == source_asin
                ):
                    continue

                if candidate_asin not in candidates:

                    candidates[candidate_asin] = {
                        "product_matches": [],
                        "search_matches": [],
                    }

                candidates[
                    candidate_asin
                ]["product_matches"].append(
                    {
                        "source_asin": source_asin,
                        "similarity": float(
                            similarity
                        ),
                        "preference": float(
                            source_preference
                        ),
                    }
                )

        return candidates

    # ========================================================
    # SEARCH CANDIDATES
    # ========================================================

    def generate_search_candidates(
        self,
        search_interests: List[str],
        top_n: int
    ) -> Dict[str, Dict]:

        candidates = {}

        if not search_interests:
            return candidates

        query_embeddings = (
            self.encode_text(
                search_interests
            )
        )

        for query, query_vector in zip(
            search_interests,
            query_embeddings
        ):

            results = self.search_faiss(
                query_vector,
                top_n
            )

            for candidate_index, similarity in results:

                candidate_asin = (
                    self.index_to_asin.get(
                        candidate_index
                    )
                )

                if not candidate_asin:
                    continue

                if candidate_asin not in candidates:

                    candidates[candidate_asin] = {
                        "product_matches": [],
                        "search_matches": [],
                    }

                candidates[
                    candidate_asin
                ]["search_matches"].append(
                    {
                        "query": query,
                        "similarity": float(
                            similarity
                        ),
                    }
                )

        return candidates

    # ========================================================
    # MERGE CANDIDATES
    # ========================================================

    def merge_candidates(
        self,
        product_candidates: Dict,
        search_candidates: Dict
    ) -> Dict:

        merged = {}

        all_asins = (
            set(product_candidates)
            | set(search_candidates)
        )

        for asin in all_asins:

            merged[asin] = {
                "product_matches": [],
                "search_matches": [],
            }

            if asin in product_candidates:

                merged[asin][
                    "product_matches"
                ].extend(
                    product_candidates[
                        asin
                    ].get(
                        "product_matches",
                        []
                    )
                )

            if asin in search_candidates:

                merged[asin][
                    "search_matches"
                ].extend(
                    search_candidates[
                        asin
                    ].get(
                        "search_matches",
                        []
                    )
                )

        return merged

    # ========================================================
    # INTERACTION SIGNAL
    # ========================================================

    def calculate_interaction_signal(
        self,
        asin: str,
        interactions: pd.DataFrame
    ) -> float:

        if interactions.empty:
            return 0.0

        if "product_asin" not in interactions.columns:
            return 0.0

        rows = interactions[
            interactions[
                "product_asin"
            ].astype(str)
            == str(asin)
        ]

        if rows.empty:
            return 0.0

        weights = {
            "SEARCH": 0.5,
            "VIEW": 1.0,
            "LIKE": 3.0,
            "SAVE": 4.0,
            "CART": 5.0,
            "PURCHASE": 8.0,
            "RATING": 2.0,
            "SKIP": -3.0,
        }

        score = 0.0

        for _, row in rows.iterrows():

            event_type = str(
                row.get(
                    "event_type",
                    ""
                )
            ).upper()

            score += weights.get(
                event_type,
                0.0
            )

        return float(score)

    # ========================================================
    # PERSONALIZED PREFERENCE TRANSFER
    # ========================================================

    def calculate_preference_transfer(
        self,
        product_matches: List[Dict]
    ) -> Tuple[float, List[Dict]]:
        """
        Transfer user preference from source products to
        candidates based on actual similarity.

        This is the key V3.1 fix.

        Example:

            source preference = 16
            candidate similarity = 0.90

        receives substantially more preference signal than:

            source preference = 16
            candidate similarity = 0.50
        """

        if not product_matches:

            return 0.0, []

        transfers = []

        for match in product_matches:

            similarity = float(
                match.get(
                    "similarity",
                    0.0
                )
            )

            preference = float(
                match.get(
                    "preference",
                    0.0
                )
            )

            # Only positive preference should transfer.
            preference = max(
                preference,
                0.0
            )

            # Similarity from normalized FAISS cosine/IP
            # is approximately in [-1, 1].
            similarity_normalized = float(
                np.clip(
                    similarity,
                    0.0,
                    1.0
                )
            )

            # Square similarity to make high-confidence
            # similarity matter more.
            similarity_strength = (
                similarity_normalized
                ** 2
            )

            transfer = (
                preference
                * similarity_strength
            )

            transfers.append(
                {
                    "source_asin": match[
                        "source_asin"
                    ],
                    "similarity": similarity,
                    "preference": preference,
                    "transfer": transfer,
                }
            )

        if not transfers:

            return 0.0, []

        # Strongest source relationship wins.
        strongest = max(
            transfers,
            key=lambda x: x[
                "transfer"
            ]
        )

        transfer_score = strongest[
            "transfer"
        ]

        # Convert to approximately 0-1.
        #
        # 16 preference is the current maximum in the
        # demo profile. log1p prevents large values from
        # dominating the complete score.
        normalized = (
            np.log1p(
                max(
                    transfer_score,
                    0.0
                )
            )
            / np.log1p(
                20.0
            )
        )

        normalized = float(
            np.clip(
                normalized,
                0.0,
                1.0
            )
        )

        return normalized, transfers

    # ========================================================
    # SEARCH RELEVANCE
    # ========================================================

    def calculate_search_relevance(
        self,
        search_matches: List[Dict]
    ) -> float:

        if not search_matches:

            return 0.0

        similarities = [
            float(
                match.get(
                    "similarity",
                    0.0
                )
            )
            for match in search_matches
        ]

        if not similarities:

            return 0.0

        return float(
            np.clip(
                max(
                    similarities
                ),
                0.0,
                1.0
            )
        )

    # ========================================================
    # MAIN RECOMMENDATION
    # ========================================================

    def recommend(
        self,
        user_id: str,
        top_k: int = DEFAULT_TOP_K
    ) -> Dict:

        start = time.perf_counter()

        user_id = str(user_id)

        profile = self.get_user_profile(
            user_id
        )

        interactions = (
            self.get_user_interactions(
                user_id
            )
        )

        positive_products = (
            self.get_positive_products(
                profile
            )
        )

        negative_products = (
            self.get_negative_products(
                profile
            )
        )

        search_interests = (
            self.get_search_interests(
                profile
            )
        )

        preference_map = (
            self.get_preference_map(
                profile
            )
        )

        # ----------------------------------------------------
        # Existing interaction history
        # ----------------------------------------------------

        interacted_products = set()

        if (
            not interactions.empty
            and "product_asin"
            in interactions.columns
        ):

            interacted_products = {
                str(value).strip()
                for value in interactions[
                    "product_asin"
                ].dropna()
                if str(value).strip()
            }

        # ----------------------------------------------------
        # Candidate pool
        # ----------------------------------------------------

        candidate_count = max(
            top_k
            * CANDIDATE_MULTIPLIER,
            MIN_CANDIDATES
        )

        print(
            "\nGenerating up to "
            f"{candidate_count:,} candidates..."
        )

        product_candidates = (
            self.generate_product_candidates(
                positive_products,
                preference_map,
                candidate_count
            )
        )

        search_candidates = (
            self.generate_search_candidates(
                search_interests,
                candidate_count
            )
        )

        merged = self.merge_candidates(
            product_candidates,
            search_candidates
        )

        print(
            f"Raw candidates: "
            f"{len(merged):,}"
        )

        # ----------------------------------------------------
        # Source product families
        # ----------------------------------------------------

        source_families = set()

        for asin in positive_products:

            source_families.update(
                self.classify_product(
                    asin
                )
            )

        # ----------------------------------------------------
        # Search families
        # ----------------------------------------------------

        search_families = set()

        for query in search_interests:

            search_families.update(
                self.classify_query(
                    query
                )
            )

        # If there are search families but no positive product
        # families, search intent becomes the compatibility
        # source.
        if (
            not source_families
            or source_families == {"other"}
        ):

            source_families = (
                search_families
                if search_families
                else {"other"}
            )

        print(
            f"User product families: "
            f"{sorted(source_families)}"
        )

        if search_families:

            print(
                f"Search intent families: "
                f"{sorted(search_families)}"
            )

        # ----------------------------------------------------
        # Score candidates
        # ----------------------------------------------------

        scored = []

        filtered_negative = 0
        filtered_interacted = 0
        filtered_incompatible = 0

        for asin, data in merged.items():

            # ------------------------------------------------
            # Negative filter
            # ------------------------------------------------

            if asin in negative_products:

                filtered_negative += 1

                continue

            # ------------------------------------------------
            # Already interacted filter
            # ------------------------------------------------

            if asin in interacted_products:

                filtered_interacted += 1

                continue

            # ------------------------------------------------
            # Source product filter
            # ------------------------------------------------

            if asin in positive_products:

                filtered_interacted += 1

                continue

            metadata = (
                self.product_lookup.get(
                    asin
                )
            )

            if metadata is None:

                continue

            candidate_families = (
                self.classify_product(
                    asin
                )
            )

            compatibility = (
                self.compatibility_score(
                    source_families,
                    candidate_families
                )
            )

            # ------------------------------------------------
            # Search compatibility
            # ------------------------------------------------

            if search_families:

                search_compatibilities = []

                for search_family in search_families:

                    score = (
                        self.compatibility_score(
                            {search_family},
                            candidate_families
                        )
                    )

                    search_compatibilities.append(
                        score
                    )

                if search_compatibilities:

                    search_compatibility = max(
                        search_compatibilities
                    )

                else:

                    search_compatibility = 0.35

                # Blend behavioral source compatibility
                # and search compatibility.
                compatibility = (
                    0.55 * compatibility
                    +
                    0.45 * search_compatibility
                )

            # ------------------------------------------------
            # HARD compatibility filter
            # ------------------------------------------------
            #
            # Only apply this when we know what the user's
            # product family is.
            #
            # Unknown products remain possible but unrelated
            # known families are removed.
            # ------------------------------------------------

            known_user_family = (
                "other"
                not in source_families
                or
                "other"
                not in search_families
            )

            if known_user_family:

                if compatibility < 0.20:

                    filtered_incompatible += 1

                    continue

            # ------------------------------------------------
            # Semantic similarity
            # ------------------------------------------------

            product_matches = data.get(
                "product_matches",
                []
            )

            search_matches = data.get(
                "search_matches",
                []
            )

            semantic_values = []

            for match in product_matches:

                semantic_values.append(
                    safe_float(
                        match.get(
                            "similarity",
                            0.0
                        )
                    )
                )

            for match in search_matches:

                semantic_values.append(
                    safe_float(
                        match.get(
                            "similarity",
                            0.0
                        )
                    )
                )

            if semantic_values:

                semantic_score = float(
                    np.clip(
                        max(
                            semantic_values
                        ),
                        0.0,
                        1.0
                    )
                )

            else:

                semantic_score = 0.0

            # ------------------------------------------------
            # Preference transfer
            # ------------------------------------------------

            (
                preference_score,
                preference_details,
            ) = (
                self.calculate_preference_transfer(
                    product_matches
                )
            )

            # ------------------------------------------------
            # Search relevance
            # ------------------------------------------------

            search_score = (
                self.calculate_search_relevance(
                    search_matches
                )
            )

            # ------------------------------------------------
            # Interaction signal
            # ------------------------------------------------

            interaction_score_raw = (
                self.calculate_interaction_signal(
                    asin,
                    interactions
                )
            )

            interaction_score = float(
                np.clip(
                    interaction_score_raw / 8.0,
                    0.0,
                    1.0
                )
            )

            # ------------------------------------------------
            # Final score
            # ------------------------------------------------

            final_score = (
                SEMANTIC_WEIGHT
                * semantic_score
                +
                PREFERENCE_WEIGHT
                * preference_score
                +
                SEARCH_WEIGHT
                * search_score
                +
                COMPATIBILITY_WEIGHT
                * compatibility
                +
                INTERACTION_WEIGHT
                * interaction_score
            )

            # ------------------------------------------------
            # Explanation
            # ------------------------------------------------

            reasons = []

            if product_matches:

                strongest_product = max(
                    product_matches,
                    key=lambda x: x[
                        "similarity"
                    ]
                )

                reasons.append(
                    "similar to a product you positively interacted with"
                )

                based_on_products = [
                    x["source_asin"]
                    for x in product_matches
                ]

            else:

                strongest_product = None

                based_on_products = []

            if search_matches:

                strongest_search = max(
                    search_matches,
                    key=lambda x: x[
                        "similarity"
                    ]
                )

                reasons.append(
                    f"matches your search for "
                    f"'{strongest_search['query']}'"
                )

                based_on_searches = [
                    x["query"]
                    for x in search_matches
                ]

            else:

                based_on_searches = []

            if preference_score >= 0.50:

                reasons.append(
                    "strongly aligned with your preferences"
                )

            elif preference_score > 0:

                reasons.append(
                    "aligned with your preferences"
                )

            if compatibility >= 0.80:

                reasons.append(
                    "compatible product category"
                )

            elif compatibility >= 0.60:

                reasons.append(
                    "related product category"
                )

            if not reasons:

                reasons.append(
                    "matches your personalized profile"
                )

            scored.append(
                {
                    "product_asin": asin,
                    "final_score": float(
                        final_score
                    ),
                    "semantic_score": float(
                        semantic_score
                    ),
                    "preference_score": float(
                        preference_score
                    ),
                    "search_score": float(
                        search_score
                    ),
                    "compatibility_score": float(
                        compatibility
                    ),
                    "interaction_score": float(
                        interaction_score
                    ),
                    "interaction_score_raw": float(
                        interaction_score_raw
                    ),
                    "candidate_families": sorted(
                        candidate_families
                    ),
                    "source_families": sorted(
                        source_families
                    ),
                    "preference_transfer": (
                        preference_details
                    ),
                    "based_on_products": list(
                        dict.fromkeys(
                            based_on_products
                        )
                    ),
                    "based_on_searches": list(
                        dict.fromkeys(
                            based_on_searches
                        )
                    ),
                    "reason": " + ".join(
                        reasons
                    ),
                    "metadata": metadata,
                }
            )

        # ----------------------------------------------------
        # Sort
        # ----------------------------------------------------

        scored.sort(
            key=lambda x: x[
                "final_score"
            ],
            reverse=True
        )

        recommendations = (
            scored[:top_k]
        )

        # ----------------------------------------------------
        # Final output formatting
        # ----------------------------------------------------

        final_recommendations = []

        for rank, item in enumerate(
            recommendations,
            start=1
        ):

            metadata = item.pop(
                "metadata"
            )

            price = metadata.get(
                "price",
                None
            )

            rating = metadata.get(
                "average_rating",
                metadata.get(
                    "rating",
                    None
                )
            )

            try:

                if pd.isna(price):

                    price = None

            except Exception:

                pass

            try:

                if pd.isna(rating):

                    rating = None

            except Exception:

                pass

            final_recommendations.append(
                {
                    "rank": rank,
                    "product_asin": item[
                        "product_asin"
                    ],
                    "title": clean_text(
                        metadata.get(
                            "title",
                            ""
                        )
                    ),
                    "store": clean_text(
                        metadata.get(
                            "store",
                            ""
                        )
                    ),
                    "categories": clean_text(
                        metadata.get(
                            "categories",
                            ""
                        )
                    ),
                    "price": price,
                    "rating": rating,
                    "product_families": item[
                        "candidate_families"
                    ],
                    "final_score": round(
                        item[
                            "final_score"
                        ],
                        6
                    ),
                    "score_components": {
                        "semantic": round(
                            item[
                                "semantic_score"
                            ],
                            6
                        ),
                        "preference": round(
                            item[
                                "preference_score"
                            ],
                            6
                        ),
                        "search": round(
                            item[
                                "search_score"
                            ],
                            6
                        ),
                        "compatibility": round(
                            item[
                                "compatibility_score"
                            ],
                            6
                        ),
                        "interaction": round(
                            item[
                                "interaction_score"
                            ],
                            6
                        ),
                    },
                    "based_on_products": item[
                        "based_on_products"
                    ],
                    "based_on_searches": item[
                        "based_on_searches"
                    ],
                    "preference_transfer": item[
                        "preference_transfer"
                    ],
                    "reason": item[
                        "reason"
                    ],
                }
            )

        # ----------------------------------------------------
        # Execution statistics
        # ----------------------------------------------------

        elapsed = (
            time.perf_counter()
            - start
        )

        result = {
            "engine": (
                "ShopGraph Recommendation Engine V3.1"
            ),
            "version": "3.1",
            "user_id": user_id,
            "generated_at": (
                pd.Timestamp.now(
                    tz="UTC"
                ).isoformat()
            ),
            "top_k": int(top_k),
            "user_profile_summary": {
                "total_interactions": profile.get(
                    "total_interactions",
                    0
                ),
                "positive_products": sorted(
                    positive_products
                ),
                "negative_products": sorted(
                    negative_products
                ),
                "search_interests": (
                    search_interests
                ),
                "interacted_products_count": (
                    len(
                        interacted_products
                    )
                ),
                "product_families": sorted(
                    source_families
                ),
                "search_families": sorted(
                    search_families
                ),
            },
            "scoring_weights": {
                "semantic": SEMANTIC_WEIGHT,
                "preference": PREFERENCE_WEIGHT,
                "search": SEARCH_WEIGHT,
                "compatibility": (
                    COMPATIBILITY_WEIGHT
                ),
                "interaction": (
                    INTERACTION_WEIGHT
                ),
            },
            "candidate_statistics": {
                "raw_candidates": len(
                    merged
                ),
                "negative_filtered": (
                    filtered_negative
                ),
                "already_interacted_filtered": (
                    filtered_interacted
                ),
                "incompatible_filtered": (
                    filtered_incompatible
                ),
                "scored_candidates": len(
                    scored
                ),
            },
            "recommendation_count": len(
                final_recommendations
            ),
            "generation_time_seconds": round(
                elapsed,
                4
            ),
            "recommendations": (
                final_recommendations
            ),
        }

        return result

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    def save_results(
        self,
        result: Dict
    ):

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            OUTPUT_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                result,
                file,
                indent=4,
                ensure_ascii=False,
                default=str
            )

        print(
            "\nResults saved to:"
        )

        print(
            OUTPUT_PATH
        )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    def display_results(
        self,
        result: Dict
    ):

        print("\n" + "=" * 80)
        print(
            "PERSONALIZED RECOMMENDATIONS - V3.1"
        )
        print("=" * 80)

        print(
            f"User: "
            f"{result['user_id']}"
        )

        print(
            f"Recommendations: "
            f"{result['recommendation_count']}"
        )

        print(
            f"Generation time: "
            f"{result['generation_time_seconds']}s"
        )

        stats = result[
            "candidate_statistics"
        ]

        print(
            f"Raw candidates: "
            f"{stats['raw_candidates']}"
        )

        print(
            f"Negative filtered: "
            f"{stats['negative_filtered']}"
        )

        print(
            f"Already interacted filtered: "
            f"{stats['already_interacted_filtered']}"
        )

        print(
            f"Incompatible filtered: "
            f"{stats['incompatible_filtered']}"
        )

        print(
            f"Final scored candidates: "
            f"{stats['scored_candidates']}"
        )

        recommendations = (
            result["recommendations"]
        )

        if not recommendations:

            print(
                "\nNo recommendations generated."
            )

            print(
                "\nPossible reasons:"
            )

            print(
                "  - User has no positive interactions"
            )

            print(
                "  - User profile is empty"
            )

            print(
                "  - All candidates were filtered"
            )

            return

        for item in recommendations:

            print("\n" + "-" * 80)

            print(
                f"#{item['rank']} "
                f"{item['title']}"
            )

            print(
                f"ASIN: "
                f"{item['product_asin']}"
            )

            print(
                f"Score: "
                f"{item['final_score']:.4f}"
            )

            print(
                f"Families: "
                f"{', '.join(item['product_families'])}"
            )

            components = (
                item["score_components"]
            )

            print(
                "Components: "
                f"semantic={components['semantic']:.3f}, "
                f"preference={components['preference']:.3f}, "
                f"search={components['search']:.3f}, "
                f"compatibility={components['compatibility']:.3f}, "
                f"interaction={components['interaction']:.3f}"
            )

            print(
                f"Reason: "
                f"{item['reason']}"
            )

            if item[
                "based_on_products"
            ]:

                print(
                    "Based on products: "
                    + ", ".join(
                        item[
                            "based_on_products"
                        ]
                    )
                )

            if item[
                "based_on_searches"
            ]:

                print(
                    "Based on searches: "
                    + ", ".join(
                        item[
                            "based_on_searches"
                        ]
                    )
                )

        print("\n" + "=" * 80)


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "ShopGraph Recommendation Engine V3.1"
        )
    )

    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="User ID"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of recommendations"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo using user_001"
    )

    args = parser.parse_args()

    if args.top_k <= 0:

        parser.error(
            "--top-k must be greater than zero."
        )

    user_id = args.user

    if args.demo:

        user_id = "user_001"

    if not user_id:

        parser.error(
            "Provide --user USER_ID or use --demo."
        )

    engine = (
        RecommendationEngineV3()
    )

    result = engine.recommend(
        user_id=user_id,
        top_k=args.top_k
    )

    engine.display_results(
        result
    )

    engine.save_results(
        result
    )


if __name__ == "__main__":

    main()
