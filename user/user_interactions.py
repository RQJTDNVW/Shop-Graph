"""
ShopGraph - User Interaction Engine

Component #8
----------------
Records and retrieves user interactions for the future
Personalization Engine and Recommendation Engine V3.

Supported events:
    SEARCH
    VIEW
    LIKE
    SAVE
    CART
    PURCHASE
    RATING
    SKIP

Storage:
    data/processed/user_interactions.parquet

The engine intentionally stores raw interactions.
Personalization logic will be implemented later.
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR_CANDIDATES = (
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "data" / "raw" / "processed",
)
DATA_DIR = next(
    (
        directory
        for directory in DATA_DIR_CANDIDATES
        if (directory / "user_interactions.parquet").exists()
    ),
    DATA_DIR_CANDIDATES[0],
)

INTERACTION_FILE = DATA_DIR / "user_interactions.parquet"


# ============================================================
# CONFIGURATION
# ============================================================

EVENT_TYPES = {
    "SEARCH",
    "VIEW",
    "LIKE",
    "SAVE",
    "CART",
    "PURCHASE",
    "RATING",
    "SKIP",
}


COLUMNS = [
    "interaction_id",
    "user_id",
    "session_id",
    "timestamp",
    "event_type",
    "product_asin",
    "query",
    "rating",
    "metadata",
]


# ============================================================
# USER INTERACTION ENGINE
# ============================================================

class UserInteractionEngine:
    """
    Stores and retrieves raw ShopGraph user interactions.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = (
            Path(storage_path)
            if storage_path is not None
            else INTERACTION_FILE
        )

        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.interactions = self._load()

    # --------------------------------------------------------
    # LOAD EXISTING DATA
    # --------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        """
        Load existing interaction history.

        If no file exists, create an empty DataFrame.
        """

        if not self.storage_path.exists():
            return pd.DataFrame(columns=COLUMNS)

        try:
            df = pd.read_parquet(
                self.storage_path,
                engine="fastparquet",
            )

            # Make sure all expected columns exist.
            for column in COLUMNS:
                if column not in df.columns:
                    df[column] = None

            return df[COLUMNS]

        except Exception as exc:
            raise RuntimeError(
                f"Could not load interaction data:\n"
                f"{self.storage_path}\n\n"
                f"Error: {exc}"
            ) from exc

    # --------------------------------------------------------
    # SAVE DATA
    # --------------------------------------------------------

    def _save(self) -> None:
        """
        Persist interactions to Parquet.
        """

        self.interactions.to_parquet(
            self.storage_path,
            engine="fastparquet",
            index=False,
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    @staticmethod
    def _validate_user_id(user_id: str) -> str:
        if not user_id or not str(user_id).strip():
            raise ValueError("user_id cannot be empty.")

        return str(user_id).strip()

    @staticmethod
    def _validate_event_type(event_type: str) -> str:
        event_type = str(event_type).strip().upper()

        if event_type not in EVENT_TYPES:
            raise ValueError(
                f"Invalid event type: {event_type}\n"
                f"Allowed events: {sorted(EVENT_TYPES)}"
            )

        return event_type

    @staticmethod
    def _validate_asin(product_asin: Optional[str]) -> Optional[str]:
        if product_asin is None:
            return None

        product_asin = str(product_asin).strip()

        if not product_asin:
            return None

        return product_asin

    @staticmethod
    def _validate_rating(
        rating: Optional[float],
    ) -> Optional[float]:

        if rating is None:
            return None

        try:
            rating = float(rating)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "rating must be a number between 1 and 5."
            ) from exc

        if not 1 <= rating <= 5:
            raise ValueError(
                "rating must be between 1 and 5."
            )

        return rating

    # --------------------------------------------------------
    # SESSION
    # --------------------------------------------------------

    @staticmethod
    def create_session_id() -> str:
        """
        Generate a new session ID.
        """

        return f"session_{uuid.uuid4().hex[:12]}"

    # --------------------------------------------------------
    # GENERIC EVENT LOGGER
    # --------------------------------------------------------

    def log_event(
        self,
        user_id: str,
        event_type: str,
        product_asin: Optional[str] = None,
        query: Optional[str] = None,
        session_id: Optional[str] = None,
        rating: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Record a generic interaction event.
        """

        user_id = self._validate_user_id(user_id)

        event_type = self._validate_event_type(event_type)

        product_asin = self._validate_asin(product_asin)

        rating = self._validate_rating(rating)

        if session_id is None:
            session_id = self.create_session_id()
        else:
            session_id = str(session_id).strip()

        if query is not None:
            query = str(query).strip()

        if metadata is None:
            metadata = {}

        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary.")

        interaction = {
            "interaction_id": f"interaction_{uuid.uuid4().hex}",
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "product_asin": product_asin,
            "query": query,
            "rating": rating,
            "metadata": json.dumps(
                metadata,
                ensure_ascii=False,
            ),
        }

        new_row = pd.DataFrame(
            [interaction],
            columns=COLUMNS,
        )

        self.interactions = pd.concat(
            [
                self.interactions,
                new_row,
            ],
            ignore_index=True,
        )

        self._save()

        return interaction

    # ========================================================
    # CONVENIENCE METHODS
    # ========================================================

    def log_search(
        self,
        user_id: str,
        query: str,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:

        if not query or not str(query).strip():
            raise ValueError("Search query cannot be empty.")

        return self.log_event(
            user_id=user_id,
            event_type="SEARCH",
            query=query,
            session_id=session_id,
            metadata=metadata,
        )

    def log_view(
        self,
        user_id: str,
        product_asin: str,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:

        return self.log_event(
            user_id=user_id,
            event_type="VIEW",
            product_asin=product_asin,
            session_id=session_id,
            metadata=metadata,
        )

    def log_like(
        self,
        user_id: str,
        product_asin: str,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:

        return self.log_event(
            user_id=user_id,
            event_type="LIKE",
            product_asin=product_asin,
            session_id=session_id,
            metadata=metadata,
        )

    def log_save(
        self,
        user_id: str,
        product_asin: str,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:

        return self.log_event(
            user_id=user_id,
            event_type="SAVE",
            product_asin=product_asin,
            session_id=session_id,
            metadata=metadata,
        )

    def log_cart(
        self,
        user_id: str,
        product_asin: str,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:

        return self.log_event(
            user_id=user_id,
            event_type="CART",
            product_asin=product_asin,
            session_id=session_id,
            metadata=metadata,
        )

    def log_purchase(
        self,
        user_id: str,
        product_asin: str,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:

        return self.log_event(
            user_id=user_id,
            event_type="PURCHASE",
            product_asin=product_asin,
            session_id=session_id,
            metadata=metadata,
        )

    def log_rating(
        self,
        user_id: str,
        product_asin: str,
        rating: float,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:

        return self.log_event(
            user_id=user_id,
            event_type="RATING",
            product_asin=product_asin,
            session_id=session_id,
            rating=rating,
            metadata=metadata,
        )

    def log_skip(
        self,
        user_id: str,
        product_asin: str,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:

        return self.log_event(
            user_id=user_id,
            event_type="SKIP",
            product_asin=product_asin,
            session_id=session_id,
            metadata=metadata,
        )

    # ========================================================
    # RETRIEVAL
    # ========================================================

    def get_user_history(
        self,
        user_id: str,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Return all interactions for a user.
        """

        user_id = self._validate_user_id(user_id)

        df = self.interactions[
            self.interactions["user_id"] == user_id
        ].copy()

        df = df.sort_values(
            "timestamp",
            ascending=False,
        )

        if limit is not None:
            df = df.head(int(limit))

        return df.reset_index(drop=True)

    def get_product_interactions(
        self,
        product_asin: str,
    ) -> pd.DataFrame:
        """
        Return all interactions involving a product.
        """

        product_asin = self._validate_asin(product_asin)

        if product_asin is None:
            return pd.DataFrame(columns=COLUMNS)

        df = self.interactions[
            self.interactions["product_asin"] == product_asin
        ].copy()

        return df.sort_values(
            "timestamp",
            ascending=False,
        ).reset_index(drop=True)

    def get_session_history(
        self,
        session_id: str,
    ) -> pd.DataFrame:
        """
        Return all interactions from a session.
        """

        session_id = str(session_id).strip()

        df = self.interactions[
            self.interactions["session_id"] == session_id
        ].copy()

        return df.sort_values(
            "timestamp",
            ascending=True,
        ).reset_index(drop=True)

    # ========================================================
    # STATISTICS
    # ========================================================

    def get_statistics(self) -> dict[str, Any]:
        """
        Return high-level interaction statistics.
        """

        if self.interactions.empty:
            return {
                "total_interactions": 0,
                "unique_users": 0,
                "unique_products": 0,
                "unique_sessions": 0,
                "event_counts": {},
            }

        return {
            "total_interactions": int(
                len(self.interactions)
            ),
            "unique_users": int(
                self.interactions["user_id"].nunique()
            ),
            "unique_products": int(
                self.interactions["product_asin"]
                .dropna()
                .nunique()
            ),
            "unique_sessions": int(
                self.interactions["session_id"].nunique()
            ),
            "event_counts": (
                self.interactions["event_type"]
                .value_counts()
                .to_dict()
            ),
        }

    # ========================================================
    # CLEAR DATA
    # ========================================================

    def clear_all(self) -> None:
        """
        Delete all stored interactions.

        This is intended for development/testing only.
        """

        self.interactions = pd.DataFrame(
            columns=COLUMNS
        )

        self._save()


# ============================================================
# DEMO / TEST
# ============================================================

def run_demo() -> None:
    """
    Create a small demonstration interaction history.
    """

    print("=" * 70)
    print("SHOPGRAPH - USER INTERACTION ENGINE")
    print("=" * 70)

    engine = UserInteractionEngine()

    print("\nStorage:")
    print(engine.storage_path)

    # --------------------------------------------------------
    # Create test user/session
    # --------------------------------------------------------

    user_id = "user_001"

    session_id = engine.create_session_id()

    print("\nUser:")
    print(user_id)

    print("\nSession:")
    print(session_id)

    # --------------------------------------------------------
    # Record interactions
    # --------------------------------------------------------

    print("\nRecording interactions...")

    engine.log_search(
        user_id=user_id,
        query="wireless headphones",
        session_id=session_id,
    )

    engine.log_view(
        user_id=user_id,
        product_asin="B00MCW7G9M",
        session_id=session_id,
    )

    engine.log_like(
        user_id=user_id,
        product_asin="B00MCW7G9M",
        session_id=session_id,
    )

    engine.log_save(
        user_id=user_id,
        product_asin="B00MCW7G9M",
        session_id=session_id,
    )

    engine.log_cart(
        user_id=user_id,
        product_asin="B00MCW7G9M",
        session_id=session_id,
    )

    engine.log_rating(
        user_id=user_id,
        product_asin="B00MCW7G9M",
        rating=5,
        session_id=session_id,
    )

    engine.log_skip(
        user_id=user_id,
        product_asin="B08FDM4DHF",
        session_id=session_id,
    )

    # --------------------------------------------------------
    # Display statistics
    # --------------------------------------------------------

    print("\nStatistics:")
    print(
        json.dumps(
            engine.get_statistics(),
            indent=4,
            default=str,
        )
    )

    # --------------------------------------------------------
    # Display user history
    # --------------------------------------------------------

    print("\nUser History:")
    print("-" * 70)

    history = engine.get_user_history(
        user_id=user_id
    )

    print(
        history[
            [
                "timestamp",
                "event_type",
                "product_asin",
                "query",
                "rating",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Display product interactions
    # --------------------------------------------------------

    print("\nProduct Interactions:")
    print("-" * 70)

    product_history = engine.get_product_interactions(
        "B00MCW7G9M"
    )

    print(
        product_history[
            [
                "user_id",
                "event_type",
                "product_asin",
                "rating",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("USER INTERACTION ENGINE TEST COMPLETE")
    print("=" * 70)


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="ShopGraph User Interaction Engine"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the interaction engine demo.",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Display interaction statistics.",
    )

    parser.add_argument(
        "--user",
        type=str,
        help="Display interaction history for a user.",
    )

    parser.add_argument(
        "--product",
        type=str,
        help="Display interactions for a product ASIN.",
    )

    args = parser.parse_args()

    engine = UserInteractionEngine()

    if args.demo:
        run_demo()

    elif args.stats:
        print(
            json.dumps(
                engine.get_statistics(),
                indent=4,
                default=str,
            )
        )

    elif args.user:
        history = engine.get_user_history(
            args.user
        )

        if history.empty:
            print(
                f"No interactions found for user: "
                f"{args.user}"
            )
        else:
            print(
                history.to_string(index=False)
            )

    elif args.product:
        history = engine.get_product_interactions(
            args.product
        )

        if history.empty:
            print(
                f"No interactions found for product: "
                f"{args.product}"
            )
        else:
            print(
                history.to_string(index=False)
            )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
