
"""
ShopGraph - Personalization Engine

Component #9
----------------
Transforms raw user interaction history into user preference
profiles.

Input:
    data/processed/user_interactions.parquet

Output:
    data/processed/user_profiles.parquet

The engine currently calculates:
    - Interaction counts
    - Weighted product preferences
    - Positive products
    - Negative/skipped products
    - Search interests
    - User preference scores

Recommendation Engine V3 will consume these profiles later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"

INTERACTION_FILE = DATA_DIR / "user_interactions.parquet"

PROFILE_FILE = DATA_DIR / "user_profiles.parquet"


# ============================================================
# INTERACTION WEIGHTS
# ============================================================

# Higher values represent stronger positive signals.

INTERACTION_WEIGHTS = {
    "SEARCH": 0.5,
    "VIEW": 1.0,
    "LIKE": 3.0,
    "SAVE": 4.0,
    "CART": 5.0,
    "PURCHASE": 8.0,
    "RATING": 0.0,
    "SKIP": -3.0,
}


POSITIVE_EVENTS = {
    "VIEW",
    "LIKE",
    "SAVE",
    "CART",
    "PURCHASE",
    "RATING",
}


NEGATIVE_EVENTS = {
    "SKIP",
}


# ============================================================
# PERSONALIZATION ENGINE
# ============================================================

class PersonalizationEngine:
    """
    Converts raw interaction events into user profiles.
    """

    def __init__(
        self,
        interaction_path: Optional[Path] = None,
        profile_path: Optional[Path] = None,
    ):

        self.interaction_path = (
            Path(interaction_path)
            if interaction_path is not None
            else INTERACTION_FILE
        )

        self.profile_path = (
            Path(profile_path)
            if profile_path is not None
            else PROFILE_FILE
        )

        self.profile_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.interactions = self._load_interactions()

    # ========================================================
    # LOAD INTERACTIONS
    # ========================================================

    def _load_interactions(self) -> pd.DataFrame:

        if not self.interaction_path.exists():
            raise FileNotFoundError(
                f"Interaction file not found:\n"
                f"{self.interaction_path}\n\n"
                f"Run Component #8 first."
            )

        df = pd.read_parquet(
            self.interaction_path,
            engine="fastparquet",
        )

        required_columns = {
            "user_id",
            "event_type",
            "product_asin",
            "query",
            "rating",
            "timestamp",
        }

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"Interaction file is missing columns: "
                f"{sorted(missing)}"
            )

        return df

    # ========================================================
    # RATING SCORE
    # ========================================================

    @staticmethod
    def _rating_weight(rating: Any) -> float:

        if pd.isna(rating):
            return 0.0

        try:
            rating = float(rating)
        except (TypeError, ValueError):
            return 0.0

        # Convert 1-5 rating to approximately -3 to +3.
        return (rating - 3.0) * 1.5

    # ========================================================
    # EVENT SCORE
    # ========================================================

    def _event_score(
        self,
        event_type: str,
        rating: Any = None,
    ) -> float:

        event_type = str(event_type).upper()

        if event_type == "RATING":
            return self._rating_weight(rating)

        return INTERACTION_WEIGHTS.get(
            event_type,
            0.0,
        )

    # ========================================================
    # USER PRODUCT PREFERENCES
    # ========================================================

    def get_product_preferences(
        self,
        user_id: str,
    ) -> pd.DataFrame:
        """
        Calculate weighted product preferences for one user.
        """

        user_df = self.interactions[
            self.interactions["user_id"] == user_id
        ].copy()

        if user_df.empty:
            return pd.DataFrame(
                columns=[
                    "product_asin",
                    "preference_score",
                    "interaction_count",
                    "positive_interactions",
                    "negative_interactions",
                    "average_rating",
                ]
            )

        # Only product-related events.
        user_df = user_df[
            user_df["product_asin"].notna()
        ].copy()

        if user_df.empty:
            return pd.DataFrame(
                columns=[
                    "product_asin",
                    "preference_score",
                    "interaction_count",
                    "positive_interactions",
                    "negative_interactions",
                    "average_rating",
                ]
            )

        user_df["event_score"] = user_df.apply(
            lambda row: self._event_score(
                row["event_type"],
                row["rating"],
            ),
            axis=1,
        )

        grouped = (
            user_df
            .groupby("product_asin")
            .agg(
                preference_score=(
                    "event_score",
                    "sum",
                ),
                interaction_count=(
                    "event_type",
                    "count",
                ),
                average_rating=(
                    "rating",
                    "mean",
                ),
            )
            .reset_index()
        )

        positive_counts = (
            user_df[
                user_df["event_type"].isin(
                    POSITIVE_EVENTS
                )
            ]
            .groupby("product_asin")
            .size()
            .rename("positive_interactions")
        )

        negative_counts = (
            user_df[
                user_df["event_type"].isin(
                    NEGATIVE_EVENTS
                )
            ]
            .groupby("product_asin")
            .size()
            .rename("negative_interactions")
        )

        grouped = grouped.merge(
            positive_counts,
            on="product_asin",
            how="left",
        )

        grouped = grouped.merge(
            negative_counts,
            on="product_asin",
            how="left",
        )

        grouped[
            "positive_interactions"
        ] = grouped[
            "positive_interactions"
        ].fillna(0)

        grouped[
            "negative_interactions"
        ] = grouped[
            "negative_interactions"
        ].fillna(0)

        grouped = grouped.sort_values(
            "preference_score",
            ascending=False,
        )

        return grouped.reset_index(drop=True)

    # ========================================================
    # SEARCH INTERESTS
    # ========================================================

    def get_search_interests(
        self,
        user_id: str,
    ) -> list[str]:
        """
        Return search queries made by a user.
        """

        user_df = self.interactions[
            self.interactions["user_id"] == user_id
        ]

        searches = user_df[
            (
                user_df["event_type"] == "SEARCH"
            )
            & user_df["query"].notna()
        ]["query"]

        results = []

        for query in searches:
            query = str(query).strip()

            if query and query not in results:
                results.append(query)

        return results

    # ========================================================
    # USER EVENT COUNTS
    # ========================================================

    def get_event_counts(
        self,
        user_id: str,
    ) -> dict[str, int]:
        """
        Count interaction events for a user.
        """

        user_df = self.interactions[
            self.interactions["user_id"] == user_id
        ]

        if user_df.empty:
            return {}

        counts = (
            user_df["event_type"]
            .value_counts()
            .to_dict()
        )

        return {
            str(key): int(value)
            for key, value in counts.items()
        }

    # ========================================================
    # BUILD USER PROFILE
    # ========================================================

    def build_user_profile(
        self,
        user_id: str,
    ) -> dict[str, Any]:
        """
        Build a complete profile for one user.
        """

        user_df = self.interactions[
            self.interactions["user_id"] == user_id
        ].copy()

        if user_df.empty:
            return {
                "user_id": user_id,
                "total_interactions": 0,
                "event_counts": {},
                "positive_products": [],
                "negative_products": [],
                "search_interests": [],
            }

        product_preferences = (
            self.get_product_preferences(
                user_id
            )
        )

        positive_products = []

        negative_products = []

        if not product_preferences.empty:

            positive = product_preferences[
                product_preferences[
                    "preference_score"
                ] > 0
            ]

            negative = product_preferences[
                product_preferences[
                    "preference_score"
                ] < 0
            ]

            positive_products = (
                positive[
                    "product_asin"
                ]
                .tolist()
            )

            negative_products = (
                negative[
                    "product_asin"
                ]
                .tolist()
            )

        profile = {
            "user_id": user_id,

            "total_interactions": int(
                len(user_df)
            ),

            "event_counts": self.get_event_counts(
                user_id
            ),

            "positive_products": (
                positive_products
            ),

            "negative_products": (
                negative_products
            ),

            "search_interests": (
                self.get_search_interests(
                    user_id
                )
            ),

            "product_preferences": (
                product_preferences.to_dict(
                    orient="records"
                )
            ),
        }

        return profile

    # ========================================================
    # BUILD ALL PROFILES
    # ========================================================

    def build_all_profiles(self) -> list[dict[str, Any]]:
        """
        Build profiles for every user.
        """

        users = (
            self.interactions["user_id"]
            .dropna()
            .astype(str)
            .unique()
        )

        profiles = []

        for user_id in users:

            profile = self.build_user_profile(
                user_id
            )

            profiles.append(profile)

        return profiles

    # ========================================================
    # SAVE PROFILES
    # ========================================================

    def save_profiles(
        self,
        profiles: list[dict[str, Any]],
    ) -> None:
        """
        Save profiles to Parquet.
        """

        rows = []

        for profile in profiles:

            rows.append(
                {
                    "user_id": profile[
                        "user_id"
                    ],

                    "total_interactions": profile[
                        "total_interactions"
                    ],

                    "event_counts": json.dumps(
                        profile[
                            "event_counts"
                        ],
                        ensure_ascii=False,
                    ),

                    "positive_products": json.dumps(
                        profile[
                            "positive_products"
                        ],
                        ensure_ascii=False,
                    ),

                    "negative_products": json.dumps(
                        profile[
                            "negative_products"
                        ],
                        ensure_ascii=False,
                    ),

                    "search_interests": json.dumps(
                        profile[
                            "search_interests"
                        ],
                        ensure_ascii=False,
                    ),

                    "product_preferences": json.dumps(
                        profile[
                            "product_preferences"
                        ],
                        ensure_ascii=False,
                    ),
                }
            )

        df = pd.DataFrame(rows)

        df.to_parquet(
            self.profile_path,
            engine="fastparquet",
            index=False,
        )

    # ========================================================
    # LOAD SAVED PROFILE
    # ========================================================

    def load_profile(
        self,
        user_id: str,
    ) -> Optional[dict[str, Any]]:
        """
        Load one saved user profile.
        """

        if not self.profile_path.exists():
            return None

        df = pd.read_parquet(
            self.profile_path,
            engine="fastparquet",
        )

        matches = df[
            df["user_id"] == user_id
        ]

        if matches.empty:
            return None

        row = matches.iloc[0]

        return {
            "user_id": row["user_id"],
            "total_interactions": int(
                row["total_interactions"]
            ),
            "event_counts": json.loads(
                row["event_counts"]
            ),
            "positive_products": json.loads(
                row["positive_products"]
            ),
            "negative_products": json.loads(
                row["negative_products"]
            ),
            "search_interests": json.loads(
                row["search_interests"]
            ),
            "product_preferences": json.loads(
                row["product_preferences"]
            ),
        }


# ============================================================
# DEMO
# ============================================================

def run_demo() -> None:

    print("=" * 70)
    print("SHOPGRAPH - PERSONALIZATION ENGINE")
    print("=" * 70)

    engine = PersonalizationEngine()

    print("\nInteraction source:")
    print(engine.interaction_path)

    print("\nProfile output:")
    print(engine.profile_path)

    # --------------------------------------------------------
    # Build profiles
    # --------------------------------------------------------

    profiles = engine.build_all_profiles()

    print(
        f"\nUsers discovered: {len(profiles)}"
    )

    if not profiles:
        print(
            "\nNo users found."
        )
        return

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    engine.save_profiles(
        profiles
    )

    print(
        "\nProfiles saved successfully."
    )

    # --------------------------------------------------------
    # Display profiles
    # --------------------------------------------------------

    for profile in profiles:

        print("\n" + "-" * 70)

        print(
            f"USER: {profile['user_id']}"
        )

        print(
            f"Total interactions: "
            f"{profile['total_interactions']}"
        )

        print("\nEvent counts:")

        print(
            json.dumps(
                profile["event_counts"],
                indent=4,
            )
        )

        print("\nPositive products:")

        for product in profile[
            "positive_products"
        ]:
            print(
                f"  + {product}"
            )

        print("\nNegative products:")

        for product in profile[
            "negative_products"
        ]:
            print(
                f"  - {product}"
            )

        print("\nSearch interests:")

        for query in profile[
            "search_interests"
        ]:
            print(
                f"  • {query}"
            )

        print("\nProduct preferences:")

        for product in profile[
            "product_preferences"
        ]:

            print(
                f"  {product['product_asin']}"
                f" → "
                f"{product['preference_score']:.2f}"
            )

    print("\n" + "=" * 70)
    print("PERSONALIZATION ENGINE TEST COMPLETE")
    print("=" * 70)


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "ShopGraph Personalization Engine"
        )
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Build and display user profiles.",
    )

    parser.add_argument(
        "--user",
        type=str,
        help="Build and display a specific user profile.",
    )

    args = parser.parse_args()

    engine = PersonalizationEngine()

    if args.demo:

        run_demo()

    elif args.user:

        profile = engine.build_user_profile(
            args.user
        )

        print(
            json.dumps(
                profile,
                indent=4,
                default=str,
            )
        )

    else:

        parser.print_help()


if __name__ == "__main__":
    main()
