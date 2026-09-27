from pathlib import Path
from typing import Any

import json
import os
import re
import sys
import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    SearchRequest,
    RecommendationRequest,
    InteractionRequest,
)
from backend.data_paths import processed_data_dir


# ============================================================
# SHOPGRAPH PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = processed_data_dir("products_text_fixed.parquet")

PRODUCT_FILE = DATA_DIR / "products_text_fixed.parquet"
INTERACTION_FILE = DATA_DIR / "user_interactions.parquet"
PROFILE_FILE = DATA_DIR / "user_profiles.parquet"

SEARCH_OUTPUT = DATA_DIR / "search"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="ShopGraph API",
    description=(
        "AI-powered e-commerce product discovery and "
        "personalized recommendation API."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # The API does not use cookies; wildcard origins cannot be credentialed.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL DATA
# ============================================================

products_df: pd.DataFrame | None = None
product_lookup: dict[str, dict[str, Any]] = {}

interaction_engine = None
recommendation_engine = None
search_engine = None


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    global products_df
    global product_lookup
    global interaction_engine
    global recommendation_engine

    print("\n" + "=" * 70)
    print("SHOPGRAPH BACKEND API")
    print("=" * 70)

    # --------------------------------------------------------
    # Product metadata
    # --------------------------------------------------------

    if PRODUCT_FILE.exists():
        print("[OK] Loading product metadata...")

        products_df = pd.read_parquet(
            PRODUCT_FILE,
            engine="fastparquet"
        )

        print(f"[OK] Products loaded: {len(products_df):,}")

        if "parent_asin" in products_df.columns:
            for _, row in products_df.iterrows():

                asin = str(row["parent_asin"])

                product_lookup[asin] = {
                    column: clean_value(row[column])
                    for column in products_df.columns
                }

        print(f"[OK] Product lookup: {len(product_lookup):,}")

    else:
        print("[WARNING] Product metadata not found:")
        print(PRODUCT_FILE)

    # --------------------------------------------------------
    # Interaction engine
    # --------------------------------------------------------

    try:
        from user.user_interactions import UserInteractionEngine

        interaction_engine = UserInteractionEngine(
            storage_path=INTERACTION_FILE
        )

        print("[OK] User interaction engine loaded")

    except Exception as exc:
        print("[WARNING] Interaction engine unavailable:")
        print(exc)

    # --------------------------------------------------------
    # Recommendation engine
    # --------------------------------------------------------

    if os.getenv("SHOPGRAPH_ENABLE_RECOMMENDATIONS", "").lower() in {
        "1", "true", "yes"
    }:
        try:
            from src.recommendation.recommendation_engine_v3 import (
                RecommendationEngineV3
            )

            recommendation_engine = RecommendationEngineV3()

            print("[OK] Recommendation Engine V3.1 loaded")

        except Exception as exc:
            print("[WARNING] Recommendation engine unavailable:")
            print(exc)
    else:
        print(
            "[INFO] Recommendation engine loading is disabled. "
            "Set SHOPGRAPH_ENABLE_RECOMMENDATIONS=true to enable it."
        )

    print("=" * 70)
    print("ShopGraph API startup complete")
    print("=" * 70 + "\n")


# ============================================================
# HELPERS
# ============================================================

def clean_value(value):

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, (list, tuple)):
        return list(value)

    if isinstance(value, dict):
        return value

    return value


def json_safe(value):
    """Convert pandas/NumPy values nested in engine output to JSON values."""
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    if hasattr(value, "tolist") and not isinstance(value, str):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass

    value = clean_value(value)

    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()
        except Exception:
            pass

    return value


def product_image_url(product: dict[str, Any]) -> str:
    """Extract the preferred image URL from flattened Amazon metadata."""
    for field in ("images.hi_res", "images.large", "images.thumb"):
        value = product.get(field)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not isinstance(item, str):
                continue
            match = re.search(r"https?://[^\s'\"]+", item)
            if match:
                return match.group(0)
    return ""


def success_response(data, elapsed_ms=None):

    response = {
        "success": True,
        "data": json_safe(data),
    }

    if elapsed_ms is not None:
        response["processing_time_ms"] = round(
            elapsed_ms,
            2
        )

    return response


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "success": True,
        "service": "ShopGraph API",
        "status": "online",
        "version": "1.0.0",
        "components": {
            "products": products_df is not None,
            "interaction_engine": interaction_engine is not None,
            "recommendation_engine": recommendation_engine is not None,
        },
    }


# ============================================================
# PRODUCT COUNT
# ============================================================

@app.get("/api/products")
def get_products(
    limit: int = 20,
    offset: int = 0,
):

    if products_df is None:
        raise HTTPException(
            status_code=503,
            detail="Product database is not loaded."
        )

    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 100."
        )

    if offset < 0:
        raise HTTPException(
            status_code=400,
            detail="offset cannot be negative."
        )

    selected = products_df.iloc[
        offset:offset + limit
    ]

    products = []

    for _, row in selected.iterrows():

        item = {
            column: clean_value(row[column])
            for column in selected.columns
        }

        products.append(item)

    return success_response({
        "products": products,
        "count": len(products),
        "offset": offset,
        "limit": limit,
        "total_products": len(products_df),
    })


# ============================================================
# SINGLE PRODUCT
# ============================================================

@app.get("/api/products/{asin}")
def get_product(asin: str):

    asin = asin.strip()

    if not asin:
        raise HTTPException(
            status_code=400,
            detail="ASIN cannot be empty."
        )

    product = product_lookup.get(asin)

    if product is None:
        raise HTTPException(
            status_code=404,
            detail=f"Product {asin} not found."
        )

    return success_response(product)


# ============================================================
# SEMANTIC SEARCH
# ============================================================

@app.post("/api/search")
def search_products(request: SearchRequest):

    start = time.perf_counter()

    try:
        global search_engine

        # Load the existing semantic-search resources once, then reuse them.
        if search_engine is None:
            from src.search.HybridSearch_v2 import HybridSearch
            search_engine = HybridSearch()

        results = search_engine.search(
            query=request.query,
            top_k=request.top_k,
        )

        # The optimized search metadata does not retain image columns. Enrich
        # each result from the canonical product catalog before returning it.
        for result in results:
            asin = str(result.get("parent_asin", ""))
            product = product_lookup.get(asin)
            if product:
                result["image_url"] = product_image_url(product)

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        return success_response(
            {
                "query": request.query,
                "results": results,
                "count": len(results),
            },
            elapsed,
        )

    except ImportError as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Search engine could not be imported. "
                f"Details: {exc}"
            ),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {exc}",
        )


# ============================================================
# PERSONALIZED RECOMMENDATIONS
# ============================================================

@app.post("/api/recommendations")
def recommendations(
    request: RecommendationRequest
):

    start = time.perf_counter()

    if recommendation_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation engine is unavailable."
        )

    try:

        result = recommendation_engine.recommend(
            user_id=request.user_id,
            top_k=request.top_k,
        )

        # Recommendation V3 intentionally returns a compact scoring payload.
        # Add the catalog image needed by the frontend without changing its
        # ranking or explanation data.
        for recommendation in result.get("recommendations", []):
            asin = str(recommendation.get("product_asin", ""))
            product = product_lookup.get(asin)
            if product:
                recommendation["image_url"] = product_image_url(product)
                recommendation["average_rating"] = product.get(
                    "average_rating", recommendation.get("rating")
                )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        return success_response(
            result,
            elapsed,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Recommendation generation failed: {exc}",
        )


# ============================================================
# USER PROFILE
# ============================================================

@app.get("/api/users/{user_id}/profile")
def get_user_profile(user_id: str):

    if not PROFILE_FILE.exists():

        raise HTTPException(
            status_code=503,
            detail="User profile database is unavailable."
        )

    try:

        profiles = pd.read_parquet(
            PROFILE_FILE,
            engine="fastparquet"
        )

        if "user_id" not in profiles.columns:

            raise HTTPException(
                status_code=500,
                detail="Profile file does not contain user_id."
            )

        matches = profiles[
            profiles["user_id"].astype(str) == str(user_id)
        ]

        if matches.empty:

            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found."
            )

        profile = matches.iloc[0].to_dict()

        profile = {
            key: clean_value(value)
            for key, value in profile.items()
        }

        return success_response(profile)

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Profile lookup failed: {exc}",
        )


# ============================================================
# LOG USER INTERACTION
# ============================================================

@app.post("/api/interactions")
def log_interaction(
    request: InteractionRequest
):

    if interaction_engine is None:

        raise HTTPException(
            status_code=503,
            detail="Interaction engine is unavailable."
        )

    event_type = request.event_type.upper()

    allowed_events = {
        "SEARCH",
        "VIEW",
        "LIKE",
        "SAVE",
        "CART",
        "PURCHASE",
        "RATING",
        "SKIP",
    }

    if event_type not in allowed_events:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported event type: {event_type}. "
                f"Allowed: {sorted(allowed_events)}"
            ),
        )

    try:

        result = interaction_engine.log_event(
            user_id=request.user_id,
            event_type=event_type,
            product_asin=request.product_asin,
            query=request.query,
            rating=request.rating,
            metadata=request.metadata,
        )

        return success_response({
            "event": result,
            "event_type": event_type,
            "user_id": request.user_id,
        })

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Interaction logging failed: {exc}",
        )


# ============================================================
# USER HISTORY
# ============================================================

@app.get("/api/users/{user_id}/history")
def get_user_history(user_id: str):

    if interaction_engine is None:

        raise HTTPException(
            status_code=503,
            detail="Interaction engine is unavailable."
        )

    try:

        history = interaction_engine.get_user_history(
            user_id
        )

        return success_response({
            "user_id": user_id,
            "history": history,
            "count": len(history),
        })

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"History retrieval failed: {exc}",
        )


# ============================================================
# API ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "ShopGraph API",
        "message": "AI-powered e-commerce product discovery backend",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }
