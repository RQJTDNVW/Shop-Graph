from typing import Optional, Any
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=50)


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    top_k: int = Field(default=10, ge=1, le=50)


class InteractionRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    event_type: str = Field(..., min_length=1, max_length=50)

    product_asin: Optional[str] = None
    query: Optional[str] = None
    rating: Optional[float] = Field(default=None, ge=1, le=5)
    metadata: Optional[dict[str, Any]] = None