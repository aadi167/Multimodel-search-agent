from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator


class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, json_schema_extra={"example": "a red sports car at sunset"})
    top_k: int = Field(default=5, ge=1, le=20)
    include_base64: bool = Field(default=False)

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "a dog playing in snow",
                "top_k": 5,
                "include_base64": False,
            }
        }
    }


class HybridSearchRequest(BaseModel):
    text_query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    text_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    include_base64: bool = Field(default=False)


class SearchResult(BaseModel):
    rank: int
    path: str
    filename: str
    score: float
    image_base64: Optional[str] = None
    image_format: Optional[str] = None


class SearchResponse(BaseModel):
    query: Optional[str] = None
    query_type: str
    results: List[SearchResult]
    total_results: int
    encode_ms: float
    search_ms: float


class ClassifyResponse(BaseModel):
    scores: Dict[str, float]
    top_label: str
    confidence: float
    encode_ms: float


class IndexStatsResponse(BaseModel):
    status: str
    total_images: Optional[int] = None
    embedding_dim: Optional[int] = None
    index_type: Optional[str] = None
    model: Optional[str] = None
    device: Optional[str] = None
