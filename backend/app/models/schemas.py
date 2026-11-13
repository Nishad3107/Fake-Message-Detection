"""Pydantic request and response models."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.core.config import get_settings


class PredictionRequest(BaseModel):
    text: str = Field(..., description="Social media post to analyze")

    @field_validator("text")
    @classmethod
    def text_must_contain_content(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        settings = get_settings()
        if not cleaned:
            raise ValueError("Post cannot be empty")
        if len(cleaned) < settings.min_post_length:
            raise ValueError(
                f"Post must contain at least {settings.min_post_length} non-whitespace characters"
            )
        if len(cleaned) > settings.max_post_length:
            raise ValueError(
                f"Post must not exceed {settings.max_post_length} characters"
            )
        return cleaned


class PredictionResponse(BaseModel):
    id: int | None = None
    text: str | None = None
    classification: Literal["misinformation", "likely_true"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    detected_keywords: list[str]
    explanation: str
    influential_terms: list[str] = []
    low_confidence: bool = False
    model_name: str
    model_version: str
    timestamp: datetime
    disclaimer: str = (
        "This is a probabilistic model prediction, not factual verification."
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class BatchPredictionRequest(BaseModel):
    posts: list[PredictionRequest] = Field(..., min_length=1, max_length=50)


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    algorithm: str
    dataset: str
    metrics: dict[str, object]
    ready: bool


class HistoryResponse(BaseModel):
    items: list[PredictionResponse]
    summary: dict[str, object]


class MetricsResponse(BaseModel):
    metrics: dict[str, object]
