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
    cleaned_text: str | None = None
    prediction: Literal["true", "false", "unverified"]
    classification: Literal["true", "false", "unverified"]
    display_label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_category: Literal[
        "low_confidence", "moderate_confidence", "high_confidence"
    ]
    class_probabilities: dict[str, float]
    detected_features: list[str] = []
    detected_keywords: list[str]
    explanation: str
    explanation_details: dict[str, object] = {}
    influential_terms: list[str] = []
    low_confidence: bool = False
    model_name: str
    model_version: str
    training_dataset_reference: str | list[str] | None = None
    training_split_reference: str | None = None
    model_limitations: str
    timestamp: datetime
    warning: str
    disclaimer: str = (
        "This is a probabilistic model prediction, not factual verification."
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    dataset: str | None = None


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
    dataset_reference: str | list[str] | None = None
    labels: list[str] = []
    number_of_classes: int = 0
    tfidf_configuration: dict[str, object] = {}
    model_hyperparameters: dict[str, object] = {}
    final_test_macro_f1: float | None = None
    training_timestamp: str | None = None
    limitations: str = ""
    confidence_thresholds: dict[str, float] = {}


class HistoryResponse(BaseModel):
    items: list[PredictionResponse]
    summary: dict[str, object]


class MetricsResponse(BaseModel):
    metrics: dict[str, object]
