"""Versioned API routes."""

from fastapi import APIRouter, HTTPException, Query, Request

from app.core.config import get_settings
from app.db.database import history_summary, list_analyses, save_analysis
from app.ml.model_service import FrozenModelService
from app.models.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HistoryResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable: service is not initialized",
        )
    try:
        result = save_analysis(get_settings(), service.predict(payload.text))
        return PredictionResponse(**result)
    except (RuntimeError, ValueError):
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable or could not process the request",
        ) from None


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    payload: BatchPredictionRequest, request: Request
) -> BatchPredictionResponse:
    service = getattr(request.app.state, "prediction_service", None)
    if service is None or not service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable",
        )
    try:
        results = [
            save_analysis(get_settings(), service.predict(item.text))
            for item in payload.posts
        ]
        return BatchPredictionResponse(
            results=[PredictionResponse(**item) for item in results]
        )
    except (RuntimeError, ValueError):
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable or could not process the request",
        ) from None


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        return ModelInfoResponse(
            name=FrozenModelService.model_name,
            version="unavailable",
            algorithm="TF-IDF word n-grams + Multinomial Naive Bayes",
            dataset="PHEME",
            metrics={},
            ready=False,
        )
    return ModelInfoResponse(**service.info())


@router.get("/history", response_model=HistoryResponse)
def history(limit: int = Query(50, ge=1, le=100)) -> HistoryResponse:
    settings = get_settings()
    limit = min(limit, settings.history_result_limit)
    return HistoryResponse(
        items=[PredictionResponse(**item) for item in list_analyses(settings, limit)],
        summary=history_summary(settings),
    )


@router.get("/metrics", response_model=MetricsResponse)
def metrics(request: Request) -> MetricsResponse:
    service = getattr(request.app.state, "prediction_service", None)
    return MetricsResponse(metrics=service.metrics() if service else {})
