"""FastAPI application entrypoint."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.db.database import init_db
from app.models.schemas import HealthResponse
from app.services.prediction_service import PredictionService

settings = get_settings()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)


@asynccontextmanager
async def lifespan(application: FastAPI):
    service = PredictionService(settings)
    service.load_or_train()
    init_db(settings)
    application.state.prediction_service = service
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix=settings.api_v1_prefix)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service = getattr(app.state, "prediction_service", None)
    ready = bool(service and service.is_ready)
    return HealthResponse(status="ok" if ready else "degraded", model_loaded=ready)
