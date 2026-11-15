"""FastAPI application entrypoint."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.db.database import init_db
from app.ml.model_service import FrozenModelService
from app.models.schemas import HealthResponse

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if (
            content_length
            and content_length.isdigit()
            and int(content_length) > settings.max_request_bytes
        ):
            return JSONResponse(
                status_code=413, content={"detail": "Request body is too large"}
            )
        return await call_next(request)


@asynccontextmanager
async def lifespan(application: FastAPI):
    service = FrozenModelService(settings)
    init_db(settings)
    application.state.prediction_service = service
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(RequestSizeLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


app.include_router(router, prefix=settings.api_v1_prefix)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed",
            "errors": [
                {
                    "loc": error.get("loc"),
                    "msg": error.get("msg"),
                    "type": error.get("type"),
                }
                for error in exc.errors()
            ],
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if settings.debug:
        logging.getLogger(__name__).exception("Unhandled request error", exc_info=exc)
    else:
        logging.getLogger(__name__).error(
            "Unhandled request error: %s", type(exc).__name__
        )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred"},
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service = getattr(app.state, "prediction_service", None)
    ready = bool(service and service.is_ready)
    info = service.info() if service else {}
    return HealthResponse(
        status="ok" if ready else "degraded",
        model_loaded=ready,
        model_version=info.get("version"),
        dataset=info.get("dataset"),
    )
