"""Environment-backed application settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Misinformation Detection System"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"
    cors_origins: str = "http://localhost:5173,http://localhost:4173"
    debug: bool = False
    log_level: str = "INFO"
    model_dir: Path = Path(__file__).resolve().parents[2] / "models"
    final_model_artifact: Path = (
        Path(__file__).resolve().parents[2]
        / "models"
        / "final"
        / "final_pipeline.joblib"
    )
    final_model_config: Path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "reports"
        / "model_comparison"
        / "final_model_config.json"
    )
    final_test_results: Path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "reports"
        / "model_comparison"
        / "final_test_results.json"
    )
    dataset_path: Path = (
        Path(__file__).resolve().parents[2] / "data" / "sample_dataset.csv"
    )
    database_path: Path = Path(__file__).resolve().parents[2] / "data" / "history.db"
    min_post_length: int = 10
    max_post_length: int = 5000
    low_confidence_threshold: float = 0.6
    high_confidence_threshold: float = 0.8
    max_batch_size: int = 50
    history_result_limit: int = 50
    max_request_bytes: int = 1_000_000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def resolve_backend_paths(self) -> "Settings":
        if self.max_post_length < self.min_post_length:
            raise ValueError("max_post_length must be greater than min_post_length")
        if not 1 <= self.max_batch_size <= 50:
            raise ValueError("max_batch_size must be between 1 and 50")
        if not 1 <= self.history_result_limit <= 100:
            raise ValueError("history_result_limit must be between 1 and 100")
        if self.max_request_bytes < 1024:
            raise ValueError("max_request_bytes must be at least 1024")
        if self.environment.lower() == "production":
            origins = self.cors_origin_list
            if not origins or "*" in origins:
                raise ValueError(
                    "Production requires explicit CORS_ORIGINS; wildcard origins are not allowed"
                )
        backend_root = Path(__file__).resolve().parents[2]
        if not self.model_dir.is_absolute():
            self.model_dir = backend_root / self.model_dir
        if not self.final_model_artifact.is_absolute():
            self.final_model_artifact = backend_root / self.final_model_artifact
        if not self.final_model_config.is_absolute():
            self.final_model_config = backend_root / self.final_model_config
        if not self.final_test_results.is_absolute():
            self.final_test_results = backend_root / self.final_test_results
        if not self.dataset_path.is_absolute():
            self.dataset_path = backend_root / self.dataset_path
        if not self.database_path.is_absolute():
            self.database_path = backend_root / self.database_path
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [
            origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
