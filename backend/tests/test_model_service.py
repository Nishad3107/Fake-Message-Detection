from pathlib import Path

import pytest
from app.core.config import Settings
from app.ml.model_service import FrozenModelService

ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "models" / "final" / "final_pipeline.joblib"
CONFIG = ROOT / "data" / "reports" / "model_comparison" / "final_model_config.json"
METRICS = ROOT / "data" / "reports" / "model_comparison" / "final_test_results.json"


def _settings(artifact: Path = ARTIFACT) -> Settings:
    return Settings(
        final_model_artifact=artifact,
        final_model_config=CONFIG,
        final_test_results=METRICS,
    )


@pytest.mark.skipif(
    not ARTIFACT.is_file(), reason="local final artifact is unavailable"
)
def test_frozen_model_loads_and_returns_pheme_probabilities() -> None:
    service = FrozenModelService(_settings())
    result = service.predict("A public report describes the latest event and evidence.")
    assert service.is_ready
    assert result["prediction"] in {"false", "true", "unverified"}
    assert set(result["class_probabilities"]) == {"false", "true", "unverified"}
    assert sum(result["class_probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert result["confidence_category"] in {
        "low_confidence",
        "moderate_confidence",
        "high_confidence",
    }
    assert result["warning"]
    assert result["model_version"]


def test_missing_artifact_is_reported(tmp_path: Path) -> None:
    service = FrozenModelService(_settings(tmp_path / "missing.joblib"))
    assert not service.is_ready
    with pytest.raises(RuntimeError, match="artifact"):
        service.predict("This is a sufficiently long test message.")


def test_corrupt_artifact_is_reported(tmp_path: Path) -> None:
    artifact = tmp_path / "corrupt.joblib"
    artifact.write_bytes(b"not a joblib artifact")
    service = FrozenModelService(_settings(artifact))
    assert not service.is_ready
    assert service.model_error is not None


@pytest.mark.skipif(
    not ARTIFACT.is_file(), reason="local final artifact is unavailable"
)
def test_model_info_exposes_frozen_configuration() -> None:
    info = FrozenModelService(_settings()).info()
    assert info["ready"] is True
    assert info["labels"] == ["false", "true", "unverified"]
    assert info["tfidf_configuration"]["min_df"] == 3
    assert info["model_hyperparameters"]["alpha"] == 0.1
