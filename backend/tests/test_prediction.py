from pathlib import Path

from app.core.config import Settings
from app.ml.train import train_model
from app.services.prediction_service import PredictionService


def test_trained_model_predicts_probability(tmp_path: Path) -> None:
    data = Path(__file__).parents[1] / "data" / "sample_dataset.csv"
    train_model(data, tmp_path)
    service = PredictionService(Settings(model_dir=tmp_path, dataset_path=data))
    service.load_or_train()
    result = service.predict("Breaking shocking secret claim share this now")
    assert result["classification"] in {"misinformation", "likely_true"}
    assert 0 <= result["confidence"] <= 1
