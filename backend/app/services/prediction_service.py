"""Prediction orchestration and automatic first-run model training."""

import json
import logging
from datetime import UTC, datetime

from app.core.config import Settings
from app.ml.model_utils import detect_keywords
from app.ml.predict import ModelPredictor
from app.ml.train import train_model

logger = logging.getLogger(__name__)


class PredictionService:
    model_name = "TF-IDF Logistic Regression Baseline"
    model_version = "0.1.0"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.predictor: ModelPredictor | None = None
        self.model_error: str | None = None

    def load_or_train(self) -> None:
        try:
            classifier = self.settings.model_dir / "classifier.joblib"
            vectorizer = self.settings.model_dir / "vectorizer.joblib"
            if not classifier.exists() or not vectorizer.exists():
                train_model(self.settings.dataset_path, self.settings.model_dir)
            self.predictor = ModelPredictor(self.settings.model_dir)
            self.model_error = None
        except (
            OSError,
            ValueError,
            RuntimeError,
            EOFError,
            ImportError,
            AttributeError,
        ) as first_error:
            logger.warning(
                "Model artifacts unavailable; attempting retraining: %s", first_error
            )
            try:
                train_model(self.settings.dataset_path, self.settings.model_dir)
                self.predictor = ModelPredictor(self.settings.model_dir)
                self.model_error = None
            except (
                OSError,
                ValueError,
                RuntimeError,
                EOFError,
                ImportError,
                AttributeError,
            ) as final_error:
                self.predictor = None
                self.model_error = str(final_error)
                logger.exception("Unable to load or train prediction model")

    @property
    def is_ready(self) -> bool:
        return self.predictor is not None

    def predict(self, text: str) -> dict[str, object]:
        if self.predictor is None:
            detail = self.model_error or "Prediction model is not loaded"
            raise RuntimeError(f"Prediction model is unavailable: {detail}")
        label, confidence, influential_terms = self.predictor.predict(text)
        keywords = detect_keywords(text)
        explanation = (
            "The model found language patterns associated with misinformation in the training data."
            if label == "misinformation"
            else "The model found language patterns that more closely resemble likely reliable examples in the training data."
        )
        if keywords:
            explanation += f" Detected terms: {', '.join(keywords)}."
        return {
            "text": text,
            "classification": label,
            "confidence": round(confidence, 4),
            "detected_keywords": keywords,
            "explanation": explanation,
            "influential_terms": influential_terms,
            "low_confidence": confidence < 0.65,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "timestamp": datetime.now(UTC),
        }

    def metrics(self) -> dict[str, object]:
        metrics_path = self.settings.model_dir / "metrics.json"
        if not metrics_path.exists():
            return {}
        return json.loads(metrics_path.read_text(encoding="utf-8"))
