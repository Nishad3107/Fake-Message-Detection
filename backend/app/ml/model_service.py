"""Cached inference service for the frozen PHEME final model."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np

from app.core.config import Settings
from app.ml.model_utils import detect_keywords
from app.ml.preprocess_text import PreprocessingOptions, preprocess_text

logger = logging.getLogger(__name__)
LABELS = ["false", "true", "unverified"]


class ModelArtifactError(RuntimeError):
    """Raised when the frozen inference artifact cannot be used."""


class FrozenModelService:
    model_name = "PHEME tuned Multinomial Naive Bayes"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.artifact: dict[str, Any] | None = None
        self.model: Any | None = None
        self.vectorizer: Any | None = None
        self.model_error: str | None = None
        self._load_once()

    def _load_once(self) -> None:
        path = self.settings.final_model_artifact
        if not path.is_file():
            self.model_error = (
                f"Final model artifact not found at {path}. "
                "Run the documented final evaluation workflow to regenerate it."
            )
            logger.error(self.model_error)
            return
        try:
            artifact = joblib.load(path)
            required = {"model", "vectorizer", "labels", "model_version"}
            missing = required - set(artifact)
            if missing:
                raise ModelArtifactError(
                    f"Final model artifact is missing fields: {', '.join(sorted(missing))}"
                )
            if list(artifact["labels"]) != LABELS:
                raise ModelArtifactError(
                    f"Unsupported artifact label order: {artifact['labels']}"
                )
            model = artifact["model"]
            vectorizer = artifact["vectorizer"]
            if not hasattr(model, "predict_proba") or not hasattr(
                vectorizer, "transform"
            ):
                raise ModelArtifactError(
                    "Final artifact does not contain a probability model and vectorizer"
                )
            # Force a small compatibility check during startup, not on a request.
            vector = vectorizer.transform(["model compatibility check"])
            model.predict_proba(vector)
            self.artifact = artifact
            self.model = model
            self.vectorizer = vectorizer
            self.model_error = None
        except Exception as exc:  # joblib may raise several pickle exceptions.
            self.model = None
            self.vectorizer = None
            self.artifact = None
            self.model_error = f"Unable to load final model artifact: {exc}"
            logger.exception(self.model_error)

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.vectorizer is not None

    def _require_ready(self) -> tuple[Any, Any, dict[str, Any]]:
        if not self.is_ready or self.artifact is None:
            raise ModelArtifactError(
                self.model_error or "Final model artifact is unavailable"
            )
        return self.model, self.vectorizer, self.artifact

    def _options(self) -> PreprocessingOptions:
        artifact = self.artifact or {}
        values = artifact.get("preprocessing_configuration", {})
        allowed = {
            key: bool(values[key])
            for key in PreprocessingOptions.__dataclass_fields__
            if key in values
        }
        return PreprocessingOptions(**allowed)

    def _confidence_category(self, confidence: float) -> str:
        if confidence < self.settings.low_confidence_threshold:
            return "low_confidence"
        if confidence < self.settings.high_confidence_threshold:
            return "moderate_confidence"
        return "high_confidence"

    def _important_terms(
        self, vector: Any, predicted_index: int, model: Any, vectorizer: Any
    ) -> list[str]:
        try:
            feature_names = vectorizer.get_feature_names_out()
            values = vector.toarray()[0]
            log_prob = np.asarray(model.feature_log_prob_)
            other = np.delete(log_prob, predicted_index, axis=0).mean(axis=0)
            contributions = values * (log_prob[predicted_index] - other)
            return [
                str(feature_names[index])
                for index in contributions.argsort()[::-1]
                if contributions[index] > 0
            ][:5]
        except (AttributeError, IndexError, TypeError, ValueError):
            return []

    def predict(self, text: str) -> dict[str, Any]:
        model, vectorizer, artifact = self._require_ready()
        cleaned = preprocess_text(text, self._options())
        if not cleaned:
            raise ValueError("Post is empty after preprocessing")
        vector = vectorizer.transform([cleaned])
        probabilities = np.asarray(model.predict_proba(vector)[0], dtype=float)
        classes = [str(value) for value in model.classes_]
        if classes != LABELS:
            raise ModelArtifactError(f"Unexpected model class order: {classes}")
        predicted_index = int(probabilities.argmax())
        prediction = classes[predicted_index]
        confidence = float(probabilities[predicted_index])
        category = self._confidence_category(confidence)
        terms = self._important_terms(vector, predicted_index, model, vectorizer)
        keywords = detect_keywords(text)
        explanation = (
            f"The frozen PHEME model associated this post most strongly with "
            f"the '{prediction}' class based on learned TF-IDF language patterns."
        )
        if terms:
            explanation += f" Influential terms included: {', '.join(terms)}."
        limitations = (
            "These terms are statistical associations, not evidence or causal proof. "
            "The model does not retrieve sources or verify factual truth."
        )
        warning = "This is a probabilistic NLP prediction, not factual verification."
        if category == "low_confidence":
            warning += " The model output is below the configured confidence threshold."
        elif category == "high_confidence":
            warning += " High model confidence is not evidence of factual certainty."
        if prediction == "unverified":
            warning += " The source label is unverified; seek independent evidence."
        config = self._config()
        return {
            "text": text,
            "cleaned_text": cleaned,
            "prediction": prediction,
            "classification": prediction,
            "display_label": f"Model prediction: {prediction}",
            "confidence": round(confidence, 6),
            "confidence_category": category,
            "class_probabilities": {
                label: round(float(probabilities[index]), 6)
                for index, label in enumerate(LABELS)
            },
            "detected_features": terms,
            "detected_keywords": keywords,
            "explanation": explanation,
            "explanation_details": {
                "summary": explanation,
                "important_terms": terms,
                "limitations": limitations,
            },
            "influential_terms": terms,
            "low_confidence": category == "low_confidence",
            "model_name": self.model_name,
            "model_version": str(artifact["model_version"]),
            "training_dataset_reference": artifact.get(
                "training_dataset_reference", "PHEME"
            ),
            "training_split_reference": config.get(
                "split_manifest_reference",
                "backend/data/splits/pheme_split_manifest.json",
            ),
            "model_limitations": limitations,
            "warning": warning,
            "timestamp": datetime.now(UTC),
        }

    def _config(self) -> dict[str, Any]:
        if not self.settings.final_model_config.is_file():
            return {}
        try:
            return json.loads(
                self.settings.final_model_config.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return {}

    def info(self) -> dict[str, Any]:
        config = self._config()
        final_metrics: dict[str, Any] = {}
        if self.settings.final_test_results.is_file():
            try:
                final_metrics = json.loads(
                    self.settings.final_test_results.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                logger.warning("Unable to read final test metrics")
        test_metrics = final_metrics.get("metrics", final_metrics)
        return {
            "name": self.model_name,
            "version": (
                str(self.artifact.get("model_version"))
                if self.artifact
                else "unavailable"
            ),
            "algorithm": "TF-IDF word n-grams + Multinomial Naive Bayes",
            "dataset": "PHEME Rumour Detection and Veracity Classification",
            "dataset_reference": (
                self.artifact.get("training_dataset_reference", "PHEME")
                if self.artifact
                else "PHEME"
            ),
            "labels": LABELS,
            "number_of_classes": len(LABELS),
            "tfidf_configuration": config.get("tfidf_configuration", {}),
            "model_hyperparameters": config.get("model_parameters", {}),
            "final_test_macro_f1": test_metrics.get("macro_f1"),
            "training_timestamp": (
                self.artifact.get("training_timestamp") if self.artifact else None
            ),
            "limitations": "Uncalibrated statistical language model evaluated on two held-out events; not factual verification.",
            "metrics": test_metrics,
            "ready": self.is_ready,
            "confidence_thresholds": {
                "low_below": self.settings.low_confidence_threshold,
                "high_at_or_above": self.settings.high_confidence_threshold,
            },
        }

    def metrics(self) -> dict[str, Any]:
        return self.info().get("metrics", {})
