"""Train and compare classical NLP baselines without test-set selection."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

DATA_DIR = Path("backend/data/processed/splits")
MANIFEST_PATH = Path("backend/data/splits/pheme_split_manifest.json")
REPORT_DIR = Path("backend/data/reports/model_comparison")
MODEL_DIR = Path("backend/models/baselines")
RANDOM_SEED = 42
LABELS = ["false", "true", "unverified"]
REQUIRED_COLUMNS = {
    "text_clean",
    "label",
    "event",
    "thread_id",
    "source_tweet_id",
    "split",
}


@dataclass(frozen=True)
class TfidfConfig:
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    sublinear_tf: bool = True
    lowercase: bool = False
    strip_accents: str = "unicode"


def load_split(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Split {path} is missing required columns: {', '.join(sorted(missing))}"
        )
    if frame.empty:
        raise ValueError(f"Split is empty: {path}")
    return frame


def load_splits(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    return {
        split: load_split(data_dir / f"{split}.csv")
        for split in ("train", "validation", "test")
    }


def validate_split_boundaries(splits: dict[str, pd.DataFrame]) -> None:
    event_sets = {split: set(frame["event"]) for split, frame in splits.items()}
    thread_sets = {split: set(frame["thread_id"]) for split, frame in splits.items()}
    event_overlap = {
        f"{left}:{right}": sorted(event_sets[left] & event_sets[right])
        for left in event_sets
        for right in event_sets
        if left < right and event_sets[left] & event_sets[right]
    }
    thread_overlap = {
        f"{left}:{right}": sorted(thread_sets[left] & thread_sets[right])
        for left in thread_sets
        for right in thread_sets
        if left < right and thread_sets[left] & thread_sets[right]
    }
    if event_overlap or thread_overlap:
        raise ValueError(
            f"Split leakage detected: events={event_overlap}, threads={thread_overlap}"
        )


def fit_training_vectorizer(
    train: pd.DataFrame, config: TfidfConfig | None = None
) -> TfidfVectorizer:
    settings = config or TfidfConfig()
    vectorizer = TfidfVectorizer(**asdict(settings))
    vectorizer.fit(train["text_clean"].fillna("").astype(str))
    return vectorizer


def _metric_summary(y_true: pd.Series, predictions: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=LABELS, zero_division=0
    )
    report = classification_report(
        y_true,
        predictions,
        labels=LABELS,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 6),
        "macro_precision": round(float(np.mean(precision)), 6),
        "macro_recall": round(float(np.mean(recall)), 6),
        "macro_f1": round(float(np.mean(f1)), 6),
        "weighted_precision": round(float(report["weighted avg"]["precision"]), 6),
        "weighted_recall": round(float(report["weighted avg"]["recall"]), 6),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 6),
        "per_class": {
            label: {
                "precision": round(float(precision[index]), 6),
                "recall": round(float(recall[index]), 6),
                "f1": round(float(f1[index]), 6),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=LABELS
        ).tolist(),
    }


def _model_specs() -> dict[str, Any]:
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "multinomial_naive_bayes": MultinomialNB(alpha=1.0),
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        ),
        "linear_svc": LinearSVC(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED
        ),
    }


def train_baselines(
    data_dir: Path = DATA_DIR,
    manifest_path: Path = MANIFEST_PATH,
    report_dir: Path = REPORT_DIR,
    model_dir: Path = MODEL_DIR,
    vectorizer_config: TfidfConfig | None = None,
) -> dict[str, Any]:
    splits = load_splits(data_dir)
    validate_split_boundaries(splits)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vectorizer = fit_training_vectorizer(splits["train"], vectorizer_config)
    train_x = vectorizer.transform(splits["train"]["text_clean"].fillna("").astype(str))
    validation_x = vectorizer.transform(
        splits["validation"]["text_clean"].fillna("").astype(str)
    )
    test_x = vectorizer.transform(splits["test"]["text_clean"].fillna("").astype(str))
    train_y = splits["train"]["label"].astype(str)
    validation_y = splits["validation"]["label"].astype(str)

    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.joblib")

    results: list[dict[str, Any]] = []
    model_details: dict[str, Any] = {}
    for name, model in _model_specs().items():
        started = time.perf_counter()
        training_started_at = datetime.now(UTC).isoformat()
        model.fit(train_x, train_y)
        training_duration = time.perf_counter() - started
        training_ended_at = datetime.now(UTC).isoformat()
        prediction_started = time.perf_counter()
        predictions = model.predict(validation_x)
        validation_prediction_duration = time.perf_counter() - prediction_started
        metrics = _metric_summary(validation_y, predictions)
        confidence_type = (
            "probabilities" if hasattr(model, "predict_proba") else "decision_margins"
        )
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(validation_x)
            confidence_summary = {
                "type": "probabilities",
                "mean_max_probability": round(
                    float(probabilities.max(axis=1).mean()), 6
                ),
            }
        else:
            decision_scores = model.decision_function(validation_x)
            confidence_summary = {
                "type": "decision_margins",
                "mean_max_decision_score": round(
                    float(np.max(decision_scores, axis=1).mean()), 6
                ),
                "warning": "LinearSVC decision margins are not calibrated probabilities.",
            }
        joblib.dump(model, model_dir / f"{name}.joblib")
        (report_dir / f"{name}_confusion_matrix.json").write_text(
            json.dumps(
                {"labels": LABELS, "matrix": metrics["confusion_matrix"]}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        detail = {
            "model": name,
            "configuration": model.get_params(),
            "training_started_at": training_started_at,
            "training_ended_at": training_ended_at,
            "training_duration_seconds": round(training_duration, 6),
            "validation_prediction_duration_seconds": round(
                validation_prediction_duration, 6
            ),
            "training_samples": len(train_y),
            "validation_samples": len(validation_y),
            "test_samples_transformed_only": int(test_x.shape[0]),
            "features": int(train_x.shape[1]),
            "confidence": confidence_summary,
            "confidence_output_type": confidence_type,
            "validation_metrics": metrics,
        }
        model_details[name] = detail
        results.append(
            {
                "model": name,
                **{
                    metric: metrics[metric]
                    for metric in (
                        "accuracy",
                        "macro_precision",
                        "macro_recall",
                        "macro_f1",
                        "weighted_f1",
                    )
                },
                "training_duration_seconds": detail["training_duration_seconds"],
                "validation_prediction_duration_seconds": detail[
                    "validation_prediction_duration_seconds"
                ],
                "features": detail["features"],
                "confidence_output_type": confidence_type,
            }
        )

    report: dict[str, Any] = {
        "dataset_source": "real PHEME processed split files",
        "dataset_path": str(data_dir),
        "split_manifest_reference": str(manifest_path),
        "split_manifest_seed": manifest.get("random_seed"),
        "dataset_sizes": {split: len(frame) for split, frame in splits.items()},
        "label_distributions": {
            split: {
                str(label): int(count)
                for label, count in frame["label"].value_counts().to_dict().items()
            }
            for split, frame in splits.items()
        },
        "tfidf_configuration": asdict(vectorizer_config or TfidfConfig()),
        "feature_count": int(train_x.shape[1]),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "random_seed": RANDOM_SEED,
        "test_policy": "Test text was transformed for shape verification only; no test predictions or metrics were used for model selection.",
        "models": model_details,
        "results": results,
        "best_validation_model": max(results, key=lambda row: row["macro_f1"])["model"],
        "warnings": [
            "Validation metrics are not final generalization results.",
            "One fixed event-level split is used; grouped cross-validation may provide a more robust comparison.",
            "LinearSVC decision margins are not probabilities and are not calibrated confidence.",
        ],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
        },
    }
    (report_dir / "baseline_results.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    with (report_dir / "baseline_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    report = train_baselines(
        args.data_dir, args.manifest, args.report_dir, args.model_dir
    )
    print(
        json.dumps(
            {
                "best_validation_model": report["best_validation_model"],
                "results": report["results"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
