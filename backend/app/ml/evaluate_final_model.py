"""Freeze, fit, and evaluate the final candidate once on held-out PHEME test data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB

from app.ml.preprocess_text import (
    PreprocessingOptions,
    options_as_dict,
    preprocess_text,
)

TRAIN_PATH = Path("backend/data/processed/splits/train.csv")
VALIDATION_PATH = Path("backend/data/processed/splits/validation.csv")
TEST_PATH = Path("backend/data/processed/splits/test.csv")
TUNING_REPORT = Path("backend/data/reports/model_comparison/tuning_results.json")
TUNING_SUMMARY = Path("backend/data/reports/model_comparison/tuning_summary.json")
PREPROCESSING_REPORT = Path("backend/data/reports/preprocessing_report.json")
MANIFEST_PATH = Path("backend/data/splits/pheme_split_manifest.json")
REPORT_DIR = Path("backend/data/reports/model_comparison")
MODEL_DIR = Path("backend/models/final")
CONFIG_PATH = REPORT_DIR / "final_model_config.json"
ARTIFACT_PATH = MODEL_DIR / "final_pipeline.joblib"
LABELS = ["false", "true", "unverified"]
REQUIRED_COLUMNS = {
    "text_original",
    "text_clean",
    "label",
    "event",
    "thread_id",
    "source_tweet_id",
    "split",
}
LOW_CONFIDENCE_THRESHOLD = 0.6


def _load_split(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required split file not found: {path}")
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Split {path} is missing required columns: {', '.join(sorted(missing))}"
        )
    if frame.empty:
        raise ValueError(f"Split is empty: {path}")
    return frame


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_final_artifact(path: Path = ARTIFACT_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Final model artifact not found: {path}")
    artifact = joblib.load(path)
    required = {
        "model",
        "vectorizer",
        "preprocessing_configuration",
        "labels",
        "model_version",
    }
    missing = required - set(artifact)
    if missing:
        raise ValueError(
            f"Final model artifact is missing fields: {', '.join(sorted(missing))}"
        )
    return artifact


def freeze_final_config(
    config_path: Path = CONFIG_PATH,
    tuning_report_path: Path = TUNING_REPORT,
    tuning_summary_path: Path = TUNING_SUMMARY,
    preprocessing_report_path: Path = PREPROCESSING_REPORT,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    tuning = _load_json(tuning_report_path)
    summary = _load_json(tuning_summary_path)
    preprocessing = _load_json(preprocessing_report_path)
    manifest = _load_json(manifest_path)
    selected = summary.get("selected_by_family", {}).get("multinomial_naive_bayes")
    if not selected:
        raise ValueError(
            "Tuning summary does not contain the selected Multinomial Naive Bayes configuration"
        )
    expected_model = {
        "alpha": 0.1,
        "fit_prior": False,
    }
    if selected["configuration"]["model"] != expected_model:
        raise ValueError(
            "Selected Naive Bayes configuration differs from the frozen candidate"
        )
    expected_tfidf = {
        "ngram_range": [1, 2],
        "min_df": 3,
        "sublinear_tf": False,
        "max_df": 0.95,
        "lowercase": False,
        "strip_accents": "unicode",
    }
    if selected["configuration"]["tfidf"] != expected_tfidf:
        raise ValueError(
            "Selected TF-IDF configuration differs from the frozen candidate"
        )
    config = {
        "model_family": "multinomial_naive_bayes",
        "model_parameters": expected_model,
        "tfidf_configuration": expected_tfidf,
        "preprocessing_configuration": preprocessing["preprocessing_options"],
        "labels": LABELS,
        "training_data": [str(TRAIN_PATH), str(VALIDATION_PATH)],
        "test_data": str(TEST_PATH),
        "split_manifest_reference": str(manifest_path),
        "split_manifest_seed": manifest.get("random_seed"),
        "random_seed": 42,
        "selected_by": "Grouped CV mean macro F1 on training events; test metrics were not used.",
        "tuning_validation_macro_f1": selected["validation_macro_f1"],
        "tuning_cv_macro_f1_mean": selected["mean_macro_f1"],
        "tuning_cv_macro_f1_std": selected["std_macro_f1"],
        "test_selection_used": False,
        "frozen_at": datetime.now(UTC).isoformat(),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
        },
        "source_tuning_report": str(tuning_report_path),
        "tuning_configurations_tested": tuning["configurations_tested"],
    }
    if config_path.is_file():
        existing = _load_json(config_path)
        frozen_keys = [
            "model_family",
            "model_parameters",
            "tfidf_configuration",
            "preprocessing_configuration",
            "labels",
            "training_data",
            "test_data",
            "split_manifest_reference",
            "random_seed",
            "test_selection_used",
        ]
        if any(existing.get(key) != config.get(key) for key in frozen_keys):
            raise RuntimeError(
                "Existing final_model_config.json differs from the frozen candidate"
            )
        return existing
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def _metrics(
    y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
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
    result: dict[str, Any] = {
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
    try:
        result["log_loss"] = round(
            float(log_loss(y_true, probabilities, labels=LABELS)), 6
        )
        result["roc_auc_ovr_macro"] = round(
            float(
                roc_auc_score(
                    y_true,
                    probabilities,
                    labels=LABELS,
                    multi_class="ovr",
                    average="macro",
                )
            ),
            6,
        )
        result["pr_auc_ovr_macro"] = round(
            float(
                average_precision_score(
                    pd.get_dummies(y_true).reindex(columns=LABELS, fill_value=0),
                    probabilities,
                    average="macro",
                )
            ),
            6,
        )
    except ValueError as exc:
        result["log_loss"] = None
        result["roc_auc_ovr_macro"] = None
        result["pr_auc_ovr_macro"] = None
        result["probability_metric_unavailable_reason"] = str(exc)
    one_hot = (
        pd.get_dummies(y_true)
        .reindex(columns=LABELS, fill_value=0)
        .to_numpy(dtype=float)
    )
    result["multiclass_brier_score"] = round(
        float(np.mean((probabilities - one_hot) ** 2)), 6
    )
    return result


def _calibration(
    probabilities: np.ndarray, predictions: np.ndarray, y_true: pd.Series
) -> dict[str, Any]:
    confidence = probabilities.max(axis=1)
    correct = (predictions == y_true.to_numpy()).astype(float)
    bins = np.linspace(0.0, 1.0, 11)
    entries = []
    weighted_error = 0.0
    for index in range(len(bins) - 1):
        lower, upper = bins[index], bins[index + 1]
        mask = (confidence >= lower) & (
            confidence < upper if upper < 1 else confidence <= upper
        )
        if not mask.any():
            continue
        accuracy = float(correct[mask].mean())
        mean_confidence = float(confidence[mask].mean())
        weighted_error += float(mask.mean()) * abs(mean_confidence - accuracy)
        entries.append(
            {
                "lower": lower,
                "upper": upper,
                "count": int(mask.sum()),
                "mean_confidence": round(mean_confidence, 6),
                "accuracy": round(accuracy, 6),
            }
        )
    return {
        "confidence_min": round(float(confidence.min()), 6),
        "confidence_max": round(float(confidence.max()), 6),
        "confidence_mean": round(float(confidence.mean()), 6),
        "confidence_median": round(float(np.median(confidence)), 6),
        "confidence_standard_deviation": round(float(confidence.std(ddof=1)), 6),
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "low_confidence_count": int((confidence < LOW_CONFIDENCE_THRESHOLD).sum()),
        "expected_calibration_error": round(weighted_error, 6),
        "reliability_bins": entries,
        "calibration_warning": "Calibration metrics are descriptive on one held-out event split and do not establish reliable probability calibration.",
    }


def _event_metrics(
    test: pd.DataFrame, predictions: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for event, indices in test.groupby("event").groups.items():
        positions = list(indices)
        y_true = test.loc[positions, "label"].astype(str)
        event_predictions = predictions[positions]
        present_labels = sorted(set(y_true))
        event_report = classification_report(
            y_true,
            event_predictions,
            labels=LABELS,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        )
        output[str(event)] = {
            "record_count": len(positions),
            "label_distribution": {
                str(label): int(count)
                for label, count in y_true.value_counts().to_dict().items()
            },
            "labels_present": present_labels,
            "accuracy": round(float(accuracy_score(y_true, event_predictions)), 6),
            "macro_f1": (
                round(
                    float(f1_score_present(y_true, event_predictions, present_labels)),
                    6,
                )
                if set(present_labels) == set(LABELS)
                else None
            ),
            "macro_f1_unavailable_reason": (
                None
                if set(present_labels) == set(LABELS)
                else "Not all three labels are present in this held-out event."
            ),
            "per_class": {
                label: {
                    "precision": round(float(event_report[label]["precision"]), 6),
                    "recall": round(float(event_report[label]["recall"]), 6),
                    "f1": round(float(event_report[label]["f1-score"]), 6),
                    "support": int(event_report[label]["support"]),
                }
                for label in LABELS
                if label in present_labels
            },
            "confusion_matrix": confusion_matrix(
                y_true, event_predictions, labels=LABELS
            ).tolist(),
        }
    return output


def f1_score_present(
    y_true: pd.Series, predictions: np.ndarray, labels: list[str]
) -> float:
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, predictions, labels=labels, zero_division=0
    )
    return float(np.mean(f1))


def _write_reliability_data(calibration: dict[str, Any], path: Path) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path.parent.mkdir(parents=True, exist_ok=True)
        bins = calibration["reliability_bins"]
        plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
        if bins:
            plt.plot(
                [item["mean_confidence"] for item in bins],
                [item["accuracy"] for item in bins],
                "o-",
                label="Test set",
            )
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed accuracy")
        plt.title("Final test reliability — descriptive only")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
    except ImportError:
        return


def evaluate_final(
    config_path: Path = CONFIG_PATH,
    artifact_path: Path = ARTIFACT_PATH,
    report_dir: Path = REPORT_DIR,
    model_dir: Path = MODEL_DIR,
    train_path: Path = TRAIN_PATH,
    validation_path: Path = VALIDATION_PATH,
    test_path: Path = TEST_PATH,
    baseline_report_path: Path = REPORT_DIR / "baseline_results.json",
    tuning_report_path: Path = TUNING_REPORT,
) -> dict[str, Any]:
    config = _load_json(config_path)
    if (
        config["model_family"] != "multinomial_naive_bayes"
        or config["test_selection_used"]
    ):
        raise ValueError(
            "Final configuration is not the frozen tuned Multinomial Naive Bayes candidate"
        )
    train = _load_split(train_path)
    validation = _load_split(validation_path)
    test = _load_split(test_path)
    if (
        set(train["event"]) & set(validation["event"])
        or set(train["event"]) & set(test["event"])
        or set(validation["event"]) & set(test["event"])
    ):
        raise ValueError("Event overlap detected across final fitting and test splits")
    if (
        set(train["thread_id"]) & set(validation["thread_id"])
        or set(train["thread_id"]) & set(test["thread_id"])
        or set(validation["thread_id"]) & set(test["thread_id"])
    ):
        raise ValueError("Thread overlap detected across final fitting and test splits")

    fitting = pd.concat([train, validation], ignore_index=True)
    options = PreprocessingOptions(**config["preprocessing_configuration"])
    fitting_text = fitting["text_original"].map(
        lambda text: preprocess_text(text, options)
    )
    test_text = test["text_original"].map(lambda text: preprocess_text(text, options))
    tfidf_configuration = dict(config["tfidf_configuration"])
    tfidf_configuration["ngram_range"] = tuple(tfidf_configuration["ngram_range"])
    vectorizer = TfidfVectorizer(**tfidf_configuration)
    train_x = vectorizer.fit_transform(fitting_text)
    test_x = vectorizer.transform(test_text)
    model = MultinomialNB(**config["model_parameters"])
    training_started = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    model.fit(train_x, fitting["label"].astype(str))
    training_duration = time.perf_counter() - started
    training_ended = datetime.now(UTC).isoformat()
    probabilities = model.predict_proba(test_x)
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    y_true = test["label"].astype(str)
    metrics = _metrics(y_true, predictions, probabilities)
    calibration = _calibration(probabilities, predictions, y_true)
    event_metrics = _event_metrics(
        test.reset_index(drop=True), predictions, probabilities
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "vectorizer": vectorizer,
        "preprocessing_configuration": options_as_dict(options),
        "labels": LABELS,
        "model_version": "pheme-tuned-multinomial-naive-bayes-v1",
        "training_dataset_reference": [str(train_path), str(validation_path)],
        "training_timestamp": training_ended,
        "training_duration_seconds": round(training_duration, 6),
        "configuration_reference": str(config_path),
    }
    joblib.dump(artifact, artifact_path)
    predictions_frame = test[
        ["event", "thread_id", "source_tweet_id", "label", "text_original"]
    ].copy()
    predictions_frame["prediction"] = predictions
    predictions_frame["confidence"] = probabilities.max(axis=1)
    for index, label in enumerate(LABELS):
        predictions_frame[f"probability_{label}"] = probabilities[:, index]
    predictions_frame.to_csv(report_dir / "final_predictions.csv", index=False)
    (report_dir / "final_classification_report.json").write_text(
        json.dumps(metrics["classification_report"], indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "final_confusion_matrix.json").write_text(
        json.dumps({"labels": LABELS, "matrix": metrics["confusion_matrix"]}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "event_wise_test_results.json").write_text(
        json.dumps(event_metrics, indent=2) + "\n", encoding="utf-8"
    )
    _write_reliability_data(calibration, report_dir / "final_reliability_diagram.png")
    baseline = _load_json(baseline_report_path)
    tuning = _load_json(tuning_report_path)
    result = {
        "model": "tuned_multinomial_naive_bayes",
        "artifact": str(artifact_path),
        "configuration_reference": str(config_path),
        "fit_data": {
            "train_records": len(train),
            "validation_records": len(validation),
            "combined_records": len(fitting),
        },
        "test_data": {
            "path": str(test_path),
            "records": len(test),
            "events": sorted(test["event"].unique()),
        },
        "training_started_at": training_started,
        "training_ended_at": training_ended,
        "training_duration_seconds": round(training_duration, 6),
        "feature_count": int(train_x.shape[1]),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "metrics": metrics,
        "confidence": calibration,
        "event_wise": event_metrics,
        "comparison": {
            "majority_baseline_validation_macro_f1": next(
                row["macro_f1"]
                for row in baseline["results"]
                if row["model"] == "majority_baseline"
            ),
            "original_logistic_regression_validation_macro_f1": next(
                row["macro_f1"]
                for row in baseline["results"]
                if row["model"] == "logistic_regression"
            ),
            "tuned_multinomial_naive_bayes_validation_macro_f1": tuning[
                "validation_results"
            ]["multinomial_naive_bayes"]["validation_metrics"]["macro_f1"],
            "tuned_multinomial_naive_bayes_cv_macro_f1_mean": tuning[
                "selected_by_family"
            ]["multinomial_naive_bayes"]["mean_macro_f1"],
            "tuned_multinomial_naive_bayes_cv_macro_f1_std": tuning[
                "selected_by_family"
            ]["multinomial_naive_bayes"]["std_macro_f1"],
            "final_test_macro_f1": metrics["macro_f1"],
        },
        "test_selection_used": False,
        "warnings": [
            "This is a one-time final test evaluation; do not retune from these results.",
            "Probabilities are model outputs and are not factual truth probabilities.",
            "Calibration statistics are descriptive for this single held-out event split.",
        ],
    }
    (report_dir / "final_test_results.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    csv_row = {
        "model": result["model"],
        **{
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        },
        "training_duration_seconds": result["training_duration_seconds"],
        "feature_count": result["feature_count"],
        "vocabulary_size": result["vocabulary_size"],
    }
    with (report_dir / "final_test_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_row))
        writer.writeheader()
        writer.writerow(csv_row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--validation", type=Path, default=VALIDATION_PATH)
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=REPORT_DIR / "baseline_results.json",
    )
    parser.add_argument("--tuning-report", type=Path, default=TUNING_REPORT)
    args = parser.parse_args()
    if not args.config.is_file():
        freeze_final_config(
            args.config,
            args.tuning_report,
            args.report_dir / "tuning_summary.json",
            args.report_dir.parent / "preprocessing_report.json",
            args.report_dir.parent.parent / "splits" / "pheme_split_manifest.json",
        )
    result = evaluate_final(
        args.config,
        args.artifact,
        args.report_dir,
        args.model_dir,
        args.train,
        args.validation,
        args.test,
        args.baseline_report,
        args.tuning_report,
    )
    print(
        json.dumps(
            {
                "metrics": result["metrics"],
                "confidence": result["confidence"],
                "event_wise": result["event_wise"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
