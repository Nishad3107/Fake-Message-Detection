"""Analyze errors and feature associations from the frozen final model outputs."""

from __future__ import annotations

import argparse
import json
import os
import string
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from app.ml.preprocess_text import (
    HASHTAG_PATTERN,
    MENTION_PATTERN,
    URL_PATTERN,
    PreprocessingOptions,
    preprocess_text,
)

TEST_PATH = Path("backend/data/processed/splits/test.csv")
PREDICTIONS_PATH = Path("backend/data/reports/model_comparison/final_predictions.csv")
ARTIFACT_PATH = Path("backend/models/final/final_pipeline.joblib")
REPORT_DIR = Path("backend/data/reports/model_comparison")
CHART_DIR = REPORT_DIR / "error_analysis"
LABELS = ["false", "true", "unverified"]
KEY_COLUMNS = ["event", "thread_id", "source_tweet_id"]
CONFIDENCE_BANDS = [
    (0.0, 0.49),
    (0.50, 0.59),
    (0.60, 0.69),
    (0.70, 0.79),
    (0.80, 0.89),
    (0.90, 1.00),
]


def load_aligned_data(
    test_path: Path = TEST_PATH, predictions_path: Path = PREDICTIONS_PATH
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not test_path.is_file():
        raise FileNotFoundError(f"Test split not found: {test_path}")
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Final predictions not found: {predictions_path}")
    test = pd.read_csv(test_path)
    predictions = pd.read_csv(predictions_path)
    required_test = set(
        KEY_COLUMNS
        + [
            "text_original",
            "text_clean",
            "label",
            "event",
            "thread_id",
            "source_tweet_id",
        ]
    )
    required_predictions = set(
        KEY_COLUMNS
        + [
            "label",
            "text_original",
            "prediction",
            "confidence",
            *[f"probability_{label}" for label in LABELS],
        ]
    )
    missing_test = required_test - set(test.columns)
    missing_predictions = required_predictions - set(predictions.columns)
    if missing_test:
        raise ValueError(
            f"Test split is missing columns: {', '.join(sorted(missing_test))}"
        )
    if missing_predictions:
        raise ValueError(
            f"Predictions are missing columns: {', '.join(sorted(missing_predictions))}"
        )
    if len(test) != len(predictions):
        raise ValueError(
            f"Prediction count {len(predictions)} does not match test count {len(test)}"
        )
    if (
        test[KEY_COLUMNS].duplicated().any()
        or predictions[KEY_COLUMNS].duplicated().any()
    ):
        raise ValueError("Duplicate alignment keys found in test or prediction files")
    merged = test.merge(
        predictions,
        on=KEY_COLUMNS,
        how="left",
        suffixes=("_test", "_prediction"),
        validate="one_to_one",
    )
    if merged["prediction"].isna().any():
        raise ValueError("Some test rows do not have a prediction")
    if not merged["text_original_test"].equals(merged["text_original_prediction"]):
        raise ValueError("Original text does not align with final predictions")
    if not merged["label_test"].equals(merged["label_prediction"]):
        raise ValueError("True labels do not align with final predictions")
    options = PreprocessingOptions()
    computed_clean = merged["text_original_test"].map(
        lambda value: preprocess_text(value, options)
    )
    if not computed_clean.equals(merged["text_clean"]):
        raise ValueError(
            "Cleaned text does not match the recorded preprocessing output"
        )
    if set(merged["event"]) != set(test["event"]):
        raise ValueError("Prediction events differ from the test split")
    aligned = pd.DataFrame(
        {
            "event": merged["event"],
            "thread_id": merged["thread_id"],
            "source_tweet_id": merged["source_tweet_id"],
            "text_original": merged["text_original_test"],
            "text_clean": merged["text_clean"],
            "true_label": merged["label_test"],
            "predicted_label": merged["prediction"],
            "confidence": merged["confidence"],
            **{
                f"probability_{label}": merged[f"probability_{label}"]
                for label in LABELS
            },
        }
    )
    validation = {
        "test_path": str(test_path),
        "predictions_path": str(predictions_path),
        "prediction_count": len(predictions),
        "test_count": len(test),
        "exactly_one_prediction_per_row": True,
        "original_text_aligned": True,
        "cleaned_text_aligned": True,
        "labels_aligned": True,
        "event_ids_preserved": True,
        "thread_ids_preserved": True,
        "source_tweet_ids_preserved": True,
        "predictions_from_test_split": True,
        "class_order": LABELS,
    }
    return aligned, validation


def _excerpt(text: str, limit: int = 180) -> str:
    text = str(text).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _base_example(row: pd.Series, category: str) -> dict[str, Any]:
    return {
        "event": row["event"],
        "thread_id": row["thread_id"],
        "source_tweet_id": row["source_tweet_id"],
        "true_label": row["true_label"],
        "predicted_label": row["predicted_label"],
        "confidence": round(float(row["confidence"]), 6),
        "text_original_excerpt": _excerpt(row["text_original"]),
        "text_clean_excerpt": _excerpt(row["text_clean"]),
        "error_category": category,
    }


def confusion_pair_analysis(data: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for true_label in LABELS:
        for predicted_label in LABELS:
            if true_label == predicted_label:
                continue
            subset = data[
                (data["true_label"] == true_label)
                & (data["predicted_label"] == predicted_label)
            ]
            true_total = int((data["true_label"] == true_label).sum())
            predicted_total = int((data["predicted_label"] == predicted_label).sum())
            key = f"{true_label}_to_{predicted_label}"
            output[key] = {
                "count": len(subset),
                "percentage_of_true_class": (
                    round(len(subset) / true_total * 100, 4) if true_total else 0
                ),
                "percentage_of_predicted_class": (
                    round(len(subset) / predicted_total * 100, 4)
                    if predicted_total
                    else 0
                ),
                "average_confidence": (
                    round(float(subset["confidence"].mean()), 6)
                    if len(subset)
                    else None
                ),
                "average_text_length": (
                    round(float(subset["text_original"].str.len().mean()), 2)
                    if len(subset)
                    else None
                ),
                "event_distribution": {
                    str(k): int(v)
                    for k, v in subset["event"].value_counts().to_dict().items()
                },
                "example_ids": [str(value) for value in subset["thread_id"].head(10)],
                "examples": [
                    _base_example(row, key) for _, row in subset.head(3).iterrows()
                ],
            }
    return output


def _pattern_counts(texts: pd.Series) -> dict[str, int]:
    return {
        "records_with_urls": int(
            texts.map(lambda text: bool(URL_PATTERN.search(str(text)))).sum()
        ),
        "records_with_hashtags": int(
            texts.map(lambda text: bool(HASHTAG_PATTERN.search(str(text)))).sum()
        ),
        "records_with_mentions": int(
            texts.map(lambda text: bool(MENTION_PATTERN.search(str(text)))).sum()
        ),
        "records_with_punctuation": int(
            texts.map(
                lambda text: any(char in string.punctuation for char in str(text))
            ).sum()
        ),
    }


def event_error_analysis(data: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for event, subset in data.groupby("event", sort=True):
        labels_present = sorted(set(subset["true_label"]))
        errors = subset[subset["true_label"] != subset["predicted_label"]]
        report = classification_report(
            subset["true_label"],
            subset["predicted_label"],
            labels=LABELS,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        )
        output[str(event)] = {
            "records": len(subset),
            "correct": int((subset["true_label"] == subset["predicted_label"]).sum()),
            "incorrect": len(errors),
            "accuracy": round(
                float(accuracy_score(subset["true_label"], subset["predicted_label"])),
                6,
            ),
            "macro_f1": (
                round(
                    float(
                        f1_score(
                            subset["true_label"],
                            subset["predicted_label"],
                            labels=LABELS,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    6,
                )
                if set(labels_present) == set(LABELS)
                else None
            ),
            "macro_f1_unavailable_reason": (
                None
                if set(labels_present) == set(LABELS)
                else "Not all three true labels are present in this event."
            ),
            "labels_present": labels_present,
            "per_class_recall": {
                label: round(float(report[label]["recall"]), 6)
                for label in LABELS
                if label in labels_present
            },
            "average_confidence": round(float(subset["confidence"].mean()), 6),
            "low_confidence_rate": round(float((subset["confidence"] < 0.6).mean()), 6),
            "average_text_length": round(
                float(subset["text_original"].str.len().mean()), 2
            ),
            "text_patterns": _pattern_counts(subset["text_original"]),
            "most_common_confusion_pair": (
                f"{errors.iloc[0]['true_label']}_to_{errors.iloc[0]['predicted_label']}"
                if len(errors)
                and not errors.groupby(["true_label", "predicted_label"]).size().empty
                else None
            ),
            "confusion_pairs": {
                f"{true_label}_to_{predicted_label}": len(pair)
                for (true_label, predicted_label), pair in errors.groupby(
                    ["true_label", "predicted_label"]
                )
            },
        }
        if output[str(event)]["confusion_pairs"]:
            output[str(event)]["most_common_confusion_pair"] = max(
                output[str(event)]["confusion_pairs"],
                key=output[str(event)]["confusion_pairs"].get,
            )
    return output


def confidence_analysis(data: pd.DataFrame) -> dict[str, Any]:
    bands: dict[str, Any] = {}
    for lower, upper in CONFIDENCE_BANDS:
        mask = (data["confidence"] >= lower) & (data["confidence"] <= upper)
        subset = data[mask]
        errors = subset[subset["true_label"] != subset["predicted_label"]]
        key = f"{lower:.2f}-{upper:.2f}"
        bands[key] = {
            "count": len(subset),
            "accuracy": (
                round(
                    float(
                        accuracy_score(subset["true_label"], subset["predicted_label"])
                    ),
                    6,
                )
                if len(subset)
                else None
            ),
            "macro_f1": (
                round(
                    float(
                        f1_score(
                            subset["true_label"],
                            subset["predicted_label"],
                            labels=LABELS,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    6,
                )
                if len(subset) and set(subset["true_label"]) == set(LABELS)
                else None
            ),
            "most_common_predicted_class": (
                str(subset["predicted_label"].mode().iloc[0]) if len(subset) else None
            ),
            "most_common_error_type": (
                f"{errors.iloc[0]['true_label']}_to_{errors.iloc[0]['predicted_label']}"
                if len(errors)
                else None
            ),
        }
        if len(errors):
            pair_counts = errors.groupby(["true_label", "predicted_label"]).size()
            bands[key]["most_common_error_type"] = "_to_".join(pair_counts.idxmax())
    wrong = data[data["true_label"] != data["predicted_label"]]
    correct = data[data["true_label"] == data["predicted_label"]]
    high_wrong = wrong[wrong["confidence"] >= 0.8]
    low_correct = correct[correct["confidence"] < 0.6]
    return {
        "bands": bands,
        "high_confidence_wrong_count": len(high_wrong),
        "high_confidence_wrong_rate": round(
            float(
                (
                    data["confidence"].ge(0.8)
                    & data["true_label"].ne(data["predicted_label"])
                ).mean()
            ),
            6,
        ),
        "low_confidence_correct_count": len(low_correct),
        "low_confidence_correct_rate": round(
            float(
                (
                    data["confidence"].lt(0.6)
                    & data["true_label"].eq(data["predicted_label"])
                ).mean()
            ),
            6,
        ),
        "overconfidence_observation": "High-confidence errors are present; confidence should be treated as an uncalibrated model output, not factual certainty.",
    }


def feature_explanations(artifact: dict[str, Any], top_n: int = 20) -> dict[str, Any]:
    model = artifact["model"]
    vectorizer = artifact["vectorizer"]
    names = vectorizer.get_feature_names_out()
    log_prob = model.feature_log_prob_
    result: dict[str, Any] = {
        "labels": [str(label) for label in model.classes_],
        "top_features_by_class": {},
        "pairwise_feature_differences": {},
        "interpretation_warning": "These are statistical associations in the fitted Naive Bayes model, not causal evidence or proof of truth.",
    }
    for index, label in enumerate(model.classes_):
        other_mean = np.delete(log_prob, index, axis=0).mean(axis=0)
        scores = log_prob[index] - other_mean
        top_indices = np.argsort(scores)[::-1][:top_n]
        result["top_features_by_class"][str(label)] = [
            {
                "feature": str(names[position]),
                "log_probability_difference": round(float(scores[position]), 6),
                "class_log_probability": round(float(log_prob[index, position]), 6),
            }
            for position in top_indices
        ]
    for left_index, left_label in enumerate(model.classes_):
        for right_index, right_label in enumerate(model.classes_):
            if left_index >= right_index:
                continue
            scores = log_prob[left_index] - log_prob[right_index]
            top_indices = np.argsort(np.abs(scores))[::-1][:top_n]
            result["pairwise_feature_differences"][f"{left_label}_vs_{right_label}"] = [
                {
                    "feature": str(names[position]),
                    "log_probability_difference": round(float(scores[position]), 6),
                }
                for position in top_indices
            ]
    return result


def representative_examples(data: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    correct = data[data["true_label"] == data["predicted_label"]].sort_values(
        "confidence", ascending=False
    )
    wrong = data[data["true_label"] != data["predicted_label"]].sort_values(
        "confidence", ascending=False
    )
    low_correct = data[
        (data["true_label"] == data["predicted_label"]) & (data["confidence"] < 0.6)
    ].sort_values("confidence")
    low_wrong = data[
        (data["true_label"] != data["predicted_label"]) & (data["confidence"] < 0.6)
    ].sort_values("confidence")
    categories = {
        "correct_high_confidence": (correct, "correct_high_confidence"),
        "incorrect_high_confidence": (wrong, "incorrect_high_confidence"),
        "correct_low_confidence": (low_correct, "correct_low_confidence"),
        "incorrect_low_confidence": (low_wrong, "incorrect_low_confidence"),
    }
    for true_label, predicted_label in (
        ("false", "true"),
        ("true", "false"),
        ("unverified", "true"),
        ("unverified", "false"),
    ):
        subset = data[
            (data["true_label"] == true_label)
            & (data["predicted_label"] == predicted_label)
        ].sort_values("confidence", ascending=False)
        categories[f"{true_label}_to_{predicted_label}"] = (
            subset,
            f"{true_label}_to_{predicted_label}",
        )
    return {
        name: [_base_example(row, category) for _, row in subset.head(5).iterrows()]
        for name, (subset, category) in categories.items()
    }


def _write_charts(data: pd.DataFrame, output_dir: Path) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def save(name: str) -> None:
        path = output_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(str(path))

    matrix = confusion_matrix(
        data["true_label"], data["predicted_label"], labels=LABELS
    )
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    plt.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="Fraction of true class")
    plt.xticks(range(3), LABELS)
    plt.yticks(range(3), LABELS)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Normalized confusion matrix — frozen PHEME test set")
    for row in range(3):
        for col in range(3):
            plt.text(col, row, f"{normalized[row, col]:.2f}", ha="center", va="center")
    save("normalized_confusion_matrix.png")

    event_accuracy = (
        data.assign(correct=data["true_label"] == data["predicted_label"])
        .groupby("event")["correct"]
        .mean()
        .sort_values()
    )
    event_accuracy.plot(
        kind="barh", title="Accuracy by held-out event — PHEME test set"
    )
    plt.xlabel("Accuracy")
    plt.ylabel("Event")
    save("accuracy_by_event.png")

    event_f1 = (
        data.groupby("event")
        .apply(
            lambda group: (
                f1_score(
                    group["true_label"],
                    group["predicted_label"],
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                )
                if set(group["true_label"]) == set(LABELS)
                else np.nan
            ),
            include_groups=False,
        )
        .dropna()
    )
    if len(event_f1):
        event_f1.plot(kind="bar", title="Macro F1 by event — available events only")
        plt.ylabel("Macro F1")
    else:
        plt.text(0.5, 0.5, "Unavailable", ha="center", va="center")
        plt.title("Macro F1 by event — unavailable")
    plt.xlabel("Event")
    save("macro_f1_by_event.png")

    data["confidence"].plot(
        kind="hist", bins=10, title="Confidence distribution — frozen PHEME test set"
    )
    plt.xlabel("Highest predicted class probability")
    plt.ylabel("Records")
    save("confidence_distribution.png")

    bands = confidence_analysis(data)["bands"]
    band_accuracy = pd.Series({key: value["accuracy"] for key, value in bands.items()})
    band_accuracy.plot(kind="bar", title="Accuracy by confidence band — PHEME test set")
    plt.xlabel("Confidence band")
    plt.ylabel("Accuracy")
    save("accuracy_by_confidence_band.png")

    errors = data[data["true_label"] != data["predicted_label"]].assign(
        pair=lambda frame: frame["true_label"] + " → " + frame["predicted_label"]
    )
    errors["pair"].value_counts().sort_values().plot(
        kind="barh", title="Error count by confusion pair — PHEME test set"
    )
    plt.xlabel("Errors")
    plt.ylabel("Confusion pair")
    save("error_count_by_confusion_pair.png")

    report = classification_report(
        data["true_label"],
        data["predicted_label"],
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(
        {
            "precision": [report[label]["precision"] for label in LABELS],
            "recall": [report[label]["recall"] for label in LABELS],
        },
        index=LABELS,
    ).plot(kind="bar", title="Per-class precision and recall — PHEME test set")
    plt.xlabel("Class")
    plt.ylabel("Score")
    save("per_class_precision_recall.png")

    (
        data.assign(correct=data["true_label"] == data["predicted_label"]).boxplot(
            column="text_length", by="correct"
        )
        if "text_length" in data
        else None
    )
    if "text_length" in data:
        plt.suptitle("")
        plt.title("Text length versus correctness — PHEME test set")
        plt.xlabel("Correct prediction")
        plt.ylabel("Characters")
    else:
        plt.text(0.5, 0.5, "Unavailable", ha="center", va="center")
        plt.title("Text length versus correctness — unavailable")
    save("text_length_vs_correctness.png")

    label_dist = data.groupby(["event", "true_label"]).size().unstack(fill_value=0)
    label_dist.plot(
        kind="bar", stacked=True, title="Event-wise label distribution — PHEME test set"
    )
    plt.xlabel("Event")
    plt.ylabel("Records")
    save("event_label_distribution.png")
    return paths


def analyze_errors(
    test_path: Path = TEST_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
    artifact_path: Path = ARTIFACT_PATH,
    report_dir: Path = REPORT_DIR,
    chart_dir: Path = CHART_DIR,
) -> dict[str, Any]:
    data, alignment = load_aligned_data(test_path, predictions_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Frozen final artifact not found: {artifact_path}")
    artifact = joblib.load(artifact_path)
    if artifact.get("model_version") != "pheme-tuned-multinomial-naive-bayes-v1":
        raise ValueError("Unexpected final model artifact version")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "error_analysis_validation.json").write_text(
        json.dumps(alignment, indent=2) + "\n", encoding="utf-8"
    )
    pair_report = confusion_pair_analysis(data)
    event_report = event_error_analysis(data)
    confidence_report = confidence_analysis(data)
    features = feature_explanations(artifact)
    examples = representative_examples(data)
    charts = _write_charts(
        data.assign(text_length=data["text_original"].str.len()), chart_dir
    )
    (report_dir / "confusion_pair_analysis.json").write_text(
        json.dumps(pair_report, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "event_error_analysis.json").write_text(
        json.dumps(event_report, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "confidence_analysis.json").write_text(
        json.dumps(confidence_report, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "feature_explanations.json").write_text(
        json.dumps(features, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "representative_errors.json").write_text(
        json.dumps(examples, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "alignment": alignment,
        "confusion_pairs": pair_report,
        "events": event_report,
        "confidence": confidence_report,
        "features": features,
        "examples": examples,
        "charts": charts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    args = parser.parse_args()
    report = analyze_errors(args.test, args.predictions, args.artifact)
    print(
        json.dumps(
            {
                "alignment": report["alignment"],
                "charts": report["charts"],
                "high_confidence_wrong": report["confidence"][
                    "high_confidence_wrong_count"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
