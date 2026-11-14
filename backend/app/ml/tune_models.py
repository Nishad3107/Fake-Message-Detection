"""Tune baseline sparse-text models with event-grouped cross-validation."""

from __future__ import annotations

import argparse
import csv
import itertools
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from app.ml.train_baselines import LABELS, REQUIRED_COLUMNS, _metric_summary

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - exercised only on older scikit-learn
    StratifiedGroupKFold = None

TRAIN_PATH = Path("backend/data/processed/splits/train.csv")
VALIDATION_PATH = Path("backend/data/processed/splits/validation.csv")
REPORT_DIR = Path("backend/data/reports/model_comparison")
MODEL_DIR = Path("backend/models/tuned")
RANDOM_SEED = 42
TUNING_LABELS = tuple(LABELS)


@dataclass(frozen=True)
class TuningTfidfConfig:
    ngram_range: tuple[int, int]
    min_df: int
    sublinear_tf: bool
    max_df: float = 0.95
    lowercase: bool = False
    strip_accents: str = "unicode"


def load_training_data(path: Path = TRAIN_PATH) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Training split not found: {path}")
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Training split is missing required columns: {', '.join(sorted(missing))}"
        )
    if frame.empty:
        raise ValueError("Training split is empty")
    return frame


def build_grouped_folds(
    frame: pd.DataFrame, n_splits: int = 5, seed: int = RANDOM_SEED
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    if frame["event"].nunique() < n_splits:
        raise ValueError(
            f"Need at least {n_splits} unique events for grouped CV; found {frame['event'].nunique()}"
        )
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        strategy = "StratifiedGroupKFold"
        folds = list(
            splitter.split(frame["text_clean"], frame["label"], groups=frame["event"])
        )
    else:
        splitter = GroupKFold(n_splits=n_splits)
        strategy = "GroupKFold fallback; folds are event-disjoint but not stratified"
        folds = list(
            splitter.split(frame["text_clean"], frame["label"], groups=frame["event"])
        )
    for train_indices, validation_indices in folds:
        train_events = set(frame.iloc[train_indices]["event"])
        validation_events = set(frame.iloc[validation_indices]["event"])
        if train_events & validation_events:
            raise RuntimeError(
                "Grouped CV leakage detected: an event appears in train and validation"
            )
    return folds, strategy


def default_parameter_grid() -> dict[str, list[dict[str, Any]]]:
    tfidf_configs = [
        TuningTfidfConfig(
            ngram_range=ngram_range, min_df=min_df, sublinear_tf=sublinear_tf
        )
        for ngram_range, min_df, sublinear_tf in itertools.product(
            ((1, 1), (1, 2)), (1, 2, 3), (True, False)
        )
    ]
    return {
        "logistic_regression": [
            {
                "tfidf": config,
                "model": {"C": c, "class_weight": class_weight, "max_iter": 1000},
            }
            for config in tfidf_configs
            for c, class_weight in itertools.product((0.5, 1.0), (None, "balanced"))
        ],
        "multinomial_naive_bayes": [
            {"tfidf": config, "model": {"alpha": alpha, "fit_prior": fit_prior}}
            for config in tfidf_configs
            for alpha, fit_prior in itertools.product((0.1, 1.0), (True, False))
        ],
        "linear_svc": [
            {
                "tfidf": config,
                "model": {"C": c, "class_weight": class_weight, "max_iter": 2000},
            }
            for config in tfidf_configs
            for c, class_weight in itertools.product((0.5, 1.0), (None, "balanced"))
        ],
    }


def _make_vectorizer(config: TuningTfidfConfig) -> TfidfVectorizer:
    return TfidfVectorizer(**asdict(config))


def _make_model(family: str, params: dict[str, Any]) -> Any:
    if family == "logistic_regression":
        return LogisticRegression(solver="lbfgs", random_state=RANDOM_SEED, **params)
    if family == "multinomial_naive_bayes":
        return MultinomialNB(**params)
    if family == "linear_svc":
        return LinearSVC(random_state=RANDOM_SEED, **params)
    raise ValueError(f"Unsupported model family: {family}")


def _config_to_dict(config: dict[str, Any]) -> dict[str, Any]:
    tfidf = config["tfidf"]
    return {
        "tfidf": asdict(tfidf) if isinstance(tfidf, TuningTfidfConfig) else tfidf,
        "model": config["model"],
    }


def _config_from_dict(config: dict[str, Any]) -> dict[str, Any]:
    tfidf = config["tfidf"]
    return {
        "tfidf": TuningTfidfConfig(**tfidf) if isinstance(tfidf, dict) else tfidf,
        "model": config["model"],
    }


def _std(values: list[float]) -> float:
    return round(float(np.std(values, ddof=1)), 6) if len(values) > 1 else 0.0


def _run_configuration(
    frame: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    family: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    fold_results: list[dict[str, Any]] = []
    started_at = datetime.now(UTC).isoformat()
    total_started = time.perf_counter()
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        fold_started = time.perf_counter()
        fold_train = frame.iloc[train_indices]
        fold_validation = frame.iloc[validation_indices]
        vectorizer = _make_vectorizer(config["tfidf"])
        train_x = vectorizer.fit_transform(
            fold_train["text_clean"].fillna("").astype(str)
        )
        validation_x = vectorizer.transform(
            fold_validation["text_clean"].fillna("").astype(str)
        )
        model = _make_model(family, config["model"])
        model.fit(train_x, fold_train["label"].astype(str))
        predictions = model.predict(validation_x)
        metrics = _metric_summary(fold_validation["label"].astype(str), predictions)
        fold_results.append(
            {
                "fold": fold_number,
                "train_events": sorted(set(fold_train["event"])),
                "validation_events": sorted(set(fold_validation["event"])),
                "train_samples": len(fold_train),
                "validation_samples": len(fold_validation),
                "features": int(train_x.shape[1]),
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "accuracy": metrics["accuracy"],
                "duration_seconds": round(time.perf_counter() - fold_started, 6),
            }
        )
    ended_at = datetime.now(UTC).isoformat()
    macro_f1 = [fold["macro_f1"] for fold in fold_results]
    weighted_f1 = [fold["weighted_f1"] for fold in fold_results]
    accuracy = [fold["accuracy"] for fold in fold_results]
    return {
        "model_family": family,
        "configuration": _config_to_dict(config),
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": round(time.perf_counter() - total_started, 6),
        "folds": fold_results,
        "mean_macro_f1": round(float(np.mean(macro_f1)), 6),
        "std_macro_f1": _std(macro_f1),
        "mean_weighted_f1": round(float(np.mean(weighted_f1)), 6),
        "std_weighted_f1": _std(weighted_f1),
        "mean_accuracy": round(float(np.mean(accuracy)), 6),
        "std_accuracy": _std(accuracy),
        "features_per_fold": sorted({fold["features"] for fold in fold_results}),
    }


def _select_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        results,
        key=lambda result: (
            result["mean_macro_f1"],
            -result["std_macro_f1"],
            str(result["configuration"]),
        ),
    )


def _validation_result(
    family: str,
    config: dict[str, Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    model_dir: Path,
) -> dict[str, Any]:
    config = _config_from_dict(config)
    vectorizer = _make_vectorizer(config["tfidf"])
    train_x = vectorizer.fit_transform(train["text_clean"].fillna("").astype(str))
    validation_x = vectorizer.transform(validation["text_clean"].fillna("").astype(str))
    model = _make_model(family, config["model"])
    started = time.perf_counter()
    model.fit(train_x, train["label"].astype(str))
    training_duration = time.perf_counter() - started
    predictions = model.predict(validation_x)
    metrics = _metric_summary(validation["label"].astype(str), predictions)
    artifact_name = f"{family}_tuned.joblib"
    joblib.dump(model, model_dir / artifact_name)
    joblib.dump(vectorizer, model_dir / f"{family}_tfidf.joblib")
    return {
        "model_family": family,
        "configuration": _config_to_dict(config),
        "training_duration_seconds": round(training_duration, 6),
        "features": int(train_x.shape[1]),
        "validation_metrics": metrics,
        "artifact": artifact_name,
        "confidence_output_type": (
            "decision_margins" if family == "linear_svc" else "probabilities"
        ),
    }


def tune_models(
    train_path: Path = TRAIN_PATH,
    validation_path: Path = VALIDATION_PATH,
    manifest_path: Path | None = None,
    report_dir: Path = REPORT_DIR,
    model_dir: Path = MODEL_DIR,
    n_splits: int = 5,
    parameter_grid: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    train = load_training_data(train_path)
    validation = load_training_data(validation_path)
    if set(train["event"]) & set(validation["event"]):
        raise ValueError("Training and validation events overlap")
    if set(train["thread_id"]) & set(validation["thread_id"]):
        raise ValueError("Training and validation threads overlap")
    folds, cv_strategy = build_grouped_folds(train, n_splits=n_splits)
    grid = parameter_grid or default_parameter_grid()
    if not grid:
        raise ValueError("Parameter grid cannot be empty")
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    for family, configurations in grid.items():
        if not configurations:
            raise ValueError(f"Parameter grid is empty for model family: {family}")
        family_results = [
            _run_configuration(train, folds, family, config)
            for config in configurations
        ]
        all_results.extend(family_results)
        selected[family] = _select_best(family_results)

    validation_results = {
        family: _validation_result(
            family,
            selected[family]["configuration"],
            train,
            validation,
            model_dir,
        )
        for family in selected
    }
    baseline_path = report_dir / "baseline_results.json"
    baseline = (
        json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_path.is_file()
        else None
    )
    report: dict[str, Any] = {
        "dataset_source": "real PHEME training and validation split files",
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "test_policy": "The final test split was not loaded, transformed, or evaluated.",
        "manifest_reference": str(manifest_path) if manifest_path else None,
        "cv_strategy": cv_strategy,
        "grouping_column": "event",
        "target_column": "label",
        "text_column": "text_clean",
        "random_seed": RANDOM_SEED,
        "fold_count": len(folds),
        "training_records": len(train),
        "validation_records": len(validation),
        "configurations_tested": len(all_results),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
        },
        "results": all_results,
        "selected_by_family": selected,
        "validation_results": validation_results,
        "baseline_comparison": (
            {
                row["model"]: {
                    "validation_macro_f1": row["macro_f1"],
                    "validation_weighted_f1": row["weighted_f1"],
                }
                for row in baseline["results"]
            }
            if baseline
            else None
        ),
        "warnings": [
            "Ordinary random cross-validation is unsafe because PHEME records share event context.",
            "Validation results are not final generalization results.",
            "The final test split was intentionally excluded from tuning and model selection.",
            "Five events provide only a small number of grouped folds; grouped cross-validation estimates may be unstable.",
        ],
    }
    (report_dir / "tuning_results.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    csv_rows = []
    for result in all_results:
        csv_rows.append(
            {
                "model_family": result["model_family"],
                "configuration": json.dumps(result["configuration"], sort_keys=True),
                "mean_macro_f1": result["mean_macro_f1"],
                "std_macro_f1": result["std_macro_f1"],
                "mean_weighted_f1": result["mean_weighted_f1"],
                "std_weighted_f1": result["std_weighted_f1"],
                "mean_accuracy": result["mean_accuracy"],
                "std_accuracy": result["std_accuracy"],
                "duration_seconds": result["duration_seconds"],
                "features_per_fold": json.dumps(result["features_per_fold"]),
                "folds": json.dumps(result["folds"]),
            }
        )
    with (report_dir / "tuning_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    summary = {
        "cv_strategy": cv_strategy,
        "grouping_column": "event",
        "primary_metric": "mean_macro_f1",
        "selected_by_family": {
            family: {
                "configuration": value["configuration"],
                "mean_macro_f1": value["mean_macro_f1"],
                "std_macro_f1": value["std_macro_f1"],
                "validation_macro_f1": validation_results[family]["validation_metrics"][
                    "macro_f1"
                ],
            }
            for family, value in selected.items()
        },
        "test_used": False,
    }
    (report_dir / "tuning_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--validation", type=Path, default=VALIDATION_PATH)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    report = tune_models(
        args.train,
        args.validation,
        args.manifest,
        args.report_dir,
        args.model_dir,
        args.folds,
    )
    print(
        json.dumps(
            {
                "configurations_tested": report["configurations_tested"],
                "selected_by_family": report["selected_by_family"],
                "validation_results": report["validation_results"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
