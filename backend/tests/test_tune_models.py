import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from app.ml.tune_models import (
    TuningTfidfConfig,
    build_grouped_folds,
    tune_models,
)


def _frame(split: str, events: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for event in events:
        for label in ("true", "false", "unverified"):
            identifier = f"{split}-{event}-{label}"
            rows.append(
                {
                    "text_clean": f"{split} {event} {label} trainonly",
                    "label": label,
                    "event": event,
                    "thread_id": identifier,
                    "source_tweet_id": identifier,
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def _write_data(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    _frame("train", ("event-a", "event-b", "event-c")).to_csv(
        data_dir / "train.csv", index=False
    )
    _frame("validation", ("event-validation",)).to_csv(
        data_dir / "validation.csv", index=False
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"random_seed": 42}), encoding="utf-8")
    return data_dir / "train.csv", data_dir / "validation.csv"


def _grid() -> dict:
    config = TuningTfidfConfig(ngram_range=(1, 1), min_df=1, sublinear_tf=True)
    return {
        "logistic_regression": [
            {
                "tfidf": config,
                "model": {"C": 1.0, "class_weight": "balanced", "max_iter": 100},
            }
        ],
        "multinomial_naive_bayes": [
            {"tfidf": config, "model": {"alpha": 1.0, "fit_prior": True}}
        ],
        "linear_svc": [
            {
                "tfidf": config,
                "model": {"C": 1.0, "class_weight": "balanced", "max_iter": 100},
            }
        ],
    }


def test_grouped_folds_do_not_share_events_and_cover_training_rows() -> None:
    frame = _frame("train", ("event-a", "event-b", "event-c"))
    folds, strategy = build_grouped_folds(frame, n_splits=3)
    assert strategy == "StratifiedGroupKFold"
    validation_rows = []
    for train_indices, validation_indices in folds:
        assert set(frame.iloc[train_indices]["event"]).isdisjoint(
            set(frame.iloc[validation_indices]["event"])
        )
        validation_rows.extend(validation_indices.tolist())
    assert sorted(validation_rows) == list(range(len(frame)))


def test_tuning_records_metrics_durations_and_reloadable_models(tmp_path: Path) -> None:
    train_path, validation_path = _write_data(tmp_path)
    report = tune_models(
        train_path,
        validation_path,
        tmp_path / "manifest.json",
        tmp_path / "reports",
        tmp_path / "models",
        n_splits=3,
        parameter_grid=_grid(),
    )
    assert report["configurations_tested"] == 3
    assert report["test_policy"].startswith("The final test split was not")
    for family, result in report["validation_results"].items():
        assert result["training_duration_seconds"] >= 0
        assert set(result["validation_metrics"]["per_class"]) == {
            "false",
            "true",
            "unverified",
        }
        assert (tmp_path / "models" / result["artifact"]).is_file()
        assert joblib.load(tmp_path / "models" / result["artifact"]) is not None
        assert family in report["selected_by_family"]


def test_tuning_is_reproducible_for_same_seed(tmp_path: Path) -> None:
    train_path, validation_path = _write_data(tmp_path)
    first = tune_models(
        train_path,
        validation_path,
        tmp_path / "manifest.json",
        tmp_path / "r1",
        tmp_path / "m1",
        3,
        _grid(),
    )
    second = tune_models(
        train_path,
        validation_path,
        tmp_path / "manifest.json",
        tmp_path / "r2",
        tmp_path / "m2",
        3,
        _grid(),
    )
    for family in first["selected_by_family"]:
        assert (
            first["selected_by_family"][family]["configuration"]
            == second["selected_by_family"][family]["configuration"]
        )
        assert (
            first["selected_by_family"][family]["mean_macro_f1"]
            == second["selected_by_family"][family]["mean_macro_f1"]
        )
        assert (
            first["selected_by_family"][family]["std_macro_f1"]
            == second["selected_by_family"][family]["std_macro_f1"]
        )
    assert [
        (row["model_family"], row["mean_macro_f1"]) for row in first["results"]
    ] == [(row["model_family"], row["mean_macro_f1"]) for row in second["results"]]


def test_invalid_model_family_fails_clearly(tmp_path: Path) -> None:
    train_path, validation_path = _write_data(tmp_path)
    with pytest.raises(ValueError, match="Unsupported model family"):
        tune_models(
            train_path,
            validation_path,
            tmp_path / "manifest.json",
            tmp_path / "reports",
            tmp_path / "models",
            n_splits=3,
            parameter_grid={
                "unsupported": [
                    {"tfidf": _grid()["logistic_regression"][0]["tfidf"], "model": {}}
                ]
            },
        )
