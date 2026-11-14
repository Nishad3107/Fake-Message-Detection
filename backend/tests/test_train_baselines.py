import json
from pathlib import Path

import pandas as pd
import pytest
from app.ml.train_baselines import (
    TfidfConfig,
    fit_training_vectorizer,
    load_split,
    train_baselines,
)


def _split_frame(split: str, event: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "text_clean": f"{split} true trainonly",
                "label": "true",
                "event": event,
                "thread_id": f"{split}-true",
                "source_tweet_id": f"{split}-true",
                "split": split,
            },
            {
                "text_clean": f"{split} false signal",
                "label": "false",
                "event": event,
                "thread_id": f"{split}-false",
                "source_tweet_id": f"{split}-false",
                "split": split,
            },
            {
                "text_clean": f"{split} unverified claim",
                "label": "unverified",
                "event": event,
                "thread_id": f"{split}-unverified",
                "source_tweet_id": f"{split}-unverified",
                "split": split,
            },
        ]
    )


def _write_splits(directory: Path) -> None:
    for split, event in (
        ("train", "train-event"),
        ("validation", "validation-event"),
        ("test", "test-event"),
    ):
        _split_frame(split, event).to_csv(directory / f"{split}.csv", index=False)


def _manifest() -> dict:
    return {
        "events": {
            "train": ["train-event"],
            "validation": ["validation-event"],
            "test": ["test-event"],
        },
        "record_counts": {"train": 3, "validation": 3, "test": 3},
        "event_overlap": [],
        "thread_overlap": [],
        "random_seed": 42,
    }


def test_vectorizer_is_fit_only_on_training_text(tmp_path: Path) -> None:
    train = _split_frame("train", "train-event")
    vectorizer = fit_training_vectorizer(train, TfidfConfig(min_df=1))
    assert "trainonly" in vectorizer.vocabulary_
    assert "validationonly" not in vectorizer.vocabulary_


def test_baseline_runner_trains_all_models_and_records_metrics(tmp_path: Path) -> None:
    data_dir = tmp_path / "splits"
    data_dir.mkdir()
    _write_splits(data_dir)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

    report = train_baselines(
        data_dir,
        manifest_path,
        tmp_path / "reports",
        tmp_path / "models",
        TfidfConfig(min_df=1),
    )

    assert set(report["models"]) == {
        "majority_baseline",
        "multinomial_naive_bayes",
        "logistic_regression",
        "linear_svc",
    }
    for detail in report["models"].values():
        assert detail["training_duration_seconds"] >= 0
        assert set(detail["validation_metrics"]["per_class"]) == {
            "false",
            "true",
            "unverified",
        }
        assert len(detail["validation_metrics"]["confusion_matrix"]) == 3
        assert all(
            len(row) == 3 for row in detail["validation_metrics"]["confusion_matrix"]
        )
    assert (
        report["models"]["linear_svc"]["confidence_output_type"] == "decision_margins"
    )
    assert report["models"]["linear_svc"]["confidence"]["type"] == "decision_margins"


def test_missing_split_file_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Split file not found"):
        load_split(tmp_path / "missing.csv")
