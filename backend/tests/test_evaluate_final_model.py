import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from app.ml.evaluate_final_model import (
    _event_metrics,
    _metrics,
    freeze_final_config,
    load_final_artifact,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def test_final_configuration_is_frozen_without_test_selection(tmp_path: Path) -> None:
    configuration = {
        "tfidf": {
            "ngram_range": [1, 2],
            "min_df": 3,
            "sublinear_tf": False,
            "max_df": 0.95,
            "lowercase": False,
            "strip_accents": "unicode",
        },
        "model": {"alpha": 0.1, "fit_prior": False},
    }
    tuning = {
        "configurations_tested": 144,
        "validation_results": {
            "multinomial_naive_bayes": {"validation_metrics": {"macro_f1": 0.372437}}
        },
    }
    summary = {
        "selected_by_family": {
            "multinomial_naive_bayes": {
                "configuration": configuration,
                "mean_macro_f1": 0.197411,
                "std_macro_f1": 0.113824,
                "validation_macro_f1": 0.372437,
            }
        }
    }
    preprocessing = {"preprocessing_options": {"lowercase": True}}
    manifest = {"random_seed": 42}
    paths = []
    for name, value in (
        ("tuning.json", tuning),
        ("summary.json", summary),
        ("preprocessing.json", preprocessing),
        ("manifest.json", manifest),
    ):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        paths.append(path)
    config_path = tmp_path / "final_config.json"
    frozen = freeze_final_config(config_path, paths[0], paths[1], paths[2], paths[3])
    assert frozen["model_family"] == "multinomial_naive_bayes"
    assert frozen["model_parameters"] == {"alpha": 0.1, "fit_prior": False}
    assert frozen["test_selection_used"] is False


def test_final_artifact_loads_and_contains_pipeline_parts(tmp_path: Path) -> None:
    model = MultinomialNB().fit(
        [[1, 0], [0, 1], [1, 1]], ["false", "true", "unverified"]
    )
    artifact_path = tmp_path / "final.joblib"
    joblib.dump(
        {
            "model": model,
            "vectorizer": TfidfVectorizer().fit(["alpha", "beta"]),
            "preprocessing_configuration": {},
            "labels": ["false", "true", "unverified"],
            "model_version": "test",
        },
        artifact_path,
    )
    artifact = load_final_artifact(artifact_path)
    assert artifact["model_version"] == "test"
    assert hasattr(artifact["model"], "predict_proba")
    assert hasattr(artifact["vectorizer"], "transform")


def test_final_metrics_and_event_report_have_expected_shapes() -> None:
    y_true = pd.Series(["false", "true", "unverified", "true"])
    probabilities = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7], [0.2, 0.6, 0.2]]
    )
    predictions = np.array(["false", "true", "unverified", "true"])
    metrics = _metrics(y_true, predictions, probabilities)
    assert len(metrics["confusion_matrix"]) == 3
    assert metrics["log_loss"] is not None
    test = pd.DataFrame(
        {"event": ["event-a", "event-a", "event-b", "event-b"], "label": y_true}
    )
    event_report = _event_metrics(test, predictions, probabilities)
    assert set(event_report) == {"event-a", "event-b"}
    assert event_report["event-b"]["macro_f1"] is None


def test_missing_final_artifact_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Final model artifact not found"):
        load_final_artifact(tmp_path / "missing.joblib")
