from pathlib import Path

import joblib
import pandas as pd
import pytest
from app.ml.error_analysis import (
    LABELS,
    confidence_analysis,
    confusion_pair_analysis,
    event_error_analysis,
    feature_explanations,
    load_aligned_data,
)
from app.ml.preprocess_text import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event": "event-a",
                "thread_id": "a-1",
                "source_tweet_id": "a-1",
                "true_label": "false",
                "predicted_label": "true",
                "confidence": 0.91,
                "text_original": "False story",
                "text_clean": "false story",
            },
            {
                "event": "event-a",
                "thread_id": "a-2",
                "source_tweet_id": "a-2",
                "true_label": "true",
                "predicted_label": "true",
                "confidence": 0.45,
                "text_original": "True report",
                "text_clean": "true report",
            },
            {
                "event": "event-a",
                "thread_id": "a-3",
                "source_tweet_id": "a-3",
                "true_label": "unverified",
                "predicted_label": "false",
                "confidence": 0.65,
                "text_original": "Unknown claim",
                "text_clean": "unknown claim",
            },
            {
                "event": "event-b",
                "thread_id": "b-1",
                "source_tweet_id": "b-1",
                "true_label": "true",
                "predicted_label": "unverified",
                "confidence": 0.55,
                "text_original": "Another report",
                "text_clean": "another report",
            },
        ]
    )


def test_confusion_pairs_and_event_metrics() -> None:
    data = _rows()
    pairs = confusion_pair_analysis(data)
    assert pairs["false_to_true"]["count"] == 1
    assert pairs["true_to_unverified"]["count"] == 1
    events = event_error_analysis(data)
    assert events["event-a"]["records"] == 3
    assert events["event-a"]["macro_f1"] == 0.222222
    assert events["event-b"]["macro_f1"] is None


def test_confidence_bands_and_rates() -> None:
    result = confidence_analysis(_rows())
    assert sum(band["count"] for band in result["bands"].values()) == 4
    assert result["high_confidence_wrong_count"] == 1
    assert result["low_confidence_correct_count"] == 1


def test_feature_explanations_preserve_class_order() -> None:
    texts = ["false claim", "true report", "unverified claim", "true claim"]
    labels = ["false", "true", "unverified", "true"]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    model = MultinomialNB().fit(matrix, labels)
    result = feature_explanations(
        {"model": model, "vectorizer": vectorizer, "labels": LABELS}, top_n=3
    )
    assert result["labels"] == LABELS
    assert all(result["top_features_by_class"][label] for label in LABELS)


def test_alignment_validates_original_and_cleaned_text(tmp_path: Path) -> None:
    original = "A useful #Claim https://example.test"
    cleaned = preprocess_text(original)
    test = pd.DataFrame(
        [
            {
                "event": "event-a",
                "thread_id": "thread-a",
                "source_tweet_id": "tweet-a",
                "text_original": original,
                "text_clean": cleaned,
                "label": "true",
                "split": "test",
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "event": "event-a",
                "thread_id": "thread-a",
                "source_tweet_id": "tweet-a",
                "label": "true",
                "text_original": original,
                "prediction": "true",
                "confidence": 0.8,
                "probability_false": 0.1,
                "probability_true": 0.8,
                "probability_unverified": 0.1,
            }
        ]
    )
    test_path = tmp_path / "test.csv"
    predictions_path = tmp_path / "predictions.csv"
    test.to_csv(test_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    aligned, validation = load_aligned_data(test_path, predictions_path)
    assert len(aligned) == 1
    assert validation["original_text_aligned"] is True
    assert validation["cleaned_text_aligned"] is True
    assert validation["predictions_from_test_split"] is True


def test_missing_predictions_fail_clearly(tmp_path: Path) -> None:
    test = _rows().rename(
        columns={"true_label": "label", "predicted_label": "prediction"}
    )
    test_path = tmp_path / "test.csv"
    test.to_csv(test_path, index=False)
    with pytest.raises(FileNotFoundError, match="Final predictions not found"):
        load_aligned_data(test_path, tmp_path / "missing.csv")


def test_frozen_artifact_is_not_written(tmp_path: Path) -> None:
    vectorizer = TfidfVectorizer().fit(
        ["false claim", "true report", "unverified claim"]
    )
    model = MultinomialNB().fit(
        vectorizer.transform(["false claim", "true report", "unverified claim"]),
        LABELS,
    )
    artifact_path = tmp_path / "artifact.joblib"
    joblib.dump(
        {"model": model, "vectorizer": vectorizer, "labels": LABELS}, artifact_path
    )
    before = artifact_path.read_bytes()
    feature_explanations(joblib.load(artifact_path))
    assert artifact_path.read_bytes() == before
