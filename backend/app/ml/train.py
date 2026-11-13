"""Train and evaluate the explainable TF-IDF + Logistic Regression baseline."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from app.ml.model_utils import clean_text, model_paths


def train_model(dataset_path: Path, model_dir: Path) -> dict[str, object]:
    dataset = pd.read_csv(dataset_path)
    required = {"text", "label"}
    if not required.issubset(dataset.columns):
        raise ValueError("Dataset must contain text and label columns")
    dataset = dataset.dropna(subset=["text", "label"])
    texts = dataset["text"].astype(str).map(clean_text)
    labels = dataset["label"].astype(str)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
    x_train_vectors = vectorizer.fit_transform(x_train)
    classifier = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    )
    classifier.fit(x_train_vectors, y_train)
    predictions = classifier.predict(vectorizer.transform(x_test))
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(
                y_test, predictions, pos_label="misinformation", zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, predictions, pos_label="misinformation", zero_division=0
            )
        ),
        "f1": float(
            f1_score(y_test, predictions, pos_label="misinformation", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=["likely_true", "misinformation"]
        ).tolist(),
        "labels": ["likely_true", "misinformation"],
        "test_samples": len(y_test),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    classifier_path, vectorizer_path = model_paths(model_dir)
    joblib.dump(classifier, classifier_path)
    joblib.dump(vectorizer, vectorizer_path)
    (model_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/sample_dataset.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    print(json.dumps(train_model(args.dataset, args.model_dir), indent=2))
