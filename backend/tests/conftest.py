"""CI-only model fixture for API contract tests.

The real frozen PHEME artifact is intentionally ignored and is not required in
CI. Local runs continue to use the real artifact when it is available.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def _configure_ci_fixture() -> None:
    if not os.getenv("CI"):
        return

    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB

    root = Path(tempfile.mkdtemp(prefix="misinformation-ci-"))
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    texts = [
        "confirmed public report",
        "official evidence confirms event",
        "unverified claim needs evidence",
        "rumor may be misleading",
        "false claim lacks evidence",
        "verified report from officials",
    ]
    labels = ["true", "true", "unverified", "false", "false", "true"]
    features = vectorizer.fit_transform(texts)
    model = MultinomialNB(alpha=0.1, fit_prior=False).fit(features, labels)
    artifact = root / "final_pipeline.joblib"
    joblib.dump(
        {
            "model": model,
            "vectorizer": vectorizer,
            "preprocessing_configuration": {
                "lowercase": True,
                "replace_urls": True,
                "replace_mentions": True,
                "keep_hashtags": True,
                "normalize_whitespace": True,
                "remove_extra_punctuation": True,
                "remove_retweet_marker": True,
            },
            "labels": ["false", "true", "unverified"],
            "model_version": "ci-fixture-not-for-production",
            "training_dataset_reference": "CI fixture",
        },
        artifact,
    )
    (root / "config.json").write_text(
        json.dumps({"tfidf_configuration": {"ngram_range": [1, 2], "min_df": 1}}),
        encoding="utf-8",
    )
    (root / "metrics.json").write_text(
        json.dumps({"metrics": {"accuracy": 0.0}}), encoding="utf-8"
    )
    database = root / "history.db"
    os.environ.update(
        {
            "FINAL_MODEL_ARTIFACT": str(artifact),
            "FINAL_MODEL_CONFIG": str(root / "config.json"),
            "FINAL_TEST_RESULTS": str(root / "metrics.json"),
            "DATABASE_PATH": str(database),
        }
    )


_configure_ci_fixture()
