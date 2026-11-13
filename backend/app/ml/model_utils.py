"""Shared preprocessing, model paths, and keyword helpers."""

import re
from pathlib import Path

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
    "will",
}
KEYWORD_TERMS = {
    "breaking",
    "urgent",
    "shocking",
    "miracle",
    "secret",
    "exposed",
    "share",
    "viral",
    "official",
    "study",
    "research",
    "evidence",
    "report",
    "claim",
    "warning",
    "confirmed",
}


def clean_text(text: str) -> str:
    """Normalize text while retaining words that are useful to the baseline model."""
    normalized = text.lower()
    normalized = re.sub(r"https?://\S+|www\.\S+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def detect_keywords(text: str) -> list[str]:
    tokens = set(clean_text(text).split())
    return sorted(tokens.intersection(KEYWORD_TERMS))


def model_paths(model_dir: Path) -> tuple[Path, Path]:
    return model_dir / "classifier.joblib", model_dir / "vectorizer.joblib"
