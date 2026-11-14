"""Stateless, explainable text preprocessing for PHEME source posts."""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PreprocessingOptions:
    """Options that do not learn from labels or corpus vocabulary."""

    lowercase: bool = True
    replace_urls: bool = True
    replace_mentions: bool = True
    keep_hashtags: bool = True
    normalize_whitespace: bool = True
    remove_extra_punctuation: bool = True
    remove_retweet_marker: bool = True


URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<![\w@])@[A-Za-z0-9_]+")
HASHTAG_PATTERN = re.compile(r"#([\w]+)", re.UNICODE)
RETWEET_PATTERN = re.compile(r"^\s*(?:rt|via)\s+", re.IGNORECASE)
REPEATED_PUNCTUATION_PATTERN = re.compile(r"([!?.,:;])\1{1,}")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
ESCAPED_ENTITY_PATTERN = re.compile(
    r"&(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-f]+);", re.IGNORECASE
)


def preprocess_text(text: Any, options: PreprocessingOptions | None = None) -> str:
    """Clean one text value deterministically without using labels or corpus state."""

    settings = options or PreprocessingOptions()
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = html.unescape(value)
    if settings.remove_retweet_marker:
        value = RETWEET_PATTERN.sub("", value)
    if settings.replace_urls:
        value = URL_PATTERN.sub(" URL ", value)
    if settings.replace_mentions:
        value = MENTION_PATTERN.sub(" USER ", value)
    if settings.keep_hashtags:
        value = HASHTAG_PATTERN.sub(r"\1", value)
    else:
        value = re.sub(r"#[\w]+", " ", value, flags=re.UNICODE)
    if settings.lowercase:
        value = value.casefold()
    if settings.remove_extra_punctuation:
        value = REPEATED_PUNCTUATION_PATTERN.sub(r"\1", value)
    if settings.normalize_whitespace:
        value = re.sub(r"\s+", " ", value).strip()
        value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    return value


def options_as_dict(options: PreprocessingOptions | None = None) -> dict[str, bool]:
    """Return serializable options for reports and experiment manifests."""

    return {
        key: bool(value)
        for key, value in asdict(options or PreprocessingOptions()).items()
    }


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value)


def _word_count(text: str) -> int:
    return len(re.findall(r"(?u)\b\w[\w'-]*\b", text))


def inspect_texts(
    texts: Iterable[Any], short_threshold: int = 5, long_threshold: int = 280
) -> dict[str, Any]:
    """Measure raw text patterns before any cleaning is applied."""

    values = [_safe_text(text) for text in texts]
    normalized = [re.sub(r"\s+", " ", text).strip().casefold() for text in values]
    lengths = [len(text) for text in values]
    words = [_word_count(text) for text in values]
    duplicate_counts: dict[str, int] = {}
    for text in normalized:
        if text:
            duplicate_counts[text] = duplicate_counts.get(text, 0) + 1
    duplicate_groups = [count for count in duplicate_counts.values() if count > 1]
    return {
        "records": len(values),
        "empty_messages": sum(not text for text in values),
        "whitespace_only_messages": sum(
            bool(text) and not text.strip() for text in values
        ),
        "urls": sum(bool(URL_PATTERN.search(text)) for text in values),
        "mentions": sum(bool(MENTION_PATTERN.search(text)) for text in values),
        "hashtags": sum(bool(HASHTAG_PATTERN.search(text)) for text in values),
        "numbers": sum(bool(re.search(r"\d", text)) for text in values),
        "emojis": sum(
            bool(re.search(r"[\U0001F1E6-\U0001FAFF\u2600-\u27BF]", text))
            for text in values
        ),
        "repeated_punctuation": sum(
            bool(REPEATED_PUNCTUATION_PATTERN.search(text)) for text in values
        ),
        "html_or_escaped_characters": sum(
            bool(HTML_TAG_PATTERN.search(text) or ESCAPED_ENTITY_PATTERN.search(text))
            for text in values
        ),
        "unicode_characters": sum(
            any(ord(character) > 127 for character in text) for text in values
        ),
        "very_short_messages": {
            "threshold_characters": short_threshold,
            "count": sum(len(text.strip()) <= short_threshold for text in values),
        },
        "very_long_messages": {
            "threshold_characters": long_threshold,
            "count": sum(len(text) >= long_threshold for text in values),
        },
        "retweet_or_platform_prefixes": sum(
            bool(RETWEET_PATTERN.search(text)) for text in values
        ),
        "duplicate_text_groups": len(duplicate_groups),
        "duplicate_text_rows": sum(duplicate_groups),
        "character_statistics": _statistics(lengths),
        "word_statistics": _statistics(words),
    }


def _statistics(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {
            "minimum": 0,
            "maximum": 0,
            "mean": 0,
            "median": 0,
            "standard_deviation": 0,
        }
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    mean = sum(values) / len(values)
    variance = (
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        if len(values) > 1
        else 0
    )
    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": round(mean, 2),
        "median": median,
        "standard_deviation": round(variance**0.5, 2),
    }
