"""Run reproducible exploratory data analysis for the misinformation datasets.

The analyzer prefers processed PHEME data and falls back to the synthetic
development sample when PHEME is unavailable. It never merges the datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

PHEME_PATH = Path("backend/data/processed/pheme_veracity.csv")
SAMPLE_PATH = Path("backend/data/sample_dataset.csv")
REPORT_PATH = Path("backend/data/reports/eda_report.json")
CHART_DIR = Path("backend/data/reports/eda")
REQUIRED_COLUMNS = {"text", "label"}
SHORT_TEXT_THRESHOLD = 5
NEAR_DUPLICATE_THRESHOLD = 0.9
STOP_WORDS = {
    "a",
    "about",
    "after",
    "all",
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
    "have",
    "if",
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
    "were",
    "with",
}


def detect_dataset_source(path: Path) -> str:
    """Return a stable source identifier without inspecting record content."""

    if path.resolve() == PHEME_PATH.resolve() or "pheme" in path.stem.lower():
        return "real_pheme"
    if path.resolve() == SAMPLE_PATH.resolve() or path.name == SAMPLE_PATH.name:
        return "synthetic_sample"
    return "user_supplied"


def select_dataset(
    pheme_path: Path = PHEME_PATH, sample_path: Path = SAMPLE_PATH
) -> tuple[Path, str]:
    """Select PHEME when prepared, otherwise use the development sample."""

    if pheme_path.is_file():
        return pheme_path, "real_pheme"
    if sample_path.is_file():
        return sample_path, "synthetic_sample"
    raise FileNotFoundError(
        "Neither processed PHEME nor the synthetic sample dataset was found."
    )


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV and validate the normalized dataset schema."""

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(sorted(missing))}"
        )
    return frame


def _safe_text(value: Any) -> str:
    return "" if pd.isna(value) else str(value)


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", _safe_text(value)).strip().casefold()


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"(?u)\b\w[\w'-]*\b", text.casefold())


def _top_ngrams(
    texts: list[str], ngram_range: tuple[int, int], limit: int = 15
) -> list[dict[str, Any]]:
    if not any(texts):
        return []
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words=list(STOP_WORDS),
        ngram_range=ngram_range,
        token_pattern=r"(?u)\b\w[\w'-]*\b",
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    order = np.argsort(-counts, kind="stable")[:limit]
    return [{"term": terms[index], "count": int(counts[index])} for index in order]


def _common_words(texts: list[str], limit: int = 15) -> list[dict[str, Any]]:
    counter = Counter(
        token
        for text in texts
        for token in _word_tokens(text)
        if token not in STOP_WORDS and len(token) > 1
    )
    return [
        {"term": word, "count": count} for word, count in counter.most_common(limit)
    ]


def _near_duplicate_pairs(texts: list[str]) -> list[dict[str, Any]]:
    """Find practical near-duplicates using token-set Jaccard similarity."""

    token_sets = [set(_word_tokens(text)) for text in texts]
    pairs: list[dict[str, Any]] = []
    for left in range(len(token_sets)):
        for right in range(left + 1, len(token_sets)):
            union = token_sets[left] | token_sets[right]
            similarity = (
                len(token_sets[left] & token_sets[right]) / len(union) if union else 1.0
            )
            if similarity >= NEAR_DUPLICATE_THRESHOLD and texts[left] != texts[right]:
                pairs.append(
                    {
                        "left_index": left,
                        "right_index": right,
                        "jaccard_similarity": round(similarity, 4),
                    }
                )
    return pairs


def _pattern_report(texts: list[str]) -> dict[str, Any]:
    joined = "\n".join(texts)
    emoji_pattern = r"[\U0001F1E6-\U0001FAFF\u2600-\u27BF]"
    punctuation = Counter(
        character for character in joined if character in string.punctuation
    )
    return {
        "records_with_urls": int(
            sum(
                bool(re.search(r"https?://\S+|www\.\S+", text, re.IGNORECASE))
                for text in texts
            )
        ),
        "records_with_hashtags": int(sum("#" in text for text in texts)),
        "records_with_mentions": int(sum("@" in text for text in texts)),
        "records_with_numbers": int(
            sum(bool(re.search(r"\d", text)) for text in texts)
        ),
        "records_with_emojis": int(
            sum(bool(re.search(emoji_pattern, text)) for text in texts)
        ),
        "punctuation_counts": dict(punctuation.most_common()),
    }


def _duplicate_report(frame: pd.DataFrame) -> dict[str, Any]:
    text_keys = frame["text"].map(_normalized_text)
    duplicate_text_mask = text_keys.duplicated(keep=False)
    by_text = frame.assign(_text_key=text_keys).groupby("_text_key", dropna=False)
    repeated_groups: list[dict[str, Any]] = []
    cross_label_groups: list[dict[str, Any]] = []
    for key, group in by_text:
        if not key or len(group) < 2:
            continue
        labels = sorted({_safe_text(value) for value in group["label"]})
        item = {"text": key, "count": len(group), "labels": labels}
        repeated_groups.append(item)
        if len(labels) > 1:
            cross_label_groups.append(item)
    return {
        "duplicate_rows": int(frame.duplicated(keep=False).sum()),
        "duplicate_text_rows": int(duplicate_text_mask.sum()),
        "duplicate_text_groups": len(repeated_groups),
        "repeated_text_groups": repeated_groups,
        "same_text_with_different_labels": cross_label_groups,
    }


def _leakage_report(
    frame: pd.DataFrame, source: str, duplicates: dict[str, Any]
) -> dict[str, Any]:
    metadata_columns = [
        column for column in frame.columns if column not in REQUIRED_COLUMNS
    ]
    event_column = "source_event" if "source_event" in frame.columns else "event"
    thread_column = (
        "source_tweet_id" if "source_tweet_id" in frame.columns else "thread_id"
    )
    return {
        "exact_duplicate_texts": duplicates["duplicate_text_groups"],
        "near_duplicate_pairs": _near_duplicate_pairs(
            frame["text"].map(_safe_text).tolist()
        ),
        "same_text_with_different_labels": duplicates[
            "same_text_with_different_labels"
        ],
        "metadata_columns_that_require_review": metadata_columns,
        "random_split_risk": (
            "High if related PHEME event/thread records are split across partitions."
            if source == "real_pheme"
            else "Unknown for a synthetic sample; inspect duplicates before any split."
        ),
        "recommended_split": (
            f"Group by {event_column} and {thread_column} before train/validation/test splitting."
            if source == "real_pheme" and event_column in frame.columns
            else "Use a fixed, documented split after reviewing duplicates."
        ),
        "metadata_label_leakage_note": (
            "Do not use source_event, source_tweet_id, annotation metadata, or future reactions as text features."
            if source == "real_pheme"
            else "No source metadata is present in the sample; verify feature columns before training."
        ),
    }


def _write_charts(frame: pd.DataFrame, source: str, output_dir: Path) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    source_title = (
        "PHEME real data" if source == "real_pheme" else "synthetic development data"
    )
    chart_paths: list[str] = []

    def save(name: str) -> None:
        path = output_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        chart_paths.append(str(path))

    labels = frame["label"].fillna("<missing>").astype(str).value_counts()
    labels.plot(
        kind="bar", color="#2f6fed", title=f"Label distribution — {source_title}"
    )
    plt.xlabel("Label")
    plt.ylabel("Records")
    save("label_distribution.png")

    lengths = frame["text"].map(_safe_text).str.len()
    lengths.plot(
        kind="hist",
        bins=min(15, max(1, len(frame))),
        color="#24a88f",
        title=f"Text length distribution — {source_title}",
    )
    plt.xlabel("Characters")
    plt.ylabel("Records")
    save("text_length_distribution.png")

    frame.assign(_length=lengths).boxplot(column="_length", by="label", grid=False)
    plt.suptitle("")
    plt.title(f"Text length by label — {source_title}")
    plt.xlabel("Label")
    plt.ylabel("Characters")
    save("text_length_by_label.png")

    top_by_label = []
    for label, group in frame.groupby("label", dropna=False):
        for item in _common_words(group["text"].map(_safe_text).tolist(), limit=8):
            top_by_label.append({"label": str(label), **item})
    words = pd.DataFrame(top_by_label)
    if not words.empty:
        words["term_label"] = words["term"] + " (" + words["label"] + ")"
        words.set_index("term_label")["count"].sort_values().plot(
            kind="barh", color="#f59e0b", title=f"Top words by label — {source_title}"
        )
    else:
        plt.text(0.5, 0.5, "No vocabulary available", ha="center", va="center")
        plt.title(f"Top words by label — {source_title}")
    plt.xlabel("Occurrences")
    save("top_words_by_label.png")

    bigrams = _top_ngrams(frame["text"].map(_safe_text).tolist(), (2, 2), limit=15)
    if bigrams:
        bigram_frame = pd.DataFrame(bigrams).set_index("term")
        bigram_frame["count"].sort_values().plot(
            kind="barh", color="#8b5cf6", title=f"Top bigrams — {source_title}"
        )
    else:
        plt.text(0.5, 0.5, "No bigrams available", ha="center", va="center")
        plt.title(f"Top bigrams — {source_title}")
    plt.xlabel("Occurrences")
    save("top_bigrams.png")

    missing = frame.isna().sum().sort_values(ascending=False)
    missing.plot(kind="bar", color="#ef4444", title=f"Missing values — {source_title}")
    plt.xlabel("Column")
    plt.ylabel("Missing values")
    save("missing_values.png")

    duplicate_values = {
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_text_rows": int(
            frame["text"].map(_normalized_text).duplicated().sum()
        ),
        "near_duplicate_pairs": len(
            _near_duplicate_pairs(frame["text"].map(_safe_text).tolist())
        ),
    }
    pd.Series(duplicate_values).plot(
        kind="bar", color="#64748b", title=f"Duplicate summary — {source_title}"
    )
    plt.xlabel("Check")
    plt.ylabel("Count")
    save("duplicate_summary.png")
    return chart_paths


def analyze_dataset(
    frame: pd.DataFrame, source: str, dataset_path: Path
) -> dict[str, Any]:
    texts = frame["text"].map(_safe_text)
    char_lengths = texts.str.len()
    word_lengths = texts.map(lambda text: len(_word_tokens(text)))
    labels = frame["label"].fillna("<missing>").astype(str)
    distribution = labels.value_counts().to_dict()
    total = len(frame)
    duplicate_info = _duplicate_report(frame)
    common_words = {
        str(label): _common_words(group["text"].map(_safe_text).tolist())
        for label, group in frame.assign(label=labels).groupby("label", dropna=False)
    }
    report: dict[str, Any] = {
        "dataset": {
            "path": str(dataset_path),
            "source": source,
            "source_description": (
                "real PHEME data"
                if source == "real_pheme"
                else "synthetic development data"
            ),
        },
        "records": total,
        "columns": [
            {"name": column, "dtype": str(dtype)}
            for column, dtype in frame.dtypes.items()
        ],
        "missing_values": {
            column: int(value) for column, value in frame.isna().sum().items()
        },
        "duplicate_analysis": duplicate_info,
        "label_distribution": {
            str(key): int(value) for key, value in distribution.items()
        },
        "label_percentages": (
            {
                str(key): round(value / total * 100, 2)
                for key, value in distribution.items()
            }
            if total
            else {}
        ),
        "text_length": {
            "characters": {
                "minimum": int(char_lengths.min()) if total else 0,
                "maximum": int(char_lengths.max()) if total else 0,
                "mean": round(float(char_lengths.mean()), 2) if total else 0,
                "median": round(float(char_lengths.median()), 2) if total else 0,
                "standard_deviation": (
                    round(float(char_lengths.std(ddof=1)), 2) if total > 1 else 0
                ),
            },
            "words": {
                "minimum": int(word_lengths.min()) if total else 0,
                "maximum": int(word_lengths.max()) if total else 0,
                "mean": round(float(word_lengths.mean()), 2) if total else 0,
                "median": round(float(word_lengths.median()), 2) if total else 0,
                "standard_deviation": (
                    round(float(word_lengths.std(ddof=1)), 2) if total > 1 else 0
                ),
            },
        },
        "short_messages": {
            "threshold_characters": SHORT_TEXT_THRESHOLD,
            "empty_messages": int((texts.str.strip() == "").sum()),
            "extremely_short_messages": int(
                (char_lengths <= SHORT_TEXT_THRESHOLD).sum()
            ),
        },
        "common_words_by_label": common_words,
        "common_unigrams": _top_ngrams(texts.tolist(), (1, 1)),
        "common_bigrams": _top_ngrams(texts.tolist(), (2, 2)),
        "linguistic_patterns": _pattern_report(texts.tolist()),
        "class_balance": {
            "largest_label_percentage": (
                round(max(distribution.values()) / total * 100, 2) if total else 0
            ),
            "severely_imbalanced": bool(
                total and max(distribution.values()) / total >= 0.8
            ),
            "threshold": "Severe imbalance is flagged when one label is at least 80% of records.",
        },
        "dataset_structure": {
            "event_column": "event" if "event" in frame.columns else None,
            "thread_column": "thread_id" if "thread_id" in frame.columns else None,
            "event_count": (
                int(frame["event"].nunique()) if "event" in frame.columns else None
            ),
            "thread_count": (
                int(frame["thread_id"].nunique())
                if "thread_id" in frame.columns
                else None
            ),
            "event_distribution": (
                {
                    str(key): int(value)
                    for key, value in frame["event"].value_counts().to_dict().items()
                }
                if "event" in frame.columns
                else {}
            ),
        },
    }
    report["leakage_analysis"] = _leakage_report(frame, source, duplicate_info)
    return report


def run_eda(
    dataset_path: Path | None = None,
    report_path: Path = REPORT_PATH,
    chart_dir: Path = CHART_DIR,
    pheme_path: Path = PHEME_PATH,
    sample_path: Path = SAMPLE_PATH,
) -> dict[str, Any]:
    selected_path, source = (
        (dataset_path, detect_dataset_source(dataset_path))
        if dataset_path is not None
        else select_dataset(pheme_path, sample_path)
    )
    frame = load_dataset(selected_path)
    report = analyze_dataset(frame, source, selected_path)
    report["charts"] = _write_charts(frame, source, chart_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Explicit CSV path; otherwise prefer PHEME then sample.",
    )
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--chart-dir", type=Path, default=CHART_DIR)
    args = parser.parse_args()
    report = run_eda(args.dataset, args.report, args.chart_dir)
    dataset = report["dataset"]
    print(f"Analyzed {dataset['source_description']}: {dataset['path']}")
    print(json.dumps(report, indent=2))
    if dataset["source"] == "synthetic_sample":
        print(
            "PHEME is unavailable. After preparation, rerun: make prepare-dataset && make analyze-dataset"
        )


if __name__ == "__main__":
    main()
