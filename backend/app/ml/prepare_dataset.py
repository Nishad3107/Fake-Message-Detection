"""Prepare extracted PHEME files without modifying the raw dataset.

The PHEME release is organized by event, then ``rumours``/``non-rumours``,
then tweet-id directories containing ``source-tweets`` JSON and annotations.
This script creates a veracity dataset from annotated rumor directories only.
Non-rumor directories are counted and skipped because the preferred academic
labels for this project are true, false, and unverified.
"""

import argparse
import csv
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
OUTPUT_COLUMNS = ["text", "label", "event", "thread_id", "source_tweet_id"]
LABELS = {"true", "false", "unverified"}


def _truthy(value: Any) -> bool:
    return value is True or value == 1 or str(value).lower() in {"true", "1", "yes"}


def _normalize_label(annotation: dict[str, Any]) -> str | None:
    direct = annotation.get("veracity") or annotation.get("label")
    if isinstance(direct, str) and direct.lower() in LABELS:
        return direct.lower()
    if _truthy(annotation.get("true")):
        return "true"
    if _truthy(annotation.get("misinformation")):
        return "false"
    if (
        ("true" in annotation and "misinformation" in annotation)
        and not _truthy(annotation.get("true"))
        and not _truthy(annotation.get("misinformation"))
    ):
        return "unverified"
    return None


def _find_source_text(tweet_dir: Path) -> str | None:
    candidates = sorted((tweet_dir / "source-tweets").glob("*.json"))
    candidates += sorted((tweet_dir / "source-tweet").glob("*.json"))
    candidates += sorted(tweet_dir.glob("*.json"))
    for candidate in candidates:
        if candidate.name in {"annotation.json", "structure.json"}:
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Unable to parse tweet file: %s", candidate)
            continue
        text = payload.get("text") or payload.get("full_text")
        if isinstance(text, str):
            return text
    return None


def prepare_dataset(
    raw_dir: Path, output_path: Path, report_path: Path
) -> dict[str, Any]:
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"PHEME raw directory not found: {raw_dir}. Download and extract the official release first."
        )
    annotation_files = sorted(raw_dir.rglob("annotation.json"))
    if not annotation_files:
        raise FileNotFoundError(
            f"No annotation.json files found under {raw_dir}. Expected the extracted PHEME event tree."
        )

    rows: list[dict[str, str]] = []
    labels = Counter()
    seen_text: set[str] = set()
    duplicate_count = 0
    missing_values = 0
    skipped_non_rumours = 0
    skipped_unlabelled = 0
    raw_records = len(annotation_files)

    for annotation_path in annotation_files:
        tweet_dir = annotation_path.parent
        category_dir = tweet_dir.parent
        event_dir = category_dir.parent
        if category_dir.name != "rumours":
            skipped_non_rumours += 1
            continue
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            missing_values += 1
            continue
        label = _normalize_label(annotation)
        text = _find_source_text(tweet_dir)
        if not label or not text or not text.strip():
            skipped_unlabelled += 1
            missing_values += 1
            continue
        normalized_key = re.sub(r"\s+", " ", text).strip().casefold()
        if normalized_key in seen_text:
            duplicate_count += 1
            continue
        seen_text.add(normalized_key)
        row = {
            "text": re.sub(r"\s+", " ", text).strip(),
            "label": label,
            "event": event_dir.name,
            "thread_id": tweet_dir.name,
            "source_tweet_id": tweet_dir.name,
        }
        rows.append(row)
        labels[label] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    lengths = [len(row["text"]) for row in rows]
    report: dict[str, Any] = {
        "raw_records": raw_records,
        "processed_records": len(rows),
        "removed_records": raw_records - len(rows),
        "duplicate_count": duplicate_count,
        "missing_value_count": missing_values,
        "skipped_non_rumour_records": skipped_non_rumours,
        "skipped_unlabelled_records": skipped_unlabelled,
        "label_distribution": dict(sorted(labels.items())),
        "average_text_length": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "minimum_text_length": min(lengths) if lengths else 0,
        "maximum_text_length": max(lengths) if lengths else 0,
        "language_assumption": "English; language detection is not performed by this preparation script.",
        "schema": OUTPUT_COLUMNS,
        "source_preserved": True,
        "label_policy": "Only annotated rumours with true, false, or unverified veracity labels are included.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("backend/data/raw/pheme"))
    parser.add_argument(
        "--output", type=Path, default=Path("backend/data/processed/pheme_veracity.csv")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("backend/data/reports/pheme_quality.json")
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(json.dumps(prepare_dataset(args.raw_dir, args.output, args.report), indent=2))


if __name__ == "__main__":
    main()
