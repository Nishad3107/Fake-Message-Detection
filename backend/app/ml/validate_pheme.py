"""Validate the extracted PHEME directory without modifying it."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.ml.prepare_dataset import _normalize_label


def _json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(value, dict):
        return None, "JSON value is not an object"
    return value, None


def _source_files(thread_dir: Path) -> list[Path]:
    files = sorted((thread_dir / "source-tweets").glob("*.json"))
    files += sorted((thread_dir / "source-tweet").glob("*.json"))
    return files


def validate_pheme(raw_dir: Path) -> dict[str, Any]:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"PHEME raw directory not found: {raw_dir}")

    events = sorted(path for path in raw_dir.iterdir() if path.is_dir())
    report: dict[str, Any] = {
        "raw_directory": str(raw_dir),
        "events_found": [event.name for event in events],
        "event_count": len(events),
        "total_threads": 0,
        "rumour_thread_count": 0,
        "non_rumour_thread_count": 0,
        "valid_source_tweet_count": 0,
        "missing_source_tweet_count": 0,
        "valid_annotation_count": 0,
        "invalid_annotation_count": 0,
        "missing_annotation_count": 0,
        "label_counts": Counter(),
        "unsupported_annotation_count": 0,
        "empty_text_count": 0,
        "duplicate_tweet_ids": [],
        "unusable_record_count": 0,
        "warnings": [],
        "assumptions": [
            "Only source tweet JSON is considered for the text dataset; reactions are preserved but not used as records.",
            "PHEME true=1 maps to true, misinformation=1 maps to false, and both flags equal to 0 maps to unverified.",
            "Non-rumour threads are validated and counted but excluded from the veracity processed CSV.",
        ],
    }
    labels: Counter[str] = Counter()
    seen_ids: dict[str, str] = {}
    for event in events:
        categories = {path.name: path for path in event.iterdir() if path.is_dir()}
        for category_name in ("rumours", "non-rumours"):
            category = categories.get(category_name)
            if category is None:
                report["warnings"].append(f"{event.name} is missing {category_name}/")
                continue
            for thread in sorted(path for path in category.iterdir() if path.is_dir()):
                report["total_threads"] += 1
                is_rumour = category_name == "rumours"
                report[
                    "rumour_thread_count" if is_rumour else "non_rumour_thread_count"
                ] += 1
                thread_id = thread.name
                if thread_id in seen_ids:
                    report["duplicate_tweet_ids"].append(
                        {
                            "tweet_id": thread_id,
                            "first_event": seen_ids[thread_id],
                            "duplicate_event": event.name,
                        }
                    )
                else:
                    seen_ids[thread_id] = event.name

                sources = _source_files(thread)
                if not sources:
                    report["missing_source_tweet_count"] += 1
                    report["unusable_record_count"] += 1
                else:
                    source_valid = False
                    for source in sources:
                        payload, error = _json_object(source)
                        if error:
                            report["warnings"].append(
                                f"Malformed source tweet {source}: {error}"
                            )
                            continue
                        text = payload.get("text") or payload.get("full_text")
                        if isinstance(text, str) and text.strip():
                            source_valid = True
                            break
                    if source_valid:
                        report["valid_source_tweet_count"] += 1
                    else:
                        report["empty_text_count"] += 1
                        report["unusable_record_count"] += 1

                annotation_path = thread / "annotation.json"
                if not annotation_path.is_file():
                    report["missing_annotation_count"] += 1
                    if is_rumour:
                        report["unusable_record_count"] += 1
                    continue
                annotation, error = _json_object(annotation_path)
                if error or annotation is None:
                    report["invalid_annotation_count"] += 1
                    report["warnings"].append(
                        f"Invalid annotation {annotation_path}: {error}"
                    )
                    if is_rumour:
                        report["unusable_record_count"] += 1
                    continue
                report["valid_annotation_count"] += 1
                if is_rumour:
                    label = _normalize_label(annotation)
                    if label is None:
                        report["unsupported_annotation_count"] += 1
                        report["unusable_record_count"] += 1
                    else:
                        labels[label] += 1

    report["label_counts"] = dict(sorted(labels.items()))
    if not report["events_found"]:
        report["warnings"].append("No event directories were found.")
    if report["duplicate_tweet_ids"]:
        report["warnings"].append(
            "Duplicate thread IDs were found across events; preserve event and thread grouping."
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("backend/data/raw/pheme"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("backend/data/reports/pheme_validation.json"),
    )
    args = parser.parse_args()
    report = validate_pheme(args.raw_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
