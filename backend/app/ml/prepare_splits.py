"""Apply stateless preprocessing to the leakage-safe PHEME split manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.create_splits import DATASET_PATH, MANIFEST_PATH, REQUIRED_COLUMNS
from app.ml.preprocess_text import (
    PreprocessingOptions,
    _statistics,
    inspect_texts,
    options_as_dict,
    preprocess_text,
)

OUTPUT_DIR = Path("backend/data/processed/splits")
REPORT_PATH = Path("backend/data/reports/preprocessing_report.json")
OUTPUT_COLUMNS = [
    "record_id",
    "text_original",
    "text_clean",
    "label",
    "event",
    "thread_id",
    "source_tweet_id",
    "split",
]


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"events", "record_counts", "event_overlap", "thread_overlap"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(
            f"Split manifest is missing required fields: {', '.join(sorted(missing))}"
        )
    if manifest["event_overlap"] or manifest["thread_overlap"]:
        raise ValueError("Split manifest contains event or thread overlap")
    manifest_events = [
        event for events in manifest["events"].values() for event in events
    ]
    if len(manifest_events) != len(set(manifest_events)):
        raise ValueError("Split manifest assigns an event to more than one split")
    return manifest


def _assign_splits(frame: pd.DataFrame, manifest: dict[str, Any]) -> pd.Series:
    event_to_split = {
        event: split for split, events in manifest["events"].items() for event in events
    }
    assignments = frame["event"].map(event_to_split)
    if assignments.isna().any():
        missing = sorted(frame.loc[assignments.isna(), "event"].unique())
        raise ValueError(f"Events missing from split manifest: {', '.join(missing)}")
    if len(event_to_split) != frame["event"].nunique():
        raise ValueError(
            "Split manifest includes events not present in the processed dataset"
        )
    return assignments


def prepare_splits(
    dataset_path: Path = DATASET_PATH,
    manifest_path: Path = MANIFEST_PATH,
    output_dir: Path = OUTPUT_DIR,
    report_path: Path = REPORT_PATH,
    options: PreprocessingOptions | None = None,
) -> dict[str, Any]:
    frame = pd.read_csv(dataset_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Processed dataset is missing required columns: {', '.join(sorted(missing))}"
        )
    manifest = load_manifest(manifest_path)
    assignments = _assign_splits(frame, manifest)

    original_text = frame["text"].fillna("").astype(str)
    cleaned_text = original_text.map(lambda value: preprocess_text(value, options))
    prepared = pd.DataFrame(
        {
            "record_id": range(len(frame)),
            "text_original": original_text,
            "text_clean": cleaned_text,
            "label": frame["label"].astype(str),
            "event": frame["event"].astype(str),
            "thread_id": frame["thread_id"].astype(str),
            "source_tweet_id": frame["source_tweet_id"].astype(str),
            "split": assignments.astype(str),
        }
    )
    if len(prepared) != len(frame) or set(prepared["record_id"]) != set(
        range(len(frame))
    ):
        raise RuntimeError(
            "Preprocessing did not preserve every input record exactly once"
        )
    if not prepared["label"].equals(frame["label"].astype(str)):
        raise RuntimeError("Preprocessing changed labels")
    for column in ("event", "thread_id", "source_tweet_id"):
        if not prepared[column].equals(frame[column].astype(str)):
            raise RuntimeError(f"Preprocessing changed metadata column: {column}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        prepared.loc[prepared["split"] == split, OUTPUT_COLUMNS].to_csv(
            output_dir / f"{split}.csv", index=False
        )

    before = inspect_texts(original_text)
    after_lengths = prepared["text_clean"].str.len().tolist()
    after_words = prepared["text_clean"].map(lambda value: len(value.split())).tolist()
    report: dict[str, Any] = {
        "input_path": str(dataset_path),
        "output_directory": str(output_dir),
        "input_record_count": len(frame),
        "output_record_count": len(prepared),
        "raw_text_inspection": before,
        "empty_text_count_before": before["empty_messages"],
        "empty_text_count_after": int(prepared["text_clean"].eq("").sum()),
        "url_count": before["urls"],
        "mention_count": before["mentions"],
        "hashtag_count": before["hashtags"],
        "character_statistics_before": before["character_statistics"],
        "character_statistics_after": _statistics(after_lengths),
        "word_statistics_before": before["word_statistics"],
        "word_statistics_after": _statistics(after_words),
        "changed_text_count": int(
            (prepared["text_original"] != prepared["text_clean"]).sum()
        ),
        "unchanged_text_count": int(
            (prepared["text_original"] == prepared["text_clean"]).sum()
        ),
        "split_counts": {
            split: int((prepared["split"] == split).sum())
            for split in ("train", "validation", "test")
        },
        "label_counts": {
            split: {
                str(label): int(count)
                for label, count in prepared.loc[prepared["split"] == split, "label"]
                .value_counts()
                .to_dict()
                .items()
            }
            for split in ("train", "validation", "test")
        },
        "preprocessing_options": options_as_dict(options),
        "labels_preserved": prepared["label"].equals(frame["label"].astype(str)),
        "metadata_preserved": all(
            prepared[column].equals(frame[column].astype(str))
            for column in ("event", "thread_id", "source_tweet_id")
        ),
        "split_assignments_preserved": set(prepared["split"])
        == {"train", "validation", "test"},
        "vocabulary_learning": "None; this command performs stateless cleaning only. Fit vectorizers on train text in a later training step.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    report = prepare_splits(args.dataset, args.manifest, args.output_dir, args.report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
