"""Create deterministic leakage-safe event-level PHEME splits."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import pandas as pd

DATASET_PATH = Path("backend/data/processed/pheme_veracity.csv")
MANIFEST_PATH = Path("backend/data/splits/pheme_split_manifest.json")
GROUPING_REPORT_PATH = Path("backend/data/reports/grouping_report.json")
REQUIRED_COLUMNS = {"event", "thread_id", "source_tweet_id", "label", "text"}
LABELS = ("true", "false", "unverified")
RANDOM_SEED = 42


def load_processed_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Processed dataset is missing required columns: {', '.join(sorted(missing))}"
        )
    if frame.empty:
        raise ValueError("Processed dataset is empty")
    return frame


def build_grouping_report(frame: pd.DataFrame) -> dict[str, Any]:
    event_stats: dict[str, Any] = {}
    for event, group in frame.groupby("event", sort=True):
        event_stats[str(event)] = {
            "records": len(group),
            "labels": {
                str(k): int(v)
                for k, v in group["label"].value_counts().to_dict().items()
            },
            "unique_threads": int(group["thread_id"].nunique()),
        }
    thread_sizes = frame.groupby("thread_id").size()
    thread_label_counts = frame.groupby("thread_id")["label"].nunique()
    return {
        "dataset_records": len(frame),
        "columns": list(frame.columns),
        "unique_events": int(frame["event"].nunique()),
        "records_per_event": {
            event: stats["records"] for event, stats in event_stats.items()
        },
        "labels_per_event": event_stats,
        "unique_threads": int(frame["thread_id"].nunique()),
        "records_per_thread": {
            "minimum": int(thread_sizes.min()),
            "maximum": int(thread_sizes.max()),
            "threads_with_multiple_records": int((thread_sizes > 1).sum()),
        },
        "threads_with_multiple_labels": int((thread_label_counts > 1).sum()),
        "events_with_only_one_label": {
            event: stats["labels"]
            for event, stats in event_stats.items()
            if len(stats["labels"]) == 1
        },
        "thread_ids_appearing_more_than_once": [
            str(thread_id) for thread_id, count in thread_sizes.items() if count > 1
        ],
    }


def _score_groups(frame: pd.DataFrame, groups: list[tuple[str, ...]]) -> float:
    target_sizes = (0.65, 0.175, 0.175)
    total = len(frame)
    global_distribution = frame["label"].value_counts(normalize=True)
    score = 0.0
    for index, events in enumerate(groups):
        subset = frame[frame["event"].isin(events)]
        size_ratio = len(subset) / total
        score += (size_ratio - target_sizes[index]) ** 2
        distribution = subset["label"].value_counts(normalize=True)
        score += 0.2 * sum(
            (distribution.get(label, 0.0) - global_distribution.get(label, 0.0)) ** 2
            for label in LABELS
        )
    return score


def choose_event_groups(frame: pd.DataFrame) -> tuple[list[tuple[str, ...]], list[str]]:
    events = tuple(sorted(str(event) for event in frame["event"].unique()))
    if len(events) != 9:
        raise ValueError(
            f"Expected 9 PHEME events for the documented split, found {len(events)}"
        )
    candidates: list[tuple[float, tuple[tuple[str, ...], ...]]] = []
    for train in itertools.combinations(events, 5):
        remaining = tuple(event for event in events if event not in train)
        for validation in itertools.combinations(remaining, 2):
            test = tuple(event for event in remaining if event not in validation)
            groups = [train, validation, test]
            if any(
                set(frame[frame["event"].isin(group)]["label"].unique()) != set(LABELS)
                for group in groups
            ):
                continue
            candidates.append((_score_groups(frame, groups), tuple(groups)))
    if not candidates:
        raise ValueError(
            "No event-level 5/2/2 split contains all three labels in every split"
        )
    _, selected = min(candidates, key=lambda item: (item[0], item[1]))
    warnings = [
        "Only 9 events are available, so the split uses 5 train, 2 validation, and 2 test events.",
        "Event sizes are uneven; validation and test proportions will not exactly match 15-20%.",
        "Three events contain only one label and were allocated without fabricating or removing labels.",
        "The exhaustive selection is deterministic; random_seed is recorded for experiment reproducibility but no random sampling is used.",
    ]
    return [tuple(group) for group in selected], warnings


def create_split_manifest(
    frame: pd.DataFrame,
    manifest_path: Path = MANIFEST_PATH,
    grouping_report_path: Path = GROUPING_REPORT_PATH,
) -> dict[str, Any]:
    grouping_report = build_grouping_report(frame)
    grouping_report_path.parent.mkdir(parents=True, exist_ok=True)
    grouping_report_path.write_text(
        json.dumps(grouping_report, indent=2) + "\n", encoding="utf-8"
    )

    groups, warnings = choose_event_groups(frame)
    split_names = ("train", "validation", "test")
    assignments = {
        split: {event for event in events} for split, events in zip(split_names, groups)
    }
    split_frames = {
        split: frame[frame["event"].isin(events)]
        for split, events in zip(split_names, groups)
    }
    event_sets = list(assignments.values())
    event_overlap = (
        sorted(
            set().union(
                *(left & right for left, right in itertools.combinations(event_sets, 2))
            )
        )
        if event_sets
        else []
    )
    thread_sets = [set(split["thread_id"]) for split in split_frames.values()]
    thread_overlap = (
        sorted(
            set().union(
                *(
                    left & right
                    for left, right in itertools.combinations(thread_sets, 2)
                )
            )
        )
        if thread_sets
        else []
    )
    assigned_rows = sum(len(split) for split in split_frames.values())
    record_assignment_complete = assigned_rows == len(frame)
    if event_overlap or thread_overlap or not record_assignment_complete:
        raise RuntimeError(
            f"Leakage check failed: event_overlap={event_overlap} "
            f"thread_overlap={thread_overlap} "
            f"record_assignment_complete={record_assignment_complete}"
        )

    manifest: dict[str, Any] = {
        "dataset": str(DATASET_PATH),
        "random_seed": RANDOM_SEED,
        "split_strategy": "Deterministic exhaustive event-level 5/2/2 split; all labels required in every subset.",
        "split_order": list(split_names),
        "events": {split: list(events) for split, events in zip(split_names, groups)},
        "record_counts": {split: len(split_frames[split]) for split in split_names},
        "label_counts": {
            split: {
                str(label): int(count)
                for label, count in split_frames[split]["label"]
                .value_counts()
                .to_dict()
                .items()
            }
            for split in split_names
        },
        "event_record_counts": {
            split: {
                str(event): len(
                    split_frames[split][split_frames[split]["event"] == event]
                )
                for event in groups[index]
            }
            for index, split in enumerate(split_names)
        },
        "all_labels_present": {
            split: set(split_frames[split]["label"]) == set(LABELS)
            for split in split_names
        },
        "event_overlap": event_overlap,
        "thread_overlap": thread_overlap,
        "record_assignment_complete": record_assignment_complete,
        "total_assigned_records": assigned_rows,
        "warnings": warnings,
        "limitations": [
            "This is a single held-out event allocation, not a substitute for grouped cross-validation.",
            "Model performance will measure generalization to the selected events, not all future social-media events.",
            "The current processed dataset has one source record per thread; the thread check remains enforced for future revisions.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--grouping-report", type=Path, default=GROUPING_REPORT_PATH)
    args = parser.parse_args()
    frame = load_processed_dataset(args.dataset)
    manifest = create_split_manifest(frame, args.manifest, args.grouping_report)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
