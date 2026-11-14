import json
from pathlib import Path

import pandas as pd
import pytest
from app.ml.create_splits import (
    build_grouping_report,
    create_split_manifest,
    load_processed_dataset,
)


def sample_frame() -> pd.DataFrame:
    rows = []
    for event in ("a", "b", "c", "d", "e", "f", "g", "h", "i"):
        for label in ("true", "false", "unverified"):
            rows.append(
                {
                    "text": f"{event} {label}",
                    "label": label,
                    "event": event,
                    "thread_id": f"{event}-{label}",
                    "source_tweet_id": f"{event}-{label}",
                }
            )
    return pd.DataFrame(rows)


def test_grouping_report_identifies_event_and_thread_structure() -> None:
    frame = sample_frame()
    report = build_grouping_report(frame)
    assert report["unique_events"] == 9
    assert report["unique_threads"] == 27
    assert report["events_with_only_one_label"] == {}
    assert report["records_per_thread"]["threads_with_multiple_records"] == 0


def test_create_split_manifest_is_disjoint_and_reproducible(tmp_path: Path) -> None:
    frame = sample_frame()
    first = create_split_manifest(
        frame, tmp_path / "first.json", tmp_path / "grouping.json"
    )
    second = create_split_manifest(
        frame, tmp_path / "second.json", tmp_path / "grouping2.json"
    )
    assert first["events"] == second["events"]
    assert first["event_overlap"] == []
    assert first["thread_overlap"] == []
    assert first["record_assignment_complete"] is True
    assert all(first["all_labels_present"].values())


def test_missing_grouping_column_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"
    sample_frame().drop(columns=["event"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns: event"):
        load_processed_dataset(path)


def test_manifest_is_valid_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    create_split_manifest(sample_frame(), manifest_path, tmp_path / "grouping.json")
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["total_assigned_records"] == 27
