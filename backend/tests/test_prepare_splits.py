import json
from pathlib import Path

import pandas as pd
from app.ml.create_splits import create_split_manifest
from app.ml.prepare_splits import prepare_splits


def _frame() -> pd.DataFrame:
    rows = []
    for event in ("a", "b", "c", "d", "e", "f", "g", "h", "i"):
        for label in ("true", "false", "unverified"):
            identifier = f"{event}-{label}"
            rows.append(
                {
                    "text": f"RT @source: {label} #News https://example.com/{identifier}",
                    "label": label,
                    "event": event,
                    "thread_id": identifier,
                    "source_tweet_id": identifier,
                }
            )
    return pd.DataFrame(rows)


def test_prepare_splits_preserves_records_labels_and_metadata(tmp_path: Path) -> None:
    frame = _frame()
    dataset = tmp_path / "dataset.csv"
    frame.to_csv(dataset, index=False)
    manifest = tmp_path / "manifest.json"
    create_split_manifest(frame, manifest, tmp_path / "grouping.json")
    report_path = tmp_path / "preprocessing.json"

    report = prepare_splits(dataset, manifest, tmp_path / "splits", report_path)

    assert report["input_record_count"] == 27
    assert report["output_record_count"] == 27
    assert report["labels_preserved"] is True
    assert report["metadata_preserved"] is True
    assert report["changed_text_count"] == 27
    assert report["split_counts"] == {"train": 15, "validation": 6, "test": 6}
    assert json.loads(report_path.read_text(encoding="utf-8"))[
        "vocabulary_learning"
    ].startswith("None")
    for split in ("train", "validation", "test"):
        saved = pd.read_csv(tmp_path / "splits" / f"{split}.csv")
        assert list(saved.columns) == [
            "record_id",
            "text_original",
            "text_clean",
            "label",
            "event",
            "thread_id",
            "source_tweet_id",
            "split",
        ]
        assert saved["split"].eq(split).all()
        assert not saved["record_id"].duplicated().any()
