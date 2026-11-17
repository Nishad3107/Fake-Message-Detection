from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from app.prepare_verification_benchmarks import normalize_record, prepare_benchmark
from app.verification_dataset_metadata import (
    DATASET_METADATA,
    inspect_local_files,
)


def test_metadata_defines_separate_tracks() -> None:
    assert set(DATASET_METADATA) == {"fever", "averitec"}
    assert DATASET_METADATA["fever"].label_set
    assert DATASET_METADATA["averitec"].languages == ["en"]


def test_missing_local_benchmarks_remain_incomplete(
    tmp_path: Path, monkeypatch
) -> None:
    import app.prepare_verification_benchmarks as adapter

    monkeypatch.setattr(adapter, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(adapter, "PROCESSED_ROOT", tmp_path / "processed")
    monkeypatch.setattr(adapter, "REPORT_ROOT", tmp_path / "reports")
    report = prepare_benchmark("fever")
    assert report["status"] == "incomplete"
    assert report["records_written"] == 0
    assert "No download" in report["manual_action"]


def test_normalize_fever_like_record_preserves_evidence() -> None:
    raw = {
        "id": 12,
        "claim": "A fictional claim.",
        "label": "SUPPORTS",
        "evidence": [[[1, 2, "Fictional page", 0]]],
    }
    normalized = normalize_record("fever", raw, 1)
    assert normalized.record_id == "12"
    assert normalized.label == "SUPPORTS"
    assert normalized.evidence == raw["evidence"]


def test_normalize_averitec_like_record() -> None:
    normalized = normalize_record(
        "averitec",
        {
            "claim_id": "a-1",
            "claim": "A fictional claim",
            "verdict": "TRUE",
            "answers": [],
        },
        1,
    )
    assert normalized.benchmark == "averitec"
    assert normalized.label == "TRUE"


def test_normalization_rejects_missing_id_or_claim() -> None:
    with pytest.raises(ValueError, match="Missing record ID"):
        normalize_record("fever", {"claim": "text"}, 1)
    with pytest.raises(ValueError, match="Missing claim text"):
        normalize_record("fever", {"id": "x", "claim": ""}, 1)


def test_supplied_local_file_is_hashed_and_normalized(
    tmp_path: Path, monkeypatch
) -> None:
    import app.prepare_verification_benchmarks as adapter

    raw_dir = tmp_path / "raw" / "fever"
    raw_dir.mkdir(parents=True)
    data = {"id": "f-1", "claim": "A fictional claim", "label": "NOT ENOUGH INFO"}
    file_path = raw_dir / "train.jsonl"
    file_path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    monkeypatch.setattr(adapter, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(adapter, "PROCESSED_ROOT", tmp_path / "processed")
    monkeypatch.setattr(adapter, "REPORT_ROOT", tmp_path / "reports")
    report = prepare_benchmark("fever")
    assert report["status"] == "complete"
    assert (
        report["source_metadata"]["file_hashes"]["train.jsonl"]
        == hashlib.sha256(file_path.read_bytes()).hexdigest()
    )
    output = Path(report["output_path"])
    assert output.is_file()
    assert json.loads(output.read_text().splitlines()[0])["benchmark"] == "fever"


def test_empty_local_directory_is_incomplete(tmp_path: Path) -> None:
    metadata = inspect_local_files("averitec", tmp_path)
    assert metadata.status == "incomplete"
    assert metadata.file_names == []
