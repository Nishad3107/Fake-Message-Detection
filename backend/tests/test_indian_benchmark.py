from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from app.ml import create_indian_splits, prepare_indian_benchmark


def test_indian_preparation_preserves_labels_and_excludes_url_from_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    headers = ["URL", "TITLE", "CONTENT", "BOOL"]
    records = [
        {
            "URL": "https://real.example/a",
            "TITLE": "A",
            "CONTENT": "Article A",
            "BOOL": "1",
            "source_row": "2",
        },
        {
            "URL": "https://fake.example/b",
            "TITLE": "B",
            "CONTENT": "Article B",
            "BOOL": "0",
            "source_row": "3",
        },
    ]
    monkeypatch.setattr(
        prepare_indian_benchmark, "_load_records", lambda _: (headers, records)
    )
    output = tmp_path / "prepared.csv"
    report = tmp_path / "preparation.json"
    result = prepare_indian_benchmark.prepare_dataset(
        tmp_path / "input.xlsx", output, report, tmp_path / "leakage.json"
    )
    assert result["final_row_count"] == 2
    assert result["label_distribution"] == {"fake": 1, "real": 1}
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["text_original"] == "Article A"
    assert "https://real.example/a" not in rows[0]["text_original"]
    assert rows[0]["label_original"] == "1"
    assert rows[0]["label"] == "real"
    assert json.loads(report.read_text(encoding="utf-8"))["license_status"].startswith(
        "unknown"
    )


def test_indian_preparation_reports_conflicting_duplicate_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    headers = ["URL", "TITLE", "CONTENT", "BOOL"]
    records = [
        {
            "URL": "https://example.com/a",
            "TITLE": "Same",
            "CONTENT": "Same content",
            "BOOL": "0",
            "source_row": "2",
        },
        {
            "URL": "https://example.com/b",
            "TITLE": "Same",
            "CONTENT": "Same content",
            "BOOL": "1",
            "source_row": "3",
        },
    ]
    monkeypatch.setattr(
        prepare_indian_benchmark, "_load_records", lambda _: (headers, records)
    )
    result = prepare_indian_benchmark.prepare_dataset(
        tmp_path / "input.xlsx",
        tmp_path / "prepared.csv",
        tmp_path / "preparation.json",
        tmp_path / "leakage.json",
    )
    assert result["conflicting_label_duplicate_group_count"] == 1
    assert result["final_row_count"] == 1


def test_indian_split_has_no_domain_or_text_overlap(tmp_path: Path) -> None:
    input_path = tmp_path / "prepared.csv"
    rows = []
    for index, domain in enumerate(
        ("a.example", "b.example", "c.example", "d.example", "e.example", "f.example")
    ):
        for offset in range(3):
            rows.append(
                {
                    "record_id": f"r{index}-{offset}",
                    "text_original": f"unique text {index} {offset}",
                    "label": "fake" if offset == 0 else "real",
                    "domain": domain,
                }
            )
    with input_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["record_id", "text_original", "label", "domain"]
        )
        writer.writeheader()
        writer.writerows(rows)
    manifest = create_indian_splits.create_splits(
        input_path, tmp_path / "manifest.json"
    )
    assert manifest["assignment_count"] == len(rows)
    assert manifest["domain_overlap"] == []
    assert manifest["exact_text_overlap"] == []
    assert sum(part["record_count"] for part in manifest["splits"].values()) == len(
        rows
    )


def test_indian_split_rejects_missing_grouping_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.csv"
    input_path.write_text(
        "record_id,label,text_original\nr1,fake,text\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing required columns"):
        create_indian_splits.create_splits(input_path, tmp_path / "manifest.json")


def test_indian_paths_are_separate_from_pheme() -> None:
    assert "indian" in prepare_indian_benchmark.OUTPUT.parts
    assert "pheme" not in prepare_indian_benchmark.OUTPUT.parts
