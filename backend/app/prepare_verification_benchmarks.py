"""Audit and normalize local FEVER/AVeriTeC files without downloading data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.verification_dataset_metadata import (
    DATASET_METADATA,
    BenchmarkName,
    inspect_local_files,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data/raw/verification"
PROCESSED_ROOT = ROOT / "data/processed/verification"
REPORT_ROOT = ROOT / "data/reports/verification"


class NormalizedVerificationRecord(BaseModel):
    benchmark: BenchmarkName
    record_id: str
    claim: str
    label: str | None = None
    evidence: Any = None
    language: str = "en"
    source_record: dict[str, Any] = Field(default_factory=dict)


def _read_json_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("data", "claims", "examples", "records"):
            if isinstance(value.get(key), list):
                return value[key]
    raise ValueError(f"Unsupported JSON structure in {path.name}")


def _first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    return next((record[key] for key in keys if record.get(key) is not None), None)


def normalize_record(
    benchmark: BenchmarkName, record: dict[str, Any], position: int
) -> NormalizedVerificationRecord:
    record_id = _first(record, ("id", "claim_id", "uid", "record_id"))
    claim = _first(record, ("claim", "statement", "text"))
    if not record_id:
        raise ValueError(f"Missing record ID at position {position}")
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError(f"Missing claim text for record {record_id}")
    return NormalizedVerificationRecord(
        benchmark=benchmark,
        record_id=str(record_id),
        claim=claim.strip(),
        label=_first(record, ("label", "verdict", "gold_label")),
        evidence=_first(record, ("evidence", "evidence_sets", "answers")),
        language=str(record.get("language", "en")),
        source_record=record,
    )


def prepare_benchmark(benchmark: BenchmarkName) -> dict[str, Any]:
    raw_dir = RAW_ROOT / benchmark
    output_dir = PROCESSED_ROOT / benchmark
    report_path = REPORT_ROOT / f"{benchmark}_preparation.json"
    metadata = inspect_local_files(benchmark, raw_dir)
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark": benchmark,
        "offline_only": True,
        "source_metadata": metadata.model_dump(),
        "status": "incomplete",
        "records_read": 0,
        "records_written": 0,
        "label_counts": {},
        "errors": [],
        "manual_action": (
            f"Place the licensed {metadata.dataset_name} files in {raw_dir} "
            "and rerun the adapter. No download is performed automatically."
        ),
    }
    if not metadata.file_names:
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    normalized: list[NormalizedVerificationRecord] = []
    seen_ids: set[str] = set()
    for path in sorted(raw_dir.glob("*.json*")):
        try:
            raw_records = _read_json_file(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            report["errors"].append(f"{path.name}: {exc}")
            continue
        report["records_read"] += len(raw_records)
        for position, raw_record in enumerate(raw_records, start=1):
            try:
                item = normalize_record(benchmark, raw_record, position)
            except ValueError as exc:
                report["errors"].append(f"{path.name}:{position}: {exc}")
                continue
            if item.record_id in seen_ids:
                report["errors"].append(f"Duplicate record ID: {item.record_id}")
                continue
            seen_ids.add(item.record_id)
            normalized.append(item)
    if normalized and not report["errors"]:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{benchmark}.jsonl"
        output_path.write_text(
            "\n".join(item.model_dump_json() for item in normalized) + "\n",
            encoding="utf-8",
        )
        report["records_written"] = len(normalized)
        report["label_counts"] = dict(Counter(item.label for item in normalized))
        report["output_path"] = str(output_path)
        report["status"] = "complete"
    else:
        report["status"] = "invalid" if report["errors"] else "incomplete"
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=sorted(DATASET_METADATA))
    args = parser.parse_args()
    print(json.dumps(prepare_benchmark(args.benchmark), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
