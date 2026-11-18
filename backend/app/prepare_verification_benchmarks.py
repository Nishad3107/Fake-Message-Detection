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
    questions: list[dict[str, Any]] = Field(default_factory=list)
    language: str = "en"
    claim_date: str | None = None
    speaker: str | None = None
    reporting_source: str | None = None
    location: str | None = None
    claim_types: list[str] = Field(default_factory=list)
    fact_checking_strategies: list[str] = Field(default_factory=list)
    fact_checking_article: str | None = None
    original_claim_url: str | None = None
    cached_original_claim_url: str | None = None
    justification: str | None = None
    required_reannotation: bool | None = None
    split: str | None = None
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
    benchmark: BenchmarkName,
    record: dict[str, Any],
    position: int,
    split: str | None = None,
) -> NormalizedVerificationRecord:
    record_id = _first(record, ("id", "claim_id", "uid", "record_id"))
    claim = _first(record, ("claim", "statement", "text"))
    if not record_id:
        if benchmark == "averitec" and split:
            record_id = f"{split}-{position:06d}"
        else:
            raise ValueError(f"Missing record ID at position {position}")
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError(f"Missing claim text for record {record_id}")
    return NormalizedVerificationRecord(
        benchmark=benchmark,
        record_id=str(record_id),
        claim=claim.strip(),
        label=_first(record, ("label", "verdict", "gold_label")),
        evidence=_first(record, ("evidence", "evidence_sets", "answers")),
        questions=list(record.get("questions") or []),
        language=str(record.get("language", "en")),
        claim_date=record.get("claim_date"),
        speaker=record.get("speaker"),
        reporting_source=record.get("reporting_source"),
        location=record.get("location_ISO_code"),
        claim_types=list(record.get("claim_types") or []),
        fact_checking_strategies=list(record.get("fact_checking_strategies") or []),
        fact_checking_article=record.get("fact_checking_article"),
        original_claim_url=record.get("original_claim_url"),
        cached_original_claim_url=record.get("cached_original_claim_url"),
        justification=record.get("justification"),
        required_reannotation=record.get("required_reannotation"),
        split=split,
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
        "record_id_policy": (
            "AVeriTeC source records have no ID field; IDs are deterministic split-qualified "
            "source-row identifiers and do not replace the original source record."
            if benchmark == "averitec"
            else "Source IDs are required."
        ),
        "source_metadata": metadata.model_dump(),
        "status": "incomplete",
        "records_read": 0,
        "records_written": 0,
        "label_counts": {},
        "errors": [],
        "excluded_records": [],
        "manual_action": (
            f"Place the licensed {metadata.dataset_name} files in {raw_dir} "
            "and rerun the adapter. No download is performed automatically."
        ),
    }
    if not metadata.file_names:
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    normalized_by_split: dict[str, list[NormalizedVerificationRecord]] = {}
    seen_ids: set[str] = set()
    input_paths = (
        sorted(
            path
            for path in raw_dir.glob("*.json")
            if path.name in {"train.json", "dev.json"}
        )
        if benchmark == "averitec"
        else sorted(raw_dir.glob("*.json*"))
    )
    for path in input_paths:
        split = path.stem if benchmark == "averitec" else None
        normalized_by_split.setdefault(split or "all", [])
        try:
            raw_records = _read_json_file(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            report["errors"].append(f"{path.name}: {exc}")
            continue
        report["records_read"] += len(raw_records)
        for position, raw_record in enumerate(raw_records, start=1):
            try:
                item = normalize_record(benchmark, raw_record, position, split)
            except ValueError as exc:
                message = f"{path.name}:{position}: {exc}"
                if benchmark == "averitec" and "Missing claim text" in str(exc):
                    report["excluded_records"].append(
                        {"file": path.name, "position": position, "reason": str(exc)}
                    )
                else:
                    report["errors"].append(message)
                continue
            if item.record_id in seen_ids:
                report["errors"].append(f"Duplicate record ID: {item.record_id}")
                continue
            seen_ids.add(item.record_id)
            normalized_by_split[split or "all"].append(item)
    normalized = [item for items in normalized_by_split.values() for item in items]
    if normalized and not report["errors"]:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths: dict[str, str] = {}
        for split, items in normalized_by_split.items():
            output_path = output_dir / f"{split}.jsonl"
            output_path.write_text(
                "\n".join(item.model_dump_json() for item in items) + "\n",
                encoding="utf-8",
            )
            output_paths[split] = str(output_path)
        report["records_written"] = len(normalized)
        report["records_per_split"] = {
            split: len(items) for split, items in normalized_by_split.items()
        }
        report["label_counts"] = dict(Counter(item.label for item in normalized))
        report["output_paths"] = output_paths
        metadata.label_set = sorted(
            {item.label for item in normalized if item.label is not None}
        )
        metadata.claim_count = len(normalized)
        report["source_metadata"] = metadata.model_dump()
        if benchmark != "averitec" and "all" in output_paths:
            report["output_path"] = output_paths["all"]
        report["status"] = "partial" if report["excluded_records"] else "complete"
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
