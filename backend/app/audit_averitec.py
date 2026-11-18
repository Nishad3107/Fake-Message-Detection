"""Audit local AVeriTeC train/dev files without fetching evidence."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.averitec_utils import answer_records, evidence_urls, normalized
from app.verification_dataset_metadata import sha256_file

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw/verification/averitec"
REPORT = ROOT / "data/reports/verification/averitec_dataset_audit.json"


def _load(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    claims = [str(row.get("claim", "")) for row in rows]
    answer_rows = [answer for row in rows for answer in answer_records(row)]
    urls = [url for row in rows for url in evidence_urls(row)]
    duplicate_claims = sum(
        count - 1
        for count in Counter(normalized(claim) for claim in claims).values()
        if count > 1
    )
    duplicate_urls = sum(count - 1 for count in Counter(urls).values() if count > 1)
    marker_patterns = (
        "supported",
        "refuted",
        "not enough evidence",
        "conflicting",
        "fact-check",
        "fact check",
    )
    marker_counts = {
        marker: sum(marker in normalized(row.get("justification")) for row in rows)
        for marker in marker_patterns
    }
    lengths = [len(claim) for claim in claims]
    return {
        "record_count": len(rows),
        "label_counts": dict(Counter(row.get("label") for row in rows)),
        "missing_fields": {
            field: sum(row.get(field) in (None, "", []) for row in rows)
            for field in (
                "claim",
                "justification",
                "claim_date",
                "speaker",
                "location_ISO_code",
                "fact_checking_article",
                "questions",
            )
        },
        "claim_length_characters": {
            "minimum": min(lengths) if lengths else 0,
            "maximum": max(lengths) if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 4) if lengths else 0,
        },
        "question_count": sum(len(row.get("questions") or []) for row in rows),
        "answer_count": len(answer_rows),
        "source_url_count": sum(
            bool(answer.get("source_url")) for answer in answer_rows
        ),
        "cached_url_count": sum(
            bool(answer.get("cached_source_url")) for answer in answer_rows
        ),
        "claim_date_available": sum(bool(row.get("claim_date")) for row in rows),
        "location_distribution": dict(
            Counter(row.get("location_ISO_code") or "__missing__" for row in rows)
        ),
        "claim_type_distribution": dict(
            Counter(item for row in rows for item in (row.get("claim_types") or []))
        ),
        "fact_checking_strategy_distribution": dict(
            Counter(
                item
                for row in rows
                for item in (row.get("fact_checking_strategies") or [])
            )
        ),
        "duplicate_claim_count": duplicate_claims,
        "duplicate_url_count": duplicate_urls,
        "potential_leakage_markers_in_justifications": marker_counts,
    }


def audit() -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark": "averitec",
        "offline_only": True,
        "license_metadata": "requires confirmation for the supplied release",
        "research_readiness": "not_ready_without_independent_evidence_corpus",
        "splits": {},
        "files": {},
        "notes": [
            "This audit preserves nested questions and answers; it does not fetch URLs.",
            "Justification markers are potential leakage signals, not automatic deletions.",
        ],
    }
    for split in ("train", "dev"):
        path = RAW_DIR / f"{split}.json"
        if not path.is_file():
            report["splits"][split] = {"status": "missing"}
            continue
        rows = _load(path)
        report["files"][path.name] = {"sha256": sha256_file(path), "records": len(rows)}
        report["splits"][split] = _stats(rows)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, ensure_ascii=False))
