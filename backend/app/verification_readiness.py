"""Readiness gates for AVeriTeC verification experiments."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/reports/verification/averitec_readiness.json"
CORPUS = (
    ROOT / "data/processed/verification/corpora/averitec_train_evidence/evidence.jsonl"
)
LEAKAGE = ROOT / "data/reports/verification/averitec_corpus_leakage.json"


def check_readiness() -> dict[str, Any]:
    leakage = (
        json.loads(LEAKAGE.read_text(encoding="utf-8")) if LEAKAGE.is_file() else {}
    )
    safe_train_corpus = CORPUS.is_file() and leakage.get("status") == "review_required"
    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark": "averitec",
        "offline_only": True,
        "statuses": {
            "ready_for_schema_tests": True,
            "ready_for_oracle_tests": True,
            "ready_for_train_corpus_retrieval": safe_train_corpus,
            "ready_for_development_retrieval": safe_train_corpus,
            "ready_for_research_metrics": False,
            "ready_for_live_retrieval": False,
        },
        "blocking_reasons": [
            "An independent knowledge store or evidence corpus is unavailable.",
            "Live web retrieval is disabled by policy.",
            "No blind test set is available.",
        ],
        "corpus_mode": "train_evidence_corpus" if CORPUS.is_file() else None,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


readiness = check_readiness


if __name__ == "__main__":
    print(json.dumps(check_readiness(), indent=2))
