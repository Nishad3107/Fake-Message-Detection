"""Leakage checks for AVeriTeC train-derived evidence corpora."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.averitec_utils import (
    answer_records,
    content_hash,
    evidence_urls,
    load_jsonl,
    normalized,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/processed/verification/averitec/train.jsonl"
DEV = ROOT / "data/processed/verification/averitec/dev.jsonl"
CORPUS = (
    ROOT / "data/processed/verification/corpora/averitec_train_evidence/evidence.jsonl"
)
REPORT = ROOT / "data/reports/verification/averitec_corpus_leakage.json"
PROCESSED = ROOT / "data/processed/verification/averitec"


def _evidence_hashes(rows: list[dict[str, Any]]) -> set[str]:
    return {
        content_hash(answer["answer"].strip())
        for row in rows
        for answer in answer_records(row)
        if isinstance(answer.get("answer"), str) and answer["answer"].strip()
    }


def audit_corpus_leakage() -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark": "averitec",
        "corpus_mode": "train_evidence_corpus",
        "offline_only": True,
        "status": "incomplete",
        "checks": {},
        "temporal_leakage_status": "not_assessable_without_evidence_publication_dates",
        "notes": [
            "No web pages are fetched and no development labels are used.",
            "Verdict words in answers or justifications are reported as review signals, not removed automatically.",
        ],
    }
    if not TRAIN.is_file() or not DEV.is_file() or not CORPUS.is_file():
        report["manual_action"] = (
            "Prepare train/dev files and build the train evidence corpus first."
        )
    else:
        train = load_jsonl(TRAIN)
        dev = load_jsonl(DEV)
        corpus = load_jsonl(CORPUS)
        train_claims = {normalized(row.get("claim")) for row in train}
        dev_claims = {normalized(row.get("claim")) for row in dev}
        train_urls = {url for row in train for url in evidence_urls(row)}
        dev_urls = {url for row in dev for url in evidence_urls(row)}
        train_evidence = _evidence_hashes(train)
        dev_evidence = _evidence_hashes(dev)
        corpus_hashes = {row.get("content_hash") for row in corpus}
        dev_record_ids = {str(row.get("record_id")) for row in dev}
        corpus_claim_ids = {
            str(claim_id)
            for row in corpus
            for claim_id in row.get("source_claim_ids", [])
        }
        verdict_pattern = re.compile(
            r"\b(supported|refuted|true|false|not enough evidence|conflicting)\b",
            re.IGNORECASE,
        )
        verdict_hits = sum(
            bool(verdict_pattern.search(str(row.get("text", "")))) for row in corpus
        )
        claim_overlap = train_claims & dev_claims
        corpus_dev_claims = corpus_claim_ids & dev_record_ids
        evidence_overlap = corpus_hashes & dev_evidence
        url_overlap = train_urls & dev_urls
        checks = {
            "development_claim_in_train_evidence_corpus": len(corpus_dev_claims),
            "development_evidence_in_train_corpus": len(evidence_overlap),
            "duplicate_claims_across_train_dev": len(claim_overlap),
            "duplicate_urls_across_train_dev": len(url_overlap),
            "duplicate_evidence_content_across_train_dev": len(
                train_evidence & dev_evidence
            ),
            "verdict_or_label_markers_in_train_evidence": verdict_hits,
            "corpus_documents": len(corpus),
        }
        report["checks"] = checks
        report["status"] = (
            "unsafe"
            if corpus_dev_claims or claim_overlap or evidence_overlap or url_overlap
            else "review_required"
        )
        report["claim_overlap_examples"] = sorted(claim_overlap)[:10]
        report["corpus_development_claim_examples"] = sorted(corpus_dev_claims)[:10]
        report["evidence_overlap_examples"] = sorted(evidence_overlap)[:10]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


audit = audit_corpus_leakage


if __name__ == "__main__":
    print(json.dumps(audit_corpus_leakage(), indent=2, ensure_ascii=False))
