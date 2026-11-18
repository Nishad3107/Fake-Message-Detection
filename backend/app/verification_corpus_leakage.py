"""Leakage checks for AVeriTeC train-derived evidence corpora."""

from __future__ import annotations

import argparse
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
from app.verification_canonicalize import record_signatures

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/processed/verification/averitec/train.jsonl"
DEV = ROOT / "data/processed/verification/averitec/dev.jsonl"
CORPUS = (
    ROOT / "data/processed/verification/corpora/averitec_train_evidence/evidence.jsonl"
)
REPORT = ROOT / "data/reports/verification/averitec_corpus_leakage.json"
PROCESSED = ROOT / "data/processed/verification/averitec"
CLEAN_ROOT = ROOT / "data/processed/verification/averitec_clean"
CLEAN_REPORT = ROOT / "data/reports/verification/averitec_clean_corpus_leakage.json"


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


def audit_clean_benchmark() -> dict[str, Any]:
    train_path = CLEAN_ROOT / "internal_train.jsonl"
    dev_path = CLEAN_ROOT / "internal_dev.jsonl"
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark": "averitec",
        "derived_benchmark": True,
        "status": "incomplete",
        "checks": {},
        "offline_only": True,
        "statement": "This validates a derived internal benchmark, not official AVeriTeC evaluation.",
    }
    if not train_path.is_file() or not dev_path.is_file():
        report["manual_action"] = "Run the clean benchmark generator first."
    else:
        train = load_jsonl(train_path)
        dev = load_jsonl(dev_path)
        train_sigs = [record_signatures(row) for row in train]
        dev_sigs = [record_signatures(row) for row in dev]
        train_claims = {item["claim_hash"] for item in train_sigs if item["claim_hash"]}
        dev_claims = {item["claim_hash"] for item in dev_sigs if item["claim_hash"]}
        train_urls = {value for item in train_sigs for value in item["url_hashes"]}
        dev_urls = {value for item in dev_sigs for value in item["url_hashes"]}
        train_evidence = {
            value for item in train_sigs for value in item["evidence_hashes"]
        }
        dev_evidence = {value for item in dev_sigs for value in item["evidence_hashes"]}
        train_documents = {
            value for item in train_sigs for value in item["evidence_document_hashes"]
        }
        dev_documents = {
            value for item in dev_sigs for value in item["evidence_document_hashes"]
        }
        checks = {
            "claim_hash_overlap": len(train_claims & dev_claims),
            "canonical_url_overlap": len(train_urls & dev_urls),
            "evidence_hash_overlap": len(train_evidence & dev_evidence),
            "evidence_document_overlap": len(train_documents & dev_documents),
            "gold_evidence_in_internal_train_corpus": len(
                train_evidence & dev_evidence
            ),
            "internal_train_records": len(train),
            "internal_dev_records": len(dev),
        }
        report["checks"] = checks
        overlap_keys = (
            "claim_hash_overlap",
            "canonical_url_overlap",
            "evidence_hash_overlap",
            "evidence_document_overlap",
            "gold_evidence_in_internal_train_corpus",
        )
        report["status"] = (
            "safe" if not any(checks[key] for key in overlap_keys) else "unsafe"
        )
    CLEAN_REPORT.parent.mkdir(parents=True, exist_ok=True)
    CLEAN_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    result = audit_clean_benchmark() if args.clean else audit_corpus_leakage()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.clean and result["status"] != "safe":
        raise SystemExit(1)
