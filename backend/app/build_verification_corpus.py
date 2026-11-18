"""Build a train-only AVeriTeC evidence corpus from local answer fields."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.averitec_utils import answer_records, content_hash, load_jsonl

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed/verification/averitec"
CORPUS_ROOT = ROOT / "data/processed/verification/corpora"
REPORT_ROOT = ROOT / "data/reports/verification"


def build_averitec_train_corpus() -> dict[str, Any]:
    input_path = PROCESSED / "train.jsonl"
    output_dir = CORPUS_ROOT / "averitec_train_evidence"
    output_path = output_dir / "evidence.jsonl"
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark": "averitec",
        "corpus_mode": "train_evidence_corpus",
        "offline_only": True,
        "status": "incomplete",
        "source_split": "train",
        "source_path": str(input_path),
        "documents_written": 0,
        "duplicate_evidence_removed": 0,
        "notes": [
            "Evidence is extracted from local answer fields only; URLs are not fetched.",
            "The development split is never read by this builder.",
        ],
    }
    if not input_path.is_file():
        report["manual_action"] = (
            "Run make prepare-verification-benchmark BENCHMARK=averitec first."
        )
    else:
        documents: dict[str, dict[str, Any]] = {}
        source_claims: dict[str, set[str]] = {}
        for record in load_jsonl(input_path):
            for index, answer in enumerate(answer_records(record)):
                text = answer.get("answer")
                if not isinstance(text, str) or not text.strip():
                    continue
                digest = content_hash(text.strip())
                document_id = f"averitec-train-{digest.removeprefix('sha256:')[:20]}"
                source_claims.setdefault(document_id, set()).add(
                    str(record["record_id"])
                )
                if document_id in documents:
                    continue
                documents[document_id] = {
                    "document_id": document_id,
                    "title": str(
                        answer.get("question") or "AVeriTeC training evidence"
                    ),
                    "text": text.strip(),
                    "url": answer.get("source_url"),
                    "publisher": None,
                    "source_category": "unknown",
                    "language": "en",
                    "published_at": None,
                    "retrieved_at": None,
                    "content_hash": digest,
                    "license": "requires confirmation",
                    "corpus_mode": "train_evidence_corpus",
                    "source_claim_ids": [str(record["record_id"])],
                    "question_index": index,
                    "source_medium": answer.get("source_medium"),
                    "cached_source_url": answer.get("cached_source_url"),
                }
        for document_id, claim_ids in source_claims.items():
            if document_id in documents:
                documents[document_id]["source_claim_ids"] = sorted(claim_ids)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "\n".join(
                json.dumps(item, ensure_ascii=False) for item in documents.values()
            )
            + "\n",
            encoding="utf-8",
        )
        answer_count = sum(
            len(answer_records(record)) for record in load_jsonl(input_path)
        )
        report.update(
            {
                "status": "complete",
                "documents_written": len(documents),
                "answer_records_seen": answer_count,
                "duplicate_evidence_removed": answer_count - len(documents),
                "output_path": str(output_path),
                "language_counts": dict(
                    Counter(item["language"] for item in documents.values())
                ),
            }
        )
    report_path = REPORT_ROOT / "averitec_train_corpus.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["averitec"], default="averitec")
    parser.add_argument(
        "--mode", choices=["train_evidence_corpus"], default="train_evidence_corpus"
    )
    parser.parse_args()
    print(json.dumps(build_averitec_train_corpus(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
