"""Build deterministic overlap groups without changing benchmark records."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.averitec_utils import load_jsonl
from app.verification_canonicalize import record_signatures, stable_hash

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed/verification/averitec"
REPORT = ROOT / "data/reports/verification/averitec_overlap_groups.json"


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def build_overlap_groups() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "dev"):
        path = PROCESSED / f"{split}.jsonl"
        if path.is_file():
            for row in load_jsonl(path):
                rows.append(
                    {
                        "split": split,
                        "record": row,
                        "signatures": record_signatures(row),
                    }
                )
    ids = [str(item["record"].get("record_id")) for item in rows]
    uf = _UnionFind(ids)
    key_records: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for item in rows:
        record_id = str(item["record"].get("record_id"))
        signatures = item["signatures"]
        for key_type, values in (
            ("claim", [signatures["claim_hash"]]),
            ("url", signatures["url_hashes"]),
            ("evidence", signatures["evidence_hashes"]),
            ("document", signatures["evidence_document_hashes"]),
        ):
            for value in values:
                if value:
                    key_records[(key_type, value)].append(record_id)
    for record_ids in key_records.values():
        for record_id in record_ids[1:]:
            uf.union(record_ids[0], record_id)

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in rows:
        grouped[uf.find(str(item["record"].get("record_id")))].append(item)
    output_groups: list[dict[str, Any]] = []
    for members in grouped.values():
        if len(members) < 2:
            continue
        record_ids = sorted(str(item["record"].get("record_id")) for item in members)
        group_id = (
            "overlap-" + stable_hash("|".join(record_ids)).removeprefix("sha256:")[:16]
        )
        reasons = set()
        claims, urls, evidence, documents = set(), set(), set(), set()
        labels = Counter()
        for item in members:
            signatures = item["signatures"]
            claims.add(signatures["claim_hash"])
            urls.update(signatures["url_hashes"])
            evidence.update(signatures["evidence_hashes"])
            documents.update(signatures["evidence_document_hashes"])
            labels[str(item["record"].get("label"))] += 1
        if len(claims) < len(members):
            reasons.add("exact_normalized_claim")
        if sum(len(item["signatures"]["url_hashes"]) for item in members) > len(urls):
            reasons.add("canonical_url")
        if sum(len(item["signatures"]["evidence_hashes"]) for item in members) > len(
            evidence
        ):
            reasons.add("evidence_text")
        if sum(
            len(item["signatures"]["evidence_document_hashes"]) for item in members
        ) > len(documents):
            reasons.add("evidence_document_identity")
        output_groups.append(
            {
                "group_id": group_id,
                "source_splits": sorted({item["split"] for item in members}),
                "record_ids": record_ids,
                "claim_hashes": sorted(item for item in claims if item),
                "url_hashes": sorted(urls),
                "evidence_hashes": sorted(evidence),
                "evidence_document_hashes": sorted(documents),
                "label_distribution": dict(labels),
                "overlap_reason": sorted(reasons),
            }
        )
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark": "averitec",
        "offline_only": True,
        "source_files": [str(PROCESSED / "train.jsonl"), str(PROCESSED / "dev.jsonl")],
        "record_count": len(rows),
        "overlap_group_count": len(output_groups),
        "groups": sorted(output_groups, key=lambda item: item["group_id"]),
        "note": "Groups are diagnostic and do not modify official or prepared records.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(build_overlap_groups(), indent=2, ensure_ascii=False))
