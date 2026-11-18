"""Create a derived, leakage-safe internal AVeriTeC benchmark."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.averitec_utils import load_jsonl
from app.verification_canonicalize import record_signatures, stable_hash

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data/processed/verification/averitec"
OUTPUT_ROOT = ROOT / "data/processed/verification/averitec_clean"
REPORT = ROOT / "data/reports/verification/averitec_clean_benchmark_report.json"
SEED = 42


def _group_records(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    seen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        signatures = record_signatures(row)
        keys = [("claim", signatures["claim_hash"])]
        keys.extend(("url", value) for value in signatures["url_hashes"])
        keys.extend(("evidence", value) for value in signatures["evidence_hashes"])
        keys.extend(
            ("document", value) for value in signatures["evidence_document_hashes"]
        )
        for key in keys:
            if key[1] and key in seen:
                union(index, seen[key])
            elif key[1]:
                seen[key] = index
    groups: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[find(index)].append(row)
    return list(groups.values())


def _assign_groups(
    groups: list[list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(groups) < 2:
        raise ValueError(
            "At least two independent groups are required for an internal split"
        )
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"{SEED}|{min(str(row['record_id']) for row in group)}".encode()
        ).hexdigest(),
    )
    target_dev = max(1, round(sum(len(group) for group in groups) * 0.2))
    internal_dev: list[dict[str, Any]] = []
    internal_train: list[dict[str, Any]] = []
    for group in ordered:
        if len(internal_dev) < target_dev:
            internal_dev.extend(group)
        else:
            internal_train.extend(group)
    if not internal_dev:
        internal_dev.append(internal_train.pop())
    return internal_train, internal_dev


def _with_derived_fields(
    row: dict[str, Any], derived_split: str, group_id: str
) -> dict[str, Any]:
    copied = dict(row)
    copied["derived_split"] = derived_split
    copied["clean_benchmark"] = True
    copied["overlap_group_id"] = group_id
    return copied


def _cross_overlap(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, int]:
    left_sigs = [record_signatures(row) for row in left]
    right_sigs = [record_signatures(row) for row in right]
    return {
        "claim": len(
            {item["claim_hash"] for item in left_sigs if item["claim_hash"]}
            & {item["claim_hash"] for item in right_sigs if item["claim_hash"]}
        ),
        "url": len(
            {value for item in left_sigs for value in item["url_hashes"]}
            & {value for item in right_sigs for value in item["url_hashes"]}
        ),
        "evidence_text": len(
            {value for item in left_sigs for value in item["evidence_hashes"]}
            & {value for item in right_sigs for value in item["evidence_hashes"]}
        ),
        "evidence_document": len(
            {value for item in left_sigs for value in item["evidence_document_hashes"]}
            & {
                value
                for item in right_sigs
                for value in item["evidence_document_hashes"]
            }
        ),
    }


def create_clean_benchmark() -> dict[str, Any]:
    source_train_path = SOURCE_ROOT / "train.jsonl"
    source_dev_path = SOURCE_ROOT / "dev.jsonl"
    if not source_train_path.is_file() or not source_dev_path.is_file():
        raise FileNotFoundError(
            "Run make prepare-verification-benchmark BENCHMARK=averitec first"
        )
    source_train = load_jsonl(source_train_path)
    source_dev = load_jsonl(source_dev_path)
    groups = _group_records(source_train)
    internal_train, internal_dev = _assign_groups(groups)
    group_ids = {}
    for index, group in enumerate(groups):
        group_id = (
            "internal-group-"
            + stable_hash(
                "|".join(sorted(str(row["record_id"]) for row in group))
            ).removeprefix("sha256:")[:16]
        )
        for row in group:
            group_ids[str(row["record_id"])] = group_id
    internal_train = [
        _with_derived_fields(row, "internal_train", group_ids[str(row["record_id"])])
        for row in internal_train
    ]
    internal_dev = [
        _with_derived_fields(row, "internal_dev", group_ids[str(row["record_id"])])
        for row in internal_dev
    ]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("source_train.jsonl", source_train),
        ("source_dev.jsonl", source_dev),
        ("internal_train.jsonl", internal_train),
        ("internal_dev.jsonl", internal_dev),
    ):
        (OUTPUT_ROOT / name).write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
    official_overlap = _cross_overlap(source_train, source_dev)
    internal_overlap = _cross_overlap(internal_train, internal_dev)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark": "averitec",
        "derived_benchmark": True,
        "official_evaluation": False,
        "statement": "This is a derived internal development benchmark and is not the official AVeriTeC evaluation split.",
        "seed": SEED,
        "source_train_count": len(source_train),
        "source_dev_count": len(source_dev),
        "internal_train_count": len(internal_train),
        "internal_dev_count": len(internal_dev),
        "excluded_record_count": 0,
        "exclusion_reasons": {},
        "official_train_dev_overlap": official_overlap,
        "internal_train_dev_overlap": internal_overlap,
        "group_count": len(groups),
        "internal_train_group_count": len(
            {row["overlap_group_id"] for row in internal_train}
        ),
        "internal_dev_group_count": len(
            {row["overlap_group_id"] for row in internal_dev}
        ),
        "label_distributions": {
            "source_train": dict(Counter(row.get("label") for row in source_train)),
            "source_dev": dict(Counter(row.get("label") for row in source_dev)),
            "internal_train": dict(Counter(row.get("label") for row in internal_train)),
            "internal_dev": dict(Counter(row.get("label") for row in internal_dev)),
        },
        "limitations": [
            "The official AVeriTeC files are preserved but already contain cross-split overlap.",
            "The derived benchmark uses only source-training records and is not an official score set.",
            "Evidence publication dates are unavailable, so temporal leakage cannot be assessed.",
            "An independent knowledge store is still required for realistic retrieval evaluation.",
        ],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(create_clean_benchmark(), indent=2, ensure_ascii=False))
