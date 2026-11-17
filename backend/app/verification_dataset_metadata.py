"""Metadata and local availability records for separate verification tracks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

BenchmarkName = Literal["fever", "averitec"]


class VerificationDatasetMetadata(BaseModel):
    benchmark: BenchmarkName
    dataset_name: str
    official_source: str
    dataset_version: str
    expected_file_names: list[str]
    file_names: list[str] = Field(default_factory=list)
    file_hashes: dict[str, str] = Field(default_factory=dict)
    license: str
    languages: list[str]
    claim_count: int | None = None
    evidence_available: bool | None = None
    label_set: list[str] = Field(default_factory=list)
    status: Literal["available", "incomplete", "invalid"] = "incomplete"
    notes: list[str] = Field(default_factory=list)


DATASET_METADATA: dict[BenchmarkName, VerificationDatasetMetadata] = {
    "fever": VerificationDatasetMetadata(
        benchmark="fever",
        dataset_name="FEVER: Fact Extraction and VERification",
        official_source="https://fever.ai/dataset/fever.html",
        dataset_version="not locally verified",
        expected_file_names=["train.jsonl", "dev.jsonl", "shared_task_dev.jsonl"],
        license="requires confirmation for the supplied release",
        languages=["en"],
        evidence_available=True,
        label_set=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
        notes=[
            "Expected labels are documented by the FEVER format; local files are required for verification.",
            "Do not mix FEVER with PHEME, HinFakeNews, or application history.",
        ],
    ),
    "averitec": VerificationDatasetMetadata(
        benchmark="averitec",
        dataset_name="AVeriTeC: Automated Verification of Textual Claims",
        official_source="https://fever.ai/dataset/averitec.html",
        dataset_version="not locally verified",
        expected_file_names=["train.json", "dev.json", "test.json", "averitec.json"],
        license="requires confirmation for the supplied release",
        languages=["en"],
        evidence_available=True,
        label_set=[],
        notes=[
            "The exact local release schema and labels must be inspected before preparation.",
            "Do not infer missing labels or evidence from claim text.",
            "Do not mix AVeriTeC with FEVER, PHEME, or HinFakeNews.",
        ],
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_local_files(
    benchmark: BenchmarkName, raw_dir: Path
) -> VerificationDatasetMetadata:
    metadata = DATASET_METADATA[benchmark].model_copy(deep=True)
    files = sorted(
        path for path in raw_dir.glob("*") if path.is_file() and path.name != ".gitkeep"
    )
    metadata.file_names = [path.name for path in files]
    metadata.file_hashes = {path.name: sha256_file(path) for path in files}
    if files:
        metadata.status = "available"
        metadata.dataset_version = "local files supplied; release not declared"
    else:
        metadata.status = "incomplete"
        metadata.notes.append("No local benchmark files were supplied.")
    return metadata
