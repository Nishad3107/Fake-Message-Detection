"""Local JSONL evidence-corpus ingestion with no network access."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.verification_sources import SourceCategory

ALLOWED_LANGUAGES = {"hi", "en", "hinglish", "romanized_hi"}
SOURCE_CATEGORIES = set(SourceCategory.__args__)


class EvidenceDocument(BaseModel):
    document_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    url: str | None = None
    publisher: str | None = None
    source_category: str
    language: str
    published_at: str | None = None
    retrieved_at: str | None = None
    content_hash: str = Field(..., min_length=1)
    license: str | None = None

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        return value

    @field_validator("language")
    @classmethod
    def language_supported(cls, value: str) -> str:
        if value not in ALLOWED_LANGUAGES:
            raise ValueError(f"Unsupported evidence language: {value}")
        return value

    @field_validator("source_category")
    @classmethod
    def source_category_supported(cls, value: str) -> str:
        if value not in SOURCE_CATEGORIES:
            raise ValueError(f"Invalid source category: {value}")
        return value

    @property
    def computed_content_hash(self) -> str:
        digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"


class CorpusStats(BaseModel):
    document_count: int
    language_counts: dict[str, int]
    source_category_counts: dict[str, int]
    missing_optional_metadata: dict[str, int]
    duplicate_document_id_count: int
    duplicate_content_hash_count: int
    average_document_length_characters: float
    content_hash_prefixes: dict[str, int]
    license_metadata_status: str
    sources_requiring_review: int


def _domain(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    return parsed.hostname.lower() if parsed.hostname else None


def load_corpus(
    path: Path,
    *,
    require_approved_sources: bool = False,
    approved_domains: set[str] | None = None,
) -> list[EvidenceDocument]:
    """Load local JSONL and validate IDs, hashes, and metadata.

    No URL is opened. ``require_approved_sources`` only checks local metadata.
    """
    path = resolve_local_corpus_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Evidence corpus not found: {path}")
    documents: list[EvidenceDocument] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    approved = approved_domains or set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw: dict[str, Any] = json.loads(line)
                document = EvidenceDocument.model_validate(raw)
            except (json.JSONDecodeError, ValidationError, TypeError) as exc:
                raise ValueError(f"Invalid corpus line {line_number}: {exc}") from exc
            if document.document_id in seen_ids:
                raise ValueError(f"Duplicate document ID: {document.document_id}")
            if document.content_hash in seen_hashes:
                raise ValueError(f"Duplicate content hash: {document.content_hash}")
            if document.content_hash != document.computed_content_hash:
                raise ValueError(
                    f"Content hash mismatch for document: {document.document_id}"
                )
            if (
                require_approved_sources
                and document.url
                and _domain(document.url) not in approved
            ):
                raise ValueError(f"Unapproved source metadata: {document.document_id}")
            seen_ids.add(document.document_id)
            seen_hashes.add(document.content_hash)
            documents.append(document)
    if not documents:
        raise ValueError("Evidence corpus contains no documents")
    return documents


def resolve_local_corpus_path(path: Path, allowed_root: Path | None = None) -> Path:
    """Resolve a local corpus path and reject URL-like or escaped paths."""
    candidate = path.expanduser().resolve()
    if (
        "://" in str(path)
        or path.parts
        and path.parts[0].lower()
        in {
            "http:",
            "https:",
        }
    ):
        raise ValueError("Evidence corpus must be a local filesystem path")
    if allowed_root is not None:
        root = allowed_root.expanduser().resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("Evidence corpus path escapes the allowed root") from exc
    return candidate


def corpus_stats(
    documents: list[EvidenceDocument], approved_domains: set[str] | None = None
) -> CorpusStats:
    approved = approved_domains or set()
    optional = ("url", "publisher", "published_at", "retrieved_at", "license")
    missing = {
        field: sum(getattr(doc, field) is None for doc in documents)
        for field in optional
    }
    review_count = sum(
        bool(doc.url and _domain(doc.url) not in approved) for doc in documents
    )
    return CorpusStats(
        document_count=len(documents),
        language_counts=dict(Counter(doc.language for doc in documents)),
        source_category_counts=dict(Counter(doc.source_category for doc in documents)),
        missing_optional_metadata=missing,
        duplicate_document_id_count=0,
        duplicate_content_hash_count=0,
        average_document_length_characters=round(
            sum(len(doc.text) for doc in documents) / len(documents), 4
        ),
        content_hash_prefixes=dict(Counter(doc.content_hash[:15] for doc in documents)),
        license_metadata_status=(
            "complete" if all(doc.license for doc in documents) else "incomplete"
        ),
        sources_requiring_review=review_count,
    )


def write_corpus_audit(
    documents: list[EvidenceDocument],
    output_path: Path,
    *,
    corpus_scope: str = "local_corpus",
) -> CorpusStats:
    stats = corpus_stats(documents)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "offline_only": True,
                "corpus_scope": corpus_scope,
                "stats": stats.model_dump(),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("backend/data/reports/verification/local_corpus_audit.json"),
    )
    parser.add_argument(
        "--test-fixture",
        action="store_true",
        help="Mark the audit explicitly as test-fixture-only",
    )
    args = parser.parse_args()
    documents = load_corpus(args.corpus)
    scope = "test_fixture_only" if args.test_fixture else "local_corpus"
    print(
        write_corpus_audit(documents, args.report, corpus_scope=scope).model_dump_json(
            indent=2
        )
    )


if __name__ == "__main__":
    main()
