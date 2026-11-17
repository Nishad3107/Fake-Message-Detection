from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest
from app.evidence_validation import validate_evidence
from app.verification_chunking import chunk_documents
from app.verification_corpus import (
    load_corpus,
    resolve_local_corpus_path,
    write_corpus_audit,
)
from app.verification_retrieval import OfflineBM25Retriever
from app.verification_retrieval_eval import evaluate_retrieval

FIXTURE = Path("backend/tests/fixtures/evidence_corpus.jsonl")


def content_hash(text: str) -> str:
    return "sha256:" + sha256(text.encode("utf-8")).hexdigest()


def record(document_id: str, text: str, **overrides: object) -> dict[str, object]:
    return {
        "document_id": document_id,
        "title": "Fictional title",
        "text": text,
        "source_category": "unknown",
        "language": "en",
        "content_hash": content_hash(text),
        **overrides,
    }


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )


def test_valid_fixture_loads_and_audit_report_is_local(tmp_path: Path) -> None:
    documents = load_corpus(FIXTURE)
    assert len(documents) == 4
    report = tmp_path / "audit.json"
    stats = write_corpus_audit(documents, report, corpus_scope="test_fixture_only")
    assert stats.document_count == 4
    assert stats.language_counts == {"en": 3, "hi": 1}
    assert stats.license_metadata_status == "incomplete"
    assert report.is_file()
    assert "test_fixture_only" in report.read_text(encoding="utf-8")


def test_invalid_jsonl_and_empty_text_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid corpus line"):
        load_corpus(bad)
    write_jsonl(tmp_path / "empty.jsonl", [record("x", "   ")])
    with pytest.raises(ValueError, match="Invalid corpus line"):
        load_corpus(tmp_path / "empty.jsonl")


def test_duplicate_ids_hashes_and_invalid_metadata_are_rejected(tmp_path: Path) -> None:
    duplicate_id = [record("same", "one"), record("same", "two")]
    write_jsonl(tmp_path / "ids.jsonl", duplicate_id)
    with pytest.raises(ValueError, match="Duplicate document ID"):
        load_corpus(tmp_path / "ids.jsonl")
    duplicate_hash = [record("one", "same"), record("two", "same")]
    write_jsonl(tmp_path / "hashes.jsonl", duplicate_hash)
    with pytest.raises(ValueError, match="Duplicate content hash"):
        load_corpus(tmp_path / "hashes.jsonl")
    write_jsonl(tmp_path / "language.jsonl", [record("x", "text", language="fr")])
    with pytest.raises(ValueError, match="Unsupported evidence language"):
        load_corpus(tmp_path / "language.jsonl")
    write_jsonl(
        tmp_path / "category.jsonl",
        [record("x", "text", source_category="bad_category")],
    )
    with pytest.raises(ValueError, match="Invalid source category"):
        load_corpus(tmp_path / "category.jsonl")


def test_local_path_safety() -> None:
    with pytest.raises(ValueError, match="local filesystem"):
        resolve_local_corpus_path(Path("https://example.org/evidence.jsonl"))


def test_chunk_metadata_and_ids_are_deterministic() -> None:
    documents = load_corpus(FIXTURE)
    first = chunk_documents(documents, max_characters=50, overlap=10)
    second = chunk_documents(documents, max_characters=50, overlap=10)
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert first[0].document_id == "fixture-support-en"
    assert first[0].content_hash.startswith("sha256:")


def test_bm25_retrieves_english_and_hindi_without_network() -> None:
    chunks = chunk_documents(load_corpus(FIXTURE), max_characters=500)
    retriever = OfflineBM25Retriever(chunks)
    english = retriever.retrieve("library opens Monday", top_k=2)
    hindi = retriever.retrieve("पुस्तकालय सोमवार खुलेगा", top_k=2)
    assert english[0].document_id == "fixture-support-en"
    assert hindi[0].document_id == "fixture-hindi"
    assert len(retriever.retrieve("library opens Monday", top_k=1)) == 1


def test_retrieval_evaluation_contract() -> None:
    metrics = evaluate_retrieval(["a", "b", "c"], {"b"})
    assert metrics["recall_at_1"] == 0
    assert metrics["recall_at_3"] == 1
    assert metrics["mean_reciprocal_rank"] == 0.5


def test_citation_validation_marks_missing_optional_metadata() -> None:
    result = OfflineBM25Retriever(
        chunk_documents(load_corpus(FIXTURE), max_characters=500)
    ).retrieve("weather note", top_k=1)[0]
    validation = validate_evidence(result)
    assert validation.complete
    assert validation.missing_optional_fields == ["url", "publisher"]


def test_fixture_is_not_a_research_claim() -> None:
    assert "fixtures" in FIXTURE.parts
