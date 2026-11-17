"""Deterministic local document chunking with provenance preservation."""

from __future__ import annotations

import hashlib

from pydantic import BaseModel

from app.verification_corpus import EvidenceDocument


class EvidenceChunk(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    text: str
    url: str | None = None
    publisher: str | None = None
    source_category: str
    language: str
    published_at: str | None = None
    content_hash: str


def chunk_documents(
    documents: list[EvidenceDocument],
    *,
    max_characters: int = 600,
    overlap: int = 80,
) -> list[EvidenceChunk]:
    if max_characters < 1:
        raise ValueError("max_characters must be positive")
    if overlap < 0 or overlap >= max_characters:
        raise ValueError("overlap must be non-negative and smaller than max_characters")
    chunks: list[EvidenceChunk] = []
    step = max_characters - overlap
    for document in documents:
        text = document.text
        starts = range(0, max(1, len(text)), step)
        for index, start in enumerate(starts):
            excerpt = text[start : start + max_characters]
            if not excerpt:
                continue
            digest = hashlib.sha256(
                f"{document.document_id}:{index}:{excerpt}".encode()
            ).hexdigest()[:16]
            chunks.append(
                EvidenceChunk(
                    chunk_id=f"{document.document_id}:chunk-{index:04d}-{digest}",
                    document_id=document.document_id,
                    title=document.title,
                    text=excerpt,
                    url=document.url,
                    publisher=document.publisher,
                    source_category=document.source_category,
                    language=document.language,
                    published_at=document.published_at,
                    content_hash=document.content_hash,
                )
            )
    return chunks
