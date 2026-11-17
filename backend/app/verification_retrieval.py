"""Offline BM25 retrieval over explicitly loaded local evidence chunks."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime

from pydantic import BaseModel

from app.verification_chunking import EvidenceChunk

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    text: str
    url: str | None = None
    publisher: str | None = None
    source_category: str
    language: str
    retrieval_score: float
    retrieved_at: datetime


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


class OfflineBM25Retriever:
    def __init__(
        self,
        chunks: list[EvidenceChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError("At least one evidence chunk is required")
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("BM25 parameters are invalid")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(chunk.text) for chunk in chunks]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths)
        document_frequency: dict[str, int] = {}
        for tokens in self.tokens:
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        self.idf = {
            token: math.log(1 + (len(chunks) - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("Retrieval query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_terms = tokenize(query)
        scored: list[tuple[float, int]] = []
        for index, tokens in enumerate(self.tokens):
            frequencies = {term: tokens.count(term) for term in set(query_terms)}
            score = 0.0
            for term, frequency in frequencies.items():
                if not frequency:
                    continue
                idf = self.idf.get(term, 0.0)
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * self.lengths[index] / self.average_length
                )
                score += idf * frequency * (self.k1 + 1) / denominator
            scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], self.chunks[item[1]].chunk_id))
        timestamp = datetime.now(UTC)
        return [
            RetrievalResult(
                **self.chunks[index].model_dump(),
                retrieval_score=round(score, 8),
                retrieved_at=timestamp,
            )
            for score, index in scored[:top_k]
            if score > 0
        ]
