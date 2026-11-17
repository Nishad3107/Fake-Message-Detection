"""Typed contracts for deterministic claim extraction."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ClaimType = Literal[
    "event",
    "policy",
    "person",
    "organization",
    "place",
    "health",
    "science",
    "finance",
    "statistics",
    "quote",
    "historical",
    "prediction",
    "opinion",
    "unknown",
]
ClaimRelation = Literal[
    "independent",
    "supports",
    "contradicts",
    "depends_on",
    "unknown",
]


class QueryCandidate(BaseModel):
    query_text: str = Field(..., min_length=1)
    query_type: Literal[
        "original",
        "normalized",
        "entity_focused",
        "date_location_focused",
        "language_variant",
    ]
    source_claim_id: str
    language: str
    reason: str


class ExtractedClaim(BaseModel):
    claim_id: str
    claim_text: str = Field(..., min_length=1)
    language: str
    script: Literal["devanagari", "latin", "mixed", "unknown"]
    mixed_language: bool = False
    normalized_text: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    location: str | None = None
    date_or_time: str | None = None
    entities: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    claim_type: ClaimType = "unknown"
    verification_queries: list[QueryCandidate] = Field(default_factory=list)
    extraction_method: Literal["rule_based"] = "rule_based"
    extraction_confidence: float = Field(..., ge=0.0, le=1.0)
    needs_human_review: bool = False
    is_question: bool = False
    negated: bool = False
    sentence_index: int
    relation_to_previous: ClaimRelation = "independent"


class ClaimExtractionResult(BaseModel):
    input_type: str
    status: Literal["ready", "verification_unavailable"]
    language: str
    script: str
    mixed_language: bool
    claims: list[ExtractedClaim] = Field(default_factory=list)
    reason: str | None = None
