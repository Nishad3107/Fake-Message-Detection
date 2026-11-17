"""Future evidence-verification contracts, intentionally not wired to the API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.claim_schema import ExtractedClaim

EvidenceStatus = Literal[
    "SUPPORTED",
    "REFUTED",
    "INSUFFICIENT_EVIDENCE",
    "CONFLICTING_EVIDENCE",
    "TEMPORALLY_UNCLEAR",
    "VERIFICATION_UNAVAILABLE",
]
EvidenceRelation = Literal["supports", "contradicts", "neutral"]
SourceCategory = Literal[
    "official_government",
    "official_institutional",
    "primary_research",
    "court_or_legal",
    "reputable_news",
    "professional_fact_check",
    "expert_technical",
    "user_generated",
    "unknown_low_context",
]
PatternLabel = Literal["true", "false", "unverified", "pattern_model_uncertain"]
VerificationInputType = Literal[
    "plain_text",
    "social_media_text",
    "article_text",
    "url",
    "image",
    "video",
]


class Claim(BaseModel):
    text: str = Field(..., min_length=1)
    language: str | None = None


class PatternAnalysis(BaseModel):
    label: PatternLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    warning: str = "Pattern classification is not factual verification."


class EvidenceCitation(BaseModel):
    source_title: str = Field(..., min_length=1)
    source_url: str = Field(..., min_length=1)
    publisher: str | None = None
    source_category: SourceCategory
    publication_date: datetime | None = None
    retrieved_at: datetime
    excerpt: str = Field(..., min_length=1)
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    entailment_score: float = Field(..., ge=0.0, le=1.0)
    relation: EvidenceRelation


class TemporalContext(BaseModel):
    claim_time: datetime | None = None
    event_time: datetime | None = None
    verification_time: datetime
    evidence_time_policy: str


class EvidenceVerification(BaseModel):
    status: EvidenceStatus
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: list[EvidenceCitation] = Field(default_factory=list)
    reason: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_evidence_requirements(self) -> EvidenceVerification:
        if self.status in {"SUPPORTED", "REFUTED"} and not self.evidence:
            raise ValueError(f"{self.status} requires at least one citation")
        if self.status == "CONFLICTING_EVIDENCE":
            relations = {item.relation for item in self.evidence}
            if not {"supports", "contradicts"}.issubset(relations):
                raise ValueError(
                    "CONFLICTING_EVIDENCE requires supporting and contradicting citations"
                )
        return self


class VerificationResponse(BaseModel):
    claim: Claim
    input_type: VerificationInputType = "plain_text"
    extracted_claims: list[ExtractedClaim] = Field(default_factory=list)
    pattern_analysis: PatternAnalysis | None = None
    evidence_verification: EvidenceVerification
    temporal_context: TemporalContext
    verification_readiness: Literal[
        "ready", "needs_human_review", "verification_unavailable"
    ] = "ready"
    human_review_required: bool = False
    unsupported_modality_status: str | None = None
    limitations: list[str] = Field(default_factory=list)
