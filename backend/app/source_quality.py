"""Contextual source-quality factors; deliberately no universal truth score."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from app.verification_sources import VerificationSource


class SourceQualityFactors(BaseModel):
    source_category: str
    publisher_identity: str | None
    domain_match: bool
    publication_date: datetime | None
    retrieval_date: datetime
    evidence_freshness_days: int | None
    primary_source_indicator: bool | None
    article_text_available: bool
    citation_complete: bool
    conflicting_source_count: int = Field(ge=0)
    human_review_required: bool


def assess_source(
    source: VerificationSource,
    *,
    publication_date: datetime | None = None,
    article_text_available: bool = True,
    citation_complete: bool = False,
    conflicting_source_count: int = 0,
    primary_source_indicator: bool | None = None,
    retrieved_at: datetime | None = None,
) -> SourceQualityFactors:
    retrieved = retrieved_at or datetime.now(UTC)
    freshness = None
    if publication_date is not None:
        freshness = max(0, (retrieved.date() - publication_date.date()).days)
    return SourceQualityFactors(
        source_category=source.category,
        publisher_identity=source.name,
        domain_match=True,
        publication_date=publication_date,
        retrieval_date=retrieved,
        evidence_freshness_days=freshness,
        primary_source_indicator=primary_source_indicator,
        article_text_available=article_text_available,
        citation_complete=citation_complete,
        conflicting_source_count=conflicting_source_count,
        human_review_required=source.requires_manual_review
        or not source.allowed_for_citation,
    )
