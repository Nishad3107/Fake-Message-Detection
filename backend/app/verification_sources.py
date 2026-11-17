"""India-first source records for the future verifier.

All initial records are pending review and disabled. This registry is not used
by the current PHEME prediction API or any live retrieval process.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

SourceCategory = Literal[
    "official_government",
    "official_institution",
    "primary_document",
    "court_or_legal_record",
    "research_paper",
    "professional_fact_checker",
    "reputable_news",
    "international_reference",
    "user_generated",
    "unknown",
]
ApprovalStatus = Literal["approved", "pending_review", "blocked"]


class VerificationSource(BaseModel):
    name: str = Field(..., min_length=1)
    domain: str = Field(..., min_length=1)
    category: SourceCategory
    country_or_region: str
    languages: list[str] = Field(min_length=1)
    allowed_for_retrieval: bool = False
    allowed_for_citation: bool = False
    requires_manual_review: bool = True
    approval_status: ApprovalStatus = "pending_review"
    notes: str = ""
    last_policy_review_date: date


# These are source candidates, not approvals. Domains remain disabled until a
# human confirms provenance, terms, language coverage, and citation suitability.
SOURCE_REGISTRY: tuple[VerificationSource, ...] = (
    VerificationSource(
        name="Press Information Bureau India",
        domain="pib.gov.in",
        category="official_government",
        country_or_region="India",
        languages=["hi", "en"],
        notes="Candidate official government source; policy approval pending.",
        last_policy_review_date=date(2026, 11, 17),
    ),
    VerificationSource(
        name="Election Commission of India",
        domain="eci.gov.in",
        category="official_government",
        country_or_region="India",
        languages=["hi", "en"],
        notes="Candidate official institutional source; policy approval pending.",
        last_policy_review_date=date(2026, 11, 17),
    ),
    VerificationSource(
        name="Supreme Court of India",
        domain="sci.gov.in",
        category="court_or_legal_record",
        country_or_region="India",
        languages=["en"],
        notes="Candidate legal record source; policy approval pending.",
        last_policy_review_date=date(2026, 11, 17),
    ),
    VerificationSource(
        name="IndiaAI dataset catalogue",
        domain="aikosh.indiaai.gov.in",
        category="official_institution",
        country_or_region="India",
        languages=["en"],
        notes="Candidate institutional catalogue; not an evidence source by default.",
        last_policy_review_date=date(2026, 11, 17),
    ),
)
