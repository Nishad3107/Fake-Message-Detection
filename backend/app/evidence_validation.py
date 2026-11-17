"""Citation completeness checks for offline retrieval results."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.verification_retrieval import RetrievalResult


class EvidenceValidationResult(BaseModel):
    complete: bool
    missing_fields: list[str] = Field(default_factory=list)
    missing_optional_fields: list[str] = Field(default_factory=list)
    reason: str


def validate_evidence(result: RetrievalResult) -> EvidenceValidationResult:
    missing: list[str] = []
    for field in (
        "chunk_id",
        "document_id",
        "title",
        "text",
        "source_category",
        "retrieval_score",
    ):
        if not getattr(result, field):
            missing.append(field)
    if not result.retrieved_at:
        missing.append("retrieved_at")
    optional_missing = [
        field for field in ("url", "publisher") if getattr(result, field) is None
    ]
    complete = not missing
    return EvidenceValidationResult(
        complete=complete,
        missing_fields=missing,
        missing_optional_fields=optional_missing,
        reason=(
            "Evidence contains the fields required for citation."
            if complete
            else "Evidence is incomplete and must not become a final citation automatically."
        ),
    )
