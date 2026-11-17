from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from app.verification_contract import (
    EvidenceCitation,
    EvidenceVerification,
    TemporalContext,
    VerificationResponse,
)
from pydantic import ValidationError

NOW = datetime.now(UTC)


def citation(relation: str = "supports") -> EvidenceCitation:
    return EvidenceCitation(
        source_title="Official report",
        source_url="https://example.org/report",
        publisher="Example institution",
        source_category="official_institutional",
        publication_date=NOW,
        retrieved_at=NOW,
        excerpt="The report describes the relevant event.",
        relevance_score=0.9,
        entailment_score=0.8,
        relation=relation,
    )


def test_supported_requires_citation() -> None:
    with pytest.raises(ValidationError):
        EvidenceVerification(
            status="SUPPORTED", confidence=0.8, evidence=[], reason="No source"
        )


def test_invalid_status_is_rejected() -> None:
    with pytest.raises(ValidationError):
        EvidenceVerification(
            status="TRUE", confidence=0.8, evidence=[citation()], reason="Invalid"
        )


def test_insufficient_evidence_allows_empty_evidence() -> None:
    result = EvidenceVerification(
        status="INSUFFICIENT_EVIDENCE",
        confidence=0.0,
        evidence=[],
        reason="No sufficient independent evidence was found.",
    )
    assert result.evidence == []


def test_conflicting_evidence_requires_both_relations() -> None:
    with pytest.raises(ValidationError):
        EvidenceVerification(
            status="CONFLICTING_EVIDENCE",
            confidence=0.4,
            evidence=[citation("supports")],
            reason="Sources disagree.",
        )
    result = EvidenceVerification(
        status="CONFLICTING_EVIDENCE",
        confidence=0.4,
        evidence=[citation("supports"), citation("contradicts")],
        reason="Sources disagree.",
    )
    assert len(result.evidence) == 2


def test_citation_and_temporal_metadata_are_required() -> None:
    item = citation()
    assert item.source_url.startswith("https://")
    context = TemporalContext(
        claim_time=None,
        event_time=None,
        verification_time=NOW,
        evidence_time_policy="Evidence must be available by verification_time.",
    )
    assert context.verification_time == NOW


def test_pattern_and_verification_are_separate() -> None:
    response = VerificationResponse(
        claim={"text": "A claim", "language": "en"},
        pattern_analysis={
            "label": "unverified",
            "confidence": 0.61,
            "warning": "Pattern classification is not factual verification.",
        },
        evidence_verification={
            "status": "INSUFFICIENT_EVIDENCE",
            "confidence": 0.0,
            "evidence": [],
            "reason": "No sufficient independent evidence was found.",
        },
        temporal_context={
            "verification_time": NOW,
            "evidence_time_policy": "Use evidence available by verification time.",
        },
    )
    assert response.pattern_analysis is not None
    assert response.evidence_verification.status == "INSUFFICIENT_EVIDENCE"


def test_contract_does_not_reference_training_datasets() -> None:
    contract_text = VerificationResponse.model_json_schema()
    assert "PHEME" not in str(contract_text)
    assert "HinFakeNews" not in str(contract_text)


def test_verification_contract_isolated_from_pheme_and_indian_data() -> None:
    source = Path("backend/app/verification_contract.py").read_text(encoding="utf-8")
    assert "PHEME" not in source
    assert "HinFakeNews" not in source
    assert "backend/data" not in source
