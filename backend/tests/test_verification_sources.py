from __future__ import annotations

from pathlib import Path

import pytest
from app.source_quality import assess_source
from app.verification_source_validator import (
    decide_source,
    load_policy,
    validate_policy,
)
from app.verification_sources import SOURCE_REGISTRY

POLICY = Path("config/verification_sources.yaml")


def test_conservative_policy_loads_and_live_retrieval_is_disabled() -> None:
    policy = load_policy(POLICY)
    assert policy.defaults["live_retrieval_allowed"] is False
    assert policy.defaults["full_page_storage_allowed"] is False
    assert policy.defaults["external_search_api_allowed"] is False


def test_pending_source_cannot_be_retrieved_or_cited() -> None:
    decision = decide_source(load_policy(POLICY), "pib.gov.in", "hi")
    assert not decision.can_retrieve_candidate
    assert not decision.can_cite
    assert decision.requires_manual_review


def test_explicitly_approved_source_can_be_enabled_in_a_future_policy() -> None:
    policy = load_policy(POLICY)
    approved = policy.sources[0].model_copy(
        update={
            "approval_status": "approved",
            "allowed_for_retrieval": True,
            "allowed_for_citation": True,
            "requires_manual_review": False,
        }
    )
    future_policy = policy.model_copy(
        update={
            "defaults": {**policy.defaults, "live_retrieval_allowed": True},
            "sources": [approved],
        }
    )
    validate_policy(future_policy)
    decision = decide_source(future_policy, "pib.gov.in", "hi")
    assert decision.can_retrieve_candidate
    assert decision.can_cite
    assert not decision.requires_manual_review


def test_blocked_source_is_rejected() -> None:
    policy = load_policy(POLICY).model_copy(update={"blocked_domains": ["blocked.in"]})
    decision = decide_source(policy, "blocked.in", "en")
    assert not decision.can_retrieve_candidate
    assert not decision.can_cite


def test_unknown_source_is_not_citable() -> None:
    decision = decide_source(load_policy(POLICY), "unknown.example", "en")
    assert not decision.can_cite
    assert decision.requires_manual_review


def test_unsupported_language_is_rejected() -> None:
    decision = decide_source(load_policy(POLICY), "pib.gov.in", "ta")
    assert not decision.can_retrieve_candidate
    assert not decision.can_cite
    assert "Unsupported" in decision.reason


def test_invalid_domain_and_conflict_are_rejected() -> None:
    policy = load_policy(POLICY)
    with pytest.raises(ValueError, match="Invalid source domain"):
        validate_policy(policy.model_copy(update={"allowed_domains": ["not a domain"]}))
    with pytest.raises(ValueError, match="both allowed and blocked"):
        validate_policy(
            policy.model_copy(
                update={
                    "allowed_domains": ["example.in"],
                    "blocked_domains": ["example.in"],
                }
            )
        )


def test_duplicate_source_domains_are_rejected() -> None:
    policy = load_policy(POLICY)
    duplicate = policy.sources[0].model_copy(update={"name": "Duplicate"})
    with pytest.raises(ValueError, match="Duplicate source domain"):
        validate_policy(
            policy.model_copy(update={"sources": [policy.sources[0], duplicate]})
        )


def test_secret_in_policy_is_rejected() -> None:
    policy = load_policy(POLICY)
    unsafe = policy.model_copy(
        update={"defaults": {**policy.defaults, "external_search_api_key": "secret"}}
    )
    with pytest.raises(ValueError, match="Secrets"):
        validate_policy(unsafe)


def test_quality_is_structured_not_a_truth_score() -> None:
    factors = assess_source(SOURCE_REGISTRY[0], citation_complete=True)
    assert factors.source_category == "official_government"
    assert factors.citation_complete
    assert not hasattr(factors, "truth_score")


def test_registry_records_are_pending_and_disabled() -> None:
    assert SOURCE_REGISTRY
    assert all(item.approval_status == "pending_review" for item in SOURCE_REGISTRY)
    assert all(not item.allowed_for_citation for item in SOURCE_REGISTRY)
