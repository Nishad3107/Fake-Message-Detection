"""Validation and policy decisions for the future source registry."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field, ValidationError

from app.verification_sources import VerificationSource

DOMAIN_RE = re.compile(
    r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}$"
)
LANGUAGES = {"hi", "en", "hinglish", "romanized_hi"}


class SourcePolicy(BaseModel):
    policy_version: str
    last_policy_review_date: str
    defaults: dict[str, Any]
    allowed_languages: list[str]
    allowed_domains: list[str] = Field(default_factory=list)
    blocked_domains: list[str] = Field(default_factory=list)
    sources: list[VerificationSource] = Field(default_factory=list)


class SourceDecision(BaseModel):
    domain: str
    language: str
    can_retrieve_candidate: bool
    can_cite: bool
    requires_manual_review: bool
    reason: str


def load_policy(path: Path) -> SourcePolicy:
    if not path.is_file():
        raise FileNotFoundError(f"Verification source policy not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        policy = SourcePolicy.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid verification source policy: {exc}") from exc
    validate_policy(policy, require_conservative_defaults=True)
    return policy


def _validate_domain(domain: str) -> None:
    if not DOMAIN_RE.fullmatch(domain):
        raise ValueError(f"Invalid source domain: {domain}")


def validate_policy(
    policy: SourcePolicy, *, require_conservative_defaults: bool = False
) -> None:
    defaults = policy.defaults
    required_defaults = {
        "live_retrieval_allowed",
        "external_search_api_allowed",
        "full_page_storage_allowed",
        "excerpt_storage_allowed",
        "unknown_domains_candidate_retrieval",
        "unknown_domains_citation_allowed",
        "require_url_title_publisher_excerpt_timestamp",
    }
    missing = required_defaults - defaults.keys()
    if missing:
        raise ValueError(f"Missing policy defaults: {', '.join(sorted(missing))}")
    for name in required_defaults:
        if not isinstance(defaults[name], bool):
            raise TypeError(f"Policy default must be boolean: {name}")
    if require_conservative_defaults and any(
        defaults[name]
        for name in (
            "live_retrieval_allowed",
            "external_search_api_allowed",
            "full_page_storage_allowed",
            "unknown_domains_citation_allowed",
        )
    ):
        raise ValueError("Initial source policy contains an unsafe enabled default")
    if not set(policy.allowed_languages).issubset(LANGUAGES):
        raise ValueError("Policy contains an unsupported language code")
    for collection_name, domains in (
        ("allowed_domains", policy.allowed_domains),
        ("blocked_domains", policy.blocked_domains),
    ):
        if len(domains) != len(set(domains)):
            raise ValueError(f"Duplicate domains in {collection_name}")
        for domain in domains:
            _validate_domain(domain)
    conflict = set(policy.allowed_domains) & set(policy.blocked_domains)
    if conflict:
        raise ValueError(
            f"Domains cannot be both allowed and blocked: {sorted(conflict)}"
        )
    seen: set[str] = set()
    for source in policy.sources:
        _validate_domain(source.domain)
        if source.domain in seen:
            raise ValueError(f"Duplicate source domain: {source.domain}")
        seen.add(source.domain)
        if not set(source.languages).issubset(LANGUAGES):
            raise ValueError(f"Unsupported language on source: {source.domain}")
        if source.approval_status != "approved" and (
            source.allowed_for_retrieval or source.allowed_for_citation
        ):
            raise ValueError(f"Unapproved source cannot be enabled: {source.domain}")
    if defaults.get("external_search_api_key"):
        raise ValueError("Secrets must not be stored in the source policy")


def decide_source(policy: SourcePolicy, domain: str, language: str) -> SourceDecision:
    parsed = urlparse(domain if "://" in domain else f"https://{domain}")
    normalized = (parsed.hostname or domain).lower().rstrip(".")
    if language not in policy.allowed_languages:
        return SourceDecision(
            domain=normalized,
            language=language,
            can_retrieve_candidate=False,
            can_cite=False,
            requires_manual_review=True,
            reason="Unsupported evidence language.",
        )
    if normalized in policy.blocked_domains:
        return SourceDecision(
            domain=normalized,
            language=language,
            can_retrieve_candidate=False,
            can_cite=False,
            requires_manual_review=True,
            reason="Domain is explicitly blocked.",
        )
    source = next((item for item in policy.sources if item.domain == normalized), None)
    if source is None:
        candidate = bool(policy.defaults["unknown_domains_candidate_retrieval"])
        return SourceDecision(
            domain=normalized,
            language=language,
            can_retrieve_candidate=candidate
            and policy.defaults["live_retrieval_allowed"],
            can_cite=False,
            requires_manual_review=True,
            reason="Unknown domains may not be cited automatically.",
        )
    return SourceDecision(
        domain=normalized,
        language=language,
        can_retrieve_candidate=(
            source.allowed_for_retrieval
            and policy.defaults["live_retrieval_allowed"]
            and language in source.languages
        ),
        can_cite=(
            source.allowed_for_citation
            and source.approval_status == "approved"
            and language in source.languages
        ),
        requires_manual_review=source.requires_manual_review
        or source.approval_status != "approved",
        reason=(
            "Source is pending manual policy review."
            if source.approval_status != "approved"
            else "Source decision follows the configured policy."
        ),
    )
