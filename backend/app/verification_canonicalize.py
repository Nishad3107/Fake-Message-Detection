"""Canonicalization helpers for leakage checks on evidence benchmarks."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.averitec_utils import answer_records

_TRACKING_KEYS = {"fbclid", "gclid", "ref", "referrer"}
_SPACE_RE = re.compile(r"\s+")


def canonical_text(value: str | None) -> str:
    text = unicodedata.normalize("NFKC", value or "")
    return _SPACE_RE.sub(" ", text).strip().casefold()


def stable_hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_url(value: str | None) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    parsed = urlsplit(value.strip())
    scheme = parsed.scheme.casefold()
    host = (parsed.hostname or "").casefold()
    host = host.removeprefix("www.")
    netloc = host
    if parsed.port:
        netloc = f"{host}:{parsed.port}"
    query = [
        (key, val)
        for key, val in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.casefold().startswith("utm_")
        and key.casefold() not in _TRACKING_KEYS
    ]
    return urlunsplit((scheme, netloc, parsed.path or "/", urlencode(query), ""))


def canonical_url_hash(value: str | None) -> str | None:
    normalized = canonical_url(value)
    return stable_hash(normalized) if normalized else None


def evidence_document_identity(text: str | None, url: str | None) -> str:
    url_value = canonical_url(url)
    if url_value:
        return stable_hash("url:" + url_value)
    return stable_hash("text:" + canonical_text(text))


def record_signatures(record: dict[str, Any]) -> dict[str, Any]:
    claim_value = canonical_text(record.get("claim"))
    urls: list[str] = []
    evidence: list[dict[str, str | None]] = []
    for answer in answer_records(record):
        text = answer.get("answer")
        source_url = answer.get("source_url") or answer.get("cached_source_url")
        if isinstance(source_url, str) and source_url.strip():
            urls.append(source_url)
        if isinstance(text, str) and text.strip():
            evidence.append(
                {
                    "evidence_hash": stable_hash(canonical_text(text)),
                    "document_identity": evidence_document_identity(text, source_url),
                }
            )
    return {
        "claim_hash": stable_hash(claim_value) if claim_value else None,
        "url_hashes": sorted(
            {item for item in (canonical_url_hash(url) for url in urls) if item}
        ),
        "evidence_hashes": sorted({item["evidence_hash"] for item in evidence}),
        "evidence_document_hashes": sorted(
            {item["document_identity"] for item in evidence}
        ),
        "canonical_urls": sorted({canonical_url(url) for url in urls}),
    }
