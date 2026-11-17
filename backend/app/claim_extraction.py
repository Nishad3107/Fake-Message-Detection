"""Deterministic India-first claim extraction; not a fact verifier."""

from __future__ import annotations

import re
import unicodedata

from app.claim_schema import (
    ClaimExtractionResult,
    ExtractedClaim,
    QueryCandidate,
)
from app.verification_input import VerificationInput, process_input

URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
NUMBER_RE = re.compile(
    r"(?<!\w)(?:₹|rs\.?|inr|\$|€|£)?\s*\d[\d,]*(?:\.\d+)?%?", re.IGNORECASE
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
LATIN_RE = re.compile(r"[A-Za-z]")
INDIA_PLACES = (
    "India",
    "Delhi",
    "New Delhi",
    "Mumbai",
    "Maharashtra",
    "Bengaluru",
    "Bangalore",
    "Karnataka",
    "Uttar Pradesh",
    "भारत",
    "दिल्ली",
    "मुंबई",
    "महाराष्ट्र",
    "कर्नाटक",
    "उत्तर प्रदेश",
)
HINDI_MARKERS = {
    "hai",
    "hain",
    "ka",
    "ki",
    "ke",
    "mein",
    "se",
    "ko",
    "aur",
    "par",
    "यह",
    "है",
    "हैं",
    "का",
    "की",
    "के",
    "में",
    "से",
    "को",
    "और",
}
QUESTION_MARKERS = ("?", "क्या", "क्यों", "कैसे", "who", "what", "why", "how", "is ", "are ")
OPINION_MARKERS = (
    "i think",
    "i believe",
    "in my opinion",
    "should",
    "must",
    "मुझे लगता",
    "मेरे विचार",
)
SPECULATION_MARKERS = (
    "may",
    "might",
    "could",
    "possibly",
    "allegedly",
    "शायद",
    "संभवतः",
    "कहा जा रहा",
)
NEGATION_MARKERS = ("not", "never", "no", "नहीं", "नही", "मत", "न", "false", "गलत")
TYPE_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("health", ("health", "hospital", "disease", "स्वास्थ्य", "अस्पताल", "बीमारी")),
    ("science", ("science", "research", "वैज्ञानिक", "विज्ञान", "अनुसंधान")),
    ("finance", ("rupee", "₹", "budget", "tax", "बजट", "कर")),
    ("policy", ("policy", "law", "government", "नीति", "कानून", "सरकार")),
    ("statistics", ("%", "percent", "प्रतिशत", "million", "करोड़", "लाख")),
    ("historical", ("history", "historical", "इतिहास", "ऐतिहासिक")),
    ("quote", ('"', "said", "कहा", "बताया")),
)


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def detect_language_script(text: str) -> tuple[str, str, bool]:
    devanagari = bool(DEVANAGARI_RE.search(text))
    latin = bool(LATIN_RE.search(text))
    mixed = devanagari and latin
    if mixed:
        return "hinglish", "mixed", True
    if devanagari:
        return "hi", "devanagari", False
    if latin:
        words = set(re.findall(r"[A-Za-z]+", text.casefold()))
        return (
            ("romanized_hi", "latin", False)
            if words & HINDI_MARKERS
            else ("en", "latin", False)
        )
    return "unknown", "unknown", False


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?।])\s+", text) if part.strip()]


def _entities(text: str) -> list[str]:
    found: list[str] = []
    for place in INDIA_PLACES:
        if place.casefold() in text.casefold() and place not in found:
            found.append(place)
    for match in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", text):
        if match not in found:
            found.append(match)
    return found


def _numbers(text: str) -> list[str]:
    return list(dict.fromkeys(match.strip() for match in NUMBER_RE.findall(text)))


def _date_or_time(text: str) -> str | None:
    year = YEAR_RE.search(text)
    if year:
        return year.group(0)
    month = re.search(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
        text,
        re.IGNORECASE,
    )
    return month.group(0) if month else None


def _claim_type(text: str, is_question: bool) -> str:
    lowered = text.casefold()
    if any(marker in lowered for marker in OPINION_MARKERS):
        return "opinion"
    if any(marker in lowered for marker in SPECULATION_MARKERS):
        return "prediction"
    if is_question:
        return "unknown"
    for claim_type, markers in TYPE_MARKERS:
        if any(marker.casefold() in lowered for marker in markers):
            return claim_type
    return "event"


def _query_candidates(
    claim_id: str,
    claim_text: str,
    normalized: str,
    language: str,
    entities: list[str],
    date_or_time: str | None,
    location: str | None,
) -> list[QueryCandidate]:
    candidates: list[tuple[str, str, str]] = [
        (claim_text, "original", "Preserves the original claim wording."),
        (normalized, "normalized", "Uses normalized whitespace and Unicode."),
    ]
    if entities:
        candidates.append(
            (
                " ".join(entities),
                "entity_focused",
                "Uses entities found directly in the claim.",
            )
        )
    if date_or_time or location:
        context = " ".join(item for item in (location, date_or_time) if item)
        candidates.append(
            (
                f"{context} {normalized}".strip(),
                "date_location_focused",
                "Adds date or location present in the claim.",
            )
        )
    output: list[QueryCandidate] = []
    seen: set[str] = set()
    for query, query_type, reason in candidates:
        if query and query not in seen:
            output.append(
                QueryCandidate(
                    query_text=query,
                    query_type=query_type,
                    source_claim_id=claim_id,
                    language=language,
                    reason=reason,
                )
            )
            seen.add(query)
    return output[:5]


def _extract_sentence(
    sentence: str,
    index: int,
    language: str,
    script: str,
    mixed: bool,
    previous: ExtractedClaim | None,
) -> ExtractedClaim:
    normalized = normalize_text(sentence)
    lowered = normalized.casefold()
    is_question = any(marker.casefold() in lowered for marker in QUESTION_MARKERS)
    opinion = any(marker.casefold() in lowered for marker in OPINION_MARKERS)
    speculation = any(marker.casefold() in lowered for marker in SPECULATION_MARKERS)
    entities = _entities(normalized)
    numbers = _numbers(normalized)
    urls = URL_RE.findall(normalized)
    date_or_time = _date_or_time(normalized)
    location = next(
        (place for place in INDIA_PLACES if place.casefold() in lowered), None
    )
    claim_id = f"claim-{index + 1:03d}"
    claim_type = _claim_type(normalized, is_question)
    relation = "independent"
    if previous is not None and any(
        marker in lowered
        for marker in ("because", "therefore", "लेकिन", "इसलिए", "however")
    ):
        relation = "unknown"
    confidence = 0.45
    if claim_type in {
        "event",
        "policy",
        "health",
        "science",
        "finance",
        "statistics",
        "historical",
    }:
        confidence = 0.7
    if is_question or opinion or language == "unknown":
        confidence = 0.25
    queries = (
        []
        if opinion or is_question
        else _query_candidates(
            claim_id, sentence, normalized, language, entities, date_or_time, location
        )
    )
    return ExtractedClaim(
        claim_id=claim_id,
        claim_text=sentence,
        language=language,
        script=script,
        mixed_language=mixed,
        normalized_text=normalized,
        entities=entities,
        numbers=numbers,
        urls=urls,
        location=location,
        date_or_time=date_or_time,
        claim_type=claim_type,
        verification_queries=queries,
        extraction_confidence=confidence,
        needs_human_review=is_question
        or opinion
        or speculation
        or language == "unknown",
        is_question=is_question,
        negated=any(marker in lowered.split() for marker in NEGATION_MARKERS),
        sentence_index=index,
        relation_to_previous=relation,
    )


def extract_claims(value: VerificationInput) -> ClaimExtractionResult:
    processed = process_input(value)
    if processed.status != "ready":
        return ClaimExtractionResult(
            input_type=value.input_type,
            status="verification_unavailable",
            language="unknown",
            script="unknown",
            mixed_language=False,
            reason=processed.reason,
        )
    assert processed.text is not None
    language, script, mixed = detect_language_script(processed.text)
    sentences = split_sentences(processed.text)
    claims: list[ExtractedClaim] = []
    for index, sentence in enumerate(sentences):
        claim = _extract_sentence(
            sentence, index, language, script, mixed, claims[-1] if claims else None
        )
        claims.append(claim)
    return ClaimExtractionResult(
        input_type=value.input_type,
        status="verification_unavailable" if language == "unknown" else "ready",
        language=language,
        script=script,
        mixed_language=mixed,
        claims=claims,
        reason=("Unsupported language or script." if language == "unknown" else None),
    )
