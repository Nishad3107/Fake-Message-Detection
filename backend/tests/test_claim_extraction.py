from __future__ import annotations

from app.claim_extraction import extract_claims
from app.verification_input import VerificationInput, classify_input, process_input


def extract(text: str, input_type: str = "plain_text"):
    return extract_claims(VerificationInput(input_type=input_type, text=text))


def test_english_factual_claim_and_query_generation() -> None:
    result = extract("A fictional office opened in Delhi in 2024.")
    claim = result.claims[0]
    assert result.language == "en"
    assert result.script == "latin"
    assert claim.claim_type == "event"
    assert claim.location == "Delhi"
    assert claim.date_or_time == "2024"
    assert claim.verification_queries[0].query_type == "original"
    assert claim.extraction_method == "rule_based"


def test_hindi_and_hinglish_detection() -> None:
    hindi = extract("दिल्ली में पुस्तकालय सोमवार को खुलेगा।")
    hinglish = extract("Delhi mein library Monday ko khulegi.")
    assert hindi.language == "hi"
    assert hindi.script == "devanagari"
    assert hinglish.language == "romanized_hi"
    assert hinglish.script == "latin"


def test_opinion_and_question_require_review_and_do_not_query() -> None:
    opinion = extract("I think this fictional policy is helpful.").claims[0]
    question = extract("क्या काल्पनिक पुस्तकालय सोमवार को खुलेगा?").claims[0]
    assert opinion.claim_type == "opinion"
    assert opinion.needs_human_review
    assert opinion.verification_queries == []
    assert question.is_question
    assert question.claim_type == "unknown"
    assert question.needs_human_review


def test_negation_numbers_percent_currency_and_url() -> None:
    result = extract(
        "The fictional budget did not allocate ₹25 million in 2024, or 12% of funds. "
        "Details: https://fictional.example/post"
    )
    claim = result.claims[0]
    assert claim.negated
    assert "2024" == claim.date_or_time
    assert any("25" in value for value in claim.numbers)
    assert any("12%" in value for value in claim.numbers)
    assert result.claims[1].urls == ["https://fictional.example/post"]


def test_multiple_claims_keep_positions_and_relations_conservative() -> None:
    result = extract(
        "A fictional office opened in Delhi. It published 25 notices in 2024."
    )
    assert len(result.claims) == 2
    assert [claim.sentence_index for claim in result.claims] == [0, 1]
    assert result.claims[1].relation_to_previous == "independent"
    assert all(claim.claim_id for claim in result.claims)


def test_missing_fields_are_null_and_entities_are_not_hallucinated() -> None:
    claim = extract("Maybe.").claims[0]
    assert claim.subject is None
    assert claim.predicate is None
    assert claim.object is None
    assert claim.date_or_time is None
    assert claim.entities == []
    assert claim.needs_human_review


def test_unsupported_language_is_preserved_and_unavailable() -> None:
    result = extract("هذا نص خيالي للاختبار فقط.")
    assert result.status == "verification_unavailable"
    assert result.language == "unknown"
    assert result.claims[0].claim_text == "هذا نص خيالي للاختبار فقط."
    assert result.reason


def test_url_image_and_video_are_not_analysed() -> None:
    url = process_input(classify_input("https://fictional.example/post"))
    image = process_input(
        VerificationInput(input_type="image", metadata={"name": "x.png"})
    )
    video = process_input(
        VerificationInput(input_type="video", metadata={"name": "x.mp4"})
    )
    assert url.status == "not_implemented"
    assert image.status == "not_implemented"
    assert video.status == "not_implemented"


def test_deterministic_output() -> None:
    text = "A fictional event occurred in Delhi in 2024."
    assert extract(text).model_dump() == extract(text).model_dump()
