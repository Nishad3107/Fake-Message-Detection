from app.ml.model_utils import clean_text, detect_keywords


def test_clean_text_normalizes_urls_and_punctuation() -> None:
    assert clean_text("Visit HTTPS://example.com!  New REPORT.") == "visit new report"


def test_detect_keywords_returns_known_terms() -> None:
    assert detect_keywords("Breaking warning: share this viral report") == [
        "breaking",
        "report",
        "share",
        "viral",
        "warning",
    ]


def test_clean_text_preserves_hashtag_signal() -> None:
    assert "hashtag_breaking" in clean_text("#Breaking news")
