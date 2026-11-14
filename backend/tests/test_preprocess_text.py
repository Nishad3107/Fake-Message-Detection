from app.ml.preprocess_text import PreprocessingOptions, inspect_texts, preprocess_text


def test_null_and_empty_input_are_safe() -> None:
    assert preprocess_text(None) == ""
    assert preprocess_text("") == ""
    assert preprocess_text("   ") == ""


def test_urls_mentions_and_hashtags_are_handled() -> None:
    result = preprocess_text("Visit https://example.com @Reporter #BreakingNews")
    assert result == "visit url user breakingnews"


def test_lowercase_unicode_whitespace_and_repeated_punctuation() -> None:
    result = preprocess_text("  ＣＡＦÉ!!!   News\tNow??  ")
    assert result == "café! news now?"


def test_negation_is_preserved_and_hashtag_option_is_configurable() -> None:
    options = PreprocessingOptions(keep_hashtags=False)
    result = preprocess_text("This is NOT #safe", options)
    assert result == "this is not"


def test_retweet_marker_and_html_entities_are_normalized() -> None:
    result = preprocess_text("RT @user: Truth &amp; evidence")
    assert result == "user: truth & evidence"


def test_preprocessing_is_deterministic_and_inspection_is_pre_cleaning() -> None:
    text = "Breaking!!! https://example.com #News"
    assert preprocess_text(text) == preprocess_text(text)
    report = inspect_texts([text, "   ", "short"])
    assert report["urls"] == 1
    assert report["hashtags"] == 1
    assert report["whitespace_only_messages"] == 1
    assert report["repeated_punctuation"] == 1
