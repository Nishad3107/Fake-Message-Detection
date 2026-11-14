import json
from pathlib import Path

from app.ml.validate_pheme import validate_pheme


def test_validate_pheme_reports_structure_and_labels(tmp_path: Path) -> None:
    thread = tmp_path / "event-a" / "rumours" / "123"
    (thread / "source-tweets").mkdir(parents=True)
    (thread / "reactions").mkdir()
    (thread / "source-tweets" / "123.json").write_text(
        json.dumps({"id": 123, "text": "A source post #news"}), encoding="utf-8"
    )
    (thread / "annotation.json").write_text(
        json.dumps({"true": "1", "misinformation": "0"}), encoding="utf-8"
    )
    (thread / "structure.json").write_text('{"123": []}', encoding="utf-8")

    report = validate_pheme(tmp_path)

    assert report["event_count"] == 1
    assert report["total_threads"] == 1
    assert report["valid_source_tweet_count"] == 1
    assert report["valid_annotation_count"] == 1
    assert report["label_counts"] == {"true": 1}
    assert report["unusable_record_count"] == 0
