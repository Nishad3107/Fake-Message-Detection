from __future__ import annotations

from pathlib import Path

import pytest
from app.ml.audit_indian_content import (
    _near_duplicate_report,
    _write_candidates,
    detect_flags,
    normalize_text,
)


def _row(
    record_id: str, text: str, label: str = "fake", domain: str = "example.com"
) -> dict[str, str]:
    return {
        "record_id": record_id,
        "text_original": text,
        "label": label,
        "label_original": "0" if label == "fake" else "1",
        "title": "Title",
        "url": f"https://{domain}/{record_id}",
        "domain": domain,
        "source_row": "2",
        "dataset_name": "HinFakeNews",
    }


def test_leakage_markers_are_detected() -> None:
    flags = detect_flags(
        _row("r1", "Fact Check दावा: यह fake और भ्रामक है. Verdict: false.")
    )
    assert flags["contains_fact_check_marker"]
    assert flags["contains_claim_marker"]
    assert flags["contains_debunking_language"]
    assert flags["contains_label_word"]
    assert not flags["contains_html"]


def test_variant_generation_preserves_records_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_row("r1", "  Article   text  "), _row("r2", "दूसरा लेख", "real")]
    monkeypatch.setattr("app.ml.audit_indian_content.CANDIDATE_DIR", tmp_path)
    comparison = _write_candidates(
        rows,
        {row["record_id"]: {"contains_label_word": False} for row in rows},
        {row["record_id"]: normalize_text(row["text_original"]) for row in rows},
    )
    assert comparison["content_raw"]["records"] == 2
    assert comparison["content_clean"]["changed_records_vs_original_content"] == 1


def test_near_duplicate_report_is_deterministic_and_preserves_labels() -> None:
    base = " ".join(f"article{i} news report important fact check" for i in range(20))
    first = _row("r1", base, "fake", "a.example")
    second = _row("r2", f"{base} अतिरिक्त", "fake", "b.example")
    third = _row("r3", "पूरी तरह अलग समाचार और अलग विषय", "real", "c.example")
    rows = [first, second, third]
    normalized = {
        row["record_id"]: normalize_text(row["text_original"]) for row in rows
    }
    report_a = _near_duplicate_report(
        rows, normalized, Path("/nonexistent/manifest.json")
    )
    report_b = _near_duplicate_report(
        rows, normalized, Path("/nonexistent/manifest.json")
    )
    report_a.pop("generated_at")
    report_b.pop("generated_at")
    assert report_a == report_b
    assert report_a["cross_label_near_duplicate_pair_count"] == 0
    assert report_a["cross_domain_near_duplicate_pair_count"] >= 1


def test_pheme_isolation() -> None:
    from app.ml.audit_indian_content import INPUT

    assert "indian" in INPUT.parts
    assert "pheme" not in INPUT.parts
