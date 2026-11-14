import json
from pathlib import Path

import pandas as pd
import pytest
from app.ml.analyze_dataset import (
    analyze_dataset,
    detect_dataset_source,
    load_dataset,
    run_eda,
    select_dataset,
)


def test_load_dataset_requires_text_and_label(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"message": ["hello"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="required columns"):
        load_dataset(path)


def test_analysis_reports_missing_duplicates_labels_and_lengths() -> None:
    frame = pd.DataFrame(
        {
            "text": ["Breaking news now", "Breaking news now", None, "Short"],
            "label": ["likely_true", "misinformation", "likely_true", "likely_true"],
        }
    )
    report = analyze_dataset(frame, "synthetic_sample", Path("sample.csv"))
    assert report["records"] == 4
    assert report["missing_values"]["text"] == 1
    assert report["duplicate_analysis"]["duplicate_text_groups"] == 1
    assert report["duplicate_analysis"]["same_text_with_different_labels"]
    assert report["label_distribution"]["likely_true"] == 3
    assert report["text_length"]["characters"]["minimum"] == 0


def test_source_identification_and_fallback(tmp_path: Path) -> None:
    sample = tmp_path / "sample_dataset.csv"
    sample.write_text("text,label\nA,likely_true\n", encoding="utf-8")
    selected, source = select_dataset(tmp_path / "missing.csv", sample)
    assert selected == sample
    assert source == "synthetic_sample"
    assert (
        detect_dataset_source(Path("backend/data/processed/pheme_veracity.csv"))
        == "real_pheme"
    )


def test_run_eda_writes_report_and_charts(tmp_path: Path) -> None:
    sample = tmp_path / "sample_dataset.csv"
    sample.write_text(
        "text,label\n"
        "A true update #news,likely_true\n"
        "Click https://example.com now,misinformation\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    report = run_eda(sample, report_path, tmp_path / "charts")
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["dataset"]["source"] == "synthetic_sample"
    assert saved["records"] == 2
    assert len(saved["charts"]) == 7
