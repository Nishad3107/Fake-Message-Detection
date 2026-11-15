import json
from pathlib import Path

import pandas as pd
import pytest
from app.ml import check_environment, run_experiment, validate_experiment


def test_experiment_stages_have_academic_order() -> None:
    assert [label for label, _, _ in run_experiment.STAGES] == [
        "validate dataset",
        "prepare dataset",
        "analyze dataset",
        "create event-level splits",
        "prepare text splits",
        "train baseline models",
        "tune models",
        "evaluate final model",
        "analyze errors",
    ]


def test_runner_rejects_missing_pheme_dataset(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run_experiment, "RAW_DIR", tmp_path / "missing-pheme")
    with pytest.raises(FileNotFoundError, match="PHEME raw dataset"):
        run_experiment.run_experiment(tmp_path / "summary.json")


def test_environment_report_is_non_destructive() -> None:
    report = check_environment.check_environment()
    assert "commands" in report
    assert "dataset" in report
    assert "artifact" in report


def test_validator_detects_event_and_thread_overlap(
    monkeypatch, tmp_path: Path
) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    rows = []
    for split in ("train", "validation", "test"):
        rows.append(
            {
                "text_clean": f"{split} text",
                "label": "true",
                "event": "shared-event",
                "thread_id": "shared-thread",
                "source_tweet_id": f"{split}-tweet",
            }
        )
        pd.DataFrame([rows[-1]]).to_csv(split_dir / f"{split}.csv", index=False)
    dataset = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(dataset, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"random_seed": 42}), encoding="utf-8")
    monkeypatch.setattr(validate_experiment, "DATASET", dataset)
    monkeypatch.setattr(validate_experiment, "SPLIT_DIR", split_dir)
    monkeypatch.setattr(validate_experiment, "MANIFEST", manifest)
    monkeypatch.setattr(validate_experiment, "REQUIRED_REPORTS", [])
    result = validate_experiment.validate_experiment(
        tmp_path / "validation.json", require_artifact=False
    )
    assert result["passed"] is False
    assert result["checks"]["event_overlap"] == ["shared-event"]
    assert result["checks"]["thread_overlap"] == ["shared-thread"]
