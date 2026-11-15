"""Validate generated inputs, artifacts, reports, and research-result consistency."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

DATASET = Path("backend/data/processed/pheme_veracity.csv")
SPLIT_DIR = Path("backend/data/processed/splits")
MANIFEST = Path("backend/data/splits/pheme_split_manifest.json")
ARTIFACT = Path("backend/models/final/final_pipeline.joblib")
REPORT_DIR = Path("backend/data/reports/model_comparison")
FRONTEND_RESULTS = Path("frontend/src/data/research-results.json")
OUTPUT = Path("backend/data/reports/experiment_validation.json")
LABELS = {"false", "true", "unverified"}
REQUIRED_REPORTS = [
    Path("backend/data/reports/pheme_validation.json"),
    Path("backend/data/reports/pheme_quality.json"),
    Path("backend/data/reports/eda_report.json"),
    Path("backend/data/reports/grouping_report.json"),
    Path("backend/data/reports/preprocessing_report.json"),
    REPORT_DIR / "baseline_results.json",
    REPORT_DIR / "tuning_results.json",
    REPORT_DIR / "tuning_summary.json",
    REPORT_DIR / "final_model_config.json",
    REPORT_DIR / "final_test_results.json",
    REPORT_DIR / "final_predictions.csv",
    REPORT_DIR / "error_analysis_validation.json",
    REPORT_DIR / "confusion_pair_analysis.json",
    REPORT_DIR / "event_error_analysis.json",
    REPORT_DIR / "confidence_analysis.json",
    REPORT_DIR / "feature_explanations.json",
    REPORT_DIR / "representative_errors.json",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def validate_experiment(
    output_path: Path = OUTPUT,
    require_artifact: bool = True,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    failures: list[str] = []

    checks["dataset_exists"] = DATASET.is_file()
    if not checks["dataset_exists"]:
        failures.append(f"Missing processed dataset: {DATASET}")
    checks["manifest_exists"] = MANIFEST.is_file()
    if not checks["manifest_exists"]:
        failures.append(f"Missing split manifest: {MANIFEST}")

    split_frames: dict[str, pd.DataFrame] = {}
    for name in ("train", "validation", "test"):
        path = SPLIT_DIR / f"{name}.csv"
        exists = path.is_file()
        checks[f"{name}_split_exists"] = exists
        if not exists:
            failures.append(f"Missing {name} split: {path}")
            continue
        frame = pd.read_csv(path)
        split_frames[name] = frame
        required = {"label", "event", "thread_id", "source_tweet_id", "text_clean"}
        missing = required - set(frame.columns)
        if missing:
            failures.append(f"{path} missing columns: {sorted(missing)}")

    if split_frames:
        event_sets = {
            name: set(frame["event"].dropna()) for name, frame in split_frames.items()
        }
        thread_sets = {
            name: set(frame["thread_id"].dropna())
            for name, frame in split_frames.items()
        }
        event_overlap = (
            sorted(
                event_sets["train"] & event_sets["validation"]
                | event_sets["train"] & event_sets["test"]
                | event_sets["validation"] & event_sets["test"]
            )
            if len(event_sets) == 3
            else []
        )
        thread_overlap = (
            sorted(
                thread_sets["train"] & thread_sets["validation"]
                | thread_sets["train"] & thread_sets["test"]
                | thread_sets["validation"] & thread_sets["test"]
            )
            if len(thread_sets) == 3
            else []
        )
        checks["event_overlap"] = event_overlap
        checks["thread_overlap"] = thread_overlap
        if event_overlap:
            failures.append(f"Event overlap detected: {event_overlap}")
        if thread_overlap:
            failures.append(f"Thread overlap detected: {thread_overlap}")
        labels = set().union(
            *(set(frame["label"].dropna()) for frame in split_frames.values())
        )
        checks["labels"] = sorted(labels)
        if labels != LABELS:
            failures.append(f"Unexpected labels: {sorted(labels)}")
        checks["split_sizes"] = {
            name: len(frame) for name, frame in split_frames.items()
        }
        checks["split_label_counts"] = {
            name: frame["label"].value_counts().to_dict()
            for name, frame in split_frames.items()
        }

    checks["final_artifact_exists"] = ARTIFACT.is_file()
    if require_artifact and not ARTIFACT.is_file():
        failures.append(f"Missing final model artifact: {ARTIFACT}")
    report_status = {}
    for path in REQUIRED_REPORTS:
        report_status[str(path)] = path.is_file()
        if not path.is_file():
            failures.append(f"Missing generated output: {path}")
    checks["required_outputs"] = report_status

    if (REPORT_DIR / "final_test_results.json").is_file():
        final = _read_json(REPORT_DIR / "final_test_results.json")
        checks["final_test_records"] = final.get("test_data", {}).get("records")
        checks["final_metrics"] = final.get("metrics", {})
        config = (
            _read_json(REPORT_DIR / "final_model_config.json")
            if (REPORT_DIR / "final_model_config.json").is_file()
            else {}
        )
        artifact_reference = final.get("artifact")
        checks["artifact_reference_matches"] = artifact_reference in (
            None,
            str(ARTIFACT),
        )
        if not checks["artifact_reference_matches"]:
            failures.append(
                "Final report artifact path does not match the backend artifact path"
            )
        checks["frozen_test_selection"] = config.get("test_selection_used") is False
        if not checks["frozen_test_selection"]:
            failures.append(
                "Final configuration does not confirm test metrics were excluded from selection"
            )

        if FRONTEND_RESULTS.is_file():
            research = _read_json(FRONTEND_RESULTS)
            frontend_macro_f1 = research.get("final_test", {}).get("macro_f1")
            generated_macro_f1 = final.get("metrics", {}).get("macro_f1")
            matches = frontend_macro_f1 == generated_macro_f1
            checks["frontend_research_metrics_match"] = matches
            if not matches:
                failures.append(
                    "Frontend research-results.json is stale versus final_test_results.json"
                )
        else:
            checks["frontend_research_metrics_match"] = False
            failures.append(f"Missing frontend research data: {FRONTEND_RESULTS}")

    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "warnings": [
            "Generated datasets, reports, and model artifacts are ignored by Git and must be recreated in a fresh clone.",
            "The frozen test evaluation must not be rerun with changed configuration for model selection.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-missing-artifact", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = validate_experiment(
        args.output, require_artifact=not args.allow_missing_artifact
    )
    print(json.dumps(result, indent=2, default=str))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
