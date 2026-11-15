"""Run the complete PHEME experiment in a deterministic, staged order."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.ml.validate_experiment import (
    DATASET,
    MANIFEST,
    REPORT_DIR,
    SPLIT_DIR,
    validate_experiment,
)

RAW_DIR = Path("backend/data/raw/pheme")
OUTPUT = Path("backend/data/reports/experiment_summary.json")
STAGES = [
    (
        "validate dataset",
        "app.ml.validate_pheme",
        [
            "--raw-dir",
            str(RAW_DIR),
            "--report",
            "backend/data/reports/pheme_validation.json",
        ],
    ),
    (
        "prepare dataset",
        "app.ml.prepare_dataset",
        [
            "--raw-dir",
            str(RAW_DIR),
            "--output",
            str(DATASET),
            "--report",
            "backend/data/reports/pheme_quality.json",
        ],
    ),
    ("analyze dataset", "app.ml.analyze_dataset", []),
    ("create event-level splits", "app.ml.create_splits", []),
    ("prepare text splits", "app.ml.prepare_splits", []),
    ("train baseline models", "app.ml.train_baselines", []),
    ("tune models", "app.ml.tune_models", ["--manifest", str(MANIFEST)]),
    ("evaluate final model", "app.ml.evaluate_final_model", []),
    ("analyze errors", "app.ml.error_analysis", []),
]


def _versions() -> dict[str, str | None]:
    names = [
        "fastapi",
        "pydantic",
        "scikit-learn",
        "pandas",
        "numpy",
        "joblib",
        "pytest",
    ]
    return {name: _package_version(name) for name in names}


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _run_stage(label: str, module: str, args: list[str]) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    command = [sys.executable, "-m", module, *args]
    print(f"\n=== {label} ===")
    print("$", " ".join(command))
    subprocess.run(command, check=True, env={**os.environ, "PYTHONPATH": "backend"})
    ended_at = datetime.now(UTC)
    return {
        "stage": label,
        "module": module,
        "command": command,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_seconds": round(time.perf_counter() - started, 6),
        "status": "passed",
    }


def _summary(
    started_at: datetime,
    ended_at: datetime,
    stages: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "total_duration_seconds": round((ended_at - started_at).total_seconds(), 6),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": _versions(),
        "git_commit": _git_commit(),
        "dataset_path": str(DATASET),
        "stages": stages,
        "warnings": warnings
        + [
            "This command performs the one-time final evaluation stage; do not use its test results to retune the model.",
            "Raw PHEME data, processed data, reports, and model artifacts are generated or ignored files.",
        ],
    }
    if DATASET.is_file():
        import pandas as pd

        frame = pd.read_csv(DATASET)
        summary["dataset_records"] = len(frame)
        summary["dataset_labels"] = frame["label"].value_counts().to_dict()
    if SPLIT_DIR.is_dir():
        import pandas as pd

        summary["split_sizes"] = {
            name: len(pd.read_csv(SPLIT_DIR / f"{name}.csv"))
            for name in ("train", "validation", "test")
            if (SPLIT_DIR / f"{name}.csv").is_file()
        }
    for name, path in [
        ("baseline", REPORT_DIR / "baseline_results.json"),
        ("tuning", REPORT_DIR / "tuning_summary.json"),
        ("final", REPORT_DIR / "final_test_results.json"),
    ]:
        if path.is_file():
            summary[f"{name}_results"] = json.loads(path.read_text(encoding="utf-8"))
    summary["error_analysis_outputs"] = [
        str(path) for path in REPORT_DIR.glob("*analysis*.json")
    ]
    return summary


def run_experiment(output_path: Path = OUTPUT) -> dict[str, Any]:
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(
            f"PHEME raw dataset is required at {RAW_DIR}; synthetic data is not accepted"
        )
    stages: list[dict[str, Any]] = []
    warnings: list[str] = []
    started_at = datetime.now(UTC)
    try:
        for label, module, args in STAGES:
            stages.append(_run_stage(label, module, args))
    except subprocess.CalledProcessError as exc:
        ended_at = datetime.now(UTC)
        stages.append({"status": "failed", "returncode": exc.returncode})
        result = _summary(started_at, ended_at, stages, warnings)
        result["status"] = "failed"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
        )
        raise
    ended_at = datetime.now(UTC)
    result = _summary(started_at, ended_at, stages, warnings)
    validation = validate_experiment(require_artifact=True)
    result["validation"] = validation
    result["status"] = "passed" if validation["passed"] else "failed"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if not validation["passed"]:
        raise RuntimeError(
            f"Experiment output validation failed: {validation['failures']}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run_experiment(args.output)
    print(
        json.dumps(
            {"status": result["status"], "duration": result["total_duration_seconds"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
