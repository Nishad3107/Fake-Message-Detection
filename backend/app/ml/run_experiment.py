"""Run the complete PHEME experiment into an isolated, versioned directory."""

from __future__ import annotations

import argparse
import hashlib
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

RAW_DIR = Path("backend/data/raw/pheme")
FROZEN_PATHS = [
    Path("backend/models/final/final_pipeline.joblib"),
    Path("backend/data/reports/model_comparison/final_test_results.json"),
    Path("backend/data/reports/model_comparison/final_confusion_matrix.json"),
    Path("frontend/src/data/research-results.json"),
]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _versions() -> dict[str, str | None]:
    return {
        name: _package_version(name)
        for name in [
            "fastapi",
            "pydantic",
            "scikit-learn",
            "pandas",
            "numpy",
            "joblib",
            "pytest",
        ]
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _hash_paths(paths: list[Path]) -> dict[str, str | None]:
    return {
        str(path): (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        for path in paths
    }


def _run_stage(label: str, module: str, args: list[str]) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    command = [sys.executable, "-m", module, *args]
    print(f"\n=== {label} ===", flush=True)
    print("$", " ".join(command), flush=True)
    try:
        subprocess.run(command, check=True, env={**os.environ, "PYTHONPATH": "backend"})
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            json.dumps(
                {
                    "stage": label,
                    "status": "failed",
                    "returncode": exc.returncode,
                    "started_at": started_at.isoformat(),
                    "ended_at": datetime.now(UTC).isoformat(),
                    "duration_seconds": round(time.perf_counter() - started, 6),
                }
            )
        ) from exc
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


def _paths(output_dir: Path) -> dict[str, Path]:
    reports = output_dir / "reports"
    comparison = reports / "model_comparison"
    return {
        "dataset": output_dir / "processed" / "pheme_veracity.csv",
        "splits": output_dir / "processed" / "splits",
        "manifest": output_dir / "splits" / "pheme_split_manifest.json",
        "reports": reports,
        "comparison": comparison,
        "models": output_dir / "models",
    }


def _stage_plan(paths: dict[str, Path]) -> list[tuple[str, str, list[str]]]:
    report = paths["reports"]
    comparison = paths["comparison"]
    splits = paths["splits"]
    return [
        (
            "validate dataset",
            "app.ml.validate_pheme",
            [
                "--raw-dir",
                str(RAW_DIR),
                "--report",
                str(report / "pheme_validation.json"),
            ],
        ),
        (
            "prepare dataset",
            "app.ml.prepare_dataset",
            [
                "--raw-dir",
                str(RAW_DIR),
                "--output",
                str(paths["dataset"]),
                "--report",
                str(report / "pheme_quality.json"),
            ],
        ),
        (
            "analyze dataset",
            "app.ml.analyze_dataset",
            [
                "--dataset",
                str(paths["dataset"]),
                "--report",
                str(report / "eda_report.json"),
                "--chart-dir",
                str(report / "eda"),
            ],
        ),
        (
            "create event-level splits",
            "app.ml.create_splits",
            [
                "--dataset",
                str(paths["dataset"]),
                "--manifest",
                str(paths["manifest"]),
                "--grouping-report",
                str(report / "grouping_report.json"),
            ],
        ),
        (
            "prepare text splits",
            "app.ml.prepare_splits",
            [
                "--dataset",
                str(paths["dataset"]),
                "--manifest",
                str(paths["manifest"]),
                "--output-dir",
                str(splits),
                "--report",
                str(report / "preprocessing_report.json"),
            ],
        ),
        (
            "train baseline models",
            "app.ml.train_baselines",
            [
                "--data-dir",
                str(splits),
                "--manifest",
                str(paths["manifest"]),
                "--report-dir",
                str(comparison),
                "--model-dir",
                str(paths["models"] / "baselines"),
            ],
        ),
        (
            "tune models",
            "app.ml.tune_models",
            [
                "--train",
                str(splits / "train.csv"),
                "--validation",
                str(splits / "validation.csv"),
                "--manifest",
                str(paths["manifest"]),
                "--report-dir",
                str(comparison),
                "--model-dir",
                str(paths["models"] / "tuned"),
            ],
        ),
        (
            "evaluate final model",
            "app.ml.evaluate_final_model",
            [
                "--config",
                str(comparison / "final_model_config.json"),
                "--artifact",
                str(paths["models"] / "final" / "final_pipeline.joblib"),
                "--report-dir",
                str(comparison),
                "--model-dir",
                str(paths["models"] / "final"),
                "--train",
                str(splits / "train.csv"),
                "--validation",
                str(splits / "validation.csv"),
                "--test",
                str(splits / "test.csv"),
                "--baseline-report",
                str(comparison / "baseline_results.json"),
                "--tuning-report",
                str(comparison / "tuning_results.json"),
            ],
        ),
        (
            "analyze errors",
            "app.ml.error_analysis",
            [
                "--test",
                str(splits / "test.csv"),
                "--predictions",
                str(comparison / "final_predictions.csv"),
                "--artifact",
                str(paths["models"] / "final" / "final_pipeline.joblib"),
                "--report-dir",
                str(comparison),
                "--chart-dir",
                str(comparison / "error_analysis"),
            ],
        ),
    ]


def _write_comparison(output_dir: Path, paths: dict[str, Path]) -> dict[str, Any]:
    import pandas as pd

    comparison = paths["comparison"]
    baseline = json.loads((comparison / "baseline_results.json").read_text())
    tuning = json.loads((comparison / "tuning_summary.json").read_text())
    final = json.loads((comparison / "final_test_results.json").read_text())
    frozen = json.loads(
        Path(
            "backend/data/reports/model_comparison/final_test_results.json"
        ).read_text()
    )
    result = {
        "dataset_records_match": len(pd.read_csv(paths["dataset"])) == 2271,
        "split_sizes": {
            name: len(pd.read_csv(paths["splits"] / f"{name}.csv"))
            for name in ("train", "validation", "test")
        },
        "baseline_results": baseline,
        "tuning_summary": tuning,
        "reproduced_final_metrics": final.get("metrics", {}),
        "frozen_final_metrics": frozen.get("metrics", {}),
        "reproduced_event_wise": final.get("event_wise", {}),
        "differences": {},
    }
    for key, value in result["reproduced_final_metrics"].items():
        frozen_value = result["frozen_final_metrics"].get(key)
        if (
            isinstance(value, (int, float))
            and isinstance(frozen_value, (int, float))
            and value != frozen_value
        ):
            result["differences"][key] = {"reproduced": value, "frozen": frozen_value}
    path = output_dir / "reproduction_comparison.json"
    path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    return result


def run_experiment(output_dir: Path) -> dict[str, Any]:
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(f"PHEME raw dataset is required at {RAW_DIR}")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to reuse existing experiment directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    paths = _paths(output_dir)
    before_hashes = _hash_paths(FROZEN_PATHS)
    started_at = datetime.now(UTC)
    stages: list[dict[str, Any]] = []
    try:
        for label, module, args in _stage_plan(paths):
            stages.append(_run_stage(label, module, args))
        import pandas as pd

        dataset = pd.read_csv(paths["dataset"])
        split_sizes = {
            name: len(pd.read_csv(paths["splits"] / f"{name}.csv"))
            for name in ("train", "validation", "test")
        }
        after_hashes = _hash_paths(FROZEN_PATHS)
        frozen = {
            path: {
                "before": before_hashes[path],
                "after": after_hashes[path],
                "unchanged": before_hashes[path] == after_hashes[path],
            }
            for path in before_hashes
        }
        comparison = _write_comparison(output_dir, paths)
        result = {
            "status": "passed",
            "run_id": output_dir.name,
            "output_dir": str(output_dir),
            "started_at": started_at.isoformat(),
            "ended_at": datetime.now(UTC).isoformat(),
            "total_duration_seconds": round(
                (datetime.now(UTC) - started_at).total_seconds(), 6
            ),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "package_versions": _versions(),
            "git_commit": _git_commit(),
            "dataset_path": str(paths["dataset"]),
            "dataset_records": len(dataset),
            "dataset_labels": dataset["label"].value_counts().to_dict(),
            "split_sizes": split_sizes,
            "stages": stages,
            "final_test_results": json.loads(
                (paths["comparison"] / "final_test_results.json").read_text()
            ).get("metrics", {}),
            "error_analysis_output_dir": str(paths["comparison"] / "error_analysis"),
            "comparison_report": str(output_dir / "reproduction_comparison.json"),
            "comparison": comparison,
            "frozen_artifacts": frozen,
            "warnings": [
                "This isolated test evaluation is for reproducibility only; do not retune from it."
            ],
        }
        (output_dir / "experiment_summary.json").write_text(
            json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return result
    except Exception:
        failure = {
            "status": "failed",
            "run_id": output_dir.name,
            "output_dir": str(output_dir),
            "started_at": started_at.isoformat(),
            "ended_at": datetime.now(UTC).isoformat(),
            "stages_completed": stages,
            "frozen_artifacts_after_failure": _hash_paths(FROZEN_PATHS),
        }
        (output_dir / "experiment_summary.json").write_text(
            json.dumps(failure, indent=2, default=str) + "\n", encoding="utf-8"
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, help="Unique isolated output directory"
    )
    args = parser.parse_args()
    output_dir = (
        args.output_dir
        or Path("backend/data/experiments")
        / f"run_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    result = run_experiment(output_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_dir": result["output_dir"],
                "duration": result["total_duration_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
