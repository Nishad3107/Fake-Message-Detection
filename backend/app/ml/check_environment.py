"""Report whether a fresh clone has the inputs needed for each project stage."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REQUIRED_DIRS = [
    Path("backend/data/raw/pheme"),
    Path("backend/data/processed"),
    Path("backend/data/reports"),
    Path("frontend"),
]
REQUIRED_FILES = [
    Path(".env.example"),
    Path("backend/requirements.txt"),
    Path("frontend/package.json"),
]


def check_environment() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
        "commands": {
            name: shutil.which(name) is not None
            for name in ("git", "make", "node", "npm", "docker")
        },
        "directories": {str(path): path.is_dir() for path in REQUIRED_DIRS},
        "files": {str(path): path.is_file() for path in REQUIRED_FILES},
        "dataset": {
            "raw_pheme": Path("backend/data/raw/pheme").is_dir(),
            "processed_pheme": Path(
                "backend/data/processed/pheme_veracity.csv"
            ).is_file(),
        },
        "artifact": Path("backend/models/final/final_pipeline.joblib").is_file(),
        "frontend_dependencies": Path("frontend/node_modules").is_dir(),
        "backend_dependencies": Path(".venv/bin/python").is_file(),
    }
    missing = [
        *[name for name, present in report["directories"].items() if not present],
        *[name for name, present in report["files"].items() if not present],
    ]
    report["missing_static_requirements"] = missing
    report["notes"] = [
        "The raw PHEME dataset and generated model artifacts are intentionally not committed.",
        "Missing dataset or artifact is expected in a fresh clone until the documented workflow is run.",
    ]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict", action="store_true", help="Fail when runtime inputs are missing"
    )
    args = parser.parse_args()
    report = check_environment()
    print(json.dumps(report, indent=2))
    if args.strict and (
        report["missing_static_requirements"] or not report["backend_dependencies"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
