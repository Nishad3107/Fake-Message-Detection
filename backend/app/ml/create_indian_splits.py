"""Create deterministic domain-grouped splits for the prepared Indian benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

BACKEND_ROOT = Path(__file__).resolve().parents[2]
INPUT = BACKEND_ROOT / "data/processed/indian/hinfakenews_content.csv"
OUTPUT = BACKEND_ROOT / "data/splits/indian/hinfakenews_split_manifest.json"
SEED = 42
SPLITS = ("train", "validation", "test")


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared Indian dataset not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"record_id", "label", "domain", "text_original"}
    missing = sorted(required - set(rows[0]) if rows else required)
    if missing:
        raise ValueError(
            f"Prepared dataset missing required columns: {', '.join(missing)}"
        )
    return rows


def _distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row["label"] for row in rows).items()))


def _score(
    parts: list[list[dict[str, str]]], targets: list[float], overall: Counter[str]
) -> float:
    score = sum(
        abs(len(part) - target) / max(1, target) for part, target in zip(parts, targets)
    )
    expected = {
        label: count / sum(overall.values()) for label, count in overall.items()
    }
    for part in parts:
        if not part:
            return float("inf")
        counts = Counter(row["label"] for row in part)
        for label, fraction in expected.items():
            score += 2.0 * abs(counts[label] / len(part) - fraction)
    return score


def create_splits(
    input_path: Path = INPUT, output_path: Path = OUTPUT, seed: int = SEED
) -> dict:
    rows = _read_rows(input_path)
    domains = [row["domain"] or "__missing_domain__" for row in rows]
    unique_domains = sorted(set(domains))
    if len(unique_domains) < 3:
        raise ValueError(
            "At least three domain groups are required for train/validation/test splits"
        )
    overall = Counter(row["label"] for row in rows)
    best: tuple[float, tuple[list[int], list[int], list[int]]] | None = None
    targets = [len(rows) * 0.70, len(rows) * 0.15, len(rows) * 0.15]
    # GroupShuffleSplit supplies reproducible, domain-disjoint candidates. The
    # deterministic candidate search selects the closest label/size balance.
    for trial in range(200):
        first = next(
            GroupShuffleSplit(
                n_splits=1, test_size=0.30, random_state=seed + trial
            ).split(rows, groups=domains)
        )
        train_idx, remainder_idx = first
        remainder_rows = [rows[index] for index in remainder_idx]
        remainder_domains = [domains[index] for index in remainder_idx]
        second = next(
            GroupShuffleSplit(
                n_splits=1, test_size=0.50, random_state=seed + trial
            ).split(remainder_rows, groups=remainder_domains)
        )
        validation_local, test_local = second
        train_rows = [rows[index] for index in train_idx]
        validation_rows = [remainder_rows[index] for index in validation_local]
        test_rows = [remainder_rows[index] for index in test_local]
        candidate = (train_rows, validation_rows, test_rows)
        value = _score([*candidate], targets, overall)
        if best is None or value < best[0]:
            best = (value, tuple([list(range(len(part))) for part in candidate]))
            best_rows = candidate
    if best is None:
        raise RuntimeError("Unable to create grouped split candidates")
    parts = {name: part for name, part in zip(SPLITS, best_rows)}
    assignments = {
        row["record_id"]: name for name, part in parts.items() for row in part
    }
    if len(assignments) != len(rows):
        raise RuntimeError("Every prepared record must be assigned exactly once")
    event_overlap = []
    domain_sets = {
        name: {row["domain"] for row in part} for name, part in parts.items()
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            event_overlap.extend(sorted(domain_sets[left] & domain_sets[right]))
    if event_overlap:
        raise RuntimeError(f"Domain overlap detected: {event_overlap[:10]}")

    duplicate_sets = {
        name: {row["text_original"].strip().casefold() for row in part}
        for name, part in parts.items()
    }
    duplicate_overlap = []
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            duplicate_overlap.extend(
                sorted(duplicate_sets[left] & duplicate_sets[right])
            )
    if duplicate_overlap:
        raise RuntimeError("Exact text overlap detected between splits")

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": "HinFakeNews",
        "input_path": str(input_path),
        "split_method": "deterministic domain-grouped split; no reliable date column available",
        "group_column": "domain",
        "time_column": None,
        "random_seed": seed,
        "split_proportions_target": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "splits": {
            name: {
                "record_count": len(parts[name]),
                "label_distribution": _distribution(parts[name]),
                "domain_count": len(domain_sets[name]),
                "domains": sorted(domain_sets[name]),
            }
            for name in SPLITS
        },
        "all_record_count": len(rows),
        "assignment_count": len(assignments),
        "domain_overlap": sorted(set(event_overlap)),
        "exact_text_overlap": sorted(set(duplicate_overlap)),
        "near_duplicate_overlap": None,
        "warnings": [
            "No publication/date field was available; temporal generalization was not evaluated.",
            "Domain grouping reduces source memorization but does not remove semantic or near-duplicate leakage.",
            "Fake-row justifications in CONTENT must be separated before model training.",
            "License is unresolved; files remain local and ignored.",
        ],
        "training_performed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(
        json.dumps(
            create_splits(args.input, args.output, args.seed),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
