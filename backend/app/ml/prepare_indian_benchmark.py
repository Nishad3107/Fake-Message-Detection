"""Prepare a local, deduplicated HinFakeNews content benchmark.

This command is intentionally separate from the PHEME pipeline. It records the
original numeric label, creates a documented normalized label from the supplied
catalog mapping, and keeps URL/domain values as metadata rather than model text.
It does not train a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from app.ml.prepare_indian_dataset import _read_sheets

BACKEND_ROOT = Path(__file__).resolve().parents[2]
SOURCE = BACKEND_ROOT / "data/raw/indian/hinfakenews/HinFakeNews-V1.xlsx"
OUTPUT = BACKEND_ROOT / "data/processed/indian/hinfakenews_content.csv"
REPORT = BACKEND_ROOT / "data/reports/indian/hinfakenews_preparation.json"
LEAKAGE_REPORT = BACKEND_ROOT / "data/reports/indian/hinfakenews_pre_split_leakage.json"

SHEET = "Real&Fake Combined"
TEXT_COLUMN = "CONTENT"
TITLE_COLUMN = "TITLE"
URL_COLUMN = "URL"
LABEL_COLUMN = "BOOL"
LABEL_MAPPING = {"0": "fake", "1": "real"}


def _domain(url: str) -> str:
    value = url.strip()
    if not value:
        return ""
    parsed = urlsplit(
        value if re.match(r"^[a-z]+://", value, re.IGNORECASE) else f"https://{value}"
    )
    host = parsed.netloc.lower().removeprefix("www.")
    if not host or any(character.isspace() for character in host) or "." not in host:
        return ""
    return host


def _normalized_content(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _load_records(source: Path) -> tuple[list[str], list[dict[str, str]]]:
    sheets = _read_sheets(source)
    if SHEET not in sheets:
        raise ValueError(
            f"Required sheet {SHEET!r} not found; available: {', '.join(sheets)}"
        )
    rows = sheets[SHEET]
    if not rows:
        raise ValueError(f"Sheet {SHEET!r} is empty")
    headers = [str(value).strip() for value in rows[0]]
    required = {TEXT_COLUMN, TITLE_COLUMN, URL_COLUMN, LABEL_COLUMN}
    missing = sorted(required - set(headers))
    if missing:
        raise ValueError(f"Required columns missing: {', '.join(missing)}")
    records = []
    for index, row in enumerate(rows[1:], start=2):
        record = {
            header: row[pos] if pos < len(row) else ""
            for pos, header in enumerate(headers)
        }
        record["source_row"] = str(index)
        records.append(record)
    return headers, records


def _group_labels(records: list[dict[str, str]], column: str) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        groups[record[column]].add(record[LABEL_COLUMN].strip())
    return groups


def prepare_dataset(
    source: Path = SOURCE,
    output: Path = OUTPUT,
    report_path: Path = REPORT,
    leakage_path: Path = LEAKAGE_REPORT,
) -> dict:
    _, raw_records = _load_records(source)
    empty_content = sum(not record[TEXT_COLUMN].strip() for record in raw_records)
    duplicate_rows = len(raw_records) - len(
        {
            tuple(
                record.get(key, "")
                for key in (URL_COLUMN, TITLE_COLUMN, TEXT_COLUMN, LABEL_COLUMN)
            )
            for record in raw_records
        }
    )
    content_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in raw_records:
        content_groups[_normalized_content(record[TEXT_COLUMN])].append(record)
    duplicate_content_rows = sum(
        len(group) - 1 for group in content_groups.values() if group
    )
    conflicting_groups = [
        group
        for group in content_groups.values()
        if len({item[LABEL_COLUMN].strip() for item in group}) > 1
    ]

    kept: list[dict[str, str]] = []
    seen_content: set[str] = set()
    removed_reasons = Counter()
    for record in raw_records:
        content = record[TEXT_COLUMN].strip()
        label_original = record[LABEL_COLUMN].strip()
        if not content:
            removed_reasons["empty_content"] += 1
            continue
        if label_original not in LABEL_MAPPING:
            removed_reasons["unsupported_label"] += 1
            continue
        fingerprint = _normalized_content(content)
        if fingerprint in seen_content:
            removed_reasons["duplicate_content"] += 1
            continue
        seen_content.add(fingerprint)
        kept.append(record)

    domains = Counter(_domain(record[URL_COLUMN]) for record in kept)
    label_counts = Counter(
        LABEL_MAPPING[record[LABEL_COLUMN].strip()] for record in kept
    )
    content_labels = _group_labels(kept, TEXT_COLUMN)
    title_labels = _group_labels(kept, TITLE_COLUMN)
    url_labels = _group_labels(kept, URL_COLUMN)
    label_tokens = re.compile(
        r"\b(fake|real|false|true|fact[- ]?check)\b|फर्जी|फेक|झूठ|नकली|सच",
        re.IGNORECASE,
    )
    domain_groups: dict[str, Counter[str]] = defaultdict(Counter)
    for record in kept:
        domain_groups[_domain(record[URL_COLUMN]) or "__missing_domain__"][
            LABEL_MAPPING[record[LABEL_COLUMN].strip()]
        ] += 1
    leakage_report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_name": "HinFakeNews",
        "input_text_column": TEXT_COLUMN,
        "same_content_multiple_labels": sum(
            len(labels) > 1 for labels in content_labels.values()
        ),
        "same_title_multiple_labels": sum(
            len(labels) > 1 for labels in title_labels.values()
        ),
        "same_url_multiple_labels": sum(
            len(labels) > 1 for labels in url_labels.values()
        ),
        "duplicate_content_groups": sum(
            count > 1
            for count in Counter(
                _normalized_content(record[TEXT_COLUMN]) for record in raw_records
            ).values()
        ),
        "duplicate_title_groups": sum(
            count > 1
            for count in Counter(
                record[TITLE_COLUMN] for record in raw_records
            ).values()
        ),
        "duplicate_url_groups": sum(
            count > 1
            for count in Counter(record[URL_COLUMN] for record in raw_records).values()
        ),
        "domain_count": len([domain for domain in domains if domain]),
        "missing_or_invalid_domain_rows": domains.get("", 0),
        "domain_label_distribution": {
            domain: dict(counts) for domain, counts in sorted(domain_groups.items())
        },
        "rows_with_label_related_tokens": sum(
            bool(label_tokens.search(record[TEXT_COLUMN])) for record in raw_records
        ),
        "rows_with_html_like_markup": sum(
            bool(re.search(r"<[^>]+>", record[TEXT_COLUMN])) for record in raw_records
        ),
        "near_duplicate_count": None,
        "near_duplicate_note": "Not calculated; use a documented similarity threshold before modeling.",
        "potential_signals": [
            "Fake-row justifications may encode the label in CONTENT.",
            "URL domains and fact-checker branding may act as source shortcuts.",
            "Invalid URL values indicate scraping artifacts and must not be treated as domains.",
            "Repeated titles/content require grouped splitting.",
        ],
        "training_performed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "record_id",
                "text_original",
                "label_original",
                "label",
                "title",
                "url",
                "domain",
                "source_row",
                "dataset_name",
            ],
        )
        writer.writeheader()
        for number, record in enumerate(kept, start=1):
            label_original = record[LABEL_COLUMN].strip()
            writer.writerow(
                {
                    "record_id": f"hinfakenews_{number:06d}",
                    "text_original": record[TEXT_COLUMN],
                    "label_original": label_original,
                    "label": LABEL_MAPPING[label_original],
                    "title": record[TITLE_COLUMN],
                    "url": record[URL_COLUMN],
                    "domain": _domain(record[URL_COLUMN]),
                    "source_row": record["source_row"],
                    "dataset_name": "HinFakeNews",
                }
            )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_name": "HinFakeNews",
        "source": str(source),
        "sheet": SHEET,
        "text_column": TEXT_COLUMN,
        "label_column": LABEL_COLUMN,
        "label_mapping": LABEL_MAPPING,
        "license_status": "unknown; catalog displays NA",
        "raw_row_count": len(raw_records),
        "empty_content_count": empty_content,
        "exact_duplicate_row_count": duplicate_rows,
        "duplicate_content_row_count": duplicate_content_rows,
        "conflicting_label_duplicate_group_count": len(conflicting_groups),
        "near_duplicate_count": None,
        "near_duplicate_note": "Not calculated in this preparation pass; exact normalized-content groups were removed.",
        "final_row_count": len(kept),
        "label_distribution": dict(sorted(label_counts.items())),
        "domain_distribution": dict(domains.most_common()),
        "invalid_or_missing_domain_count": domains.get("", 0),
        "date_columns": [],
        "removed_record_reasons": dict(removed_reasons),
        "url_is_metadata_only": True,
        "title_preserved_not_used_as_text": True,
        "fake_justification_warning": "CONTENT may include post-hoc justification in fake rows; separate it before training.",
        "training_performed": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    leakage_path.parent.mkdir(parents=True, exist_ok=True)
    leakage_path.write_text(
        json.dumps(leakage_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--leakage-report", type=Path, default=LEAKAGE_REPORT)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_dataset(args.source, args.output, args.report, args.leakage_report),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
