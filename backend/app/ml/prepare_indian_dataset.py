"""Audit and explicitly normalize the HinFakeNews XLSX workbook.

This module intentionally uses only the Python standard library so that a fresh
audit does not depend on a network-only Excel package. Preparation requires an
explicit sheet, text column, and label column; it never guesses a training
schema silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any
from xml.etree import ElementTree as ET

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = BACKEND_ROOT / "data/raw/indian/hinfakenews/HinFakeNews-V1.xlsx"
DEFAULT_SOURCE_METADATA = (
    BACKEND_ROOT / "data/raw/indian/hinfakenews/source_metadata.json"
)
DEFAULT_WORKBOOK_REPORT = (
    BACKEND_ROOT / "data/reports/indian/hinfakenews_workbook_audit.json"
)
DEFAULT_TEXT_REPORT = BACKEND_ROOT / "data/reports/indian/hinfakenews_text_audit.json"
DEFAULT_LEAKAGE_REPORT = (
    BACKEND_ROOT / "data/reports/indian/hinfakenews_leakage_audit.json"
)
DEFAULT_OUTPUT = BACKEND_ROOT / "data/processed/indian/hinfakenews.csv"

NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
Q = lambda namespace, name: f"{{{namespace}}}{name}"

TEXT_HINTS = ("text", "content", "article", "news", "title", "headline", "statement")
LABEL_HINTS = ("label", "class", "fake", "real", "target", "category", "bool")
URL_HINTS = ("url", "link", "source", "website", "web")


def _column_number(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference.upper())
    if not letters:
        raise ValueError(f"Invalid XLSX cell reference: {reference}")
    number = 0
    for character in letters.group(0):
        number = number * 26 + ord(character) - ord("A") + 1
    return number


def _text_from_xml(node: ET.Element) -> str:
    return "".join(node.itertext())


class WorkbookReader:
    """Small streaming reader for the XLSX structures needed by this audit."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.archive = zipfile.ZipFile(path)
        self.shared_strings = self._load_shared_strings()
        self.sheets = self._load_sheets()

    def _load_shared_strings(self) -> list[str]:
        try:
            stream = self.archive.open("xl/sharedStrings.xml")
        except KeyError:
            return []
        values: list[str] = []
        for _, element in ET.iterparse(stream, events=("end",)):
            if element.tag == Q(NS_MAIN, "si"):
                values.append(_text_from_xml(element))
                element.clear()
        stream.close()
        return values

    def _load_sheets(self) -> list[dict[str, str]]:
        workbook = ET.fromstring(self.archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(self.archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in relationships.findall(Q(NS_PKG_REL, "Relationship"))
        }
        result: list[dict[str, str]] = []
        for sheet in workbook.find(Q(NS_MAIN, "sheets")) or []:
            relation = sheet.attrib[f"{{{NS_REL}}}id"]
            target = targets[relation]
            if not target.startswith("/"):
                target = f"xl/{target.lstrip('./')}"
            result.append({"name": sheet.attrib["name"], "path": target})
        return result

    def rows(self, sheet: dict[str, str]) -> Iterable[list[str]]:
        stream = self.archive.open(sheet["path"])
        for _, row in ET.iterparse(stream, events=("end",)):
            if row.tag != Q(NS_MAIN, "row"):
                continue
            cells: dict[int, str] = {}
            for cell in row.findall(Q(NS_MAIN, "c")):
                index = _column_number(cell.attrib.get("r", "A1"))
                value = cell.find(Q(NS_MAIN, "v"))
                inline = cell.find(Q(NS_MAIN, "is"))
                if inline is not None:
                    text = _text_from_xml(inline)
                elif value is None or value.text is None:
                    text = ""
                elif cell.attrib.get("t") == "s":
                    text = self.shared_strings[int(value.text)]
                else:
                    text = value.text
                cells[index] = text
            width = max(cells, default=0)
            yield [cells.get(index, "") for index in range(1, width + 1)]
            row.clear()
        stream.close()

    def close(self) -> None:
        self.archive.close()


def _read_sheets(path: Path) -> dict[str, list[list[str]]]:
    reader = WorkbookReader(path)
    try:
        return {sheet["name"]: list(reader.rows(sheet)) for sheet in reader.sheets}
    finally:
        reader.close()


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _column_candidates(columns: list[str], hints: tuple[str, ...]) -> list[str]:
    return [
        column
        for column in columns
        if any(hint in _normalize_header(column) for hint in hints)
    ]


def _is_hindi(value: str) -> bool:
    return bool(re.search(r"[\u0900-\u097f]", value))


def _is_latin(value: str) -> bool:
    return bool(re.search(r"[A-Za-z]", value))


def _word_count(value: str) -> int:
    return len(re.findall(r"\S+", value))


def _infer_data_type(values: list[str]) -> str:
    non_empty = [value.strip() for value in values if value.strip()]
    if not non_empty:
        return "empty"
    if all(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value) for value in non_empty):
        return "numeric-looking-string-or-number"
    return "string"


def _stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {
            "minimum": 0,
            "maximum": 0,
            "mean": 0,
            "median": 0,
            "standard_deviation": 0,
        }
    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": round(mean(values), 4),
        "median": median(values),
        "standard_deviation": round(pstdev(values), 4),
    }


def _core_properties(path: Path) -> dict[str, str]:
    properties: dict[str, str] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            root = ET.fromstring(archive.read("docProps/core.xml"))
    except (KeyError, zipfile.BadZipFile):
        return properties
    for element in root:
        if element.text and element.text.strip():
            properties[element.tag.rsplit("}", 1)[-1]] = element.text.strip()
    return properties


def _metadata(
    path: Path, original_path: Path | None, sheets: list[str]
) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    source = original_path or path
    source_stat = source.stat() if source.exists() else stat
    return {
        "original_file_path": str(source),
        "copied_file_path": str(path),
        "file_size_bytes": stat.st_size,
        "file_extension": path.suffix.lower(),
        "sha256": digest,
        "source_sha256_matches_copy": source.exists()
        and hashlib.sha256(source.read_bytes()).hexdigest() == digest,
        "filesystem_created_at": datetime.fromtimestamp(
            source_stat.st_birthtime, UTC
        ).isoformat(),
        "filesystem_modified_at": datetime.fromtimestamp(
            source_stat.st_mtime, UTC
        ).isoformat(),
        "workbook_sheet_names": sheets,
        "workbook_properties": _core_properties(path),
        "audit_generated_at": datetime.now(UTC).isoformat(),
        "license_status": "Unknown; not stated in the workbook metadata",
    }


def audit_workbook(
    source: Path = DEFAULT_SOURCE,
    original_path: Path | None = None,
    metadata_path: Path = DEFAULT_SOURCE_METADATA,
    workbook_report: Path = DEFAULT_WORKBOOK_REPORT,
    text_report: Path = DEFAULT_TEXT_REPORT,
    leakage_report: Path = DEFAULT_LEAKAGE_REPORT,
) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(f"HinFakeNews workbook not found: {source}")
    sheets = _read_sheets(source)
    metadata = _metadata(source, original_path, list(sheets))
    workbook_details: dict[str, Any] = {}
    text_details: dict[str, Any] = {}
    leakage_details: dict[str, Any] = {}
    for sheet_name, rows in sheets.items():
        headers = [str(value).strip() for value in (rows[0] if rows else [])]
        records = [
            {
                headers[index] if index < len(headers) else f"column_{index + 1}": (
                    row[index] if index < len(row) else ""
                )
                for index in range(len(headers))
            }
            for row in rows[1:]
        ]
        columns = headers
        missing = {
            column: sum(not str(record.get(column, "")).strip() for record in records)
            for column in columns
        }
        candidates = {
            "text": _column_candidates(columns, TEXT_HINTS),
            "label": _column_candidates(columns, LABEL_HINTS),
            "url": _column_candidates(columns, URL_HINTS),
        }
        workbook_details[sheet_name] = {
            "rows_including_header": len(rows),
            "records": len(records),
            "columns": columns,
            "column_count": len(columns),
            "data_types": {
                column: _infer_data_type(
                    [str(record.get(column, "")) for record in records]
                )
                for column in columns
            },
            "missing_values": missing,
            "candidate_columns": candidates,
            "contains_labels": bool(candidates["label"]),
            "contains_article_text": bool(candidates["text"]),
            "contains_urls": bool(candidates["url"]),
            "contains_images_or_image_paths": any(
                "image" in _normalize_header(column) for column in columns
            ),
            "contains_metadata": any(
                token in _normalize_header(column)
                for column in columns
                for token in ("date", "source", "category", "author", "id")
            ),
            "example_values": {
                column: [record.get(column, "")[:160] for record in records[:3]]
                for column in columns
            },
        }
        for text_column in candidates["text"]:
            values = [str(record.get(text_column, "")) for record in records]
            lengths = [len(value) for value in values]
            words = [_word_count(value) for value in values]
            sheet_text_details = {
                "candidate_text_column": text_column,
                "records": len(records),
                "empty_values": sum(not value.strip() for value in values),
                "duplicate_text_values": len(values) - len(set(values)),
                "duplicate_text_groups": sum(
                    count > 1 for count in Counter(values).values()
                ),
                "exact_duplicate_rows": len(records)
                - len({tuple(row) for row in rows[1:]}),
                "character_statistics": _stats(lengths),
                "word_statistics": _stats(words),
                "devanagari_rows": sum(_is_hindi(value) for value in values),
                "latin_rows": sum(_is_latin(value) for value in values),
                "mixed_hindi_latin_rows": sum(
                    _is_hindi(value) and _is_latin(value) for value in values
                ),
                "urls": sum(
                    bool(re.search(r"https?://|www\\.", value, re.IGNORECASE))
                    for value in values
                ),
                "hashtags": sum("#" in value for value in values),
                "mentions": sum(
                    bool(re.search(r"(?<![\w@])@[A-Za-z0-9_]+", value))
                    for value in values
                ),
                "emojis": sum(
                    bool(re.search(r"[\U0001F1E6-\U0001FAFF\u2600-\u27BF]", value))
                    for value in values
                ),
                "html_rows": sum(
                    bool(re.search(r"<[^>]+>", value)) for value in values
                ),
                "boilerplate_candidates": [],
                "repeated_template_note": "Template detection is heuristic; inspect repeated prefixes and source text before modeling.",
            }
            label_columns = candidates["label"]
            if len(label_columns) == 1:
                label_column = label_columns[0]
                labels = [
                    str(record.get(label_column, "")).strip() for record in records
                ]
                sheet_text_details["candidate_label_column"] = label_column
                sheet_text_details["label_values"] = dict(Counter(labels))
                leakage_details.setdefault(
                    sheet_name, {"text_columns": {}, "label_column": label_column}
                )["text_columns"][text_column] = {
                    "signals": [
                        "Source website/domain may correlate with labels and must be grouped or audited.",
                        "URL paths and fact-checker branding may reveal the label.",
                        "Repeated templates and exact/near duplicates may cross partitions.",
                        "Sheet names, filenames, and metadata fields must not become features.",
                        "Label-related words in titles/content require manual review.",
                    ],
                    "checks": {
                        "same_text_with_multiple_labels": [
                            value
                            for value, counts in _group_label_counts(
                                values, labels
                            ).items()
                            if len(counts) > 1
                        ],
                        "label_values": dict(Counter(labels)),
                    },
                    "required_actions": [
                        "Audit duplicate and near-duplicate groups before splitting.",
                        "Compare source/domain distributions by label.",
                        "Use source/time/group-aware partitions rather than random rows.",
                        "Keep URLs and source metadata out of the first text-only baseline.",
                    ],
                }
            text_details.setdefault(sheet_name, {})[text_column] = sheet_text_details
    workbook_output = {"source_metadata": metadata, "sheets": workbook_details}
    text_output = {"source_metadata": metadata, "sheets": text_details}
    leakage_output = {"source_metadata": metadata, "sheets": leakage_details}
    for path, value in (
        (metadata_path, metadata),
        (workbook_report, workbook_output),
        (text_report, text_output),
        (leakage_report, leakage_output),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return workbook_output


def _group_label_counts(values: list[str], labels: list[str]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for value, label in zip(values, labels):
        result.setdefault(value, set()).add(label)
    return result


def prepare_dataset(
    source: Path,
    sheet_name: str,
    text_column: str,
    label_column: str,
    output: Path = DEFAULT_OUTPUT,
) -> None:
    sheets = _read_sheets(source)
    if sheet_name not in sheets:
        raise ValueError(
            f"Sheet {sheet_name!r} not found; available sheets: {', '.join(sheets)}"
        )
    rows = sheets[sheet_name]
    if not rows:
        raise ValueError(f"Sheet {sheet_name!r} is empty")
    headers = [str(value).strip() for value in rows[0]]
    for required in (text_column, label_column):
        if required not in headers:
            raise ValueError(
                f"Column {required!r} not found; available columns: {', '.join(headers)}"
            )
    records = []
    for row in rows[1:]:
        record = {
            headers[index]: row[index] if index < len(row) else ""
            for index in range(len(headers))
        }
        records.append(record)
    labels = [str(record[label_column]).strip() for record in records]
    if not all(labels):
        raise ValueError(
            "Label column contains empty values; approve a missing-label policy before preparation"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "text_original",
                "label_original",
                "label",
                "source_url",
                "sheet_name",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "text_original": record[text_column],
                    "label_original": record[label_column],
                    "label": record[label_column],
                    "source_url": record.get(
                        "URL", record.get("url", record.get("Link", ""))
                    ),
                    "sheet_name": sheet_name,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--original-path", type=Path)
    parser.add_argument(
        "--audit", action="store_true", help="Audit all sheets and write JSON reports"
    )
    parser.add_argument("--sheet")
    parser.add_argument("--text-column")
    parser.add_argument("--label-column")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.audit:
        report = audit_workbook(args.source, args.original_path)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if not all((args.sheet, args.text_column, args.label_column)):
        parser.error(
            "Preparation requires --sheet, --text-column, and --label-column; use --audit first"
        )
    prepare_dataset(
        args.source, args.sheet, args.text_column, args.label_column, args.output
    )
    print(
        json.dumps(
            {"output": str(args.output), "status": "prepared", "training": "not run"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
