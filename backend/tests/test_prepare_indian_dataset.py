from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from app.ml.prepare_indian_dataset import (
    DEFAULT_SOURCE,
    audit_workbook,
    prepare_dataset,
)


def _write_xlsx(path: Path) -> None:
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg = "http://schemas.openxmlformats.org/package/2006/relationships"
    sheet = """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheetData>
<row r="1"><c r="A1" t="inlineStr"><is><t>URL</t></is></c><c r="B1" t="inlineStr"><is><t>TITLE</t></is></c><c r="C1" t="inlineStr"><is><t>CONTENT</t></is></c><c r="D1" t="inlineStr"><is><t>BOOL</t></is></c></row>
<row r="2"><c r="A2" t="inlineStr"><is><t>https://example.test/a</t></is></c><c r="B2" t="inlineStr"><is><t>हिंदी खबर</t></is></c><c r="C2" t="inlineStr"><is><t>हिंदी content</t></is></c><c r="D2"><v>0</v></c></row>
<row r="3"><c r="A3" t="inlineStr"><is><t>https://example.test/b</t></is></c><c r="B3" t="inlineStr"><is><t>English title</t></is></c><c r="C3" t="inlineStr"><is><t>English content</t></is></c><c r="D3"><v>1</v></c></row>
</sheetData></worksheet>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{ns}" xmlns:r="{rel}"><sheets><sheet name="Real&amp;Fake Combined" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            f'<Relationships xmlns="{pkg}"><Relationship Id="rId1" Type="sheet" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        archive.writestr("xl/worksheets/sheet1.xml", sheet)


def test_workbook_loading_and_sheet_discovery(tmp_path: Path) -> None:
    workbook = tmp_path / "sample.xlsx"
    _write_xlsx(workbook)
    output = audit_workbook(
        workbook,
        workbook_report=tmp_path / "workbook.json",
        text_report=tmp_path / "text.json",
        leakage_report=tmp_path / "leakage.json",
        metadata_path=tmp_path / "metadata.json",
    )
    assert list(output["sheets"]) == ["Real&Fake Combined"]
    assert output["sheets"]["Real&Fake Combined"]["records"] == 2
    assert (tmp_path / "workbook.json").is_file()


def test_audit_reports_labels_language_and_duplicates(tmp_path: Path) -> None:
    workbook = tmp_path / "sample.xlsx"
    _write_xlsx(workbook)
    audit_workbook(
        workbook,
        workbook_report=tmp_path / "workbook.json",
        text_report=tmp_path / "text.json",
        leakage_report=tmp_path / "leakage.json",
        metadata_path=tmp_path / "metadata.json",
    )
    report = json.loads((tmp_path / "text.json").read_text(encoding="utf-8"))
    sheet = report["sheets"]["Real&Fake Combined"]
    assert sheet["TITLE"]["candidate_label_column"] == "BOOL"
    assert sheet["TITLE"]["label_values"] == {"0": 1, "1": 1}
    assert sheet["TITLE"]["devanagari_rows"] == 1
    assert sheet["CONTENT"]["latin_rows"] == 2


def test_missing_sheet_and_ambiguous_mapping_fail_clearly(tmp_path: Path) -> None:
    workbook = tmp_path / "sample.xlsx"
    _write_xlsx(workbook)
    with pytest.raises(ValueError, match="Sheet 'missing' not found"):
        prepare_dataset(workbook, "missing", "CONTENT", "BOOL", tmp_path / "out.csv")

    with pytest.raises(ValueError, match="Column '' not found"):
        prepare_dataset(
            workbook, "Real&Fake Combined", "", "BOOL", tmp_path / "out.csv"
        )


def test_explicit_preparation_preserves_original_labels_and_source(
    tmp_path: Path,
) -> None:
    workbook = tmp_path / "sample.xlsx"
    _write_xlsx(workbook)
    output = tmp_path / "processed.csv"
    prepare_dataset(workbook, "Real&Fake Combined", "CONTENT", "BOOL", output)
    content = output.read_text(encoding="utf-8")
    assert "text_original,label_original,label,source_url,sheet_name" in content
    assert "हिंदी content,0,0,https://example.test/a,Real&Fake Combined" in content
    assert "English content,1,1,https://example.test/b,Real&Fake Combined" in content


def test_indian_default_source_is_isolated_from_pheme() -> None:
    assert "indian" in DEFAULT_SOURCE.parts
    assert "pheme" not in DEFAULT_SOURCE.parts
