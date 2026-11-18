"""Shared local helpers for preserving AVeriTeC question-answer evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized(value: str | None) -> str:
    return " ".join((value or "").casefold().split())


def answer_records(record: dict[str, Any]) -> list[dict[str, Any]]:
    answers: list[dict[str, Any]] = []
    questions = record.get("questions") or []
    if not questions and isinstance(record.get("evidence"), list):
        questions = record["evidence"]
    for question in questions:
        for answer in question.get("answers") or []:
            answers.append(
                {
                    "question": question.get("question"),
                    **answer,
                }
            )
    return answers


def evidence_texts(record: dict[str, Any]) -> list[str]:
    return [
        answer["answer"]
        for answer in answer_records(record)
        if isinstance(answer.get("answer"), str) and answer["answer"].strip()
    ]


def evidence_urls(record: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for answer in answer_records(record):
        for key in ("source_url", "cached_source_url"):
            value = answer.get(key)
            if isinstance(value, str) and value.strip():
                urls.append(value.strip())
    return urls


def content_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
