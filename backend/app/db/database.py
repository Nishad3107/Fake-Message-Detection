"""SQLite repository layer, ready to be replaced by PostgreSQL later."""

import json
import sqlite3
from typing import Any

from app.core.config import Settings


def _connect(settings: Settings) -> sqlite3.Connection:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.database_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(settings: Settings) -> None:
    with _connect(settings) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                cleaned_text TEXT NOT NULL DEFAULT '',
                classification TEXT NOT NULL,
                confidence REAL NOT NULL,
                confidence_category TEXT NOT NULL DEFAULT 'moderate_confidence',
                class_probabilities TEXT NOT NULL DEFAULT '{}',
                detected_keywords TEXT NOT NULL,
                detected_features TEXT NOT NULL DEFAULT '',
                explanation TEXT NOT NULL,
                explanation_details TEXT NOT NULL DEFAULT '{}',
                influential_terms TEXT NOT NULL DEFAULT '',
                low_confidence INTEGER NOT NULL DEFAULT 0,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                warning TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )"""
        )
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(analyses)")
        }
        if "influential_terms" not in columns:
            connection.execute(
                "ALTER TABLE analyses ADD COLUMN influential_terms TEXT NOT NULL DEFAULT ''"
            )
        if "low_confidence" not in columns:
            connection.execute(
                "ALTER TABLE analyses ADD COLUMN low_confidence INTEGER NOT NULL DEFAULT 0"
            )
        migrations = {
            "cleaned_text": "TEXT NOT NULL DEFAULT ''",
            "confidence_category": "TEXT NOT NULL DEFAULT 'moderate_confidence'",
            "class_probabilities": "TEXT NOT NULL DEFAULT '{}'",
            "detected_features": "TEXT NOT NULL DEFAULT ''",
            "explanation_details": "TEXT NOT NULL DEFAULT '{}'",
            "warning": "TEXT NOT NULL DEFAULT ''",
        }
        for name, definition in migrations.items():
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE analyses ADD COLUMN {name} {definition}"
                )


def save_analysis(settings: Settings, result: dict[str, Any]) -> dict[str, Any]:
    with _connect(settings) as connection:
        cursor = connection.execute(
            """INSERT INTO analyses
            (text, cleaned_text, classification, confidence, confidence_category,
             class_probabilities, detected_keywords, detected_features, explanation,
             explanation_details, influential_terms, low_confidence, model_name,
             model_version, warning, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result["text"],
                result.get("cleaned_text", ""),
                result["classification"],
                result["confidence"],
                result.get("confidence_category", "moderate_confidence"),
                json.dumps(result.get("class_probabilities", {}), sort_keys=True),
                ",".join(result["detected_keywords"]),
                ",".join(result.get("detected_features", [])),
                result["explanation"],
                json.dumps(result.get("explanation_details", {}), sort_keys=True),
                ",".join(result.get("influential_terms", [])),
                int(result.get("low_confidence", False)),
                result["model_name"],
                result["model_version"],
                result.get("warning", ""),
                result["timestamp"].isoformat(),
            ),
        )
        return {"id": cursor.lastrowid, **result}


def list_analyses(settings: Settings, limit: int = 50) -> list[dict[str, Any]]:
    with _connect(settings) as connection:
        rows = connection.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    def normalize_label(value: str) -> str:
        return {"likely_true": "true", "misinformation": "false"}.get(value, value)

    return [
        {
            "id": row["id"],
            "text": row["text"],
            "cleaned_text": row["cleaned_text"],
            "prediction": normalize_label(row["classification"]),
            "classification": normalize_label(row["classification"]),
            "display_label": f"Model prediction: {normalize_label(row['classification'])}",
            "confidence": row["confidence"],
            "confidence_category": row["confidence_category"],
            "class_probabilities": json.loads(row["class_probabilities"] or "{}"),
            "detected_features": [x for x in row["detected_features"].split(",") if x],
            "detected_keywords": [x for x in row["detected_keywords"].split(",") if x],
            "explanation": row["explanation"],
            "explanation_details": json.loads(row["explanation_details"] or "{}"),
            "influential_terms": [x for x in row["influential_terms"].split(",") if x],
            "low_confidence": bool(row["low_confidence"]),
            "model_name": row["model_name"],
            "model_version": row["model_version"],
            "training_dataset_reference": "PHEME",
            "training_split_reference": "backend/data/splits/pheme_split_manifest.json",
            "model_limitations": "Historical analysis record; model output is not factual verification.",
            "timestamp": row["created_at"],
            "warning": row["warning"]
            or "This is a probabilistic NLP prediction, not factual verification.",
        }
        for row in rows
    ]


def history_summary(settings: Settings) -> dict[str, Any]:
    with _connect(settings) as connection:
        row = connection.execute(
            """SELECT COUNT(*) AS total,
                SUM(CASE WHEN classification IN ('true', 'likely_true') THEN 1 ELSE 0 END) AS true_count,
                SUM(CASE WHEN classification IN ('false', 'misinformation') THEN 1 ELSE 0 END) AS false_count,
                SUM(CASE WHEN classification = 'unverified' THEN 1 ELSE 0 END) AS unverified,
                SUM(CASE WHEN classification = 'likely_true' THEN 1 ELSE 0 END) AS likely_true,
                SUM(CASE WHEN classification = 'misinformation' THEN 1 ELSE 0 END) AS misinformation,
                SUM(CASE WHEN low_confidence = 1 THEN 1 ELSE 0 END) AS low_confidence,
                COALESCE(AVG(confidence), 0) AS average_confidence FROM analyses"""
        ).fetchone()
    return {
        "total_analyses": row["total"] or 0,
        "true": row["true_count"] or 0,
        "false": row["false_count"] or 0,
        "unverified": row["unverified"] or 0,
        "likely_true": row["likely_true"] or 0,
        "misinformation": row["misinformation"] or 0,
        "low_confidence": row["low_confidence"] or 0,
        "average_confidence": round(float(row["average_confidence"] or 0), 4),
    }
