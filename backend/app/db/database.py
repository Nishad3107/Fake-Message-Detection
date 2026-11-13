"""SQLite repository layer, ready to be replaced by PostgreSQL later."""

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
                classification TEXT NOT NULL,
                confidence REAL NOT NULL,
                detected_keywords TEXT NOT NULL,
                explanation TEXT NOT NULL,
                influential_terms TEXT NOT NULL DEFAULT '',
                low_confidence INTEGER NOT NULL DEFAULT 0,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
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


def save_analysis(settings: Settings, result: dict[str, Any]) -> dict[str, Any]:
    with _connect(settings) as connection:
        cursor = connection.execute(
            """INSERT INTO analyses
            (text, classification, confidence, detected_keywords, explanation, influential_terms, low_confidence, model_name, model_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result["text"],
                result["classification"],
                result["confidence"],
                ",".join(result["detected_keywords"]),
                result["explanation"],
                ",".join(result.get("influential_terms", [])),
                int(result.get("low_confidence", False)),
                result["model_name"],
                result["model_version"],
                result["timestamp"].isoformat(),
            ),
        )
        return {"id": cursor.lastrowid, **result}


def list_analyses(settings: Settings, limit: int = 50) -> list[dict[str, Any]]:
    with _connect(settings) as connection:
        rows = connection.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [
        {
            "id": row["id"],
            "text": row["text"],
            "classification": row["classification"],
            "confidence": row["confidence"],
            "detected_keywords": [x for x in row["detected_keywords"].split(",") if x],
            "explanation": row["explanation"],
            "influential_terms": [x for x in row["influential_terms"].split(",") if x],
            "low_confidence": bool(row["low_confidence"]),
            "model_name": row["model_name"],
            "model_version": row["model_version"],
            "timestamp": row["created_at"],
        }
        for row in rows
    ]


def history_summary(settings: Settings) -> dict[str, Any]:
    with _connect(settings) as connection:
        row = connection.execute(
            """SELECT COUNT(*) AS total,
                SUM(CASE WHEN classification = 'likely_true' THEN 1 ELSE 0 END) AS likely_true,
                SUM(CASE WHEN classification = 'misinformation' THEN 1 ELSE 0 END) AS misinformation,
                COALESCE(AVG(confidence), 0) AS average_confidence FROM analyses"""
        ).fetchone()
    return {
        "total_analyses": row["total"] or 0,
        "likely_true": row["likely_true"] or 0,
        "misinformation": row["misinformation"] or 0,
        "average_confidence": round(float(row["average_confidence"] or 0), 4),
    }
