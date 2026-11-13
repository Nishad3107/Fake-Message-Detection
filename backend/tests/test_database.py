from datetime import UTC, datetime
from pathlib import Path

from app.core.config import Settings
from app.db.database import history_summary, init_db, list_analyses, save_analysis


def test_sqlite_history_repository(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "history.db")
    init_db(settings)
    result = {
        "text": "A useful example post for history.",
        "classification": "likely_true",
        "confidence": 0.8,
        "detected_keywords": [],
        "explanation": "A test explanation.",
        "model_name": "test",
        "model_version": "test",
        "timestamp": datetime.now(UTC),
    }
    saved = save_analysis(settings, result)
    items = list_analyses(settings)
    summary = history_summary(settings)
    assert saved["id"]
    assert items[0]["text"] == result["text"]
    assert summary["total_analyses"] == 1
