import json


def _row(record_id: str, claim: str, evidence: str) -> dict:
    return {
        "record_id": record_id,
        "claim": claim,
        "label": "Supported",
        "questions": [
            {"question": "What happened?", "answers": [{"answer": evidence}]}
        ],
        "location": "IN",
        "claim_date": "2020-01-01",
    }


def test_internal_retrieval_benchmark_is_deterministic_and_labeled(
    tmp_path, monkeypatch
):
    import app.run_averitec_internal_retrieval as module

    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "internal_train.jsonl").write_text(
        json.dumps(_row("train-1", "A fictional train claim", "Train evidence")) + "\n",
        encoding="utf-8",
    )
    (clean / "internal_dev.jsonl").write_text(
        json.dumps(_row("dev-1", "A fictional dev claim", "Dev evidence")) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "CLEAN_ROOT", clean)
    monkeypatch.setattr(module, "REPORT_ROOT", tmp_path / "reports")
    monkeypatch.setattr(module, "CHART_ROOT", tmp_path / "charts")
    first = module.run_benchmark()
    second = module.run_benchmark()
    assert first["benchmark_status"] == "INTERNAL DEVELOPMENT RESULT"
    assert first["official_performance"] is False
    assert first["live_web_performance"] is False
    assert first["gold_evidence_coverage_rate"] == 0.0
    assert first["query_count"] == second["query_count"]
