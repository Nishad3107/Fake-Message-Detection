import json

from app.prepare_verification_benchmarks import prepare_benchmark


def _record(record_id: str, claim: str, answer: str, label: str = "Supported") -> dict:
    return {
        "benchmark": "averitec",
        "record_id": record_id,
        "claim": claim,
        "label": label,
        "questions": [{"question": "Why?", "answers": [{"answer": answer}]}],
        "claim_date": None,
        "speaker": None,
        "reporting_source": None,
        "location": None,
        "claim_types": [],
        "fact_checking_strategies": [],
        "fact_checking_article": "justification",
        "original_claim_url": None,
        "cached_original_claim_url": None,
        "justification": "justification",
        "required_reannotation": False,
        "split": "train",
        "source_record": {},
    }


def test_averitec_preparation_preserves_train_and_dev(tmp_path, monkeypatch):
    raw = tmp_path / "averitec"
    processed = tmp_path / "processed"
    reports = tmp_path / "reports"
    raw.mkdir()
    (raw / "train.json").write_text(
        json.dumps(
            [
                {
                    "claim_id": "train-1",
                    "claim": "A fictional claim",
                    "label": "Supported",
                    "questions": [],
                    "fact_checking_strategies": [],
                    "claim_types": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    (raw / "dev.json").write_text(
        json.dumps(
            [
                {
                    "claim_id": "dev-1",
                    "claim": "Another fictional claim",
                    "label": "Refuted",
                    "questions": [],
                    "fact_checking_strategies": [],
                    "claim_types": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    import app.prepare_verification_benchmarks as module

    monkeypatch.setattr(module, "RAW_ROOT", raw.parent)
    monkeypatch.setattr(module, "PROCESSED_ROOT", processed)
    monkeypatch.setattr(module, "REPORT_ROOT", reports)
    result = prepare_benchmark("averitec")
    assert result["status"] == "complete"
    assert result["records_per_split"] == {"train": 1, "dev": 1}
    assert (
        json.loads((processed / "averitec" / "train.jsonl").read_text())["split"]
        == "train"
    )


def test_train_corpus_never_reads_development_split(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    averitec = processed / "averitec"
    averitec.mkdir(parents=True)
    (averitec / "train.jsonl").write_text(
        json.dumps(_record("train-1", "train claim", "train evidence")) + "\n",
        encoding="utf-8",
    )
    (averitec / "dev.jsonl").write_text(
        json.dumps(_record("dev-1", "dev claim", "dev evidence")) + "\n",
        encoding="utf-8",
    )
    import app.build_verification_corpus as module

    monkeypatch.setattr(module, "PROCESSED", averitec)
    monkeypatch.setattr(module, "CORPUS_ROOT", tmp_path / "corpora")
    monkeypatch.setattr(module, "REPORT_ROOT", tmp_path / "reports")
    report = module.build_averitec_train_corpus()
    assert report["status"] == "complete"
    corpus = (
        tmp_path / "corpora" / "averitec_train_evidence" / "evidence.jsonl"
    ).read_text()
    assert "train-1" in corpus
    assert "dev-1" not in corpus


def test_leakage_report_blocks_development_claim_overlap(tmp_path, monkeypatch):
    processed = tmp_path / "processed" / "averitec"
    corpus = tmp_path / "corpus"
    processed.mkdir(parents=True)
    corpus.mkdir()
    train = _record("train-1", "same claim", "same evidence")
    dev = _record("dev-1", "same claim", "different evidence", label="Refuted")
    (processed / "train.jsonl").write_text(json.dumps(train) + "\n", encoding="utf-8")
    (processed / "dev.jsonl").write_text(json.dumps(dev) + "\n", encoding="utf-8")
    (corpus / "evidence.jsonl").write_text(json.dumps({"text": "same evidence"}) + "\n")
    import app.verification_corpus_leakage as module

    monkeypatch.setattr(module, "TRAIN", processed / "train.jsonl")
    monkeypatch.setattr(module, "DEV", processed / "dev.jsonl")
    monkeypatch.setattr(module, "CORPUS", corpus / "evidence.jsonl")
    monkeypatch.setattr(module, "REPORT", tmp_path / "report.json")
    report = module.audit()
    assert report["status"] == "unsafe"
    assert report["checks"]["duplicate_claims_across_train_dev"] == 1


def test_readiness_blocks_research_without_independent_store(tmp_path, monkeypatch):
    import app.verification_readiness as module

    monkeypatch.setattr(module, "CORPUS", tmp_path / "missing-corpus.json")
    monkeypatch.setattr(module, "LEAKAGE", tmp_path / "missing-leakage.json")
    monkeypatch.setattr(module, "REPORT", tmp_path / "readiness.json")
    report = module.readiness()
    assert report["statuses"]["ready_for_research_metrics"] is False
    assert report["statuses"]["ready_for_live_retrieval"] is False
