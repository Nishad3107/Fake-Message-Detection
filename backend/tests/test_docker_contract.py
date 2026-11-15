from app.main import app
from fastapi.testclient import TestClient


def test_health_contract() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    body = response.json()
    assert response.status_code == 200
    assert {"status", "model_loaded", "model_version", "dataset"} <= body.keys()


def test_model_and_prediction_contract() -> None:
    with TestClient(app) as client:
        model_info = client.get("/api/v1/model-info")
        prediction = client.post(
            "/api/v1/predict",
            json={"text": "A public report describes the latest event and evidence."},
        )
    assert model_info.status_code == 200
    assert model_info.json()["ready"] is True
    body = prediction.json()
    assert prediction.status_code == 200
    assert {"prediction", "warning", "model_version", "explanation"} <= body.keys()
    assert set(body["class_probabilities"]) == {"false", "true", "unverified"}
    assert abs(sum(body["class_probabilities"].values()) - 1.0) < 1e-4
    assert body["model_version"] in {
        "pheme-tuned-multinomial-naive-bayes-v1",
        "ci-fixture-not-for-production",
    }


def test_history_contract_after_prediction() -> None:
    with TestClient(app) as client:
        client.post(
            "/api/v1/predict",
            json={"text": "Independent evidence is needed before sharing this report."},
        )
        response = client.get("/api/v1/history")
    assert response.status_code == 200
    assert {"items", "summary"} <= response.json().keys()
    assert response.json()["summary"]["total_analyses"] >= 1
