from app.main import app
from fastapi.testclient import TestClient


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_validates_and_returns_result() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            json={"text": "Breaking news share this shocking claim now"},
        )
    assert response.status_code == 200
    assert {
        "classification",
        "confidence",
        "detected_keywords",
        "explanation",
    }.issubset(response.json())


def test_predict_endpoint_rejects_short_input() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json={"text": "short"})
    assert response.status_code == 422


def test_predict_endpoint_rejects_empty_input() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json={"text": "   "})
    assert response.status_code == 422


def test_predict_endpoint_rejects_excessively_long_input() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json={"text": "x" * 5001})
    assert response.status_code == 422


def test_predict_endpoint_rejects_non_string_input() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json={"text": 12345})
    assert response.status_code == 422


def test_batch_endpoint_enforces_configured_limit() -> None:
    posts = [
        {"text": f"A sufficiently long test message number {index}."}
        for index in range(51)
    ]
    with TestClient(app) as client:
        response = client.post("/api/v1/predict/batch", json={"posts": posts})
    assert response.status_code == 422


def test_malformed_json_has_safe_validation_error() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            content='{"text":',
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 422
    assert "Traceback" not in response.text
    assert ("/" + "Users/") not in response.text


def test_request_size_limit_and_security_headers() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            json={"text": "A valid message that is not processed."},
            headers={"Content-Length": "1000001"},
        )
    assert response.status_code == 413
    assert response.json() == {"detail": "Request body is too large"}
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


def test_cors_allows_documented_frontend_and_rejects_unknown_origin() -> None:
    with TestClient(app) as client:
        allowed = client.get("/health", headers={"Origin": "http://localhost:5173"})
        unknown = client.get("/health", headers={"Origin": "https://unknown.example"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-origin" not in unknown.headers


def test_predict_endpoint_reports_unavailable_model() -> None:
    original_service = getattr(app.state, "prediction_service", None)

    class UnavailableService:
        def predict(self, text: str) -> dict[str, object]:
            raise RuntimeError("Prediction model is unavailable: test")

    try:
        with TestClient(app) as client:
            app.state.prediction_service = UnavailableService()
            response = client.post(
                "/api/v1/predict", json={"text": "This is a valid post for testing"}
            )
        assert response.status_code == 503
        assert "model is unavailable" in response.json()["detail"].lower()
        assert "test" not in response.json()["detail"]
        assert "/" not in response.json()["detail"]
    finally:
        app.state.prediction_service = original_service


def test_model_info_batch_history_and_metrics_endpoints() -> None:
    with TestClient(app) as client:
        model_info = client.get("/api/v1/model-info")
        batch = client.post(
            "/api/v1/predict/batch",
            json={
                "posts": [{"text": "A public health report cites laboratory results."}]
            },
        )
        history = client.get("/api/v1/history")
        metrics = client.get("/api/v1/metrics")
    assert model_info.status_code == 200
    assert model_info.json()["ready"] is True
    assert batch.status_code == 200
    assert len(batch.json()["results"]) == 1
    assert history.status_code == 200
    assert history.json()["summary"]["total_analyses"] >= 1
    assert metrics.status_code == 200
    assert "accuracy" in metrics.json()["metrics"]
