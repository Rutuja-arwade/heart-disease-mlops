"""API contract tests using FastAPI's in-process TestClient."""

from __future__ import annotations


def test_root_lists_endpoints(client):
    body = client.get("/").json()
    assert body["service"] == "heart-disease-risk-api"
    assert "/predict" in body["endpoints"]


def test_health_and_ready(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["model_version"] == "test-1"


def test_predict_returns_prediction_and_confidence(client, patient):
    response = client.post("/predict", json=patient)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1
    assert body["label"] == "heart_disease"
    assert 0.5 <= body["confidence"] <= 1
    assert body["confidence"] == body["probability_disease"]
    assert body["risk_level"] in {"low", "moderate", "high"}
    assert response.headers["X-Model-Version"] == "test-1"


def test_predict_low_risk_patient(client, healthy_patient):
    body = client.post("/predict", json=healthy_patient).json()
    assert body["prediction"] == 0
    assert body["confidence"] == round(1 - body["probability_disease"], 4)


def test_predict_accepts_float_codes_and_missing_optional(client, patient):
    patient.update({"sex": 1.0, "cp": "4"})
    patient.pop("ca")
    patient.pop("thal")
    assert client.post("/predict", json=patient).status_code == 200


def test_request_id_is_propagated(client, patient):
    response = client.post("/predict", json=patient, headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"
    assert response.json()["request_id"] == "abc123"


def test_invalid_payloads_are_rejected(client, patient):
    bad_code = {**patient, "cp": 7}
    out_of_range = {**patient, "age": 300}
    missing = {k: v for k, v in patient.items() if k != "chol"}
    extra = {**patient, "weight": 80}
    for payload in (bad_code, out_of_range, missing, extra):
        assert client.post("/predict", json=payload).status_code == 422


def test_batch_prediction(client, patient, healthy_patient):
    response = client.post("/predict/batch", json={"records": [patient, healthy_patient]})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert [p["prediction"] for p in body["predictions"]] == [1, 0]


def test_batch_rejects_empty(client):
    assert client.post("/predict/batch", json={"records": []}).status_code == 422


def test_metrics_endpoint_exposes_custom_metrics(client, patient):
    client.post("/predict", json=patient)
    client.post("/predict", json={**patient, "cp": 9})
    text = client.get("/metrics").text
    assert "heart_api_requests_total" in text
    assert "heart_api_predictions_total" in text
    assert "heart_api_prediction_probability_bucket" in text
    assert 'heart_api_validation_errors_total{endpoint="/predict"}' in text
    assert "heart_api_model_loaded 1.0" in text


def test_model_info(client):
    info = client.get("/model/info").json()
    assert info["model_family"] == "logistic_regression"


def test_service_unavailable_without_model(tmp_path, monkeypatch, patient):
    from fastapi.testclient import TestClient

    from src.api.app import app

    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(app) as test_client:
        assert test_client.get("/health").json()["model_loaded"] is False
        assert test_client.get("/ready").status_code == 503
        assert test_client.post("/predict", json=patient).status_code == 503
