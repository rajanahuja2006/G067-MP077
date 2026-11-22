"""API contract tests; they use inline data and do not require DVC CSV files."""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "flu_cases_reported": 12,
    "temp_c": 24.5,
    "nearby_clinics": 4,
    "discount_pct": 10.0,
    "stock_lines": 120,
    "distance_to_hospital_km": 2.5,
    "shop_type": "chain",
    "locality": "New Town",
}


def test_health_returns_required_status():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_accepts_exact_eight_features_and_returns_number():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], (int, float))


def test_prediction_matches_champion_export_reference():
    sample_payload = {
        "flu_cases_reported": 79,
        "temp_c": 42.8,
        "nearby_clinics": 9,
        "discount_pct": 13.4,
        "stock_lines": 1637,
        "distance_to_hospital_km": 9.2,
        "shop_type": "hospital_store",
        "locality": "Cantonment",
    }
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == pytest.approx(453.3904127071465, rel=1e-6)


def test_nearby_clinics_above_25_is_rejected_before_prediction():
    payload = {**VALID_PAYLOAD, "nearby_clinics": 26}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_negative_discount_is_rejected_before_prediction():
    payload = {**VALID_PAYLOAD, "discount_pct": -0.01}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_request_rejects_fields_outside_exact_contract():
    payload = {**VALID_PAYLOAD, "record_id": "PRIVATE-123"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_prediction_log_is_one_json_line_without_input_details(capfd):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    output = capfd.readouterr().out.strip().splitlines()
    assert len(output) == 1
    logged = json.loads(output[0])
    assert logged["event"] == "prediction_generated"
    assert logged["level"] == "INFO"
    assert isinstance(logged["prediction"], float)
    assert "locality" not in logged
    assert "shop_type" not in logged
