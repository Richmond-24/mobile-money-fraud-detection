from fastapi.testclient import TestClient

from api.main import app

VALID_PAYLOAD = {
    "step": 1,
    "type": "CASH_OUT",
    "amount": 5000.0,
    "oldbalanceOrg": 10000.0,
    "newbalanceOrig": 7000.0,
    "oldbalanceDest": 2000.0,
    "newbalanceDest": 7000.0,
    "isFlaggedFraud": 0,
}


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert body["name"] == "Mobile Money Fraud Detection API"
        assert body["version"] == "1.0.0"
        assert body["status"] == "running"


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in {"healthy", "degraded"}
        assert isinstance(body["model_loaded"], bool)


def test_valid_prediction_request():
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        assert "prediction" in body
        assert "is_fraud" in body
        assert isinstance(body["is_fraud"], bool)
        assert body["prediction"] in {"Fraud", "Legitimate"}


def test_invalid_request_payload():
    with TestClient(app) as client:
        response = client.post("/predict", json={
            "step": 1,
            "type": "CASH_OUT",
            "amount": "not-a-number",
            "oldbalanceOrg": 10000.0,
            "newbalanceOrig": 7000.0,
            "oldbalanceDest": 2000.0,
            "newbalanceDest": 7000.0,
            "isFlaggedFraud": 0,
        })
        assert response.status_code == 422


def test_missing_required_field():
    with TestClient(app) as client:
        incomplete = dict(VALID_PAYLOAD)
        incomplete.pop("amount")
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422


def test_invalid_categorical_value():
    with TestClient(app) as client:
        payload = dict(VALID_PAYLOAD)
        payload["type"] = "CARD"
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
