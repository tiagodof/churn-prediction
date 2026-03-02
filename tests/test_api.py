"""
tests/test_api.py — Unit tests for the FastAPI inference endpoints.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


SAMPLE_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 6,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.50,
    "TotalCharges": 453.0,
}


@pytest.fixture
def client():
    """Create a test client with mocked model artefacts."""
    with patch("api.main.model") as mock_model, \
         patch("api.main.scaler") as mock_scaler, \
         patch("api.main.feature_names", list(range(26))):

        mock_model.predict_proba.return_value = np.array([[0.25, 0.75]])
        mock_scaler.transform.return_value = np.zeros((1, 26))

        from api.main import app
        yield TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_high_risk(client):
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_level" in data
    assert "top_risk_factors" in data
    assert data["churn_prediction"] is True
    assert data["risk_level"] == "High"


def test_predict_low_risk(client):
    low_risk = {**SAMPLE_CUSTOMER, "Contract": "Two year", "tenure": 48}
    with patch("api.main.model") as mock_model, \
         patch("api.main.scaler") as mock_scaler, \
         patch("api.main.feature_names", list(range(26))):

        mock_model.predict_proba.return_value = np.array([[0.92, 0.08]])
        mock_scaler.transform.return_value = np.zeros((1, 26))

        from api.main import app
        from fastapi.testclient import TestClient
        c = TestClient(app)
        response = c.post("/predict", json=low_risk)
        assert response.status_code == 200
        assert response.json()["risk_level"] == "Low"


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Churn Prediction API" in response.json()["message"]


def test_predict_missing_field(client):
    incomplete = {k: v for k, v in SAMPLE_CUSTOMER.items() if k != "tenure"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422
