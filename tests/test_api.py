"""Tests unitaires pour l'API FastAPI (api/main.py)"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

PATIENT = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_available"] is True


def test_predict_returns_valid_probability():
    r = client.post("/predict", json=PATIENT)
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] in (0, 1)
    assert 0 <= data["probability_disease"] <= 1
    proba_sum = data["probability_disease"] + data["probability_no_disease"]
    assert abs(proba_sum - 1.0) < 1e-3


def test_predict_rejects_invalid_input():
    bad_patient = {**PATIENT, "sex": 7}  # hors de l'intervalle [0,1]
    r = client.post("/predict", json=bad_patient)
    assert r.status_code == 422


def test_explain_returns_contributions():
    r = client.post("/explain", json=PATIENT)
    assert r.status_code == 200
    data = r.json()
    assert len(data["contributions"]) == 23
    assert "prediction" in data
