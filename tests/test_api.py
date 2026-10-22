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


def test_predict_is_medically_consistent():
    """Un profil à haut risque doit avoir une probabilité de maladie plus élevée qu'un profil à faible risque."""
    high = {"age": 62, "sex": 1, "cp": 0, "trestbps": 140, "chol": 270, "fbs": 0, "restecg": 1,
            "thalach": 110, "exang": 1, "oldpeak": 3.0, "slope": 1, "ca": 3, "thal": 3}
    low = {"age": 40, "sex": 0, "cp": 2, "trestbps": 120, "chol": 200, "fbs": 0, "restecg": 0,
           "thalach": 175, "exang": 0, "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2}
    p_high = client.post("/predict", json=high).json()["probability_disease"]
    p_low = client.post("/predict", json=low).json()["probability_disease"]
    assert p_high > 0.8 and p_low < 0.2


def test_explain_sign_convention():
    """Une angine d'effort doit AUGMENTER le risque (contribution SHAP positive)."""
    with_angina = {**PATIENT, "exang": 1}
    contribs = {c["feature"]: c["value"] for c in client.post("/explain", json=with_angina).json()["contributions"]}
    assert contribs["exang"] > 0


def test_explain_exposes_variable_contributions_and_output_space():
    r = client.post("/explain", json=PATIENT)
    data = r.json()
    assert data["output_space"] in ("log-odds", "probability")
    variables = {c["feature"] for c in data["variable_contributions"]}
    assert len(variables) == 13 and {"cp", "thal", "ca", "oldpeak"} <= variables
    # additivité : la somme par variable = la somme par colonne encodée
    total_cols = sum(c["value"] for c in data["contributions"])
    total_vars = sum(c["value"] for c in data["variable_contributions"])
    assert abs(total_cols - total_vars) < 1e-3
    assert 0 < data["base_probability"] < 1
