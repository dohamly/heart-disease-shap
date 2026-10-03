"""
main.py
-------
API FastAPI qui sert le modèle Heart Disease + SHAP.

Lancer en local :
    uvicorn api.main:app --reload

Documentation interactive auto-générée : http://127.0.0.1:8000/docs
"""

import os
import sys

import joblib
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import (  # noqa: E402
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    load_data,
    build_preprocessed_dataset,
    split_and_scale,
)

from api.schemas import (  # noqa: E402
    PatientInput,
    PredictionOutput,
    ExplanationOutput,
    FeatureContribution,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.joblib")

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "API académique de démonstration — prédiction du risque de maladie "
        "cardiaque avec explicabilité SHAP. Ne constitue pas un outil de "
        "diagnostic médical."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_bundle = None
_background = None  # échantillon du train set, utilisé comme référence SHAP


def get_bundle():
    """Charge le modèle une seule fois (lazy load, mis en cache)."""
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail="Modèle introuvable. Lance d'abord `python src/train.py`.",
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def get_background():
    """Reconstruit le X_train (même split que l'entraînement) pour servir
    de référence SHAP — sans ça, LinearExplainer prend le patient lui-même
    comme baseline et toutes les contributions tombent à 0."""
    global _background
    if _background is None:
        df = load_data()
        X, y = build_preprocessed_dataset(df)
        X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
        _background = X_train
    return _background


def build_feature_row(patient: PatientInput) -> pd.DataFrame:
    bundle = get_bundle()
    columns, scaler = bundle["columns"], bundle["scaler"]

    raw = pd.DataFrame([patient.model_dump()])
    encoded = pd.get_dummies(raw, columns=CATEGORICAL_COLS)
    for col in columns:
        if col not in encoded.columns:
            encoded[col] = 0
    encoded = encoded[columns].astype("float64")

    numeric_present = [c for c in NUMERIC_COLS if c in encoded.columns]
    encoded[numeric_present] = scaler.transform(encoded[numeric_present])
    return encoded


def run_prediction(patient: PatientInput) -> PredictionOutput:
    bundle = get_bundle()
    model, model_name = bundle["model"], bundle["model_name"]

    row = build_feature_row(patient)
    proba = model.predict_proba(row)[0]
    pred = int(model.predict(row)[0])

    return PredictionOutput(
        prediction=pred,
        prediction_label="Heart disease detected" if pred == 1 else "No heart disease detected",
        probability_disease=round(float(proba[1]), 4),
        probability_no_disease=round(float(proba[0]), 4),
        model_name=model_name,
    )


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Heart Disease Prediction API"}


@app.get("/health", tags=["Health"])
def health():
    bundle_loaded = os.path.exists(MODEL_PATH)
    return {"model_available": bundle_loaded}


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(patient: PatientInput):
    """Prédit le risque de maladie cardiaque pour un patient donné."""
    return run_prediction(patient)


@app.post("/explain", response_model=ExplanationOutput, tags=["Explainability"])
def explain(patient: PatientInput):
    """Prédiction + décomposition SHAP (contribution de chaque variable)."""
    bundle = get_bundle()
    model = bundle["model"]

    row = build_feature_row(patient)
    prediction = run_prediction(patient)

    if type(model).__name__ in ("RandomForestClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, get_background())

    shap_values = explainer(row)
    if len(shap_values.shape) == 3:
        values = shap_values.values[0, :, 1]
        base = float(shap_values.base_values[0, 1])
    else:
        values = shap_values.values[0]
        base = float(shap_values.base_values[0])

    contributions = [
        FeatureContribution(feature=f, value=round(float(v), 4))
        for f, v in sorted(zip(row.columns, values), key=lambda t: -abs(t[1]))
    ]

    return ExplanationOutput(prediction=prediction, contributions=contributions, base_value=round(base, 4))
