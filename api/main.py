"""
main.py
-------
FastAPI service exposing the Heart Disease + SHAP model.

Run locally:
    uvicorn api.main:app --reload

Interactive documentation (Swagger UI): http://127.0.0.1:8000/docs
"""

import os
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from inference import (  # noqa: E402
    MODEL_PATH, encode_patients, load_bundle, make_explainer, training_background, unpack_explanation,
)
from shap_utils import aggregate_by_variable, base_probability, output_space  # noqa: E402
from api.schemas import (  # noqa: E402
    PatientInput,
    PredictionOutput,
    ExplanationOutput,
    FeatureContribution,
)

DESCRIPTION = """
REST API serving the **Heart Disease + SHAP** model (logistic regression, UCI Cleveland).

| Endpoint | Purpose |
|---|---|
| `GET /health` | is the service running and the model available? |
| `POST /predict` | predicted class and probabilities for one patient |
| `POST /explain` | prediction + SHAP explanation (contribution of each variable) |

**Disclaimer.** For educational and research purposes only: this is not a medical diagnostic tool.
SHAP explains the behaviour of the model, not a medical cause-and-effect relationship.
"""

TAGS = [
    {"name": "Health", "description": "Check that the service and the model are available."},
    {"name": "Prediction", "description": "Predict the class and the probability of disease for one patient."},
    {"name": "Explainability", "description": "Explain a prediction with SHAP (additive values, per variable)."},
]

app = FastAPI(title="Heart Disease Prediction API", description=DESCRIPTION, version="1.0.0", openapi_tags=TAGS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_bundle = None
_background = None  # scaled training split, used as the SHAP reference population
_explainer = None


def get_bundle():
    """Load the deployed model once (lazy, then cached)."""
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run `python src/train.py` first.",
            )
        _bundle = load_bundle(MODEL_PATH)
    return _bundle


def get_background():
    """Return the SHAP reference population (cached)."""
    global _background
    if _background is None:
        _background = training_background()
    return _background


def get_explainer():
    """Return the SHAP explainer of the deployed model, built once instead of once per request."""
    global _explainer
    if _explainer is None:
        _explainer = make_explainer(get_bundle()["model"], get_background())
    return _explainer


def build_feature_row(patient: PatientInput) -> pd.DataFrame:
    """Encode one validated patient exactly like the training data (one-hot, column order, scaling)."""
    return encode_patients(patient.model_dump(), get_bundle())


def run_prediction(patient: PatientInput) -> PredictionOutput:
    """Predict the class and both probabilities for one validated patient."""
    bundle = get_bundle()
    model, model_name = bundle["model"], bundle["model_name"]

    row = build_feature_row(patient)
    proba = model.predict_proba(row)[0]
    pred = int(model.predict(row)[0])

    return PredictionOutput(
        prediction=pred,
        prediction_label="Heart disease likely (model estimate)" if pred == 1 else "Heart disease unlikely (model estimate)",
        probability_disease=round(float(proba[1]), 4),
        probability_no_disease=round(float(proba[0]), 4),
        model_name=model_name,
    )


@app.get("/", tags=["Health"], summary="API home page")
def root():
    """Return a minimal status message, handy to check that the server answers."""
    return {"status": "ok", "service": "Heart Disease Prediction API"}


@app.get("/health", tags=["Health"], summary="Service and model status")
def health():
    """Return `model_available: true` when the trained model file is present."""
    bundle_loaded = os.path.exists(MODEL_PATH)
    return {"model_available": bundle_loaded}


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"], summary="Predict for one patient")
def predict(patient: PatientInput):
    """
    Predict the **class** (1 = disease likely, 0 = unlikely, decision threshold 0.5) and both
    **probabilities** (disease / no disease) for one patient.

    Fields are validated: an out-of-range value (for example `sex = 7`) returns a 422 error.
    """
    return run_prediction(patient)


@app.post("/explain", response_model=ExplanationOutput, tags=["Explainability"], summary="Predict and explain with SHAP")
def explain(patient: PatientInput):
    """
    Prediction + SHAP explanation. `contributions` gives the effect of each encoded column and
    `variable_contributions` the effect per clinical variable (one-hot columns are summed).

    **Additivity**: `base_value` + sum of the contributions = model output (in the unit given by
    `output_space`). SHAP describes the behaviour of the model, not a medical cause-and-effect
    relationship.
    """
    bundle = get_bundle()
    model = bundle["model"]

    row = build_feature_row(patient)
    prediction = run_prediction(patient)

    all_values, base = unpack_explanation(get_explainer()(row))
    values = all_values[0]

    contributions = [
        FeatureContribution(feature=f, value=round(float(v), 4))
        for f, v in sorted(zip(row.columns, values), key=lambda t: -abs(t[1]))
    ]
    by_variable = aggregate_by_variable(pd.DataFrame([values], columns=row.columns)).iloc[0]
    variable_contributions = [
        FeatureContribution(feature=v, value=round(float(x), 4))
        for v, x in by_variable.sort_values(key=abs, ascending=False).items()
    ]

    return ExplanationOutput(
        prediction=prediction,
        contributions=contributions,
        variable_contributions=variable_contributions,
        base_value=round(base, 4),
        base_probability=round(base_probability(model, base), 4),
        output_space=output_space(model),
    )
