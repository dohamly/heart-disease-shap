"""
inference.py
------------
Shared inference helpers used by the API, the Streamlit app and the explanation scripts:
loading the deployed model, encoding patients and computing SHAP values.

Keeping them in one place guarantees that every interface scores and explains a patient
in exactly the same way.
"""

import os

import joblib
import numpy as np
import pandas as pd
import shap

from preprocessing import (
    CATEGORICAL_COLS, NUMERIC_COLS, build_preprocessed_dataset, load_data, split_and_scale,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "best_model.joblib")
DATA_PATH = os.path.join(ROOT, "data", "heart.csv")
TREE_MODELS = ("RandomForestClassifier", "XGBClassifier")


def load_bundle(path: str = MODEL_PATH) -> dict:
    """Load the deployed model bundle: model, scaler, columns, model_name, hyperparameters."""
    return joblib.load(path)


def training_background() -> pd.DataFrame:
    """
    Return the scaled training split, used as the SHAP reference population.

    Without a reference population a linear explainer would take the explained patient as
    the baseline, and every contribution would collapse to zero.
    """
    X, y = build_preprocessed_dataset(load_data(DATA_PATH))
    X_train, *_ = split_and_scale(X, y)
    return X_train


def scale_encoded(X_encoded: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Order already one-hot-encoded features as the model expects and scale the numeric ones."""
    columns = bundle["columns"]
    scaled = X_encoded.reindex(columns=columns, fill_value=0).astype("float64")
    numeric = [c for c in NUMERIC_COLS if c in columns]
    scaled[numeric] = bundle["scaler"].transform(scaled[numeric])
    return scaled


def encode_patients(raw, bundle: dict) -> pd.DataFrame:
    """
    Turn raw clinical values into model input.

    `raw` is a dict (one patient) or a DataFrame (one row per patient) holding the 13 raw
    clinical variables. Categories absent from the input become all-zero dummy columns.
    """
    frame = pd.DataFrame([raw]) if isinstance(raw, dict) else raw
    return scale_encoded(pd.get_dummies(frame, columns=CATEGORICAL_COLS), bundle)


def make_explainer(model, background: pd.DataFrame):
    """Build a TreeExplainer for tree models, otherwise a LinearExplainer referenced on `background`."""
    if type(model).__name__ in TREE_MODELS:
        return shap.TreeExplainer(model)
    return shap.LinearExplainer(model, shap.maskers.Independent(background, max_samples=len(background)))


def unpack_explanation(explanation) -> tuple[np.ndarray, float]:
    """Extract (SHAP values for the 'disease' class, base value) from a shap Explanation."""
    values, base_values = explanation.values, explanation.base_values
    if values.ndim == 3:  # tree models return one output per class
        return values[:, :, 1], float(base_values[0, 1])
    return values, float(base_values[0])


def explain_rows(model, background: pd.DataFrame, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Return SHAP values (rows x features) for the 'disease' class and the base value."""
    return unpack_explanation(make_explainer(model, background)(X))
