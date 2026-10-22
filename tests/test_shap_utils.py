"""Tests des outils SHAP : additivité, agrégation par variable, textes d'interprétation."""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from preprocessing import NUMERIC_COLS, build_preprocessed_dataset, load_data, split_data  # noqa: E402
from shap_utils import (  # noqa: E402
    CAVEAT, aggregate_by_variable, base_probability, describe_patient, output_space, sigmoid,
    value_label, variable_of,
)

ROOT = os.path.join(HERE, "..")


def _setup():
    df = load_data(os.path.join(ROOT, "data", "heart.csv"))
    X, _ = build_preprocessed_dataset(df)
    bundle = joblib.load(os.path.join(ROOT, "models", "best_model.joblib"))
    cols = bundle["columns"]
    Xs = X.copy()
    Xs[NUMERIC_COLS] = bundle["scaler"].transform(X[NUMERIC_COLS])
    return df, X, Xs[cols], bundle


def test_variable_of_maps_encoded_columns_to_clinical_variables():
    assert variable_of("cp_2") == "cp" and variable_of("thal_0") == "thal"
    assert variable_of("slope_1") == "slope" and variable_of("restecg_2") == "restecg"
    assert variable_of("age") == "age" and variable_of("exang") == "exang"


def test_shap_is_additive_for_the_deployed_model():
    """valeur de base + somme des contributions = sortie exacte du modèle (log-odds)."""
    df, X, Xs, bundle = _setup()
    model = bundle["model"]
    _, y = build_preprocessed_dataset(df)
    X_train = Xs.loc[split_data(X, y)[0].index]      # référence SHAP = jeu d'entraînement
    sv = shap.LinearExplainer(model, shap.maskers.Independent(X_train, max_samples=len(X_train)))(Xs)
    gap = np.abs(sv.values.sum(axis=1) + sv.base_values - model.decision_function(Xs)).max()
    assert gap < 1e-9


def test_aggregation_by_variable_preserves_the_total():
    _, _, Xs, bundle = _setup()
    coef = bundle["model"].coef_[0]
    contrib = pd.DataFrame((Xs.values - Xs.values.mean(axis=0)) * coef, columns=Xs.columns)
    agg = aggregate_by_variable(contrib)
    assert agg.shape[1] == 13                       # 13 variables cliniques, pas 23 colonnes
    assert np.allclose(agg.sum(axis=1), contrib.sum(axis=1))


def test_output_space_and_base_probability():
    class RandomForestClassifier: ...
    class LogisticRegression: ...
    assert output_space(RandomForestClassifier()) == "probability"
    assert output_space(LogisticRegression()) == "log-odds"
    assert base_probability(LogisticRegression(), 0.0) == 0.5
    assert abs(base_probability(LogisticRegression(), -0.2056) - sigmoid(-0.2056)) < 1e-12
    assert base_probability(RandomForestClassifier(), 0.41) == 0.41


def test_value_label_is_human_readable():
    assert value_label("cp", 0) == "asymptomatic"
    assert value_label("exang", 1) == "yes" and value_label("sex", 0) == "female"
    assert value_label("thalach", 131) == "131 bpm" and value_label("oldpeak", 2.24) == "2.2"


def test_describe_patient_lists_only_notable_effects_and_states_the_caveat():
    raw = {"age": 58, "sex": 1, "cp": 0, "trestbps": 130, "chol": 250, "fbs": 0, "restecg": 0,
           "thalach": 131, "exang": 1, "oldpeak": 2.2, "slope": 1, "ca": 3, "thal": 3}
    contrib = pd.Series({"ca": 0.91, "thal": 0.40, "oldpeak": 0.32, "cp": 0.25, "exang": 0.18,
                         "thalach": -0.30, "sex": -0.01, "fbs": -0.00, "age": 0.02, "chol": -0.02,
                         "trestbps": 0.01, "restecg": 0.0, "slope": 0.0})
    text = describe_patient(raw, contrib, proba=0.92, base_proba=0.45)
    assert "92 %" in text and "45 %" in text
    assert "Major vessels colored (3) +0.91" in text
    assert "Max heart rate (131 bpm) -0.30" in text
    assert "Fasting blood sugar" not in text           # effet de -0.00 : non listé
    assert CAVEAT in text                              # SHAP != causalité, dit à chaque explication
    assert "aucun facteur notable" in describe_patient(raw, contrib.abs(), 0.9, 0.45, top=0)
