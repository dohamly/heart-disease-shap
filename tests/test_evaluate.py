"""Tests unitaires pour src/evaluate.py et src/model_config.py (sans relancer la CV complète)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evaluate import compute_metrics, select_model, summarize  # noqa: E402
from model_config import MODEL_NAMES, make_pipeline  # noqa: E402
from preprocessing import build_preprocessed_dataset, load_data  # noqa: E402


def test_compute_metrics_known_values():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1, 0.1, 0.1])  # 3 TP, 1 FN, 1 FP, 3 TN
    m = compute_metrics(y, p)
    assert m["recall"] == 0.75
    assert m["specificity"] == 0.75
    assert m["precision"] == 0.75
    assert m["accuracy"] == 0.75


def test_summarize_mean_std():
    from evaluate import METRICS
    folds = [{m: x for m in METRICS} for x in [0.8, 0.9, 1.0]]
    s = summarize(folds)
    assert abs(s["f1"]["mean"] - 0.9) < 1e-9
    assert abs(s["f1"]["std"] - 0.1) < 1e-9


def test_select_model_prefers_simplest_within_one_se():
    base = {"std": 0.08}
    summ = {
        "Logistic Regression": {"f1": {"mean": 0.80, **base}},
        "Random Forest": {"f1": {"mean": 0.82, **base}},
        "XGBoost": {"f1": {"mean": 0.78, **base}},
    }
    selected, best, se = select_model(summ)
    assert best == "Random Forest"
    assert selected == "Logistic Regression"  # 0.82 - 0.036 < 0.80


def test_pipelines_fit_and_predict():
    df = load_data(path=os.path.join(os.path.dirname(__file__), "..", "data", "heart.csv"))
    X, y = build_preprocessed_dataset(df)
    for name in MODEL_NAMES:
        pipe = make_pipeline(name, list(X.columns)).fit(X, y)
        proba = pipe.predict_proba(X.iloc[:5])
        assert proba.shape == (5, 2)
