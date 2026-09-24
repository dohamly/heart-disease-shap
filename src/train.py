"""
train.py
--------
Entraîne et compare plusieurs modèles (Logistic Regression, Random Forest,
XGBoost) sur le dataset Heart Disease, puis sauvegarde le meilleur modèle.
"""

import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from preprocessing import load_data, build_preprocessed_dataset, split_and_scale

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
    ),
}


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def main():
    df = load_data()
    X, y = build_preprocessed_dataset(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    results = {}
    fitted_models = {}

    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[name] = metrics
        fitted_models[name] = model
        print(f"\n{name}")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                print(f"  {k}: {v}")

    # Sélection du meilleur modèle sur le F1-score
    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = fitted_models[best_name]
    print(f"\n>>> Meilleur modèle : {best_name} (F1 = {results[best_name]['f1']})")

    joblib.dump(
        {
            "model": best_model,
            "scaler": scaler,
            "columns": X_train.columns.tolist(),
            "model_name": best_name,
        },
        "models/best_model.joblib",
    )

    with open("models/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Modèle sauvegardé dans models/best_model.joblib")


if __name__ == "__main__":
    main()
