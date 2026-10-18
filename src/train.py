"""
train.py
--------
Entraîne le modèle DÉPLOYÉ (utilisé par l'API et l'app Streamlit).

  1. Le modèle à déployer est celui retenu par la validation croisée imbriquée
     (models/cv_results.json, produit par `python src/evaluate.py`).
  2. Ses hyperparamètres sont réglés par validation croisée stratifiée à 5 plis
     sur le jeu d'entraînement uniquement.
  3. Il est évalué une seule fois sur le jeu de test mis de côté (20 %).

ATTENTION : avec seulement 61 patients de test, ces chiffres sont bruités
(un seul patient change le recall de ~3 points). La référence pour comparer
les modèles est models/cv_results.json, pas ce score.

Lancer depuis la racine du projet :   python src/train.py
"""

import json
import os
import sys

import joblib
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import METRICS, compute_metrics  # noqa: E402
from model_config import SEED, base_estimator, make_search  # noqa: E402
from preprocessing import (  # noqa: E402
    NUMERIC_COLS, build_preprocessed_dataset, load_data, split_and_scale, split_data,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CV_RESULTS = os.path.join(ROOT, "models", "cv_results.json")


def selected_model_name():
    if os.path.exists(CV_RESULTS):
        with open(CV_RESULTS, encoding="utf-8") as f:
            return json.load(f)["selected_model"]
    print("models/cv_results.json introuvable : lance d'abord `python src/evaluate.py`.\n"
          "Par défaut : Logistic Regression.")
    return "Logistic Regression"


def main():
    df = load_data(os.path.join(ROOT, "data", "heart.csv"))
    X, y = build_preprocessed_dataset(df)
    name = selected_model_name()
    print(f"Modèle à déployer : {name}")

    # Réglage des hyperparamètres sur le jeu d'entraînement (scaler dans chaque pli)
    X_tr_raw, _, y_tr, _ = split_data(X, y)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    search = make_search(name, list(X.columns), cv)
    search.fit(X_tr_raw, y_tr)
    params = {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v)
              for k, v in search.best_params_.items()}
    print(f"Hyperparamètres retenus (ROC-AUC CV = {search.best_score_:.3f}) : {params}")

    # Entraînement final : le scaler est ajusté sur le train, comme attendu par l'API/app
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    model = base_estimator(name).set_params(**params).fit(X_train, y_train)

    metrics = compute_metrics(y_test, model.predict_proba(X_test)[:, 1])
    print("\nScore sur le jeu de test (61 patients, bruité) :")
    for m in METRICS:
        print(f"  {m:<12} {metrics[m]:.3f}")

    if name != "Logistic Regression":
        print("\nNote : le calcul local du dashboard HTML (web/index.html) suppose une "
              "régression logistique ; en mode API il fonctionne avec n'importe quel modèle.")

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "columns": X_train.columns.tolist(),
                 "model_name": name, "hyperparameters": params},
                os.path.join(ROOT, "models", "best_model.joblib"))
    with open(os.path.join(ROOT, "models", "results.json"), "w", encoding="utf-8") as f:
        json.dump({"model": name, "hyperparameters": params, "holdout_test_n": int(len(y_test)),
                   "holdout_metrics": {m: round(float(metrics[m]), 4) for m in METRICS},
                   "note": "Score sur 61 patients : indicatif. Référence : cv_results.json."},
                  f, indent=2, ensure_ascii=False)
    print("\nModèle sauvegardé dans models/best_model.joblib")


if __name__ == "__main__":
    main()
