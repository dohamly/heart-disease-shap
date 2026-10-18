"""
model_config.py
---------------
Définition unique des 3 modèles et de leur recherche d'hyperparamètres,
partagée par evaluate.py (validation croisée imbriquée) et train.py
(entraînement du modèle déployé).

Le Pipeline standardise les variables numériques À L'INTÉRIEUR de chaque pli
de validation croisée : le scaler n'est donc jamais ajusté sur les données de
validation (pas de fuite d'information).
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from preprocessing import NUMERIC_COLS

SEED = 42
MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost"]
INNER_SCORING = "roc_auc"   # critère utilisé pour choisir les hyperparamètres
N_ITER = 12                 # combinaisons testées par RandomizedSearchCV


def base_estimator(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=SEED)
    if name == "Random Forest":
        return RandomForestClassifier(random_state=SEED, n_jobs=1)
    if name == "XGBoost":
        return XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=1)
    raise ValueError(name)


PARAM_SPACES = {
    "Logistic Regression": {
        "clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],
        "clf__class_weight": [None, "balanced"],
    },
    "Random Forest": {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [None, 3, 5, 8],
        "clf__min_samples_leaf": [1, 2, 4, 8],
        "clf__max_features": ["sqrt", "log2"],
    },
    "XGBoost": {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [2, 3, 4],
        "clf__learning_rate": [0.01, 0.03, 0.1],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.7, 0.85, 1.0],
        "clf__min_child_weight": [1, 3, 5],
    },
}


def make_pipeline(name, columns):
    numeric = [c for c in NUMERIC_COLS if c in columns]
    prep = ColumnTransformer(
        [("num", StandardScaler(), numeric)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    return Pipeline([("prep", prep), ("clf", base_estimator(name))])


def make_search(name, columns, cv, n_jobs=1):
    """GridSearchCV (régression logistique, espace petit) ou RandomizedSearchCV."""
    pipe = make_pipeline(name, columns)
    if name == "Logistic Regression":
        return GridSearchCV(pipe, PARAM_SPACES[name], cv=cv, scoring=INNER_SCORING, n_jobs=n_jobs)
    return RandomizedSearchCV(
        pipe, PARAM_SPACES[name], n_iter=N_ITER, cv=cv,
        scoring=INNER_SCORING, random_state=SEED, n_jobs=n_jobs,
    )
