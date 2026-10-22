"""
shap_utils.py
-------------
Outils SHAP partagés par explain.py, app.py et les tests.

SHAP explique le COMPORTEMENT DU MODÈLE : ce qui a poussé sa prédiction vers
« malade » ou « sain » pour un patient donné. Ce n'est PAS une relation
causale médicale (voir CAVEAT).

Propriété utilisée partout : les valeurs SHAP sont ADDITIVES.
    sortie du modèle = valeur de base + somme des contributions
Une variable catégorielle encodée en one-hot (cp_0, cp_1, ...) est donc
résumée en additionnant les contributions de ses colonnes : on obtient UNE
contribution par variable clinique, plus lisible que 23 colonnes encodées.
"""

import numpy as np
import pandas as pd

from preprocessing import CATEGORICAL_COLS, CATEGORY_NAMES, FEATURE_NAMES_READABLE

CAVEAT = (
    "SHAP décrit comment le modèle utilise les variables pour prédire ; il "
    "n'établit aucune relation de cause à effet médicale. Une variable qui "
    "« augmente le risque » dans le modèle n'est pas pour autant une cause de "
    "la maladie."
)
DISCLAIMER = (
    "Projet académique : ce n'est pas un outil de diagnostic médical et il ne "
    "doit pas servir à une décision clinique."
)

BINARY_LABELS = {
    "sex": {0: "female", 1: "male"},
    "fbs": {0: "no", 1: "yes"},
    "exang": {0: "no", 1: "yes"},
}
UNITS = {"age": " years", "trestbps": " mmHg", "chol": " mg/dl", "thalach": " bpm"}


def variable_of(col: str) -> str:
    """Variable clinique d'origine d'une colonne encodée (cp_2 -> cp)."""
    base, _, val = col.rpartition("_")
    return base if base in CATEGORICAL_COLS and val.isdigit() else col


def aggregate_by_variable(values: pd.DataFrame) -> pd.DataFrame:
    """Additionne les contributions des colonnes one-hot d'une même variable."""
    groups = {}
    for c in values.columns:
        groups.setdefault(variable_of(c), []).append(c)
    return pd.DataFrame({v: values[cols].sum(axis=1) for v, cols in groups.items()},
                        index=values.index)


def variable_label(var: str) -> str:
    return FEATURE_NAMES_READABLE.get(var, var)


def value_label(var: str, raw) -> str:
    """Valeur brute lisible : 'asymptomatic', 'yes', '135 mmHg', '2.3'..."""
    if var in CATEGORY_NAMES:
        return CATEGORY_NAMES[var].get(int(raw), str(raw))
    if var in BINARY_LABELS:
        return BINARY_LABELS[var][int(raw)]
    if var == "oldpeak":
        return f"{float(raw):.1f}"
    return f"{float(raw):g}{UNITS.get(var, '')}"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def output_space(model) -> str:
    """Unité des valeurs SHAP : probabilité pour la forêt aléatoire, log-odds sinon."""
    return "probability" if type(model).__name__ == "RandomForestClassifier" else "log-odds"


def base_probability(model, base: float) -> float:
    """Probabilité prédite pour le « patient moyen » (valeur de base de SHAP)."""
    return float(base) if output_space(model) == "probability" else float(sigmoid(base))


def describe_patient(raw: dict, contrib_by_var: pd.Series, proba: float,
                     base_proba: float | None = None, top: int = 4,
                     min_effect: float = 0.05) -> str:
    """
    Explication en français des principaux facteurs d'une prédiction.
    `contrib_by_var` : contribution SHAP agrégée par variable (log-odds).
    Seuls les effets notables (|contribution| >= min_effect) sont listés.
    """
    c = contrib_by_var.sort_values(ascending=False)
    up = [(v, x) for v, x in c.items() if x >= min_effect][:top]
    down = [(v, x) for v, x in c.sort_values().items() if x <= -min_effect][:top]

    def fmt(items):
        return " ; ".join(
            f"{variable_label(v)} ({value_label(v, raw[v])}) {x:+.2f}" for v, x in items
        ) or "aucun facteur notable"

    shown = {v for v, _ in up} | {v for v, _ in down}
    rest = [v for v in c.index if v not in shown]
    other = float(c[rest].sum()) if rest else 0.0

    head = f"Le modèle estime une probabilité de maladie de **{proba * 100:.0f} %**"
    if base_proba is not None:
        head += f" (patient moyen du jeu d'entraînement : {base_proba * 100:.0f} %)"
    return (
        f"{head}.\n\n"
        f"- **Ce qui augmente le plus cette probabilité** : {fmt(up)}\n"
        f"- **Ce qui la réduit le plus** : {fmt(down)}\n"
        f"- Les autres variables (détail dans le graphique) pèsent ensemble {other:+.2f}.\n\n"
        f"Les chiffres sont des contributions en log-odds : leur somme, ajoutée à la "
        f"valeur de base, donne exactement la sortie du modèle. {CAVEAT}"
    )
