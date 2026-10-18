"""
encoding_ablation.py
--------------------
Teste les CHOIX de preprocessing au lieu de les affirmer. Même modèle
(régression logistique, C = 0.03), mêmes plis (validation croisée stratifiée
5 plis x 10 répétitions), seul l'encodage des variables change.

Variantes :
  A  retenue : cp/restecg/slope/thal en one-hot, ca numérique (standardisé)
  B  ca en one-hot aussi (le traiter comme purement catégoriel)
  C  slope en ordinal numérique (descendante < plate < ascendante)
  D  cp/restecg/slope/thal en codes numériques 0,1,2,3 (erreur classique :
     traiter des catégories NOMINALES comme si elles étaient ordonnées)
  E  codes hors documentation (ca = 4, thal = 0) traités comme manquants et
     imputés DANS chaque pli (médiane pour ca, mode pour thal)

Contrôle d'équivalence : le one-hot est ici calculé avant le split
(pd.get_dummies). On vérifie qu'un OneHotEncoder placé DANS le pipeline
(appris sur le pli d'entraînement seulement) donne les mêmes prédictions.

Lancer depuis la racine :   python src/encoding_ablation.py
Écrit reports/encoding_ablation.md
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_config import SEED  # noqa: E402
from preprocessing import (  # noqa: E402
    CATEGORICAL_COLS, build_preprocessed_dataset, disease_label, load_data,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = 0.03          # valeur retenue par le tuning (models/results.json)
N_REPEATS = 10
BASE_NUM = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def ohe():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def build(variant):
    """Retourne (ColumnTransformer, transformation à appliquer au DataFrame brut)."""
    ident = lambda d: d
    if variant == "A":
        ct = ColumnTransformer([
            ("num", StandardScaler(), BASE_NUM + ["ca"]),
            ("cat", ohe(), CATEGORICAL_COLS),
        ], remainder="passthrough")
        return ct, ident
    if variant == "B":
        ct = ColumnTransformer([
            ("num", StandardScaler(), BASE_NUM),
            ("cat", ohe(), CATEGORICAL_COLS + ["ca"]),
        ], remainder="passthrough")
        return ct, ident
    if variant == "C":
        ct = ColumnTransformer([
            ("num", StandardScaler(), BASE_NUM + ["ca", "slope"]),
            ("cat", ohe(), ["cp", "restecg", "thal"]),
        ], remainder="passthrough")
        return ct, ident
    if variant == "D":
        ct = ColumnTransformer([
            ("num", StandardScaler(), BASE_NUM + ["ca"] + CATEGORICAL_COLS),
        ], remainder="passthrough")
        return ct, ident
    if variant == "E":
        def to_missing(d):
            d = d.copy()
            d.loc[d["ca"] == 4, "ca"] = np.nan
            d.loc[d["thal"] == 0, "thal"] = np.nan
            return d
        ct = ColumnTransformer([
            ("num", StandardScaler(), BASE_NUM),
            ("ca", Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("sc", StandardScaler())]), ["ca"]),
            ("thal", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                               ("oh", ohe())]), ["thal"]),
            ("cat", ohe(), ["cp", "restecg", "slope"]),
        ], remainder="passthrough")
        return ct, to_missing
    raise ValueError(variant)


def model():
    return LogisticRegression(C=C, max_iter=3000, random_state=SEED)


def run_variant(variant, X, y, splits):
    ct, prep = build(variant)
    Xv = prep(X)
    aucs, f1s, oof = [], [], []
    for tr, te in splits:
        pipe = Pipeline([("prep", ct), ("clf", model())])
        pipe.fit(Xv.iloc[tr], y.iloc[tr])
        p = pipe.predict_proba(Xv.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], p))
        f1s.append(f1_score(y.iloc[te], (p >= 0.5).astype(int)))
        oof.append(p)
    return np.array(aucs), np.array(f1s), oof


def main():
    df = load_data(os.path.join(ROOT, "data", "heart.csv"))
    y = disease_label(df)
    X = df.drop(columns=["target"]).astype(float)   # données brutes, non encodées
    splits = list(RepeatedStratifiedKFold(
        n_splits=5, n_repeats=N_REPEATS, random_state=SEED).split(X, y))

    names = {
        "A": "A. **Retenue** : one-hot (cp, restecg, slope, thal) + `ca` numérique",
        "B": "B. `ca` aussi en one-hot",
        "C": "C. `slope` en ordinal numérique",
        "D": "D. cp/restecg/slope/thal en codes 0-3 (nominal traité comme ordonné)",
        "E": "E. `ca = 4` et `thal = 0` traités comme manquants, imputés dans le pli",
    }
    res = {v: run_variant(v, X, y, splits) for v in names}
    auc_a, f1_a, _ = res["A"]

    lines = ["| Variante | ROC-AUC | F1 | Δ ROC-AUC vs A (appariée) | plis où elle bat A |",
             "|---|---|---|---|---|"]
    for v, label in names.items():
        auc, f1, _ = res[v]
        if v == "A":
            lines.append(f"| {label} | {auc.mean():.3f} ± {auc.std(ddof=1):.3f} | "
                         f"{f1.mean():.3f} ± {f1.std(ddof=1):.3f} | référence | — |")
        else:
            d = auc - auc_a
            lines.append(f"| {label} | {auc.mean():.3f} ± {auc.std(ddof=1):.3f} | "
                         f"{f1.mean():.3f} ± {f1.std(ddof=1):.3f} | "
                         f"{d.mean():+.4f} ± {d.std(ddof=1):.4f} | {(d > 0).mean()*100:.0f} % |")
    table = "\n".join(lines)

    # ---- équivalence : get_dummies avant le split vs OneHotEncoder dans le pipeline
    Xd, yd = build_preprocessed_dataset(df)
    numeric_dummies = BASE_NUM + ["ca"]
    max_diff = 0.0
    for tr, te in splits[:10]:
        ct = ColumnTransformer([("num", StandardScaler(), numeric_dummies)],
                               remainder="passthrough")
        p1 = Pipeline([("prep", ct), ("clf", model())]).fit(Xd.iloc[tr], yd.iloc[tr])
        a = p1.predict_proba(Xd.iloc[te])[:, 1]
        ct2, _ = build("A")
        p2 = Pipeline([("prep", ct2), ("clf", model())]).fit(X.iloc[tr], y.iloc[tr])
        b = p2.predict_proba(X.iloc[te])[:, 1]
        max_diff = max(max_diff, float(np.abs(a - b).max()))

    n_inv = int(((df.ca == 4).sum() + (df.thal == 0).sum()))
    md = f"""# Ablation des choix d'encodage

Régression logistique (C = {C}), validation croisée stratifiée 5 plis x {N_REPEATS} répétitions
({len(splits)} évaluations par variante). Le scaler et les encodeurs sont ajustés
**dans chaque pli d'entraînement** (Pipeline). Les écarts sont calculés pli par pli sur les
mêmes partitions (comparaison appariée). Les plis se recouvrent, donc les écarts-types
sont descriptifs et ne constituent pas un test statistique.

{table}

## Contrôle d'équivalence (pas de fuite via le one-hot)

Le projet encode les catégories avec `pd.get_dummies` **avant** le split (cela fixe seulement
la liste des colonnes, aucune statistique n'est apprise). En le comparant à un
`OneHotEncoder` appris dans chaque pli d'entraînement, l'écart maximal sur les
probabilités prédites est de **{max_diff:.1e}** : c'est identique à la précision numérique près.

## Lecture

- Les valeurs hors documentation concernent {n_inv} lignes (`ca = 4` : {(df.ca == 4).sum()}, `thal = 0` : {(df.thal == 0).sum()}).
- Un écart de ROC-AUC inférieur à l'écart-type apparié n'est pas une preuve de différence.
"""
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "encoding_ablation.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(table)
    print(f"\nÉquivalence get_dummies vs OneHotEncoder dans le pipeline : écart max = {max_diff:.2e}")


if __name__ == "__main__":
    main()
