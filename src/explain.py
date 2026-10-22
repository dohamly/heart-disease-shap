"""
explain.py
----------
SHAP au cœur du projet : explications globales, individuelles, textes
d'interprétation et comparaison des trois modèles.

SHAP explique le COMPORTEMENT DU MODÈLE, pas une relation causale médicale
(voir shap_utils.CAVEAT). Projet académique, pas un outil de diagnostic.

Produit (depuis la racine du projet : `python src/explain.py`) :
  models/shap_summary_plot.png       beeswarm : effet de chaque colonne encodée
  models/shap_bar_plot.png           importance par variable clinique
  models/shap_waterfall_plot.png     un patient à risque élevé (jeu de test)
  models/shap_waterfall_low_risk.png un patient à faible risque (jeu de test)
  reports/shap_model_comparison.png  les 3 modèles s'appuient-ils sur les mêmes variables ?
  reports/shap_interpretation.md     texte d'interprétation généré à partir des chiffres
"""

import os
import sys

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_config import MODEL_NAMES, SEED, make_search  # noqa: E402
from preprocessing import (  # noqa: E402
    BINARY_COLS, CATEGORICAL_COLS, NUMERIC_COLS, build_preprocessed_dataset, load_data,
    readable_feature, split_data,
)
from shap_utils import (  # noqa: E402
    BINARY_LABELS, CAVEAT, DISCLAIMER, aggregate_by_variable, base_probability, describe_patient,
    output_space, value_label, variable_label, variable_of,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
REPORTS = os.path.join(ROOT, "reports")
RED, GREEN = "#e0546a", "#0f9488"


def load_artifacts(path=None):
    bundle = joblib.load(path or os.path.join(MODELS, "best_model.joblib"))
    return bundle["model"], bundle["scaler"], bundle["columns"], bundle["model_name"]


def get_explainer(model, background):
    """TreeExplainer pour les arbres, LinearExplainer (référence = jeu d'entraînement) sinon."""
    if type(model).__name__ in ("RandomForestClassifier", "XGBClassifier"):
        return shap.TreeExplainer(model)
    return shap.LinearExplainer(model, shap.maskers.Independent(background, max_samples=len(background)))


def shap_values(model, background, X):
    """(valeurs n x p, valeur de base) pour la classe « maladie »."""
    sv = get_explainer(model, background)(X)
    if sv.values.ndim == 3:                       # arbres : une sortie par classe
        return sv.values[:, :, 1], float(sv.base_values[0, 1])
    return sv.values, float(sv.base_values[0])



# ------------------------------------------------------------------ figures
def plot_beeswarm(vals, Xs, cols, path):
    plt.figure()
    shap.summary_plot(vals, Xs, feature_names=[readable_feature(c) for c in cols],
                      show=False, max_display=15)
    plt.title("Effet de chaque variable sur la prédiction du modèle (303 patients)", fontsize=10)
    plt.xlabel("Contribution SHAP  (← vers « sain »  |  vers « malade » →)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_importance(agg, path):
    imp = agg.abs().mean().sort_values()
    share = imp / imp.sum()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh([variable_label(v) for v in imp.index], imp.values, color=GREEN)
    for i, (v, s) in enumerate(zip(imp.values, share.values)):
        ax.text(v, i, f"  {s * 100:.0f} %", va="center", fontsize=8.5)
    ax.set_xlabel("Contribution SHAP absolue moyenne (log-odds)")
    ax.set_title("Importance des variables cliniques (somme des colonnes encodées)", fontsize=10)
    ax.set_xlim(0, imp.max() * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def plot_waterfall(contrib, base, raw, title, path, p, base_p, max_display=9):
    names = [f"{variable_label(v)}: {value_label(v, raw[v])}" for v in contrib.index]
    ex = shap.Explanation(values=contrib.values, base_values=base, data=None, feature_names=names)
    plt.figure()
    shap.plots.waterfall(ex, max_display=max_display, show=False)
    plt.title(f"{title}\nProbabilité estimée {p * 100:.0f} %  (patient moyen : {base_p * 100:.0f} %)",
              fontsize=10, loc="left")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ------------------------------------------------- interprétation du global
def direction_text(var, df, agg, shap_cols):
    """Phrase décrivant dans quel sens le modèle utilise une variable."""
    label = variable_label(var)
    if var in CATEGORICAL_COLS:
        eff = []
        for c in [c for c in shap_cols.columns if variable_of(c) == var]:
            present = df[var] == int(c.rsplit("_", 1)[1])
            if present.sum() >= 10:                        # catégories trop rares ignorées
                eff.append((value_label(var, c.rsplit("_", 1)[1]), float(shap_cols.loc[present, c].mean())))
        eff.sort(key=lambda t: t[1])
        return (f"la catégorie qui augmente le plus la probabilité estimée est « {eff[-1][0]} » ({eff[-1][1]:+.2f} en moyenne), "
                f"celle qui la réduit le plus est « {eff[0][0]} » ({eff[0][1]:+.2f})")
    x = df[var].astype(float)
    lo, hi = float(agg.loc[x <= x.quantile(0.25), var].mean()), float(agg.loc[x >= x.quantile(0.75), var].mean())
    if var in BINARY_COLS:
        up = hi > lo
        return (f"{label} = « {BINARY_LABELS[var][1]} » {'augmente' if up else 'réduit'} la probabilité estimée "
                f"par rapport à « {BINARY_LABELS[var][0]} » (contribution moyenne {hi:+.2f} contre {lo:+.2f})")
    return (f"plus la valeur est élevée, plus le modèle {'augmente' if hi > lo else 'réduit'} la probabilité : "
            f"contribution moyenne {lo:+.2f} pour les valeurs basses (≤ 25ᵉ percentile) et "
            f"{hi:+.2f} pour les valeurs hautes (≥ 75ᵉ percentile)")


# --------------------------------------------- comparaison des trois modèles
def compare_models(X, y):
    """Les 3 modèles (hyperparamètres réglés sur le train) s'appuient-ils sur les mêmes variables ?"""
    X_tr, _, y_tr, _ = split_data(X, y)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    shares = {}
    for name in MODEL_NAMES:
        best = make_search(name, list(X.columns), cv).fit(X_tr, y_tr).best_estimator_
        prep, clf = best.named_steps["prep"], best.named_steps["clf"]
        Xt, Xbg = prep.transform(X), prep.transform(X_tr)
        vals, _ = shap_values(clf, Xbg, Xt)
        agg = aggregate_by_variable(pd.DataFrame(vals, columns=Xt.columns, index=Xt.index))
        imp = agg.abs().mean()
        shares[name] = imp / imp.sum()        # parts relatives : comparables malgré des unités différentes
    return pd.DataFrame(shares)


def plot_comparison(shares, path):
    order = shares.mean(axis=1).sort_values().index
    fig, ax = plt.subplots(figsize=(8, 6))
    h, colors = 0.26, {"Logistic Regression": GREEN, "Random Forest": "#3b6fe0", "XGBoost": "#d97a1f"}
    for k, name in enumerate(MODEL_NAMES):
        ax.barh(np.arange(len(order)) + (k - 1) * h, shares.loc[order, name] * 100, h, label=name, color=colors[name])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([variable_label(v) for v in order])
    ax.set_xlabel("Part de l'importance SHAP totale du modèle (%)")
    ax.set_title("Les trois modèles s'appuient-ils sur les mêmes variables ?", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(REPORTS, exist_ok=True)
    df = load_data(os.path.join(ROOT, "data", "heart.csv"))
    X, y = build_preprocessed_dataset(df)
    model, scaler, cols, model_name = load_artifacts()
    print(f"Explication du modèle déployé : {model_name}")

    # entrée du modèle : standardisation apprise sur le train (celle du modèle déployé)
    X_tr_raw, X_te_raw, y_tr, y_te = split_data(X, y)
    Xs = X.copy()
    Xs[NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS])
    Xs = Xs[cols]
    background = Xs.loc[X_tr_raw.index]

    vals, base = shap_values(model, background, Xs)
    shap_df = pd.DataFrame(vals, columns=cols, index=Xs.index)
    agg = aggregate_by_variable(shap_df)
    space = output_space(model)
    base_p = base_probability(model, base)
    proba = pd.Series(model.predict_proba(Xs)[:, 1], index=Xs.index)

    # 1-2. global
    plot_beeswarm(vals, Xs, cols, os.path.join(MODELS, "shap_summary_plot.png"))
    plot_importance(agg, os.path.join(MODELS, "shap_bar_plot.png"))

    # 3. individuel : un patient à risque élevé et un à faible risque, tous deux dans le jeu de test
    te = X_te_raw.index
    hi = proba[te][y_te == 1].idxmax()
    lo = proba[te][y_te == 0].idxmin()
    patients = {}
    for key, idx, title, fname in [
        ("high", hi, "Patient à risque élevé (malade, jeu de test)", "shap_waterfall_plot.png"),
        ("low", lo, "Patient à faible risque (sain, jeu de test)", "shap_waterfall_low_risk.png"),
    ]:
        raw = df.loc[idx]
        plot_waterfall(agg.loc[idx].sort_values(key=np.abs, ascending=False), base, raw, title,
                       os.path.join(MODELS, fname), float(proba[idx]), base_p)
        patients[key] = (idx, raw, float(proba[idx]), describe_patient(
            raw, agg.loc[idx], float(proba[idx]), base_p))

    # 4. comparaison des modèles
    print("Comparaison des explications des 3 modèles (réglage des hyperparamètres, ~1 min)...")
    shares = compare_models(X, y)
    plot_comparison(shares, os.path.join(REPORTS, "shap_model_comparison.png"))
    rho = shares.corr(method="spearman")
    top5 = {n: set(shares[n].nlargest(5).index) for n in MODEL_NAMES}
    common = set.intersection(*top5.values())

    # 5. texte d'interprétation
    share = (agg.abs().mean() / agg.abs().mean().sum()).sort_values(ascending=False)
    lines = [
        "# Interprétation SHAP", "",
        f"> {CAVEAT}", f"> {DISCLAIMER}", "",
        f"Modèle expliqué : **{model_name}** (le modèle déployé). SHAP calculé sur les 303 patients, "
        f"avec le jeu d'entraînement comme référence. Unité : {space}. "
        f"Pour un patient moyen, le modèle prédit un risque de {base_p * 100:.0f} %.", "",
        "## Ce que le modèle utilise le plus (global)", "",
        "| Variable | Part de l'importance | Comment le modèle l'utilise |", "|---|---|---|",
    ]
    for v in share.index[:6]:
        lines.append(f"| {variable_label(v)} | {share[v] * 100:.0f} % | {direction_text(v, df, agg, shap_df)} |")
    lines += ["", f"Les 6 variables ci-dessus représentent {share.iloc[:6].sum() * 100:.0f} % de l'importance totale.", "",
              "## Deux patients expliqués (jeu de test, jamais vus à l'entraînement)", ""]
    for key, label in (("high", "Patient à risque élevé"), ("low", "Patient à faible risque")):
        idx, raw, p, text = patients[key]
        lines += [f"### {label} (patient n° {idx})", "", text, ""]
    lines += ["## Les trois modèles expliquent-ils de la même façon ?", "",
              "Part de l'importance SHAP totale par variable (hyperparamètres réglés sur le train, "
              "chaque modèle expliqué sur les 303 patients) :", "",
              "| Variable | " + " | ".join(MODEL_NAMES) + " |", "|---|" + "---|" * len(MODEL_NAMES)]
    for v in shares.mean(axis=1).sort_values(ascending=False).index[:8]:
        lines.append(f"| {variable_label(v)} | " + " | ".join(f"{shares.loc[v, n] * 100:.0f} %" for n in MODEL_NAMES) + " |")
    pairs = [("Logistic Regression", "Random Forest"), ("Logistic Regression", "XGBoost"), ("Random Forest", "XGBoost")]
    lines += ["", "Corrélation de rang de Spearman entre les importances : " +
              " ; ".join(f"{a} / {b} : {rho.loc[a, b]:.2f}" for a, b in pairs) + ".",
              f"Variables communes aux 5 premières de chaque modèle : {', '.join(variable_label(v) for v in sorted(common)) or 'aucune'}.", "",
              "Les trois modèles étant statistiquement équivalents en performance, un accord sur les variables "
              "utilisées renforce la confiance dans l'explication ; un désaccord signalerait qu'elle dépend du "
              "modèle choisi. Les valeurs SHAP des forêts sont en probabilité et celles des deux autres en "
              "log-odds : on compare donc des parts relatives, pas des valeurs brutes.", ""]
    with open(os.path.join(REPORTS, "shap_interpretation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Graphiques : models/shap_summary_plot.png, shap_bar_plot.png, shap_waterfall_plot.png, shap_waterfall_low_risk.png")
    print("Rapports   : reports/shap_model_comparison.png, reports/shap_interpretation.md")
    print("Spearman entre modèles :", {f"{a[:3]}/{b[:3]}": round(float(rho.loc[a, b]), 2) for a, b in pairs})


if __name__ == "__main__":
    matplotlib.use("Agg")      # script : pas de fenêtre ; à l'import on ne touche pas au backend
    main()
