"""
evaluate.py
-----------
Évaluation rigoureuse des 3 modèles :

  * validation croisée IMBRIQUÉE : 5 plis externes stratifiés (estimation de
    la performance) x 3 plis internes (choix des hyperparamètres). Le réglage
    n'a donc jamais accès au pli de test -> pas d'optimisme caché.
  * métriques : accuracy, precision, recall (sensibilité), spécificité, F1,
    ROC-AUC, PR-AUC, présentées en moyenne ± écart-type sur les 5 plis ;
  * matrices de confusion, courbes ROC et PR sur les prédictions "hors pli"
    (chaque patient est prédit par un modèle qui ne l'a jamais vu) ;
  * analyse des erreurs (faux positifs / faux négatifs) et interprétation de
    ces erreurs avec SHAP.

Lancer depuis la racine du projet :   python src/evaluate.py
Sorties : models/cv_results.json  et  reports/*.png / *.md
"""

import json
import os
import sys
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_config import MODEL_NAMES, SEED, make_search  # noqa: E402
from preprocessing import (  # noqa: E402
    CATEGORY_NAMES, build_preprocessed_dataset, load_data, readable_feature,
)

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, "reports")
MODELS = os.path.join(ROOT, "models")
METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall (sensibilité)",
    "specificity": "Spécificité", "f1": "F1-score", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
}
N_OUTER, N_INNER = 5, 3
SELECTION_METRIC = "f1"
THRESHOLD = 0.5


# ----------------------------------------------------------------- métriques
def compute_metrics(y_true, proba, threshold=THRESHOLD):
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def summarize(folds):
    out = {}
    for m in METRICS:
        vals = np.array([f[m] for f in folds], dtype=float)
        out[m] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}
    return out


def _py(v):
    return v.item() if hasattr(v, "item") else v


# ------------------------------------------------------- validation imbriquée
def nested_cv(X, y):
    outer = StratifiedKFold(N_OUTER, shuffle=True, random_state=SEED)
    results = {}
    for name in MODEL_NAMES:
        t0 = time.time()
        oof = np.full(len(X), np.nan)
        folds, fitted = [], []
        for k, (tr, te) in enumerate(outer.split(X, y)):
            inner = StratifiedKFold(N_INNER, shuffle=True, random_state=SEED + k)
            search = make_search(name, list(X.columns), inner)
            search.fit(X.iloc[tr], y.iloc[tr])
            proba = search.predict_proba(X.iloc[te])[:, 1]
            oof[te] = proba
            m = compute_metrics(y.iloc[te], proba)
            m["best_params"] = {p.replace("clf__", ""): _py(v) for p, v in search.best_params_.items()}
            folds.append(m)
            fitted.append((tr, te, search.best_estimator_))
        results[name] = {"folds": folds, "oof": oof, "fitted": fitted}
        s = summarize(folds)
        print(f"  {name:<20} F1 = {s['f1']['mean']:.3f} ± {s['f1']['std']:.3f}   "
              f"ROC-AUC = {s['roc_auc']['mean']:.3f} ± {s['roc_auc']['std']:.3f}   ({time.time()-t0:.0f}s)")
    return results


# ------------------------------------------------------------------- figures
def plot_confusions(results, y):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    for ax, name in zip(axes, MODEL_NAMES):
        pred = (results[name]["oof"] >= THRESHOLD).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Pred: sain", "Pred: malade"])
        ax.set_yticks([0, 1], ["Réel: sain", "Réel: malade"])
        ax.set_title(name)
    fig.suptitle("Matrices de confusion — prédictions hors-pli sur les 303 patients", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "confusion_matrices_cv.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_curves(results, y):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for name in MODEL_NAMES:
        p = results[name]["oof"]
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y, p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        a2.plot(rc, pr, label=f"{name} (AP = {average_precision_score(y, p):.3f})")
    a1.plot([0, 1], [0, 1], "--", color="gray")
    a1.set(xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs (recall)", title="Courbes ROC")
    a2.axhline(y.mean(), ls="--", color="gray")
    a2.set(xlabel="Recall", ylabel="Precision", title="Courbes Precision-Recall")
    a1.legend(loc="lower right"); a2.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "roc_pr_curves_cv.png"), dpi=150)
    plt.close(fig)


def plot_metric_bars(summaries):
    fig, ax = plt.subplots(figsize=(11, 4.4))
    w = 0.26
    for i, name in enumerate(MODEL_NAMES):
        means = [summaries[name][m]["mean"] for m in METRICS]
        stds = [summaries[name][m]["std"] for m in METRICS]
        ax.bar(np.arange(len(METRICS)) + (i - 1) * w, means, w, yerr=stds, capsize=3, label=name)
    ax.set_xticks(np.arange(len(METRICS)), [METRIC_LABELS[m] for m in METRICS], rotation=15)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Performance en validation croisée imbriquée (moyenne ± écart-type, 5 plis)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "cv_metrics.png"), dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------- SHAP
def oof_shap(name, result, X):
    """SHAP de chaque patient calculé par le modèle du pli où il était en test."""
    parts, bases = [], {}
    for tr, te, pipe in result["fitted"]:
        prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
        Xtr, Xte = prep.transform(X.iloc[tr]), prep.transform(X.iloc[te])
        if name == "Logistic Regression":
            expl = shap.LinearExplainer(clf, shap.maskers.Independent(Xtr, max_samples=len(Xtr)))
            sv = expl.shap_values(Xte)
        else:
            expl = shap.TreeExplainer(clf)
            sv = expl.shap_values(Xte)
        if isinstance(sv, list):
            sv = sv[1]
        elif np.ndim(sv) == 3:
            sv = sv[:, :, 1]
        ev = np.atleast_1d(expl.expected_value)
        base = float(ev[-1])
        parts.append(pd.DataFrame(sv, index=X.index[te], columns=Xte.columns))
        for i in X.index[te]:
            bases[i] = base
    return pd.concat(parts).sort_index(), pd.Series(bases).sort_index()


def md_table(df, floatfmt="{:.2f}"):
    head = "| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    rows = []
    for idx, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        rows.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join([head, sep] + rows)


def error_analysis(name, result, X, y, df_raw):
    proba = pd.Series(result["oof"], index=X.index)
    pred = (proba >= THRESHOLD).astype(int)
    group = pd.Series(np.select(
        [(y == 1) & (pred == 1), (y == 0) & (pred == 0), (y == 0) & (pred == 1), (y == 1) & (pred == 0)],
        ["TP", "TN", "FP", "FN"], default=""), index=X.index)
    counts = group.value_counts().reindex(["TP", "TN", "FP", "FN"]).fillna(0).astype(int)

    # Profil des groupes (valeurs d'origine)
    raw = df_raw.drop(columns=["target"])
    prof = pd.DataFrame({
        "age (moy.)": raw.age, "pression repos (moy.)": raw.trestbps, "cholestérol (moy.)": raw.chol,
        "fréq. card. max (moy.)": raw.thalach, "ST depression (moy.)": raw.oldpeak,
        "% hommes": (raw.sex == 1) * 100.0, "% angine d'effort": (raw.exang == 1) * 100.0,
        "% >=1 vaisseau coloré": (raw.ca >= 1) * 100.0,
        "% douleur asymptomatique": (raw.cp == 0) * 100.0,
        "% thal. réversible": (raw.thal == 3) * 100.0,
    })
    prof = prof.groupby(group).mean().reindex(["TN", "FP", "FN", "TP"]).T
    prof.index.name = "Caractéristique"

    # Écarts standardisés des erreurs par rapport au groupe bien classé de même vérité terrain
    std_all = X.std().replace(0, np.nan)
    def top_diff(err, ref, k=5):
        d = ((X[group == err].mean() - X[group == ref].mean()) / std_all).dropna()
        d = d.reindex(d.abs().sort_values(ascending=False).index)[:k]
        return [(readable_feature(c), v) for c, v in d.items()]

    # SHAP des erreurs
    sh, base = oof_shap(name, result, X)
    fn_idx, fp_idx = group[group == "FN"].index, group[group == "FP"].index
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, idx, title in [(axes[0], fn_idx, "Faux négatifs (malades prédits sains)"),
                           (axes[1], fp_idx, "Faux positifs (sains prédits malades)")]:
        m = sh.loc[idx].mean()
        m = m.reindex(m.abs().sort_values(ascending=False).index)[:8][::-1]
        ax.barh([readable_feature(c) for c in m.index], m.values,
                color=["#e0546a" if v > 0 else "#0f9488" for v in m.values])
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_title(f"{title} — n = {len(idx)}")
        ax.set_xlabel("SHAP moyen  (→ vers 'malade' | ← vers 'sain')")
    fig.suptitle(f"{name} : ce qui a poussé le modèle à se tromper (SHAP hors-pli)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "shap_errors.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Waterfall des erreurs les plus "sûres d'elles"
    worst = {"FN": proba[fn_idx].idxmin() if len(fn_idx) else None,
             "FP": proba[fp_idx].idxmax() if len(fp_idx) else None}
    for kind, i in worst.items():
        if i is None:
            continue
        exp = shap.Explanation(values=sh.loc[i].values, base_values=base[i],
                               data=X.loc[i, sh.columns].values,
                               feature_names=[readable_feature(c) for c in sh.columns])
        plt.figure()
        shap.plots.waterfall(exp, max_display=10, show=False)
        plt.title(f"Erreur {kind} la plus confiante — patient #{i}  (P(malade) = {proba[i]:.2f})", fontsize=10)
        plt.savefig(os.path.join(REPORTS, f"shap_waterfall_{kind}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    def case_rows(idx_list):
        out = raw.loc[idx_list, ["age", "sex", "cp", "thalach", "oldpeak", "ca", "thal"]].copy()
        out["cp"] = out["cp"].map(CATEGORY_NAMES["cp"])
        out["thal"] = out["thal"].map(CATEGORY_NAMES["thal"])
        out["sex"] = out["sex"].map({1: "M", 0: "F"})
        out["P(malade)"] = proba[idx_list].round(2)
        out.index.name = "patient"
        return out

    # Liste complète des patients mal classés + 3 facteurs SHAP principaux
    err_idx = group[group.isin(["FP", "FN"])].index
    def top3(i):
        r = sh.loc[i]
        t = r.reindex(r.abs().sort_values(ascending=False).index)[:3]
        return "; ".join(f"{readable_feature(c)} ({v:+.2f})" for c, v in t.items())
    mis = raw.loc[err_idx].copy()
    mis.insert(0, "erreur", group[err_idx].map({"FP": "faux positif", "FN": "faux négatif"}))
    mis.insert(1, "P_maladie", proba[err_idx].round(3))
    mis["top3_shap"] = [top3(i) for i in err_idx]
    mis.index.name = "patient"
    mis.sort_values(["erreur", "P_maladie"]).to_csv(
        os.path.join(REPORTS, "misclassified_patients.csv"), encoding="utf-8")

    fn_top = proba[fn_idx].sort_values().index[:5]
    fp_top = proba[fp_idx].sort_values(ascending=False).index[:5]

    lines = [
        f"# Analyse des erreurs — {name}", "",
        f"Prédictions hors-pli (validation croisée imbriquée, seuil {THRESHOLD}) sur les {len(X)} patients. "
        "Chaque patient est prédit par un modèle qui ne l'a pas vu à l'entraînement. Classe positive = maladie présente.", "",
        "## Comptage", "",
        f"- Vrais positifs : **{counts['TP']}** · Vrais négatifs : **{counts['TN']}**",
        f"- Faux positifs (sain prédit malade) : **{counts['FP']}** · Faux négatifs (malade prédit sain) : **{counts['FN']}**", "",
        "En santé, un faux négatif (malade non détecté) est en général plus grave qu'un faux positif : "
        "il faudrait donc abaisser le seuil de décision au-dessus de 0.5 si l'on privilégiait le recall.", "",
        "## Profil moyen de chaque groupe", "", md_table(prof, "{:.1f}"), "",
        "## Ce qui distingue les erreurs (écart standardisé, variables d'origine)", "",
        "**Faux négatifs vs vrais positifs** (malades : pourquoi certains passent inaperçus) :", "",
        *[f"- {n} : {v:+.2f}" for n, v in top_diff("FN", "TP")], "",
        "**Faux positifs vs vrais négatifs** (sains : pourquoi certains ressemblent à des malades) :", "",
        *[f"- {n} : {v:+.2f}" for n, v in top_diff("FP", "TN")], "",
        "## Interprétation avec SHAP", "",
        "![SHAP des erreurs](shap_errors.png)", "",
        "Les barres montrent la contribution moyenne de chaque variable pour les patients mal classés "
        "(rouge : pousse vers « malade », vert : vers « sain »). Les deux erreurs les plus confiantes en détail :", "",
        "![Faux négatif](shap_waterfall_FN.png)", "", "![Faux positif](shap_waterfall_FP.png)", "",
        "## Les 5 faux négatifs les plus confiants", "", md_table(case_rows(fn_top), "{:.1f}"), "",
        "## Les 5 faux positifs les plus confiants", "", md_table(case_rows(fp_top), "{:.1f}"), "",
        "## Limites", "",
        "- Seulement 303 patients : les moyennes par groupe d'erreurs reposent sur de petits effectifs "
        "(quelques dizaines de cas) et sont indicatives, pas des preuves statistiques.",
        "- Dataset Cleveland unique, une seule source hospitalière : aucune validation externe.",
    ]
    with open(os.path.join(REPORTS, "error_analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return counts.to_dict()


# ------------------------------------------------------------ sélection
COMPLEXITY_ORDER = ["Logistic Regression", "Random Forest", "XGBoost"]  # du plus simple au plus complexe


def select_model(summaries):
    """
    Règle de l'erreur standard (Hastie et al.) : on retient le modèle LE PLUS
    SIMPLE dont le score moyen est à moins d'une erreur standard (écart-type /
    racine du nb de plis) du meilleur. Avec 303 patients, les écarts entre
    modèles sont du même ordre que la variabilité entre plis : choisir le
    "gagnant" au centième près reviendrait à suivre du bruit.
    """
    stats = {n: summaries[n][SELECTION_METRIC] for n in MODEL_NAMES}
    best = max(MODEL_NAMES, key=lambda n: stats[n]["mean"])
    se = stats[best]["std"] / np.sqrt(N_OUTER)
    seuil = stats[best]["mean"] - se
    selected = next(n for n in COMPLEXITY_ORDER if stats[n]["mean"] >= seuil)
    return selected, best, se


# ---------------------------------------------------------------------- main
def main():
    os.makedirs(REPORTS, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    df = load_data(os.path.join(ROOT, "data", "heart.csv"))
    X, y = build_preprocessed_dataset(df)
    print(f"Données : {X.shape[0]} patients, {X.shape[1]} variables, "
          f"{int(y.sum())} malades / {int((1 - y).sum())} sains\n")
    print(f"Validation croisée imbriquée ({N_OUTER} plis externes x {N_INNER} internes) :")
    results = nested_cv(X, y)

    summaries = {n: summarize(results[n]["folds"]) for n in MODEL_NAMES}
    selected, best_by_mean, se = select_model(summaries)
    best = selected

    # Résultats sérialisables
    grid = np.linspace(0, 1, 41)
    out = {"settings": {"outer_folds": N_OUTER, "inner_folds": N_INNER, "seed": SEED,
                        "threshold": THRESHOLD, "selection_metric": SELECTION_METRIC,
                        "n_patients": int(len(X))},
           "selected_model": selected, "best_by_mean": best_by_mean,
           "selection_rule": "modèle le plus simple à moins d'une erreur standard du meilleur "
                             f"{SELECTION_METRIC} moyen (SE = {se:.3f})",
           "models": {}}
    for n in MODEL_NAMES:
        oof = results[n]["oof"]
        pooled = compute_metrics(y, oof)
        cm = confusion_matrix(y, (oof >= THRESHOLD).astype(int), labels=[0, 1])
        fpr, tpr, _ = roc_curve(y, oof)
        out["models"][n] = {
            "metrics": summaries[n],
            "folds": [{k: (float(v) if k != "best_params" else v) for k, v in f.items()} for f in results[n]["folds"]],
            "pooled_oof": {**{k: float(v) for k, v in pooled.items()}, "confusion_matrix": cm.tolist()},
            "roc_curve": {"fpr": grid.round(3).tolist(), "tpr": np.interp(grid, fpr, tpr).round(3).tolist()},
        }

    plot_confusions(results, y)
    plot_curves(results, y)
    plot_metric_bars(summaries)
    errors = error_analysis(best, results[best], X, y, df)
    out["error_counts"] = errors
    with open(os.path.join(MODELS, "cv_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Tableau markdown (moyenne ± écart-type)
    head = "| Modèle | " + " | ".join(METRIC_LABELS[m] for m in METRICS) + " |"
    rows = [head, "|" + "---|" * (len(METRICS) + 1)]
    for n in MODEL_NAMES:
        cells = [f"{summaries[n][m]['mean']:.3f} ± {summaries[n][m]['std']:.3f}" for m in METRICS]
        label = f"**{n}**" if n == best else n
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    with open(os.path.join(REPORTS, "cv_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    print("\n" + "\n".join(rows))
    print(f"\nMeilleur {SELECTION_METRIC} moyen : {best_by_mean} | modèle retenu (règle 1-SE, SE = {se:.3f}) : {selected}")
    print(f"Erreurs hors-pli de {best} : {errors}")


if __name__ == "__main__":
    # Backend sans fenêtre, uniquement quand le fichier est lancé comme script.
    # (Le faire à l'import coupait l'affichage des figures dans le notebook.)
    matplotlib.use("Agg")
    main()
