"""
shap_plots.py
-------------
Waterfall SHAP officiel (shap.plots.waterfall) regroupé par variable clinique,
partagé par explain.py (figures du README) et app.py (Streamlit), pour que les
deux produisent exactement le même graphique.

SHAP explique le comportement du modèle, pas une relation causale médicale.
"""

import threading

import matplotlib.pyplot as plt
import shap

from shap_utils import value_label, variable_label

# pyplot est un état global : on sérialise la fabrication des figures pour que deux
# sessions Streamlit simultanées ne dessinent pas dans la même figure.
_LOCK = threading.Lock()


def waterfall_figure(contrib, base, raw, title, p, base_p, max_display=9):
    """
    contrib : pd.Series de contributions SHAP par variable clinique (log-odds)
    base    : valeur de base du modèle (même unité)
    raw     : valeurs brutes du patient (dict ou Series)
    p, base_p : probabilité estimée du patient et du patient moyen
    Retourne une figure matplotlib (à fermer avec plt.close après usage).
    """
    contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
    names = [f"{variable_label(v)}: {value_label(v, raw[v])}" for v in contrib.index]
    ex = shap.Explanation(values=contrib.values, base_values=float(base), data=None, feature_names=names)
    with _LOCK:
        fig = plt.figure()
        shap.plots.waterfall(ex, max_display=max_display, show=False)
        plt.title(f"{title}\nProbabilité estimée {p * 100:.0f} %  (patient moyen : {base_p * 100:.0f} %)",
                  fontsize=10, loc="left")
        plt.tight_layout()
    return fig


def save_waterfall(path, *args, **kwargs):
    fig = waterfall_figure(*args, **kwargs)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
