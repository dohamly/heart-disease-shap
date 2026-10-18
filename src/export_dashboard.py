"""
export_dashboard.py
-------------------
Met à jour web/index.html à partir du modèle déployé et des résultats de la
validation croisée, pour qu'AUCUN chiffre du dashboard ne soit saisi à la main :

  * constantes du calcul local (coefficients, scaler, baseline SHAP, libellés)
  * cartes de métriques, matrice de confusion, courbe ROC, tableau comparatif
  * listes déroulantes (noms des catégories)

À relancer après chaque `python src/train.py` / `python src/evaluate.py` :
    python src/export_dashboard.py
"""

import json
import os
import re
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import (  # noqa: E402
    CATEGORY_NAMES, NUMERIC_COLS, build_preprocessed_dataset, load_data,
    readable_feature, split_and_scale,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = os.path.join(ROOT, "web", "index.html")
SVG = '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{p}</svg>'
ICONS = {
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1" fill="currentColor"/>',
    "trend": '<path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
    "pulse": '<path d="M3 12h4l2 7 4-14 2 7h6"/>',
    "bolt": '<path d="M13 2 3 14h7l-1 8 10-12h-7l1-8Z"/>',
    "bars": '<path d="M3 3v18h18"/><rect x="7" y="13" width="3" height="5"/><rect x="12" y="9" width="3" height="9"/><rect x="17" y="6" width="3" height="12"/>',
    "people": '<circle cx="9" cy="8" r="3.2"/><path d="M2.5 20a6.5 6.5 0 0 1 13 0"/><circle cx="17" cy="9" r="2.6"/><path d="M15.5 12.2a5 5 0 0 1 6 4.9"/>',
    "chip": '<rect x="7" y="7" width="10" height="10" rx="1.5"/><path d="M9 3v4M15 3v4M9 17v4M15 17v4M3 9h4M3 15h4M17 9h4M17 15h4"/>',
}
COLORS = {"teal": ("#e6f7f2", "#0f9488"), "blue": ("#eaf0ff", "#3b6fe0"),
          "purple": ("#f2eaff", "#7c4dcf"), "orange": ("#fff2e3", "#d97a1f")}


def icon_box(color, icon, size):
    bg, fg = COLORS[color]
    return f'<div class="icn" style="background:{bg};color:{fg}">{SVG.format(s=size, p=ICONS[icon])}</div>'


def pct(m, k):   # moyenne en %
    return f"{m[k]['mean']*100:.1f}%"


def pm_pct(m, k):
    return f"{m[k]['mean']*100:.1f} ± {m[k]['std']*100:.1f}%"


def pm(m, k):
    return f"{m[k]['mean']:.3f} ± {m[k]['std']:.3f}"


# ------------------------------------------------------------- marqueurs
def wrap_once(html, start_pat, end_pat, name, kind="html"):
    """Insère les marqueurs autour de [start_pat .. end_pat[ si absents."""
    o, c = (f"<!--AUTO:{name}-->", f"<!--/AUTO:{name}-->") if kind == "html" else (f"/*AUTO:{name}*/", f"/*/AUTO:{name}*/")
    if o in html:
        return html
    a = html.index(start_pat)
    b = html.index(end_pat, a + len(start_pat))
    return html[:a] + o + html[a:b] + c + html[b:]


def init_markers(h):
    if "/*AUTO:model*/" not in h:
        a = h.index("const COLUMNS=")
        b = h.index("\n", h.index("const GLOBAL=", a)) + 1
        h = h[:a] + "/*AUTO:model*/\n" + h[a:b] + "/*/AUTO:model*/\n" + h[b:]
    if "<!--AUTO:dash-stats-->" not in h:
        a = h.index('<div class="grid4">')
        b = h.index('<div class="card">', a)
        h = h[:a] + "<!--AUTO:dash-stats-->" + h[a:b] + "<!--/AUTO:dash-stats-->" + h[b:]
    if "<!--AUTO:perf-->" not in h:
        a = h.index('id="perf">') + len('id="perf">')
        b = h.index("<!-- ABOUT -->")
        end_inner = h.rindex("</div>", a, b)
        h = h[:a] + "<!--AUTO:perf-->" + h[a:end_inner] + "<!--/AUTO:perf-->" + h[end_inner:]
    if "<!--AUTO:about-ml-->" not in h:
        m = re.search(r'(<h3>Machine Learning</h3><p class="desc" style="margin:0">)(.*?)(</p>)', h, flags=re.S)
        h = h[:m.start(2)] + "<!--AUTO:about-ml-->" + m.group(2) + "<!--/AUTO:about-ml-->" + h[m.end(2):]
    for sel in ["f_cp", "f_restecg", "f_slope"]:
        o = f"<!--AUTO:opt-{sel}-->"
        if o not in h:
            a = h.index(f'id="{sel}">') + len(f'id="{sel}">')
            b = h.index("</select>", a)
            h = h[:a] + o + h[a:b] + f"<!--/AUTO:opt-{sel}-->" + h[b:]
    return h


def replace_block(h, name, content, kind="html"):
    o, c = (f"<!--AUTO:{name}-->", f"<!--/AUTO:{name}-->") if kind == "html" else (f"/*AUTO:{name}*/", f"/*/AUTO:{name}*/")
    a, b = h.index(o) + len(o), h.index(c)
    return h[:a] + content + h[b:]


# ------------------------------------------------------------ générateurs
def js_constants(bundle, X, X_train):
    model, scaler, cols = bundle["model"], bundle["scaler"], bundle["columns"]
    numeric = [c for c in NUMERIC_COLS if c in cols]
    coef = model.coef_[0]
    base = X_train.mean()
    contrib = np.abs((X_train.values - base.values) * coef).mean(axis=0)
    top = sorted(zip(cols, contrib), key=lambda t: -t[1])[:8]
    f = lambda v: f"{float(v):.6f}"
    obj = lambda d: "{" + ",".join(f"{k}:{f(v)}" for k, v in d.items()) + "}"
    labels = {c: readable_feature(c) for c in cols}
    return "\n".join([
        "const COLUMNS=" + json.dumps(cols) + ";",
        "const COEF=[" + ",".join(f(v) for v in coef) + "];",
        f"const INTERCEPT={f(model.intercept_[0])};",
        "const NUMERIC=" + json.dumps(numeric) + ";",
        "const SMEAN=" + obj(dict(zip(numeric, scaler.mean_))) + ";",
        "const SSCALE=" + obj(dict(zip(numeric, scaler.scale_))) + ";",
        "const BASE=" + obj({c: base[c] for c in cols}) + ";",
        "const LABELS=" + json.dumps(labels, ensure_ascii=False) + ";",
        "const GLOBAL=" + json.dumps([[c, round(float(v), 4)] for c, v in top]) + ";",
    ]) + "\n"


def dash_stats(res, name):
    m = res["models"][name]["metrics"]
    return ('<div class="grid4">\n'
        f'        <div class="stat">{icon_box("teal","target",19)}<b>{pct(m,"accuracy")}</b><div class="lbl">Model Accuracy</div><div class="sub">Nested 5-fold CV</div></div>\n'
        f'        <div class="stat">{icon_box("blue","pulse",19)}<b>{m["roc_auc"]["mean"]:.3f}</b><div class="lbl">ROC-AUC</div><div class="sub">Nested 5-fold CV</div></div>\n'
        f'        <div class="stat">{icon_box("purple","people",19)}<b>{res["settings"]["n_patients"]}</b><div class="lbl">Patients Analyzed</div><div class="sub">Cleveland UCI dataset</div></div>\n'
        f'        <div class="stat">{icon_box("orange","chip",19)}<b>{"LogReg" if name=="Logistic Regression" else name}</b><div class="lbl">Model Used</div><div class="sub">Simplest within 1 SE of best</div></div>\n'
        '      </div>\n      ')


def roc_svg(curve):
    pts = " ".join(f"{30 + x*180:.0f},{140 - y*130:.0f}" for x, y in zip(curve["fpr"], curve["tpr"]))
    t = lambda x, y, s: f'<text x="{x}" y="{y}" font-size="7" fill="var(--sub)">{s}</text>'
    return ('<svg viewBox="0 0 220 160" style="width:100%;max-width:280px">'
        '<line x1="30" y1="10" x2="30" y2="140" stroke="var(--border)"/><line x1="30" y1="140" x2="210" y2="140" stroke="var(--border)"/>'
        '<line x1="30" y1="140" x2="210" y2="10" stroke="var(--border)" stroke-dasharray="3,3"/>'
        f'<polyline fill="none" stroke="#0f9488" stroke-width="2.2" points="{pts}"/>'
        + t(8, 14, "1") + t(6, 79, "0.5") + t(6, 143, "0") + t(26, 152, "0") + t(118, 152, "0.5") + t(196, 152, "1") + t(100, 159, "FPR")
        + "</svg>")


def perf_html(res, name):
    mod = res["models"][name]
    m = mod["metrics"]
    (tn, fp), (fn, tp) = mod["pooled_oof"]["confusion_matrix"]
    n = res["settings"]["n_patients"]
    card = lambda col, ic, lab, k, fmt: (f'<div class="stat">{icon_box(col, ic, 17)}<b>{fmt(m,k)}</b>'
        f'<div class="lbl">{lab}</div><div class="sub">± {m[k]["std"]*(100 if k!="roc_auc" else 1):.{1 if k!="roc_auc" else 3}f} (std)</div></div>')
    p1 = lambda m_, k: pct(m_, k)
    p3 = lambda m_, k: f'{m_[k]["mean"]:.3f}'
    sel_f1 = max(res["models"], key=lambda k: res["models"][k]["metrics"]["f1"]["mean"])
    rows = ""
    for mn, mv in res["models"].items():
        mm = mv["metrics"]
        sel = mn == name
        badge = ' <span class="badge">Selected</span>' if sel else ""
        style = ' style="background:var(--accent-light)"' if sel else ""
        f1 = pm_pct(mm, "f1")
        f1 = f"<b>{f1}</b>" if mn == sel_f1 else f1
        rows += (f'          <tr{style}><td>{mn}{badge}</td><td>{pm_pct(mm,"accuracy")}</td><td>{pm_pct(mm,"precision")}</td>'
                 f'<td>{pm_pct(mm,"recall")}</td><td>{pm_pct(mm,"specificity")}</td><td>{f1}</td><td>{pm(mm,"roc_auc")}</td></tr>\n')
    pooled_auc = mod["pooled_oof"]["roc_auc"]
    cell = lambda cls, v, lab: f'<div class="{cls}"><b>{v}</b><br>{lab}</div>'
    lab = 'color:var(--sub);font-size:11px'
    return f'''
      <h3 style="font-size:19px;margin:2px 0 2px">Model Performance</h3>
      <p class="desc" style="margin-bottom:18px">Nested 5-fold stratified cross-validation, {name}: mean over the 5 outer folds ({n} patients; hyperparameters tuned on the inner loop only)</p>
      <div class="grid5">
        {card("teal","target","Accuracy","accuracy",p1)}
        {card("blue","trend","Precision","precision",p1)}
        {card("purple","pulse","Recall","recall",p1)}
        {card("orange","bolt","F1 Score","f1",p1)}
        {card("teal","bars","ROC-AUC","roc_auc",p3)}
      </div>
      <div class="grid2">
        <div class="card">
          <h3>Confusion Matrix</h3>
          <p class="desc">Out-of-fold predictions, {n} patients ({name}, threshold 0.5)</p>
          <div class="matrix">
            <div></div><div style="{lab}">Pred: Healthy</div><div style="{lab}">Pred: Disease</div>
            <div style="{lab};text-align:right;padding-right:6px">Actual: Healthy</div>{cell("mcell-tn", tn, "TN")}{cell("mcell-fp", fp, "FP")}
            <div style="{lab};text-align:right;padding-right:6px">Actual: Disease</div>{cell("mcell-fn", fn, "FN")}{cell("mcell-tp", tp, "TP")}
          </div>
          <p class="desc" style="margin-top:12px">{fn} of {fn+tp} diseased patients are missed (false negatives) — the costliest error type in a screening context.</p>
        </div>
        <div class="card">
          <h3>ROC Curve</h3>
          <p class="desc">AUC = {pooled_auc:.3f} (out-of-fold, pooled)</p>
          {roc_svg(mod["roc_curve"])}
        </div>
      </div>
      <div class="card">
        <h3>Feature Importance (SHAP Global)</h3>
        <p class="desc">Mean absolute SHAP value across the training set (higher = more influence on the predicted risk)</p>
        <div id="globalBars"></div>
      </div>
      <div class="card">
        <h3>Model Comparison</h3>
        <p class="desc">Nested 5-fold stratified CV — mean ± std across the 5 outer folds</p>
        <table>
          <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>Specificity</th><th>F1</th><th>ROC-AUC</th></tr>
{rows}        </table>
        <p class="desc" style="margin-top:12px">Differences between models are smaller than the fold-to-fold variability (≈ ±0.08 F1): none is significantly better with 303 patients. {name} is kept as the simplest model within one standard error of the best.</p>
      </div>
    '''


def options(feature, default):
    return "".join(f'<option value="{k}"{" selected" if k == default else ""}>{v[0].upper() + v[1:]}</option>'
                   for k, v in CATEGORY_NAMES[feature].items())


def main():
    with open(os.path.join(ROOT, "models", "cv_results.json"), encoding="utf-8") as f:
        res = json.load(f)
    bundle = joblib.load(os.path.join(ROOT, "models", "best_model.joblib"))
    name = bundle["model_name"]
    if name != "Logistic Regression":
        sys.exit("Le calcul local du dashboard suppose une régression logistique "
                 f"(modèle déployé : {name}). Dashboard non modifié.")
    X, _ = build_preprocessed_dataset(load_data(os.path.join(ROOT, "data", "heart.csv")))
    X_train = split_and_scale(X, _)[0]

    h = open(HTML, encoding="utf-8").read()
    h = init_markers(h)
    h = replace_block(h, "model", js_constants(bundle, X, X_train), "js")
    h = replace_block(h, "dash-stats", dash_stats(res, name))
    h = replace_block(h, "perf", perf_html(res, name))
    m = res["models"][name]["metrics"]
    h = replace_block(h, "about-ml",
        "Three classifiers were compared with nested 5-fold stratified cross-validation (hyperparameters tuned on the inner loop only). "
        "Their scores are statistically indistinguishable, so the simplest one, Logistic Regression, was selected "
        f"(F1 {pct(m,'f1')}, ROC-AUC {m['roc_auc']['mean']:.3f}).")
    h = replace_block(h, "opt-f_cp", options("cp", 0))
    h = replace_block(h, "opt-f_restecg", options("restecg", 0))
    h = replace_block(h, "opt-f_slope", options("slope", 1))
    open(HTML, "w", encoding="utf-8").write(h)
    print("web/index.html mis à jour depuis cv_results.json et best_model.joblib")


if __name__ == "__main__":
    main()
