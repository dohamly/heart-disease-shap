"""
app.py
------
Mini-application Streamlit : l'utilisateur saisit les caractéristiques
d'un patient et obtient une prédiction + une explication SHAP.
Thème visuel aligné sur web/index.html (HeartAI).

Projet académique de démonstration — ne constitue en aucun cas un outil
de diagnostic médical.
"""

import streamlit as st
import pandas as pd
import shap
import joblib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import (
    NUMERIC_COLS, CATEGORICAL_COLS, CATEGORY_NAMES, load_data, build_preprocessed_dataset,
    split_and_scale,
)
from shap_utils import (
    CAVEAT, DISCLAIMER, aggregate_by_variable, base_probability, describe_patient, output_space,
    sigmoid, value_label, variable_label,
)

st.set_page_config(page_title="HeartAI — Explainable Prediction", page_icon="🫀", layout="wide")

# ---------------------------------------------------------------- THEME CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Fraunces:wght@700;800&display=swap" rel="stylesheet">
<style>
:root{
  --accent:#0f9488; --accent-light:#e6f7f5; --navy1:#0f172a; --navy2:#134e4a;
  --red:#e0546a; --red-bg:#fdecef; --green-bg:#e6f7f5;
}
.block-container{padding-top:2rem; max-width:1100px}
h1, .hero-title{font-family:'Fraunces',Georgia,serif !important; font-weight:800 !important}
[data-testid="stSidebar"]{background:#ffffff; border-right:1px solid #e5e9ee}
[data-testid="stSidebar"] h2{font-size:15px !important}
.stButton>button{background:var(--accent); color:#fff; border:none; border-radius:10px;
  padding:0.6rem 1.2rem; font-weight:700; width:100%}
.stButton>button:hover{background:#0b6f66; color:#fff}
.hero{border-radius:16px; padding:34px; background:linear-gradient(135deg,var(--navy1),var(--navy2));
  color:#fff; margin-bottom:22px}
.hero .tag{font-size:11px; letter-spacing:.06em; color:#7dd3c8; text-transform:uppercase; font-weight:600}
.hero h1{color:#fff !important; font-size:34px !important; margin:10px 0 8px !important; line-height:1.15}
.hero h1 span{color:#5eead4}
.hero p{color:#c9d3dc; font-size:14px; max-width:560px; margin:0}
.stat-card{background:#fff; border:1px solid #e5e9ee; border-radius:12px; padding:16px}
.stat-card .icn{width:34px; height:34px; border-radius:9px; display:flex; align-items:center;
  justify-content:center; font-size:16px; margin-bottom:8px}
.stat-card b{font-size:21px; display:block}
.stat-card .lbl{font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:.03em; margin-top:2px}
.result-banner{border-radius:12px; padding:18px 20px; margin:6px 0 16px}
.result-banner.risk{background:var(--red-bg); color:var(--red)}
.result-banner.safe{background:var(--green-bg); color:var(--accent)}
.result-banner b{font-size:17px}
.bar-row{display:grid; grid-template-columns:190px 1fr 60px; align-items:center; gap:10px;
  margin-bottom:9px; font-size:13px}
.bar-track{height:15px; background:#f4f6f8; border-radius:5px; overflow:hidden}
.bar-fill{height:100%; border-radius:5px}
.wf-head,.wf-row{display:grid; grid-template-columns:210px 1fr 52px 58px; align-items:center; gap:10px}
.wf-head{font-size:10.5px; color:#6b7280; text-transform:uppercase; letter-spacing:.04em; margin:12px 0 6px; font-weight:600}
.wf-head span:nth-child(n+3), .wf-x, .wf-p{text-align:right}
.wf-row{margin-bottom:7px; font-size:12.5px}
.wf-row.edge{font-weight:600; padding:5px 0; border-top:1px dashed #e5e9ee}
.wf-row.edge.first{border-top:none; border-bottom:1px dashed #e5e9ee}
.wf-track{position:relative; height:16px; background:#f4f6f8; border-radius:5px}
.wf-bar{position:absolute; top:0; height:100%; border-radius:4px; min-width:3px}
.wf-dot{position:absolute; top:-3px; width:4px; height:22px; background:#111827; border-radius:2px; transform:translateX(-2px)}
.wf-p{color:#6b7280}
.note{background:var(--accent-light); border:1px solid #e5e9ee; border-radius:10px; padding:12px 14px;
  font-size:12.5px; line-height:1.55; margin-top:14px}
.section-tag{font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:.04em;
  font-weight:700; margin:14px 0 2px}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- MODEL
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.joblib")

bundle = load_model()
model, scaler, columns, model_name = bundle["model"], bundle["scaler"], bundle["columns"], bundle["model_name"]


@st.cache_resource
def load_background():
    """Référence (train set) pour SHAP — sans ça LinearExplainer prend le
    patient lui-même comme baseline et toutes les contributions tombent à 0."""
    df = load_data()
    X, y = build_preprocessed_dataset(df)
    X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)
    return X_train


background = load_background()

# ---------------------------------------------------------------- HERO
st.markdown(f"""
<div class="hero">
  <div class="tag">AI Healthcare · Portfolio Project</div>
  <h1>Heart Disease<br><span>Prediction</span></h1>
  <p>Cette application prédit le risque de maladie cardiaque coronarienne à partir des données
  cliniques d'un patient, et explique chaque prédiction avec SHAP. Modèle : <b style="color:#fff">{model_name}</b>.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- SIDEBAR FORM
st.sidebar.markdown("## Patient Information")
age = st.sidebar.slider("Age", 20, 90, 50)
sex = st.sidebar.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")

st.sidebar.markdown("## Clinical Information")
cp = st.sidebar.selectbox("Chest pain type", [0, 1, 2, 3], format_func=lambda x: CATEGORY_NAMES["cp"][x].capitalize())
trestbps = st.sidebar.slider("Resting blood pressure (mm Hg)", 90, 200, 120)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 240)
fbs = st.sidebar.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
restecg = st.sidebar.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: CATEGORY_NAMES["restecg"][x].capitalize())
thalach = st.sidebar.slider("Max heart rate achieved", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise-induced angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

st.sidebar.markdown("## Additional Clinical Data")
oldpeak = st.sidebar.slider("ST depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
slope = st.sidebar.selectbox("Slope of peak exercise ST segment", [0, 1, 2], index=1, format_func=lambda x: CATEGORY_NAMES["slope"][x].capitalize())
ca = st.sidebar.selectbox("Major vessels colored (0-3)", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3], index=2, format_func=lambda x: CATEGORY_NAMES["thal"][x].capitalize())

predict_btn = st.sidebar.button("Predict Heart Disease Risk →")


def raw_patient():
    return {"age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
           "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
           "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal}


def build_patient_row():
    raw = raw_patient()
    df_encoded = pd.get_dummies(pd.DataFrame([raw]), columns=CATEGORICAL_COLS)
    for col in columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[columns].astype("float64")
    numeric_present = [c for c in NUMERIC_COLS if c in df_encoded.columns]
    df_encoded[numeric_present] = scaler.transform(df_encoded[numeric_present])
    return df_encoded


# ---------------------------------------------------------------- RESULT
if predict_btn:
    patient_df = build_patient_row()
    proba = model.predict_proba(patient_df)[0]
    pred = model.predict(patient_df)[0]

    col1, col2 = st.columns([1, 1.3])

    with col1:
        banner_class = "risk" if pred == 1 else "safe"
        label = "Heart disease likely (model estimate)" if pred == 1 else "Heart disease unlikely (model estimate)"
        st.markdown(f"""
        <div class="result-banner {banner_class}">
          <b>{label}</b><br>
          <span style="font-size:13px">Estimated probability: {proba[1]*100:.1f}% · complement: {proba[0]*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="stat-card"><div class="icn" style="background:#e6f7f2;color:#0f9488">◎</div>
            <b>{proba[1]*100:.1f}%</b><div class="lbl">Disease probability</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="stat-card"><div class="icn" style="background:#eaf0ff;color:#3b6fe0">◉</div>
            <b>{model_name}</b><div class="lbl">Model used</div></div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-tag">Why this prediction — SHAP</div>', unsafe_allow_html=True)

        if type(model).__name__ in ("RandomForestClassifier", "XGBClassifier"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(
                model, shap.maskers.Independent(background, max_samples=len(background))
            )
        sv = explainer(patient_df)
        if sv.values.ndim == 3:
            values, base = sv.values[0, :, 1], float(sv.base_values[0, 1])
        else:
            values, base = sv.values[0], float(sv.base_values[0])

        raw = raw_patient()
        by_var = aggregate_by_variable(pd.DataFrame([values], columns=patient_df.columns)).iloc[0]
        base_p = base_probability(model, base)
        linear = output_space(model) == "log-odds"
        order = by_var.sort_values(key=abs, ascending=False)
        head, tail = order.iloc[:8], order.iloc[8:]

        steps = [(f"{variable_label(v)}: {value_label(v, raw[v])}", float(x)) for v, x in head.items()]
        if len(tail):
            steps.append((f"{len(tail)} autres variables", float(tail.sum())))
        cum, rows = base, []
        for lab, x in steps:
            rows.append((lab, x, cum, cum + x))
            cum += x
        pts = [base, cum] + [r[2] for r in rows] + [r[3] for r in rows]
        lo, hi = min(pts), max(pts)
        pad = (hi - lo) * 0.06 or 0.1
        lo, hi = lo - pad, hi + pad
        pos = lambda v: (v - lo) / (hi - lo) * 100
        prob = lambda v: f"{sigmoid(v) * 100:.0f} %" if linear else "—"

        html = ('<p style="font-size:12px;color:#6b7280;margin-bottom:4px"><span style="color:#e0546a">■</span> '
                'pousse vers « maladie » &nbsp; <span style="color:#0f9488">■</span> pousse vers « sain »</p>'
                '<div class="wf-head"><span>Variable : valeur du patient</span><span></span><span>Effet</span>'
                '<span>Prob. cumulée</span></div>')
        html += (f'<div class="wf-row edge first"><span>Patient moyen (valeur de base)</span><div class="wf-track">'
                 f'<div class="wf-dot" style="left:{pos(base):.2f}%"></div></div><span></span>'
                 f'<span class="wf-p"><b>{base_p * 100:.0f} %</b></span></div>')
        for lab, x, a_, b_ in rows:
            color = "#e0546a" if x > 0 else "#0f9488"
            html += (f'<div class="wf-row"><span>{lab}</span><div class="wf-track"><div class="wf-bar" '
                     f'style="left:{pos(min(a_, b_)):.2f}%;width:{abs(b_ - a_) / (hi - lo) * 100:.2f}%;background:{color}"></div></div>'
                     f'<span class="wf-x" style="color:{color}">{x:+.2f}</span><span class="wf-p">{prob(b_)}</span></div>')
        html += (f'<div class="wf-row edge"><span>Prédiction du modèle</span><div class="wf-track">'
                 f'<div class="wf-dot" style="left:{pos(cum):.2f}%"></div></div><span></span>'
                 f'<span class="wf-p"><b>{proba[1] * 100:.0f} %</b></span></div>')
        html += ('<div class="note"><b>Comment lire ce graphique.</b> On part du patient moyen (en haut). Chaque barre '
                 'déplace la probabilité estimée : rouge vers « maladie », vert vers « sain », et la colonne de droite '
                 'donne la probabilité après cette étape. Les effets s\'additionnent exactement jusqu\'à la prédiction du '
                 'modèle (en bas). Les longueurs sont en log-odds, pas en points de pourcentage.</div>')
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("#### En clair")
    st.markdown(describe_patient(raw, by_var, float(proba[1]), base_p))
    st.warning(f"**{CAVEAT}** {DISCLAIMER}")
else:
    st.info("Renseigne les caractéristiques du patient dans le panneau de gauche, puis clique sur **Predict**.")
