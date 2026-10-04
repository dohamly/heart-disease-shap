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
    NUMERIC_COLS, CATEGORICAL_COLS, load_data, build_preprocessed_dataset, split_and_scale,
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
cp = st.sidebar.selectbox("Chest pain type", [0, 1, 2, 3],
    format_func=lambda x: ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"][x])
trestbps = st.sidebar.slider("Resting blood pressure (mm Hg)", 90, 200, 120)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 240)
fbs = st.sidebar.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
restecg = st.sidebar.selectbox("Resting ECG", [0, 1, 2])
thalach = st.sidebar.slider("Max heart rate achieved", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise-induced angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

st.sidebar.markdown("## Additional Clinical Data")
oldpeak = st.sidebar.slider("ST depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
slope = st.sidebar.selectbox("Slope of peak exercise ST segment", [0, 1, 2])
ca = st.sidebar.selectbox("Major vessels colored (0-3)", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3])

predict_btn = st.sidebar.button("Predict Heart Disease Risk →")


def build_patient_row():
    raw = {"age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
           "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
           "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal}
    df_encoded = pd.get_dummies(pd.DataFrame([raw]), columns=CATEGORICAL_COLS)
    for col in columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[columns].astype("float64")
    numeric_present = [c for c in NUMERIC_COLS if c in df_encoded.columns]
    df_encoded[numeric_present] = scaler.transform(df_encoded[numeric_present])
    return df_encoded


LABELS = {"age": "Age", "sex": "Sex", "trestbps": "Resting Blood Pressure", "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar", "thalach": "Max Heart Rate", "exang": "Exercise-induced Angina",
    "oldpeak": "ST Depression", "ca": "Major Vessels",
    "cp_0": "Chest Pain: Typical angina", "cp_1": "Chest Pain: Atypical angina",
    "cp_2": "Chest Pain: Non-anginal", "cp_3": "Chest Pain: Asymptomatic",
    "restecg_0": "Resting ECG: Normal", "restecg_1": "Resting ECG: ST-T abnorm.", "restecg_2": "Resting ECG: LV hypertrophy",
    "slope_0": "ST Slope: Upsloping", "slope_1": "ST Slope: Flat", "slope_2": "ST Slope: Downsloping",
    "thal_0": "Thalassemia: Unknown", "thal_1": "Thalassemia: Fixed defect",
    "thal_2": "Thalassemia: Normal", "thal_3": "Thalassemia: Reversible"}

# ---------------------------------------------------------------- RESULT
if predict_btn:
    patient_df = build_patient_row()
    proba = model.predict_proba(patient_df)[0]
    pred = model.predict(patient_df)[0]

    col1, col2 = st.columns([1, 1.3])

    with col1:
        banner_class = "risk" if pred == 1 else "safe"
        label = "Heart disease detected" if pred == 1 else "No heart disease detected"
        st.markdown(f"""
        <div class="result-banner {banner_class}">
          <b>{label}</b><br>
          <span style="font-size:13px">Disease: {proba[1]*100:.1f}% · No disease: {proba[0]*100:.1f}%</span>
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
            explainer = shap.LinearExplainer(model, background)

        shap_values = explainer(patient_df)
        if len(shap_values.shape) == 3:
            values = shap_values.values[0, :, 1]
        else:
            values = shap_values.values[0]

        contrib = list(zip(patient_df.columns, values))
        contrib.sort(key=lambda t: -abs(t[1]))
        contrib = contrib[:9]
        max_abs = max(abs(v) for _, v in contrib) or 1

        bars_html = '<p style="font-size:12px;color:#6b7280;margin-bottom:10px">' \
                    '<span style="color:#e0546a">■</span> augmente le risque &nbsp; ' \
                    '<span style="color:#0f9488">■</span> réduit le risque</p>'
        for feat, val in contrib:
            pct = abs(val) / max_abs * 100
            color = "#e0546a" if val > 0 else "#0f9488"
            label_name = LABELS.get(feat, feat)
            bars_html += f"""<div class="bar-row"><span>{label_name}</span>
              <div class="bar-track"><div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div></div>
              <span style="color:{color};text-align:right">{'+' if val>0 else ''}{val:.2f}</span></div>"""
        st.markdown(bars_html, unsafe_allow_html=True)

    st.caption("⚠️ Présenté comme un exercice académique de prédiction / explicabilité, "
               "pas comme un outil médical de diagnostic.")
else:
    st.info("Renseigne les caractéristiques du patient dans le panneau de gauche, puis clique sur **Predict**.")
