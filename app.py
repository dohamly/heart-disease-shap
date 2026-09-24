"""
app.py
------
Mini-application Streamlit : l'utilisateur saisit les caractéristiques
d'un patient et obtient une prédiction + une explication SHAP.

Projet académique de démonstration — ne constitue en aucun cas un outil
de diagnostic médical.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import NUMERIC_COLS, CATEGORICAL_COLS, BINARY_COLS

st.set_page_config(page_title="Heart Disease Prediction + SHAP", page_icon="🫀", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("models/best_model.joblib")


bundle = load_model()
model = bundle["model"]
scaler = bundle["scaler"]
columns = bundle["columns"]
model_name = bundle["model_name"]

st.title("🫀 Heart Disease Prediction + Explainable AI")
st.caption(
    "Projet académique de démonstration (dataset UCI Heart Disease) — "
    "ceci n'est PAS un outil de diagnostic médical."
)

st.markdown(f"**Modèle utilisé :** {model_name}")

st.sidebar.header("Caractéristiques du patient")

age = st.sidebar.slider("Age", 20, 90, 50)
sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme")
cp = st.sidebar.selectbox(
    "Chest pain type", [0, 1, 2, 3],
    format_func=lambda x: ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"][x],
)
trestbps = st.sidebar.slider("Resting blood pressure (mm Hg)", 90, 200, 120)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 240)
fbs = st.sidebar.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
restecg = st.sidebar.selectbox("Resting ECG", [0, 1, 2])
thalach = st.sidebar.slider("Max heart rate achieved", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise-induced angina", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
oldpeak = st.sidebar.slider("ST depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
slope = st.sidebar.selectbox("Slope of peak exercise ST segment", [0, 1, 2])
ca = st.sidebar.selectbox("Major vessels colored (0-3)", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3])

predict_btn = st.sidebar.button("Predict", type="primary")


def build_patient_row():
    raw = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }
    df_raw = pd.DataFrame([raw])
    df_encoded = pd.get_dummies(df_raw, columns=CATEGORICAL_COLS)

    # Aligner sur les colonnes vues à l'entraînement
    for col in columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[columns].astype("float64")

    numeric_present = [c for c in NUMERIC_COLS if c in df_encoded.columns]
    df_encoded[numeric_present] = scaler.transform(df_encoded[numeric_present])

    return df_encoded


if predict_btn:
    patient_df = build_patient_row()
    proba = model.predict_proba(patient_df)[0]
    pred = model.predict(patient_df)[0]

    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.subheader("Résultat")
        if pred == 1:
            st.error(f"⚠️ Heart disease detected — probability: {proba[1]*100:.1f}%")
        else:
            st.success(f"✅ No heart disease detected — probability: {proba[0]*100:.1f}%")

        st.write("**Probabilities**")
        st.progress(float(proba[1]))
        st.write(f"Disease: {proba[1]*100:.1f}% | No disease: {proba[0]*100:.1f}%")

    with col2:
        st.subheader("Pourquoi cette prédiction ? (SHAP)")
        if type(model).__name__ in ("RandomForestClassifier", "XGBClassifier"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, patient_df)

        shap_values = explainer(patient_df)
        if len(shap_values.shape) == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        fig = plt.figure()
        shap.plots.waterfall(sv, show=False)
        st.pyplot(fig, use_container_width=True)

    st.caption(
        "⚠️ Présenté comme un exercice académique de prédiction / explicabilité, "
        "pas comme un outil médical de diagnostic."
    )
else:
    st.info("Renseigne les caractéristiques du patient dans le panneau de gauche, puis clique sur **Predict**.")
