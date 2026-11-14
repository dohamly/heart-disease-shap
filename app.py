"""
Streamlit app: enter a patient, get the prediction and its SHAP explanation.

The result updates automatically whenever a field changes (no button to click). For each
patient the app shows:
  - the predicted class, the probability of disease and the complementary probability;
  - the SHAP waterfall plot of this patient;
  - the main influential features, with a short written interpretation.

Educational and research use only: this is not a medical diagnostic tool. SHAP explains the
behaviour of the model, not a medical cause-and-effect relationship.

Run with:  streamlit run app.py
"""

import inspect
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from app_ui import (  # noqa: E402
    STYLE_HTML, cascade_html, class_color, header_html, influence_rows_html, kpi_card, meter_html,
)
from inference import (  # noqa: E402
    DATA_PATH, MODEL_PATH, encode_patients, load_bundle, make_explainer, training_background,
    unpack_explanation,
)
from preprocessing import CATEGORY_NAMES, load_data  # noqa: E402
from shap_plots import waterfall_figure  # noqa: E402
from shap_utils import (  # noqa: E402
    CAVEAT, DISCLAIMER, aggregate_by_variable, base_probability, describe_patient, output_space,
)

st.set_page_config(page_title="HeartAI — Prédiction et explication SHAP", page_icon="🫀", layout="wide")

# Two TEST-set patients (never seen during training): the ones explained in the README.
PRESET_ROWS = {"high": 191, "low": 162}
DEFAULTS = dict(age=50, sex=1, cp=3, trestbps=120, chol=240, fbs=0, restecg=0, thalach=150,
                exang=0, oldpeak=1.0, slope=1, ca=0, thal=2)
RANGES = dict(age=(20, 90), trestbps=(90, 200), chol=(100, 600), thalach=(60, 220), oldpeak=(0.0, 6.0))
METRIC_KEYS = ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc")
METRIC_LABELS = ["Accuracy", "Precision", "Recall", "Specificity", "F1-score", "ROC-AUC", "PR-AUC"]

st.markdown(STYLE_HTML, unsafe_allow_html=True)


def stretch(widget):
    """Full-width keyword for `widget`, whichever Streamlit version is installed."""
    return {"width": "stretch"} if "width" in inspect.signature(widget).parameters else {"use_container_width": True}


@st.cache_resource
def load_resources():
    """Load the model, the dataset, the SHAP explainer and the cross-validation results once."""
    bundle = load_bundle(MODEL_PATH)
    explainer = make_explainer(bundle["model"], training_background())
    cv_path = os.path.join(ROOT, "models", "cv_results.json")
    cv_results = json.load(open(cv_path, encoding="utf-8")) if os.path.exists(cv_path) else None
    return bundle, load_data(DATA_PATH), explainer, cv_results


bundle, dataset, explainer, cv_results = load_resources()
model, model_name = bundle["model"], bundle["model_name"]


def preset_from_row(row_index):
    """Form values of a dataset row, clipped to the ranges of the input widgets."""
    row = dataset.loc[row_index]
    values = {}
    for field in DEFAULTS:
        value = float(row[field]) if field == "oldpeak" else int(row[field])
        if field in RANGES:
            value = min(max(value, RANGES[field][0]), RANGES[field][1])
        values[field] = value
    return values


def apply_values(values):
    """Button callback: write `values` into the widget state."""
    for field, value in values.items():
        st.session_state[field] = value


def yes_no(flag):
    """Display name of a 0/1 flag."""
    return "Yes" if flag == 1 else "No"


def sex_label(code):
    """Display name of the sex code."""
    return "Male" if code == 1 else "Female"


def category_formatter(variable):
    """Return a formatter showing the readable name of a category code of `variable`."""
    def format_code(code):
        return CATEGORY_NAMES[variable][code].capitalize()
    return format_code


for field, default in DEFAULTS.items():
    st.session_state.setdefault(field, default)

# ------------------------------------------------------------------ patient form
sidebar = st.sidebar
sidebar.markdown("### Exemples")
high_column, low_column = sidebar.columns(2)
high_column.button("Risque élevé", on_click=apply_values, args=(preset_from_row(PRESET_ROWS["high"]),), **stretch(st.button))
low_column.button("Risque faible", on_click=apply_values, args=(preset_from_row(PRESET_ROWS["low"]),), **stretch(st.button))
sidebar.button("Réinitialiser", on_click=apply_values, args=(DEFAULTS,), **stretch(st.button))

sidebar.markdown("### Patient")
sidebar.slider("Age", *RANGES["age"], key="age")
sidebar.selectbox("Sex", [1, 0], key="sex", format_func=sex_label)
sidebar.markdown("### Clinique")
sidebar.selectbox("Chest pain type", [0, 1, 2, 3], key="cp", format_func=category_formatter("cp"))
sidebar.slider("Resting blood pressure (mm Hg)", *RANGES["trestbps"], key="trestbps")
sidebar.slider("Cholesterol (mg/dl)", *RANGES["chol"], key="chol")
sidebar.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1], key="fbs", format_func=yes_no)
sidebar.selectbox("Resting ECG", [0, 1, 2], key="restecg", format_func=category_formatter("restecg"))
sidebar.slider("Max heart rate achieved", *RANGES["thalach"], key="thalach")
sidebar.selectbox("Exercise-induced angina", [0, 1], key="exang", format_func=yes_no)
sidebar.markdown("### Examens complémentaires")
sidebar.slider("ST depression (oldpeak)", *RANGES["oldpeak"], step=0.1, key="oldpeak")
sidebar.selectbox("Slope of peak exercise ST segment", [0, 1, 2], key="slope", format_func=category_formatter("slope"))
sidebar.selectbox("Major vessels colored (0-3)", [0, 1, 2, 3], key="ca")
sidebar.selectbox("Thalassemia", [0, 1, 2, 3], key="thal", format_func=category_formatter("thal"))

patient = {field: st.session_state[field] for field in DEFAULTS}

# ------------------------------------------------------------------ prediction and explanation (on every change)
patient_features = encode_patients(patient, bundle)
probability = float(model.predict_proba(patient_features)[0, 1])
predicted_class = int(model.predict(patient_features)[0])

all_values, base_value = unpack_explanation(explainer(patient_features))
contribution_by_variable = aggregate_by_variable(
    pd.DataFrame([all_values[0]], columns=patient_features.columns)).iloc[0]
average_patient_probability = base_probability(model, base_value)
is_log_odds = output_space(model) == "log-odds"

st.markdown(header_html(DISCLAIMER), unsafe_allow_html=True)

# ------------------------------------------------------------------ result cards
risk_class = "risk" if predicted_class == 1 else "safe"
class_name = "Maladie probable" if predicted_class == 1 else "Maladie peu probable"
kpi_class, kpi_probability, kpi_complement, kpi_model = st.columns(4)
kpi_class.markdown(kpi_card("Classe prédite", class_name, f"classe {predicted_class} (seuil de décision : 50 %)", css=risk_class),
                   unsafe_allow_html=True)
kpi_probability.markdown(
    kpi_card("Probabilité de maladie", f"{probability * 100:.1f} %",
             meter=meter_html(probability, class_color(predicted_class))),
    unsafe_allow_html=True)
kpi_complement.markdown(kpi_card("Probabilité d'absence de maladie", f"{(1 - probability) * 100:.1f} %", "complément à 100 %"),
                        unsafe_allow_html=True)
kpi_model.markdown(kpi_card("Modèle", model_name, f"patient moyen : {average_patient_probability * 100:.0f} %",
                            value_style="font-size:18px"), unsafe_allow_html=True)

# ------------------------------------------------------------------ waterfall + influential features
waterfall_column, influence_column = st.columns([1.15, 1], gap="large")
with waterfall_column:
    st.markdown('<h4 class="sec">SHAP Waterfall Plot de ce patient</h4>', unsafe_allow_html=True)
    figure = waterfall_figure(contribution_by_variable, base_value, patient, "Patient saisi dans le formulaire",
                              probability, average_patient_probability)
    st.pyplot(figure, **stretch(st.pyplot))
    plt.close(figure)
    st.caption("On part du patient moyen (valeur de base, en bas) ; chaque barre ajoute la contribution d'une variable "
               "(rouge : vers « maladie », bleu : vers « sain »). En haut, f(x) est la sortie du modèle pour ce patient, "
               "en log-odds.")

with influence_column:
    st.markdown('<h4 class="sec">Principales caractéristiques influentes</h4>', unsafe_allow_html=True)
    st.markdown(influence_rows_html(contribution_by_variable, patient), unsafe_allow_html=True)
    st.markdown('<div class="note"><b>Lecture.</b> Les barres rouges augmentent la probabilité de maladie estimée par le modèle, '
                'les vertes la réduisent. La valeur est la contribution SHAP en log-odds.</div>', unsafe_allow_html=True)
    st.markdown("#### En clair")
    st.markdown(describe_patient(patient, contribution_by_variable, probability, average_patient_probability))

# ------------------------------------------------------------------ details
tab_cascade, tab_shap, tab_model = st.tabs(["Cascade en probabilité", "Comprendre SHAP", "À propos du modèle"])

with tab_cascade:
    st.markdown(cascade_html(contribution_by_variable, patient, base_value, average_patient_probability, probability,
                             is_log_odds), unsafe_allow_html=True)
    st.caption("Même explication que le waterfall, traduite en probabilité : la colonne de droite donne la probabilité "
               "estimée après chaque étape.")

with tab_shap:
    st.markdown(
        "- SHAP répartit l'écart entre la prédiction de ce patient et celle d'un **patient moyen** du jeu d'entraînement.\n"
        "- Les contributions sont **additives** : valeur de base + somme des contributions = sortie du modèle.\n"
        "- Les colonnes encodées d'une même variable (type de douleur, thalassémie…) sont additionnées : "
        "**une barre par variable clinique**.")
    st.warning(f"**{CAVEAT}**")

with tab_model:
    st.markdown(f"**Modèle déployé :** {model_name} — hyperparamètres : `{bundle.get('hyperparameters')}`")
    if cv_results:
        metrics = cv_results["models"][cv_results["selected_model"]]["metrics"]
        st.markdown("Performance en validation croisée imbriquée (5 plis, 303 patients), moyenne ± écart-type :")
        st.dataframe(pd.DataFrame({
            "Métrique": METRIC_LABELS,
            "Valeur": [f"{metrics[key]['mean']:.3f} ± {metrics[key]['std']:.3f}" for key in METRIC_KEYS],
        }), hide_index=True, **stretch(st.dataframe))
    st.markdown("Données : UCI Heart Disease (Cleveland), 303 patients. Les trois modèles testés sont statistiquement "
                "équivalents ; la régression logistique est retenue pour sa simplicité et son explicabilité exacte.")

st.caption(f"{DISCLAIMER}")
