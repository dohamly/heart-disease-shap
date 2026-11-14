"""
Tests de l'application Streamlit (app.py) avec le testeur intégré de Streamlit.

Ils vérifient ce que l'application promet : résultat immédiat et automatique, classe
prédite + probabilités, SHAP Waterfall Plot affiché pour chaque patient, principales
caractéristiques influentes, exemples prêts à l'emploi et avertissements visibles.
"""

import os
import re

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")


def run_app():
    return AppTest.from_file(APP, default_timeout=180).run()


def html(at):
    return " ".join(m.value for m in at.markdown)


def disease_probability(at):
    m = re.search(r"Probabilité de maladie</div><div class=\"val\">([\d.]+) %", html(at))
    assert m, "la probabilité de maladie n'est pas affichée"
    return float(m.group(1))


def click(at, label):
    next(b for b in at.sidebar.button if b.label == label).click()
    return at.run()


def test_result_is_displayed_immediately_without_any_click():
    at = run_app()
    assert not at.exception
    text = html(at)
    assert "Classe prédite" in text
    assert "Probabilité d'absence de maladie" in text
    assert 0 <= disease_probability(at) <= 100


def test_shap_waterfall_is_displayed_automatically_for_the_patient():
    at = run_app()
    assert len(at.get("image")) >= 1                      # la figure matplotlib du waterfall
    assert "SHAP Waterfall Plot de ce patient" in html(at)


def test_main_influential_features_are_listed():
    at = run_app()
    text = html(at)
    assert "Principales caractéristiques influentes" in text
    assert text.count('class="infl"') >= 5


def test_result_updates_automatically_when_an_input_changes():
    at = run_app()
    before = disease_probability(at)
    at.slider(key="oldpeak").set_value(4.5)
    at.selectbox(key="ca").select(3)
    at.run()                                              # aucun bouton « Predict » à cliquer
    assert not at.exception
    assert disease_probability(at) > before + 10


def test_example_buttons_load_a_high_and_a_low_risk_patient():
    at = click(run_app(), "Risque élevé")
    assert not at.exception
    assert "Maladie probable" in html(at) and disease_probability(at) >= 85

    at = click(at, "Risque faible")
    assert "Maladie peu probable" in html(at) and disease_probability(at) <= 20


def test_disclaimers_are_visible():
    at = run_app()
    text = html(at).lower()
    assert "à visée éducative et de recherche" in text
    assert "pas un outil de diagnostic médical" in text
    assert any("relation de cause à effet" in w.value for w in at.warning)   # SHAP n'est pas causal
