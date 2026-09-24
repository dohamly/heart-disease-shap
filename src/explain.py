"""
explain.py
----------
Génère les visualisations SHAP (summary plot, bar plot, waterfall plot)
pour expliquer les prédictions du meilleur modèle sauvegardé.
"""

import joblib
import shap
import matplotlib.pyplot as plt

from preprocessing import load_data, build_preprocessed_dataset, split_and_scale


def load_artifacts(path="models/best_model.joblib"):
    bundle = joblib.load(path)
    return bundle["model"], bundle["scaler"], bundle["columns"], bundle["model_name"]


def get_explainer(model, X_train):
    """Choisit le bon explainer SHAP selon le type de modèle."""
    model_type = type(model).__name__
    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        return shap.TreeExplainer(model)
    return shap.LinearExplainer(model, X_train)


def main():
    df = load_data()
    X, y = build_preprocessed_dataset(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    model, _, columns, model_name = load_artifacts()
    print(f"Explication du modèle : {model_name}")

    explainer = get_explainer(model, X_train)
    shap_values = explainer(X_test)

    # Pour la classification binaire avec TreeExplainer, shap_values peut
    # avoir une dimension supplémentaire (classe 0 / classe 1)
    if len(shap_values.shape) == 3:
        shap_values_plot = shap_values[:, :, 1]
    else:
        shap_values_plot = shap_values

    # 1. Summary plot (impact global de chaque feature)
    plt.figure()
    shap.summary_plot(shap_values_plot, X_test, show=False)
    plt.tight_layout()
    plt.savefig("models/shap_summary_plot.png", dpi=150)
    plt.close()

    # 2. Bar plot (importance moyenne absolue)
    plt.figure()
    shap.summary_plot(shap_values_plot, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("models/shap_bar_plot.png", dpi=150)
    plt.close()

    # 3. Waterfall plot pour une prédiction individuelle (le 1er patient du test set)
    plt.figure()
    shap.plots.waterfall(shap_values_plot[0], show=False)
    plt.tight_layout()
    plt.savefig("models/shap_waterfall_plot.png", dpi=150)
    plt.close()

    print("Graphiques SHAP sauvegardés dans models/ :")
    print(" - shap_summary_plot.png")
    print(" - shap_bar_plot.png")
    print(" - shap_waterfall_plot.png")


def explain_single_patient(patient_df):
    """
    Utilisé par l'app Streamlit : retourne les valeurs SHAP pour UN patient
    déjà préprocessé (one-hot + scalé), sous forme de dict feature -> impact.
    """
    model, scaler, columns, model_name = load_artifacts()
    explainer = get_explainer(model, patient_df)
    shap_values = explainer(patient_df)

    if len(shap_values.shape) == 3:
        values = shap_values.values[0, :, 1]
    else:
        values = shap_values.values[0]

    return dict(zip(columns, values))


if __name__ == "__main__":
    main()
