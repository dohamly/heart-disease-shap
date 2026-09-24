"""
preprocessing.py
-----------------
Chargement et préparation des données pour le projet Heart Disease + SHAP.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Colonnes catégorielles (déjà encodées numériquement dans le dataset UCI,
# mais on les traite comme catégorielles pour le one-hot encoding)
CATEGORICAL_COLS = ["cp", "restecg", "slope", "thal"]
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_COLS = ["sex", "fbs", "exang"]
TARGET_COL = "target"

FEATURE_NAMES_READABLE = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting blood sugar > 120",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression",
    "slope": "ST slope",
    "ca": "Major vessels colored",
    "thal": "Thalassemia",
}


def load_data(path="data/heart.csv"):
    """Charge le dataset brut."""
    df = pd.read_csv(path)
    return df


def build_preprocessed_dataset(df: pd.DataFrame):
    """
    Encode les variables catégorielles en one-hot, garde le reste tel quel.
    Retourne X (DataFrame), y (Series) et la liste finale des colonnes.
    """
    df = df.copy()
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=False)
    X = X.astype("float64")

    return X, y


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Split train/test stratifié + standardisation des colonnes numériques.
    On scale uniquement NUMERIC_COLS pour garder les colonnes one-hot en 0/1.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    numeric_present = [c for c in NUMERIC_COLS if c in X_train.columns]

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numeric_present] = scaler.fit_transform(X_train[numeric_present])
    X_test[numeric_present] = scaler.transform(X_test[numeric_present])

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    df = load_data()
    X, y = build_preprocessed_dataset(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Colonnes finales:", X.columns.tolist())
