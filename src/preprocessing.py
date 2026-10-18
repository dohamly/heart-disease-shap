"""
preprocessing.py
-----------------
Chargement et préparation des données pour le projet Heart Disease + SHAP.

TRAITEMENT DE CHAQUE VARIABLE
-----------------------------
  Numériques continues  age, trestbps, chol, thalach, oldpeak
      -> standardisées (moyenne 0, écart-type 1). Obligatoire pour que le
         coefficient d'une régression logistique régularisée (L2) soit
         comparable d'une variable à l'autre (mmHg, mg/dl, bpm n'ont pas la
         même échelle).
  Numérique ordinale    ca  (nombre de vaisseaux colorés : 0, 1, 2, 3)
      -> gardée numérique et standardisée. C'est un COMPTAGE ordonné : une
         colonne au lieu de cinq, et un seul coefficient lisible
         ("par vaisseau en plus"). Tester le one-hot ne change rien de
         mesurable (reports/encoding_ablation.md, variante B).
  Catégorielles nominales  cp, restecg, slope, thal
      -> one-hot, SANS standardisation. Les codes 0/1/2/3 sont des
         étiquettes, pas des quantités : "thal = 3" n'est pas "trois fois"
         "thal = 1". Les traiter comme des nombres est légèrement pire
         (variante D de l'ablation).
  Binaires              sex, fbs, exang
      -> laissées en 0/1, non standardisées (déjà interprétables).

ANTI-FUITE (DATA LEAKAGE)
-------------------------
  * Le split train/test est stratifié, random_state = 42, fait AVANT tout
    apprentissage de statistiques.
  * Le StandardScaler est ajusté sur le jeu d'entraînement uniquement, puis
    appliqué tel quel au test (split_and_scale). En validation croisée, il est
    dans un sklearn Pipeline (model_config.make_pipeline), donc ré-ajusté dans
    chaque pli d'entraînement.
  * Le one-hot est calculé avant le split avec pd.get_dummies : cela ne fixe que
    la liste des colonnes, aucune statistique du jeu de données n'est apprise.
    On vérifie dans reports/encoding_ablation.md que l'on obtient les mêmes
    prédictions (écart ~1e-16) avec un OneHotEncoder appris dans chaque pli.
  * La cible n'apparaît jamais parmi les variables d'entrée.
  Tout cela est vérifié par tests/test_leakage.py.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CATEGORICAL_COLS = ["cp", "restecg", "slope", "thal"]   # nominales -> one-hot
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]   # standardisées
BINARY_COLS = ["sex", "fbs", "exang"]                   # 0/1, non standardisées
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


# Signification des codes dans CE fichier heart.csv. Le dictionnaire de données
# qui circule avec la version Kaggle est inversé pour `cp` et `slope`. Les noms
# ci-dessous sont déduits en comparant les effectifs de chaque code avec ceux du
# jeu UCI Cleveland d'origine (ex. cp=0 : 143 patients ~ "asymptomatique" : 144)
# et vérifiés par les taux de maladie observés. `restecg` : confiance moindre
# (le groupe de 4 patients est identifié par son effectif uniquement).
CATEGORY_NAMES = {
    "cp": {0: "asymptomatic", 1: "atypical angina", 2: "non-anginal pain", 3: "typical angina"},
    "restecg": {0: "normal", 1: "LV hypertrophy", 2: "ST-T abnormality"},
    "slope": {0: "downsloping", 1: "flat", 2: "upsloping"},
    "thal": {0: "unknown", 1: "fixed defect", 2: "normal", 3: "reversible defect"},
}


def readable_feature(col: str) -> str:
    """Nom lisible d'une colonne, y compris les colonnes one-hot (ex. cp_0)."""
    if col in FEATURE_NAMES_READABLE:
        return FEATURE_NAMES_READABLE[col]
    base, _, val = col.rpartition("_")
    name = CATEGORY_NAMES.get(base, {}).get(int(val), val) if val.isdigit() else val
    return f"{FEATURE_NAMES_READABLE.get(base, base)}: {name}"


def load_data(path="data/heart.csv"):
    """Charge le dataset brut."""
    df = pd.read_csv(path)
    return df


def disease_label(df: pd.DataFrame) -> pd.Series:
    """
    Retourne l'étiquette "maladie cardiaque présente" (1 = malade, 0 = sain).

    ATTENTION : dans ce fichier heart.csv (version Kaggle du dataset UCI
    Cleveland), la colonne `target` est INVERSÉE : target=1 correspond à
    l'ABSENCE de maladie (165 sujets) et target=0 à sa PRÉSENCE (138 sujets).
    On le vérifie sur les données : les patients avec target=0 ont plus
    souvent une angine d'effort, plus de vaisseaux atteints, un ST plus
    déprimé et une fréquence cardiaque max plus basse. Voir
    tests/test_preprocessing.py::test_disease_label_is_medically_consistent.
    """
    return (1 - df[TARGET_COL]).astype(int).rename("disease")


def build_preprocessed_dataset(df: pd.DataFrame):
    """
    Encode les variables catégorielles en one-hot, garde le reste tel quel.
    Retourne X (DataFrame), y (Series) et la liste finale des colonnes.
    """
    df = df.copy()
    y = disease_label(df)  # 1 = maladie présente (cf. disease_label)
    X = df.drop(columns=[TARGET_COL])

    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=False)
    X = X.astype("float64")

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split train/test stratifié (sans standardisation)."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Split train/test stratifié + standardisation des colonnes numériques.
    On scale uniquement NUMERIC_COLS pour garder les colonnes one-hot en 0/1.
    """
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)

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
