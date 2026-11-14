"""
preprocessing.py
----------------
Data loading and preparation for the Heart Disease + SHAP project.

How each variable is handled
----------------------------
  Continuous numeric   age, trestbps, chol, thalach, oldpeak
      Standardised (mean 0, std 1). Needed so the coefficients of a regularised (L2) logistic
      regression are comparable across variables (mmHg, mg/dl and bpm have different scales).
  Ordinal numeric      ca (number of coloured vessels: 0, 1, 2, 3)
      Kept numeric and standardised. It is an ordered COUNT: one column instead of five and
      one readable coefficient ("per extra vessel"). One-hot encoding it changes nothing
      measurable (reports/encoding_ablation.md, variant B).
  Nominal categorical  cp, restecg, slope, thal
      One-hot encoded, NOT standardised. Codes 0/1/2/3 are labels, not quantities:
      "thal = 3" is not "three times" "thal = 1". Treating them as numbers is slightly
      worse (ablation variant D).
  Binary               sex, fbs, exang
      Left as 0/1, not standardised (already interpretable).

Data leakage safeguards
-----------------------
  * The train/test split is stratified, random_state = 42, and done BEFORE anything is learned.
  * The StandardScaler is fitted on the training set only, then applied unchanged to the test
    set (split_and_scale). In cross-validation it lives inside a sklearn Pipeline
    (model_config.make_pipeline), so it is refitted on every training fold.
  * One-hot encoding happens before the split with pd.get_dummies. It only fixes the list of
    columns and learns no statistic from the data. reports/encoding_ablation.md shows the
    predictions are identical (gap ~1e-16) with a OneHotEncoder fitted inside each fold.
  * The target never appears among the input features.
  All of this is checked by tests/test_leakage.py.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CATEGORICAL_COLS = ["cp", "restecg", "slope", "thal"]  # nominal -> one-hot
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]  # standardised
BINARY_COLS = ["sex", "fbs", "exang"]  # 0/1, not standardised
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


# Meaning of the codes in THIS heart.csv. The data dictionary that circulates with the Kaggle
# version is inverted for `cp` and `slope`. The names below are inferred by comparing the
# counts of each code with the original UCI Cleveland data (e.g. cp=0: 143 patients vs
# "asymptomatic": 144) and checked against the observed disease rates. `restecg` is the least
# certain: its 4-patient group is identified by its count only.
CATEGORY_NAMES = {
    "cp": {0: "asymptomatic", 1: "atypical angina", 2: "non-anginal pain", 3: "typical angina"},
    "restecg": {0: "normal", 1: "LV hypertrophy", 2: "ST-T abnormality"},
    "slope": {0: "downsloping", 1: "flat", 2: "upsloping"},
    "thal": {0: "unknown", 1: "fixed defect", 2: "normal", 3: "reversible defect"},
}


def readable_feature(col: str) -> str:
    """Return a readable name for a column, including one-hot columns (e.g. cp_0)."""
    if col in FEATURE_NAMES_READABLE:
        return FEATURE_NAMES_READABLE[col]
    base, _, val = col.rpartition("_")
    name = CATEGORY_NAMES.get(base, {}).get(int(val), val) if val.isdigit() else val
    return f"{FEATURE_NAMES_READABLE.get(base, base)}: {name}"


def load_data(path="data/heart.csv"):
    """Load the raw dataset."""
    df = pd.read_csv(path)
    return df


def disease_label(df: pd.DataFrame) -> pd.Series:
    """
    Return the "heart disease present" label (1 = disease, 0 = healthy).

    WARNING: in this heart.csv (the Kaggle version of the UCI Cleveland data) the `target`
    column is INVERTED: target=1 means NO disease (165 patients) and target=0 means disease
    (138 patients). The data shows it: target=0 patients more often have exercise-induced
    angina, more diseased vessels, a deeper ST depression and a lower maximum heart rate.
    See tests/test_preprocessing.py::test_disease_label_is_medically_consistent.
    """
    return (1 - df[TARGET_COL]).astype(int).rename("disease")


def build_preprocessed_dataset(df: pd.DataFrame):
    """One-hot encode the categorical variables and return the features X and the label y."""
    df = df.copy()
    y = disease_label(df)
    X = df.drop(columns=[TARGET_COL])

    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=False)
    X = X.astype("float64")

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified train/test split, without scaling."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Stratified train/test split, then scale the numeric columns with a scaler fitted on the
    training set only. One-hot and binary columns stay 0/1. Returns the scaler as well.
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
    print("Final columns:", X.columns.tolist())
