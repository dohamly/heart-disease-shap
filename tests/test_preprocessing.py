"""Tests unitaires pour src/preprocessing.py"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import load_data, build_preprocessed_dataset, split_and_scale, disease_label


def test_load_data_shape():
    df = load_data(path=os.path.join(os.path.dirname(__file__), "..", "data", "heart.csv"))
    assert df.shape[0] == 303
    assert "target" in df.columns
    assert df.isna().sum().sum() == 0


def test_build_preprocessed_dataset_no_nan():
    df = load_data(path=os.path.join(os.path.dirname(__file__), "..", "data", "heart.csv"))
    X, y = build_preprocessed_dataset(df)
    assert X.isna().sum().sum() == 0
    assert set(y.unique()) <= {0, 1}
    assert "cp_0" in X.columns  # one-hot bien appliqué


def test_split_and_scale_shapes():
    df = load_data(path=os.path.join(os.path.dirname(__file__), "..", "data", "heart.csv"))
    X, y = build_preprocessed_dataset(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y, test_size=0.2, random_state=42)
    assert len(X_train) + len(X_test) == len(X)
    assert abs(len(X_test) / len(X) - 0.2) < 0.02
    # les colonnes numériques scalées doivent être centrées ~0 sur le train
    assert abs(X_train["age"].mean()) < 1e-6


def test_disease_label_is_medically_consistent():
    """
    Garde-fou contre l'inversion de `target` dans ce CSV (target=1 = SAIN).
    Avec disease = 1 - target, les facteurs de risque connus doivent être plus
    fréquents chez les malades : angine d'effort, vaisseaux atteints, ST déprimé,
    sexe masculin ; et la fréquence cardiaque max doit être plus basse.
    """
    df = load_data(path=os.path.join(os.path.dirname(__file__), "..", "data", "heart.csv"))
    d = disease_label(df)
    assert d.sum() == 138 and (1 - d).sum() == 165
    assert d[df.exang == 1].mean() > d[df.exang == 0].mean()
    assert d[df.ca >= 1].mean() > d[df.ca == 0].mean()
    assert d[df.oldpeak > 1].mean() > d[df.oldpeak <= 1].mean()
    assert d[df.sex == 1].mean() > d[df.sex == 0].mean()
    assert df[d == 1].thalach.mean() < df[d == 0].thalach.mean()
