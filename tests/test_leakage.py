"""
Tests anti-fuite de données (data leakage) et de reproductibilité.

Ces tests transforment en vérifications automatiques les affirmations du README :
le scaler n'apprend que sur le train, le test set n'influence rien, le split est
stratifié et reproductible, l'encodage one-hot ne dépend pas des autres lignes.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from model_config import make_pipeline  # noqa: E402
from preprocessing import (  # noqa: E402
    NUMERIC_COLS, build_preprocessed_dataset, load_data, split_and_scale, split_data,
)

DATA = os.path.join(HERE, "..", "data", "heart.csv")


def _data():
    df = load_data(DATA)
    X, y = build_preprocessed_dataset(df)
    return df, X, y


def test_scaler_is_fit_on_train_only():
    """Les statistiques du scaler sont celles du TRAIN, pas celles du jeu complet."""
    _, X, y = _data()
    X_tr_raw, _, _, _ = split_data(X, y)
    _, _, _, _, scaler = split_and_scale(X, y)
    assert np.allclose(scaler.mean_, X_tr_raw[NUMERIC_COLS].mean().values)
    assert np.allclose(scaler.scale_, X_tr_raw[NUMERIC_COLS].std(ddof=0).values)
    # preuve que ce n'est pas la moyenne globale (sinon le test ne prouverait rien)
    assert not np.allclose(scaler.mean_, X[NUMERIC_COLS].mean().values)


def test_test_set_values_do_not_influence_training():
    """Si l'on déforme fortement le test set, rien de ce qui touche au train ne bouge."""
    _, X, y = _data()
    _, X_te_raw, _, _ = split_data(X, y)
    X_mod = X.copy()
    X_mod.loc[X_te_raw.index, NUMERIC_COLS] *= 100

    X_train, _, _, _, scaler = split_and_scale(X, y)
    X_train_mod, _, _, _, scaler_mod = split_and_scale(X_mod, y)

    assert np.allclose(scaler.mean_, scaler_mod.mean_)
    assert np.allclose(scaler.scale_, scaler_mod.scale_)
    assert np.allclose(X_train.values, X_train_mod.values)


def test_pipeline_scaler_is_fit_on_training_fold_only():
    """Dans le Pipeline utilisé en validation croisée, le scaler n'apprend que sur le pli d'entraînement."""
    _, X, y = _data()
    X_tr, _, y_tr, _ = split_data(X, y)
    pipe = make_pipeline("Logistic Regression", list(X.columns)).fit(X_tr, y_tr)
    scaler = pipe.named_steps["prep"].named_transformers_["num"]
    assert np.allclose(scaler.mean_, X_tr[NUMERIC_COLS].mean().values)
    assert not np.allclose(scaler.mean_, X[NUMERIC_COLS].mean().values)


def test_split_is_stratified_reproducible_and_disjoint():
    _, X, y = _data()
    a = split_data(X, y, random_state=42)
    b = split_data(X, y, random_state=42)
    assert list(a[0].index) == list(b[0].index) and list(a[1].index) == list(b[1].index)

    train_idx, test_idx = set(a[0].index), set(a[1].index)
    assert train_idx.isdisjoint(test_idx)                 # aucun patient dans les deux
    assert train_idx | test_idx == set(X.index)           # personne n'est perdu
    assert len(test_idx) == 61                            # 20 % de 303
    # stratification : la prévalence reste proche dans les deux jeux
    assert abs(a[3].mean() - y.mean()) < 0.02
    assert abs(a[2].mean() - y.mean()) < 0.02


def test_only_numeric_columns_are_scaled():
    """Les colonnes one-hot et binaires restent en 0/1 (jamais standardisées)."""
    _, X, y = _data()
    X_tr_raw, _, _, _ = split_data(X, y)
    X_train, X_test, _, _, _ = split_and_scale(X, y)
    other = [c for c in X.columns if c not in NUMERIC_COLS]
    assert np.array_equal(X_train[other].values, X_tr_raw[other].values)
    assert set(np.unique(X_test[other].values)) <= {0.0, 1.0}


def test_one_hot_encoding_does_not_depend_on_other_rows():
    """Encoder un patient seul (comme le fait l'API) donne la même ligne que dans le jeu complet."""
    df, X, _ = _data()
    for i in range(0, len(df), 20):
        single, _ = build_preprocessed_dataset(df.iloc[[i]])
        aligned = single.reindex(columns=X.columns, fill_value=0.0).iloc[0]
        assert np.array_equal(aligned.values, X.iloc[i].values)


def test_target_never_appears_among_features():
    _, X, _ = _data()
    assert "target" not in X.columns and "disease" not in X.columns
