# Ablation des choix d'encodage

Régression logistique (C = 0.03), validation croisée stratifiée 5 plis x 10 répétitions
(50 évaluations par variante). Le scaler et les encodeurs sont ajustés
**dans chaque pli d'entraînement** (Pipeline). Les écarts sont calculés pli par pli sur les
mêmes partitions (comparaison appariée). Les plis se recouvrent, donc les écarts-types
sont descriptifs et ne constituent pas un test statistique.

| Variante | ROC-AUC | F1 | Δ ROC-AUC vs A (appariée) | plis où elle bat A |
|---|---|---|---|---|
| A. **Retenue** : one-hot (cp, restecg, slope, thal) + `ca` numérique | 0.907 ± 0.036 | 0.797 ± 0.064 | référence | — |
| B. `ca` aussi en one-hot | 0.912 ± 0.035 | 0.805 ± 0.063 | +0.0044 ± 0.0088 | 64 % |
| C. `slope` en ordinal numérique | 0.904 ± 0.037 | 0.790 ± 0.068 | -0.0033 ± 0.0056 | 28 % |
| D. cp/restecg/slope/thal en codes 0-3 (nominal traité comme ordonné) | 0.897 ± 0.039 | 0.796 ± 0.059 | -0.0101 ± 0.0187 | 24 % |
| E. `ca = 4` et `thal = 0` traités comme manquants, imputés dans le pli | 0.915 ± 0.035 | 0.812 ± 0.059 | +0.0075 ± 0.0080 | 82 % |

## Contrôle d'équivalence (pas de fuite via le one-hot)

Le projet encode les catégories avec `pd.get_dummies` **avant** le split (cela fixe seulement
la liste des colonnes, aucune statistique n'est apprise). En le comparant à un
`OneHotEncoder` appris dans chaque pli d'entraînement, l'écart maximal sur les
probabilités prédites est de **4.4e-16** : c'est identique à la précision numérique près.

## Lecture

- Les valeurs hors documentation concernent 7 lignes (`ca = 4` : 5, `thal = 0` : 2).
- Un écart de ROC-AUC inférieur à l'écart-type apparié n'est pas une preuve de différence.
