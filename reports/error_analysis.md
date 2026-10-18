# Analyse des erreurs — Logistic Regression

Prédictions hors-pli (validation croisée imbriquée, seuil 0.5) sur les 303 patients. Chaque patient est prédit par un modèle qui ne l'a pas vu à l'entraînement. Classe positive = maladie présente.

## Comptage

- Vrais positifs : **108** · Vrais négatifs : **144**
- Faux positifs (sain prédit malade) : **21** · Faux négatifs (malade prédit sain) : **30**

En santé, un faux négatif (malade non détecté) est en général plus grave qu'un faux positif : il faudrait donc abaisser le seuil de décision au-dessus de 0.5 si l'on privilégiait le recall.

## Profil moyen de chaque groupe

| Caractéristique | TN | FP | FN | TP |
|---|---|---|---|---|
| age (moy.) | 51.5 | 59.5 | 54.0 | 57.3 |
| pression repos (moy.) | 129.0 | 131.3 | 132.1 | 135.0 |
| cholestérol (moy.) | 241.7 | 246.0 | 247.5 | 252.1 |
| fréq. card. max (moy.) | 160.8 | 142.2 | 155.2 | 134.6 |
| ST depression (moy.) | 0.5 | 1.1 | 0.6 | 1.8 |
| % hommes | 51.4 | 90.5 | 80.0 | 83.3 |
| % angine d'effort | 11.1 | 33.3 | 13.3 | 66.7 |
| % >=1 vaisseau coloré | 19.4 | 33.3 | 36.7 | 75.9 |
| % douleur asymptomatique | 18.8 | 57.1 | 40.0 | 85.2 |
| % thal. réversible | 10.4 | 61.9 | 36.7 | 72.2 |

## Ce qui distingue les erreurs (écart standardisé, variables d'origine)

**Faux négatifs vs vrais positifs** (malades : pourquoi certains passent inaperçus) :

- Exercise-induced angina : -1.14
- ST depression : -1.04
- Major vessels colored : -0.92
- Chest pain type: asymptomatic : -0.90
- Max heart rate : +0.90

**Faux positifs vs vrais négatifs** (sains : pourquoi certains ressemblent à des malades) :

- Thalassemia: normal : -1.26
- Thalassemia: reversible defect : +1.06
- Age : +0.88
- ST slope: flat : +0.85
- Sex : +0.84

## Interprétation avec SHAP

![SHAP des erreurs](shap_errors.png)

Les barres montrent la contribution moyenne de chaque variable pour les patients mal classés (rouge : pousse vers « malade », vert : vers « sain »). Les deux erreurs les plus confiantes en détail :

![Faux négatif](shap_waterfall_FN.png)

![Faux positif](shap_waterfall_FP.png)

## Les 5 faux négatifs les plus confiants

| patient | age | sex | cp | thalach | oldpeak | ca | thal | P(malade) |
|---|---|---|---|---|---|---|---|---|
| 177 | 64 | M | non-anginal pain | 158 | 0.0 | 0 | normal | 0.0 |
| 230 | 47 | M | non-anginal pain | 152 | 0.0 | 0 | normal | 0.1 |
| 254 | 59 | M | typical angina | 125 | 0.0 | 0 | normal | 0.1 |
| 182 | 61 | F | asymptomatic | 169 | 0.0 | 0 | normal | 0.1 |
| 278 | 58 | F | atypical angina | 152 | 0.0 | 2 | normal | 0.1 |

## Les 5 faux positifs les plus confiants

| patient | age | sex | cp | thalach | oldpeak | ca | thal | P(malade) |
|---|---|---|---|---|---|---|---|---|
| 158 | 58 | M | atypical angina | 144 | 0.4 | 4 | reversible defect | 0.9 |
| 101 | 59 | M | typical angina | 145 | 4.2 | 0 | reversible defect | 0.9 |
| 139 | 64 | M | asymptomatic | 105 | 0.2 | 1 | reversible defect | 0.9 |
| 52 | 62 | M | non-anginal pain | 146 | 1.8 | 3 | reversible defect | 0.8 |
| 95 | 53 | M | asymptomatic | 111 | 0.0 | 0 | reversible defect | 0.8 |

## Limites

- Seulement 303 patients : les moyennes par groupe d'erreurs reposent sur de petits effectifs (quelques dizaines de cas) et sont indicatives, pas des preuves statistiques.
- Dataset Cleveland unique, une seule source hospitalière : aucune validation externe.