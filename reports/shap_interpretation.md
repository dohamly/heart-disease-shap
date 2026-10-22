# Interprétation SHAP

> SHAP décrit comment le modèle utilise les variables pour prédire ; il n'établit aucune relation de cause à effet médicale. Une variable qui « augmente le risque » dans le modèle n'est pas pour autant une cause de la maladie.
> Projet académique : ce n'est pas un outil de diagnostic médical et il ne doit pas servir à une décision clinique.

Modèle expliqué : **Logistic Regression** (le modèle déployé). SHAP calculé sur les 303 patients, avec le jeu d'entraînement comme référence. Unité : log-odds. Pour un patient moyen, le modèle prédit un risque de 45 %.

## Ce que le modèle utilise le plus (global)

| Variable | Part de l'importance | Comment le modèle l'utilise |
|---|---|---|
| Thalassemia | 16 % | la catégorie qui augmente le plus la probabilité estimée est « reversible defect » (+0.20 en moyenne), celle qui la réduit le plus est « normal » (-0.17) |
| Major vessels colored | 16 % | plus la valeur est élevée, plus le modèle augmente la probabilité : contribution moyenne -0.26 pour les valeurs basses (≤ 25ᵉ percentile) et +0.41 pour les valeurs hautes (≥ 75ᵉ percentile) |
| ST depression | 13 % | plus la valeur est élevée, plus le modèle augmente la probabilité : contribution moyenne -0.30 pour les valeurs basses (≤ 25ᵉ percentile) et +0.43 pour les valeurs hautes (≥ 75ᵉ percentile) |
| Chest pain type | 12 % | la catégorie qui augmente le plus la probabilité estimée est « asymptomatic » (+0.18 en moyenne), celle qui la réduit le plus est « non-anginal pain » (-0.13) |
| Max heart rate | 11 % | plus la valeur est élevée, plus le modèle réduit la probabilité : contribution moyenne +0.39 pour les valeurs basses (≤ 25ᵉ percentile) et -0.32 pour les valeurs hautes (≥ 75ᵉ percentile) |
| ST slope | 9 % | la catégorie qui augmente le plus la probabilité estimée est « flat » (+0.10 en moyenne), celle qui la réduit le plus est « upsloping » (-0.10) |

Les 6 variables ci-dessus représentent 77 % de l'importance totale.

## Deux patients expliqués (jeu de test, jamais vus à l'entraînement)

### Patient à risque élevé (patient n° 191)

Le modèle estime une probabilité de maladie de **92 %** (patient moyen du jeu d'entraînement : 45 %).

- **Ce qui augmente le plus cette probabilité** : Major vessels colored (3) +0.91 ; Thalassemia (reversible defect) +0.40 ; ST depression (2.2) +0.32 ; Chest pain type (asymptomatic) +0.25
- **Ce qui la réduit le plus** : aucun facteur notable
- Les autres variables (détail dans le graphique) pèsent ensemble +0.76.

Les chiffres sont des contributions en log-odds : leur somme, ajoutée à la valeur de base, donne exactement la sortie du modèle. SHAP décrit comment le modèle utilise les variables pour prédire ; il n'établit aucune relation de cause à effet médicale. Une variable qui « augmente le risque » dans le modèle n'est pas pour autant une cause de la maladie.

### Patient à faible risque (patient n° 162)

Le modèle estime une probabilité de maladie de **10 %** (patient moyen du jeu d'entraînement : 45 %).

- **Ce qui augmente le plus cette probabilité** : Sex (male) +0.08
- **Ce qui la réduit le plus** : Max heart rate (182 bpm) -0.40 ; Thalassemia (normal) -0.30 ; ST depression (0.0) -0.30 ; Major vessels colored (0) -0.26
- Les autres variables (détail dans le graphique) pèsent ensemble -0.78.

Les chiffres sont des contributions en log-odds : leur somme, ajoutée à la valeur de base, donne exactement la sortie du modèle. SHAP décrit comment le modèle utilise les variables pour prédire ; il n'établit aucune relation de cause à effet médicale. Une variable qui « augmente le risque » dans le modèle n'est pas pour autant une cause de la maladie.

## Les trois modèles expliquent-ils de la même façon ?

Part de l'importance SHAP totale par variable (hyperparamètres réglés sur le train, chaque modèle expliqué sur les 303 patients) :

| Variable | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|
| Thalassemia | 16 % | 26 % | 25 % |
| Chest pain type | 12 % | 15 % | 20 % |
| Major vessels colored | 16 % | 11 % | 18 % |
| ST depression | 13 % | 7 % | 11 % |
| ST slope | 9 % | 10 % | 7 % |
| Exercise-induced angina | 6 % | 9 % | 7 % |
| Max heart rate | 11 % | 7 % | 3 % |
| Sex | 5 % | 4 % | 3 % |

Corrélation de rang de Spearman entre les importances : Logistic Regression / Random Forest : 0.91 ; Logistic Regression / XGBoost : 0.83 ; Random Forest / XGBoost : 0.92.
Variables communes aux 5 premières de chaque modèle : Major vessels colored, Chest pain type, Thalassemia.

Les trois modèles étant statistiquement équivalents en performance, un accord sur les variables utilisées renforce la confiance dans l'explication ; un désaccord signalerait qu'elle dépend du modèle choisi. Les valeurs SHAP des forêts sont en probabilité et celles des deux autres en log-odds : on compare donc des parts relatives, pas des valeurs brutes.
