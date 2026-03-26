<div align="center">

---

<br>

# ◈ UNIVERSITÉ HASSAN Iᵉʳ ◈

## ÉCOLE NATIONALE DE COMMERCE ET DE GESTION — SETTAT

<br>

---

### ─── FILIÈRE : FINANCE ───

---

<br><br>

# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
# CLASSIFICATION DE L'INCLUSION FINANCIÈRE AU MAROC
# Étude Comparative : Random Forest · Arbre de Décision · Gradient Boosting
# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

<br><br>

> **Projet d'Examen — Semestre 8**

<br>

|  |  |
|:---|:---|
| **Réalisé par** | **Aymane JEBBOUR** — N° Apogée : `22005997` |
|  | **Adam LAMRAHI** — N° Apogée : `22006644` |

<br>

---

**Année Universitaire 2025 — 2026**

---

<br><br><br>

</div>

---

<div align="center">

# ◇ SOMMAIRE ◇

</div>

---

| N° | Section | 
|:--:|:--------|
| **I** | Introduction |
| **II** | Présentation du Jeu de Données |
| **III** | Méthodologie Commune : Prétraitement |
| **IV** | Méthode 1 — Random Forest |
| | IV.1 · Principe de l'algorithme |
| | IV.2 · Analyse du code |
| | IV.3 · Résultats et interprétation des graphes |
| **V** | Méthode 2 — Arbre de Décision |
| | V.1 · Principe de l'algorithme |
| | V.2 · Analyse du code |
| | V.3 · Résultats et interprétation des graphes |
| **VI** | Méthode 3 — XGBoost & LightGBM (Gradient Boosting) |
| | VI.1 · Principe des algorithmes |
| | VI.2 · Analyse du code |
| | VI.3 · Résultats et interprétation des graphes |
| **VII** | Comparaison Finale des Trois Méthodes |
| **VIII** | Conclusion |
| **IX** | Webographie |
| **X** | Annexes |

---

<br>

<div align="center">

# ── I · INTRODUCTION ──

</div>

---

L'inclusion financière constitue un enjeu stratégique majeur pour le développement économique du Maroc. Elle désigne la capacité des individus et des entreprises à accéder à des produits et services financiers utiles, abordables et adaptés à leurs besoins — transactions, paiements, épargne, crédit, assurance — délivrés de manière responsable et durable.

Au Maroc, malgré les avancées significatives portées par Bank Al-Maghrib et la Stratégie Nationale d'Inclusion Financière (SNIF), une part non négligeable de la population demeure partiellement ou totalement exclue du système financier formel. Comprendre les facteurs déterminants de cette exclusion est une nécessité pour orienter les politiques publiques.

**Objectif du projet** : Ce travail propose d'appliquer trois familles d'algorithmes de Machine Learning supervisé pour classifier le niveau d'inclusion financière d'individus marocains en trois catégories — *Included*, *Partially Included*, *Excluded* — à partir de caractéristiques socio-économiques. Les trois méthodes explorées sont :

1. **Random Forest** — Ensemble d'arbres par bagging
2. **Arbre de Décision** — Modèle à arbre unique avec élagage
3. **XGBoost & LightGBM** — Gradient Boosting avancé, avec méthodes d'ensemble (Voting / Stacking)

L'enjeu est double : identifier les variables les plus prédictives de l'inclusion financière, et évaluer quelle approche algorithmique offre le meilleur compromis entre performance, interprétabilité et robustesse.

---

<br>

<div align="center">

# ── II · PRÉSENTATION DU JEU DE DONNÉES ──

</div>

---

Le dataset utilisé est `financial_inclusion_morocco.xlsx`. Il contient **500 observations** et **6 variables** décrivant des individus marocains.

### Structure des données

| Variable | Type | Description |
|:---------|:----:|:------------|
| `Age` | Numérique | Âge de l'individu |
| `Income_Monthly_MAD` | Numérique | Revenu mensuel en dirhams marocains |
| `Education_Level` | Catégorielle | Niveau d'éducation (*Primary*, *Secondary*, *University*) |
| `Has_Bank_Account` | Binaire | Possession d'un compte bancaire (0/1) |
| `Uses_Mobile_Money` | Binaire | Utilisation du mobile money (0/1) |
| `Financial_Inclusion` | Cible | Statut d'inclusion (*Excluded*, *Partially Included*, *Included*) |

### Distribution de la variable cible

Le jeu de données révèle un déséquilibre modéré entre les trois classes, avec une prédominance de la catégorie « Partially Included ». Ce déséquilibre a motivé l'utilisation systématique du paramètre `class_weight='balanced'` dans chaque modèle.

### Exploration visuelle

L'analyse exploratoire a produit plusieurs graphes clés :

- **Bar chart + Pie chart de la variable cible** : Visualisent la répartition des trois classes. Le diagramme en barres permet de lire les effectifs absolus et pourcentages, tandis que le camembert donne une vue proportionnelle immédiate.

- **Histogrammes de l'âge et du revenu par classe** : Montrent le chevauchement important des distributions d'âge entre les trois classes, ce qui suggère que l'âge seul n'est pas suffisant pour discriminer. Le revenu mensuel présente des distributions légèrement plus différenciées, les individus « Included » ayant tendance à avoir des revenus plus élevés.

- **Éducation vs Inclusion** : Le bar chart croisé montre que les individus avec un niveau universitaire sont davantage représentés dans la catégorie « Included », tandis que le niveau primaire est surreprésenté chez les « Excluded ».

- **Matrice de corrélation** : Met en évidence les corrélations entre les features encodées. `Has_Bank_Account` et `Financial_Inclusion` présentent la corrélation la plus forte, ce qui est attendu puisque la possession d'un compte bancaire est un indicateur direct d'inclusion.

---

<br>

<div align="center">

# ── III · MÉTHODOLOGIE COMMUNE : PRÉTRAITEMENT ──

</div>

---

Le prétraitement est identique pour les trois méthodes afin de garantir la comparabilité des résultats.

### Encodage des variables

Les deux variables catégorielles ont été transformées via `LabelEncoder` :

- **Education_Level** : Primary → 0, Secondary → 1, University → 2
- **Financial_Inclusion** (cible) : Excluded → 0, Included → 1, Partially Included → 2

### Séparation Train / Test

Le split est réalisé avec `train_test_split` dans un ratio **80/20** (400 échantillons d'entraînement, 100 de test), avec stratification sur la variable cible (`stratify=y`) pour conserver les proportions de classes dans les deux ensembles. Le `random_state=42` garantit la reproductibilité.

### Validation croisée

Une **StratifiedKFold à 5 plis** est utilisée de manière systématique pour l'évaluation et l'optimisation des hyperparamètres, permettant une estimation robuste de la performance de généralisation.

---

<br>

<div align="center">

# ── IV · MÉTHODE 1 — RANDOM FOREST ──

</div>

---

## IV.1 · Principe de l'algorithme

Le Random Forest est un algorithme d'ensemble qui construit un grand nombre d'arbres de décision en parallèle, chacun entraîné sur un sous-échantillon bootstrap des données et un sous-ensemble aléatoire de features. La prédiction finale résulte d'un vote majoritaire. Cette double randomisation réduit la variance et améliore la robustesse par rapport à un arbre unique.

## IV.2 · Analyse du code

**Phase 1 — Modèle de base.** Un `RandomForestClassifier` est d'abord entraîné avec des hyperparamètres par défaut raisonnables : 200 estimateurs, `max_features='sqrt'`, `class_weight='balanced'`, et sans limite de profondeur. Ce modèle de base obtient une accuracy de **64.00%** sur le jeu de test.

```python
rf_base = RandomForestClassifier(
    n_estimators=200, max_depth=None, max_features='sqrt',
    class_weight='balanced', random_state=42, n_jobs=-1
)
```

**Phase 2 — Optimisation par GridSearchCV.** Une recherche exhaustive sur grille explore 96 combinaisons d'hyperparamètres couvrant le nombre d'estimateurs (100, 200, 300), la profondeur maximale (None, 5, 10, 15), `min_samples_split` (2, 5), `min_samples_leaf` (1, 2) et `max_features` (sqrt, log2). L'évaluation se fait par validation croisée stratifiée à 5 plis.

Les meilleurs hyperparamètres identifiés sont :

| Hyperparamètre | Valeur optimale |
|:---------------|:---------------:|
| `n_estimators` | 100 |
| `max_depth` | 15 |
| `max_features` | sqrt |
| `min_samples_leaf` | 1 |
| `min_samples_split` | 2 |

Le meilleur score CV obtenu est de **66.75%**.

**Phase 3 — Évaluation complète.** Le modèle optimisé est évalué sur le jeu de test avec un rapport de classification détaillé, matrice de confusion, courbes ROC (One-vs-Rest), et analyse de l'importance des features par deux méthodes (MDI et Permutation Importance).

**Phase 4 — Courbes d'apprentissage.** Le code trace la learning curve (accuracy train vs validation en fonction de la taille du jeu d'entraînement) et l'évolution de l'erreur OOB (Out-Of-Bag) en fonction du nombre d'estimateurs, pour diagnostiquer le surapprentissage.

## IV.3 · Résultats et interprétation des graphes

### Rapport de classification

| Classe | Précision | Rappel | F1-Score | Support |
|:-------|:---------:|:------:|:--------:|:-------:|
| Excluded | 1.00 | 1.00 | 1.00 | 25 |
| Included | 0.44 | 0.38 | 0.41 | 32 |
| Partially Included | 0.58 | 0.65 | 0.62 | 43 |
| **Accuracy globale** | | | **0.65** | **100** |
| Macro avg | 0.68 | 0.68 | 0.67 | 100 |

**Interprétation** : La classe « Excluded » est parfaitement classifiée (F1 = 1.00), ce qui s'explique par le fait que les individus sans compte bancaire et sans mobile money forment un profil très distinct. En revanche, la distinction entre « Included » et « Partially Included » est difficile (F1 respectifs de 0.41 et 0.62), ces deux populations partageant des caractéristiques socio-économiques proches.

### Matrice de confusion

La matrice de confusion absolue montre que les 25 « Excluded » sont tous correctement classés. Les erreurs se concentrent sur la confusion entre « Included » et « Partially Included » : le modèle tend à classer certains « Included » comme « Partially Included » (taux de rappel de seulement 38% pour « Included »). La matrice normalisée confirme que 65% des « Partially Included » sont correctement identifiés, tandis que seulement 38% des « Included » le sont.

### Courbes ROC (One-vs-Rest)

Les courbes ROC montrent un AUC Macro de **0.807**, ce qui indique une capacité discriminante globalement bonne malgré l'accuracy modérée. La classe « Excluded » a un AUC proche de 1.0 (discrimination parfaite). Les classes « Included » et « Partially Included » affichent des AUC plus modestes mais respectables, confirmant que le modèle capture une information probabiliste utile même lorsque la classification binaire échoue.

### Importance des features

L'analyse par deux méthodes convergentes (MDI et Permutation) révèle que :

- **`Income_Monthly_MAD`** est la variable la plus importante, confirmant que le revenu est le facteur prédictif dominant de l'inclusion financière.
- **`Has_Bank_Account`** arrive en deuxième position, ce qui est cohérent avec la définition même de l'inclusion financière.
- **`Age`**, **`Education_Level`** et **`Uses_Mobile_Money`** apportent une contribution complémentaire mais plus limitée.

### Courbe d'apprentissage et OOB

La courbe d'apprentissage montre un écart modéré entre les scores d'entraînement et de validation, suggérant un léger surapprentissage. L'erreur OOB se stabilise rapidement après 50-100 estimateurs, validant le choix de 100 arbres comme optimal.

### Tableau de bord final

| Métrique | Valeur |
|:---------|:------:|
| Accuracy | 0.6500 (65.00%) |
| F1 Macro | 0.6741 (67.41%) |
| Précision Macro | 0.6759 (67.59%) |
| Rappel Macro | 0.6754 (67.54%) |
| AUC ROC Macro | 0.8071 (80.71%) |
| CV Score Moyen | 0.6600 ± 0.0518 |

---

<br>

<div align="center">

# ── V · MÉTHODE 2 — ARBRE DE DÉCISION ──

</div>

---

## V.1 · Principe de l'algorithme

L'arbre de décision est un modèle qui partitionne récursivement l'espace des features en régions homogènes vis-à-vis de la variable cible. Chaque nœud interne teste une condition sur une feature, et chaque feuille attribue une classe. Son principal avantage est l'interprétabilité : on peut lire les règles de décision directement. Son inconvénient majeur est la tendance au surapprentissage lorsque la profondeur n'est pas contrôlée.

## V.2 · Analyse du code

**Phase 1 — Arbre complet (sans élagage).** Un premier arbre est entraîné sans contrainte de profondeur. Il atteint une profondeur de **16** avec **131 feuilles**, un score d'entraînement parfait (1.0000) et un score test de seulement **0.6100**. L'écart de presque 40 points confirme un surapprentissage sévère, justifiant l'optimisation.

```python
dt_full = DecisionTreeClassifier(
    criterion='gini', max_depth=None,
    class_weight='balanced', random_state=42
)
```

**Phase 2 — Optimisation par GridSearchCV.** La recherche sur grille est beaucoup plus exhaustive que pour le Random Forest : **5 184 combinaisons** testées couvrant le critère de split (gini, entropy, log_loss), la profondeur (2 à None), `min_samples_split`, `min_samples_leaf`, `max_features`, et le paramètre de pruning `ccp_alpha`. C'est une approche rigoureuse qui explore aussi l'élagage par coût-complexité.

Les meilleurs hyperparamètres identifiés :

| Hyperparamètre | Valeur optimale |
|:---------------|:---------------:|
| `criterion` | entropy |
| `max_depth` | 10 |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 2 |
| `max_features` | None |
| `ccp_alpha` | 0.0 |

Score CV optimal : **67.50%**. L'arbre optimisé a une profondeur de **10** et **83 feuilles**.

**Phase 3 — Visualisation de l'arbre.** Le code produit deux visualisations via `plot_tree` de scikit-learn : l'arbre complet optimisé (profondeur 10, 83 feuilles) et un arbre simplifié (profondeur 3) pour la lisibilité. L'arbre simplifié atteint une accuracy de **64.00%**, très proche de l'arbre optimisé, ce qui suggère que l'information discriminante principale est captée dès les premiers niveaux de l'arbre.

**Phase 4 — Export des règles textuelles.** La fonction `export_text` produit les règles de décision en format lisible. La première règle identifiée est `Has_Bank_Account <= 0.50`, ce qui signifie que l'absence de compte bancaire est le premier critère de séparation. Pour les individus sans compte, l'utilisation du mobile money détermine ensuite le classement entre « Excluded » et les deux autres catégories.

**Phase 5 — Analyse du surapprentissage.** Deux approches sont utilisées :
- **Validation curve sur `max_depth`** (profondeurs 1 à 20) : montre que la performance test plafonne autour de profondeur 5-10, alors que le score train continue de grimper.
- **Cost Complexity Pruning** (CCP Alpha path) : trace l'évolution des scores train/test en fonction du paramètre alpha de pruning, identifiant le point optimal d'élagage.

## V.3 · Résultats et interprétation des graphes

### Rapport de classification

| Classe | Précision | Rappel | F1-Score | Support |
|:-------|:---------:|:------:|:--------:|:-------:|
| Excluded | 1.00 | 1.00 | 1.00 | 25 |
| Included | 0.46 | 0.53 | 0.49 | 32 |
| Partially Included | 0.61 | 0.53 | 0.57 | 43 |
| **Accuracy globale** | | | **0.65** | **100** |
| Macro avg | 0.69 | 0.69 | 0.69 | 100 |

**Interprétation** : L'accuracy est identique au Random Forest (65%), mais la distribution des erreurs est légèrement différente. Le rappel de la classe « Included » est meilleur (53% vs 38%), ce qui signifie que l'arbre de décision détecte mieux les individus financièrement inclus. En contrepartie, le rappel de « Partially Included » est plus faible (53% vs 65%).

### Visualisation de l'arbre

L'arbre optimisé à 10 niveaux de profondeur est trop dense pour une lecture directe, mais l'arbre simplifié (profondeur 3) offre une interprétation claire de la logique de décision :

- **Nœud racine** : `Has_Bank_Account <= 0.5` — C'est le critère le plus discriminant.
- **Branche gauche** (pas de compte) : `Uses_Mobile_Money <= 0.5` départage les « Excluded » (ni compte ni mobile money) des autres.
- **Branche droite** (avec compte) : Le revenu mensuel et le niveau d'éducation interviennent pour distinguer « Included » de « Partially Included ».

### Règles de décision extraites

Les premières règles confirment la hiérarchie :

```
|--- Has_Bank_Account <= 0.50
|   |--- Uses_Mobile_Money <= 0.50 → Excluded
|   |--- Uses_Mobile_Money > 0.50
|       |--- Income_Monthly_MAD <= 3110.50 → Partially Included
|       |--- Income_Monthly_MAD > 3110.50 → ...
```

Cette transparence est le principal atout de l'arbre de décision par rapport au Random Forest.

### Analyse du surapprentissage

La **validation curve** montre clairement que le score test plafonne à partir de `max_depth=5` environ, tandis que le score train atteint 1.0 pour des profondeurs supérieures à 10. Le **CCP Alpha path** confirme qu'un élagage léger (alpha faible) ne dégrade pas significativement la performance test tout en réduisant la complexité du modèle.

### Comparaison des critères de split

La comparaison entre Gini, Entropy et Log Loss montre des performances très similaires. L'Entropy obtient un très léger avantage en CV, ce qui explique son choix comme critère optimal.

### Importance des features

L'importance basée sur le Gini confirme la hiérarchie observée dans le Random Forest : `Has_Bank_Account` et `Income_Monthly_MAD` dominent, suivis de `Uses_Mobile_Money`, `Education_Level` et `Age`.

### Tableau de bord final

| Métrique | Valeur |
|:---------|:------:|
| Accuracy | 0.6500 (65.00%) |
| F1 Macro | 0.6869 (68.69%) |
| Précision Macro | 0.6882 (68.82%) |
| Rappel Macro | 0.6887 (68.87%) |
| AUC ROC Macro | 0.7616 (76.16%) |
| CV Score Moyen | 0.6380 ± 0.0271 |

---

<br>

<div align="center">

# ── VI · MÉTHODE 3 — XGBOOST & LIGHTGBM ──

</div>

---

## VI.1 · Principe des algorithmes

Contrairement au Random Forest qui construit des arbres en parallèle, le **Gradient Boosting** construit des arbres séquentiellement : chaque nouvel arbre corrige les erreurs résiduelles du précédent. **XGBoost** (eXtreme Gradient Boosting) optimise une fonction de coût avec régularisation L1/L2, tandis que **LightGBM** (Light Gradient Boosted Machine) utilise une stratégie de croissance « leaf-wise » (par feuille) plus agressive que la croissance « level-wise » de XGBoost, et un histogramme de gradients pour accélérer l'entraînement.

## VI.2 · Analyse du code

**Phase 1 — Modèles de base.** XGBoost et LightGBM sont d'abord entraînés avec des hyperparamètres standards (200 estimateurs, learning rate de 0.1, max_depth de 6, subsampling à 80%). Les deux obtiennent une accuracy de base de **65.00%**.

Pour XGBoost, le code gère le déséquilibre des classes via `compute_sample_weight('balanced')`, tandis que LightGBM utilise directement `class_weight='balanced'`. Un early stopping à 50 itérations est appliqué à LightGBM.

```python
xgb_base = XGBClassifier(
    objective='multi:softprob', n_estimators=200,
    learning_rate=0.1, max_depth=6, subsample=0.8, ...
)
lgbm_base = LGBMClassifier(
    objective='multiclass', n_estimators=200,
    learning_rate=0.1, num_leaves=31, class_weight='balanced', ...
)
```

**Phase 2 — Courbes de perte XGBoost.** Le code trace l'évolution du `mlogloss` (log loss multiclasse) sur les ensembles train et test au fil des itérations. Le point de divergence entre les deux courbes indique le moment optimal pour arrêter l'entraînement (early stopping implicite).

**Phase 3 — Optimisation par RandomizedSearchCV.** Pour les deux modèles, une recherche aléatoire explore **60 combinaisons** parmi un large espace d'hyperparamètres. Le choix de `RandomizedSearchCV` plutôt que `GridSearchCV` est judicieux ici car l'espace de recherche est considérablement plus grand (nombre d'estimateurs, profondeur, learning rate, subsampling, régularisation L1/L2, gamma, num_leaves, min_child_samples...).

Meilleurs hyperparamètres XGBoost :

| Hyperparamètre | Valeur |
|:---------------|:------:|
| `n_estimators` | 300 |
| `max_depth` | 3 |
| `learning_rate` | 0.2 |
| `subsample` | 0.7 |
| `colsample_bytree` | 1.0 |
| `gamma` | 0.5 |
| `reg_lambda` | 1 |

Meilleurs hyperparamètres LightGBM :

| Hyperparamètre | Valeur |
|:---------------|:------:|
| `n_estimators` | 300 |
| `max_depth` | 5 |
| `learning_rate` | 0.05 |
| `num_leaves` | 31 |
| `subsample` | 1.0 |
| `colsample_bytree` | 0.8 |
| `min_child_samples` | 20 |

**Phase 4 — Analyse SHAP.** Le code utilise la bibliothèque SHAP (SHapley Additive exPlanations) pour une interprétabilité avancée du modèle XGBoost. Deux graphiques sont produits :
- Un **bar plot** montrant l'importance SHAP globale par classe.
- Un **beeswarm plot** pour la première classe (« Excluded »), montrant l'effet de chaque feature value sur la prédiction.

**Phase 5 — Méthodes d'ensemble.** Deux stratégies d'ensemble combinent XGBoost et LightGBM :
- **Soft Voting** : Moyenne pondérée des probabilités prédites par les deux modèles.
- **Stacking** : Les probabilités prédites par XGBoost et LightGBM servent de features d'entrée à une régression logistique de second niveau.

## VI.3 · Résultats et interprétation des graphes

### Rapports de classification

**XGBoost** (Accuracy : 62.00%) :

| Classe | Précision | Rappel | F1-Score | Support |
|:-------|:---------:|:------:|:--------:|:-------:|
| Excluded | 1.00 | 1.00 | 1.00 | 25 |
| Included | 0.40 | 0.38 | 0.39 | 32 |
| Partially Included | 0.56 | 0.58 | 0.57 | 43 |

**LightGBM** (Accuracy : 60.00%) :

| Classe | Précision | Rappel | F1-Score | Support |
|:-------|:---------:|:------:|:--------:|:-------:|
| Excluded | 1.00 | 1.00 | 1.00 | 25 |
| Included | 0.36 | 0.31 | 0.33 | 32 |
| Partially Included | 0.53 | 0.58 | 0.56 | 43 |

**Interprétation** : Résultat surprenant — les méthodes de Gradient Boosting obtiennent des performances *inférieures* au Random Forest et à l'Arbre de Décision. Cela s'explique par la taille modeste du dataset (500 observations). XGBoost et LightGBM sont conçus pour des datasets plus volumineux ; sur un petit jeu de données avec peu de features, leur complexité supplémentaire n'apporte pas de gain et peut même nuire (sur-optimisation des résidus sur un échantillon limité).

### Courbes de perte XGBoost

Le graphe de l'évolution du `mlogloss` montre que le loss test se stabilise rapidement puis commence à remonter légèrement, tandis que le loss train continue de diminuer. Ce comportement classique de surapprentissage conforte l'utilisation d'un early stopping. La meilleure itération est identifiée par le point vert sur le graphe.

### Matrices de confusion comparatives

Les matrices côte à côte révèlent un schéma similaire aux méthodes précédentes : classification parfaite des « Excluded » et confusions fréquentes entre les deux autres classes. XGBoost commet légèrement moins d'erreurs que LightGBM sur cet échantillon.

### Courbes ROC

Les courbes ROC pour les deux modèles montrent des AUC macro respectifs de ~0.79 et ~0.79, légèrement en dessous du Random Forest (0.807). La classe « Excluded » obtient systématiquement un AUC proche de 1.0.

### Feature Importance comparative et SHAP

Le graphe comparatif de l'importance des features entre XGBoost et LightGBM montre des différences de hiérarchie intéressantes. Les deux modèles s'accordent sur l'importance dominante de `Income_Monthly_MAD` et `Has_Bank_Account`, mais divergent sur le poids relatif des autres features.

L'analyse SHAP fournit une compréhension plus fine : les valeurs SHAP par classe montrent que `Has_Bank_Account` a l'impact SHAP le plus élevé pour la classe « Excluded », confirmant que l'absence de compte bancaire est le prédicteur le plus fort de l'exclusion financière.

### Résultats des ensembles

| Modèle | Accuracy |
|:-------|:--------:|
| XGBoost | 0.6200 |
| LightGBM | 0.6000 |
| Soft Voting | 0.6000 |
| **Stacking** | **0.6300** |

Le Stacking obtient le meilleur résultat parmi les méthodes de Gradient Boosting, ce qui montre que la combinaison des deux modèles avec un méta-apprenant (régression logistique) capture une complémentarité entre XGBoost et LightGBM. Cependant, même le Stacking reste en deçà du Random Forest et de l'Arbre de Décision.

### Tableau comparatif final (XGBoost & LightGBM)

| Modèle | Accuracy | F1 Macro | AUC ROC | CV Mean |
|:-------|:--------:|:--------:|:-------:|:-------:|
| Stacking | 0.6300 | 0.6262 | 0.7961 | 0.6560 ± 0.038 |
| XGBoost | 0.6200 | 0.6518 | 0.7876 | 0.6725 |
| LightGBM | 0.6000 | 0.6296 | 0.7864 | 0.6750 |
| Soft Voting | 0.6000 | 0.6334 | 0.7922 | — |

---

<br>

<div align="center">

# ── VII · COMPARAISON FINALE DES TROIS MÉTHODES ──

</div>

---

### Tableau récapitulatif global

| Métrique | Random Forest | Arbre de Décision | XGBoost | LightGBM | Stacking (XGB+LGBM) |
|:---------|:------------:|:-----------------:|:-------:|:--------:|:--------------------:|
| **Accuracy** | **0.6500** | **0.6500** | 0.6200 | 0.6000 | 0.6300 |
| **F1 Macro** | 0.6741 | **0.6869** | 0.6518 | 0.6296 | 0.6262 |
| **Précision Macro** | 0.6759 | **0.6882** | 0.6533 | 0.6300 | — |
| **Rappel Macro** | 0.6754 | **0.6887** | 0.6533 | 0.6300 | — |
| **AUC ROC Macro** | **0.8071** | 0.7616 | 0.7876 | 0.7864 | 0.7961 |
| **CV Score** | 0.6600 ± 0.052 | 0.6380 ± 0.027 | **0.6725** | **0.6750** | 0.6560 ± 0.038 |
| **Interprétabilité** | Moyenne | **Élevée** | Faible | Faible | Très faible |

### Analyse comparative

**En termes d'accuracy**, le Random Forest et l'Arbre de Décision sont à égalité (65%), devançant le Gradient Boosting. C'est un résultat atypique par rapport aux benchmarks habituels où XGBoost domine. L'explication réside dans la taille du dataset : avec seulement 500 observations et 5 features, la puissance du boosting ne se manifeste pas.

**En termes de F1 Macro**, l'Arbre de Décision arrive en tête (0.6869), ce qui signifie qu'il offre le meilleur équilibre entre précision et rappel à travers les trois classes. Le Random Forest le suit de près (0.6741).

**En termes d'AUC ROC**, le Random Forest est le meilleur (0.8071), indiquant qu'il produit les probabilités de classe les plus calibrées. C'est un avantage important si on souhaite utiliser le modèle pour du scoring plutôt que de la classification dure.

**En termes de robustesse** (écart-type CV), l'Arbre de Décision présente la plus faible variance (±0.027), tandis que le Random Forest a la plus forte (±0.052). Les modèles de Gradient Boosting se situent entre les deux.

**En termes d'interprétabilité**, l'Arbre de Décision est imbattable : on peut extraire et lire les règles de décision directement. Le Random Forest offre une importance des features mais reste une « boîte noire » au niveau des prédictions individuelles. XGBoost et LightGBM nécessitent des outils spécialisés comme SHAP pour être interprétés.

### Verdict

Pour ce problème spécifique de classification de l'inclusion financière au Maroc avec un dataset de 500 observations :

- **Meilleur modèle global** : L'**Arbre de Décision optimisé** offre le meilleur compromis performance/interprétabilité, avec le F1 Macro le plus élevé et des règles de décision directement exploitables par les décideurs.
- **Meilleur modèle pour le scoring probabiliste** : Le **Random Forest**, grâce à son AUC ROC supérieur.
- **Le Gradient Boosting** (XGBoost/LightGBM) sous-performe sur ce dataset, mais resterait potentiellement le choix optimal avec un volume de données plus important.

---

<br>

<div align="center">

# ── VIII · CONCLUSION ──

</div>

---

Ce projet a permis d'explorer trois familles d'algorithmes de classification supervisée appliquées à l'inclusion financière au Maroc. L'analyse révèle plusieurs enseignements fondamentaux.

Premièrement, la **possession d'un compte bancaire** et le **revenu mensuel** sont les deux facteurs les plus déterminants de l'inclusion financière, quelle que soit la méthode utilisée. Ce résultat, convergent à travers les trois approches, suggère que les politiques publiques visant à élargir l'accès aux services bancaires et à améliorer les revenus auraient l'impact le plus direct sur l'inclusion financière.

Deuxièmement, la **taille du dataset** (500 observations) constitue une limitation significative. Elle explique les performances modérées de l'ensemble des modèles (~65% d'accuracy) et le fait que les méthodes les plus complexes (Gradient Boosting) n'apportent pas de gain par rapport aux méthodes plus simples. Cela illustre un principe fondamental en Machine Learning : la complexité du modèle doit être proportionnée à la quantité et à la richesse des données disponibles.

Troisièmement, la **difficulté principale** réside dans la distinction entre les individus « Included » et « Partially Included ». Les features disponibles ne capturent pas suffisamment les nuances entre ces deux catégories. L'ajout de variables supplémentaires — type d'emploi, localisation géographique (rural/urbain), utilisation de services financiers spécifiques, historique de crédit — pourrait améliorer significativement les performances.

En perspective, ce travail pourrait être prolongé par l'utilisation de techniques de *feature engineering* plus poussées, par l'exploration d'algorithmes de deep learning (réseaux de neurones) si le volume de données augmente, et par la mise en place d'une interface de déploiement permettant de prédire en temps réel le statut d'inclusion financière d'un individu à partir de ses caractéristiques.

---

<br>

<div align="center">

# ── IX · WEBOGRAPHIE ──

</div>

---

- **Scikit-learn** — Documentation officielle : Random Forest, Decision Trees, Métriques  
  → https://scikit-learn.org/stable/

- **XGBoost** — Documentation officielle  
  → https://xgboost.readthedocs.io/

- **LightGBM** — Documentation officielle  
  → https://lightgbm.readthedocs.io/

- **SHAP** — SHapley Additive exPlanations  
  → https://shap.readthedocs.io/

- **Bank Al-Maghrib** — Stratégie Nationale d'Inclusion Financière  
  → https://www.bkam.ma/

- **World Bank** — Global Findex Database  
  → https://www.worldbank.org/en/publication/globalfindex

- **Kaggle** — Ressources et datasets sur l'inclusion financière  
  → https://www.kaggle.com/

- **Matplotlib / Seaborn** — Documentation pour la visualisation  
  → https://matplotlib.org/ · https://seaborn.pydata.org/

---

<br>

<div align="center">

# ── X · ANNEXES ──

</div>

---

### Annexe A — Configuration de l'environnement

L'ensemble du code a été développé et exécuté sur **Google Colab** avec les bibliothèques suivantes :

| Bibliothèque | Usage |
|:-------------|:------|
| `numpy`, `pandas` | Manipulation de données |
| `matplotlib`, `seaborn`, `plotly` | Visualisation |
| `scikit-learn` | Prétraitement, modélisation, évaluation |
| `xgboost` | Algorithme XGBoost |
| `lightgbm` | Algorithme LightGBM |
| `shap` | Interprétabilité des modèles |

### Annexe B — Paramètres de reproduction

Tous les modèles utilisent `random_state=42` pour la reproductibilité. Le split train/test est de 80/20 avec stratification. La validation croisée est à 5 plis stratifiés.

### Annexe C — Résumé chiffré des résultats

| Modèle | Accuracy Test | F1 Macro | AUC ROC | CV Mean |
|:-------|:------------:|:--------:|:-------:|:-------:|
| Random Forest | 65.00% | 67.41% | 80.71% | 66.00% |
| Arbre de Décision | 65.00% | 68.69% | 76.16% | 63.80% |
| XGBoost | 62.00% | 65.18% | 78.76% | 67.25% |
| LightGBM | 60.00% | 62.96% | 78.64% | 67.50% |
| Stacking (XGB+LGBM) | 63.00% | 62.62% | 79.61% | 65.60% |

### Annexe D — Structure du notebook

Le notebook Jupyter est organisé en trois blocs principaux correspondant aux trois méthodes :

- **Cellules 0–23** : Random Forest (imports, exploration, prétraitement, entraînement, évaluation, feature importance, courbes d'apprentissage, tableau de bord)
- **Cellules 24–46** : Arbre de Décision (imports, prétraitement, entraînement/optimisation, visualisation de l'arbre, évaluation, analyse du surapprentissage, feature importance, comparaison des critères)
- **Cellules 47–72** : XGBoost & LightGBM (imports, prétraitement, entraînement XGBoost, entraînement LightGBM, évaluation comparative, SHAP, ensembles, comparaison finale)

---

<div align="center">

<br>

*Projet réalisé dans le cadre du module de Data Science — S8 Finance — ENCG Settat*

**Aymane JEBBOUR · Adam LAMRAHI**

Année Universitaire 2025–2026

<br>

---

</div>
