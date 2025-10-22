# Exercice 3 : Comparaison Automatique de Modèles

Ce projet est la correction de l'exercice 3, qui met en place un pipeline CML avancé pour la comparaison automatique de modèles et la sélection du meilleur.

## Objectif

L'objectif est d'automatiser l'entraînement et l'évaluation de plusieurs modèles avec différents hyperparamètres, de générer un rapport de comparaison complet et de sélectionner automatiquement le meilleur modèle en fonction de ses performances.

## Structure du Projet

```
.
├── .github/workflows/cml.yml       # Workflow de comparaison
├── train_experiments.py          # NOUVEAU: Script d'entraînement de la matrice
├── compare_models.py             # NOUVEAU: Script d'agrégation et de comparaison
├── requirements.txt              # Mis à jour
├── experiments/                  # NOUVEAU: Dossier pour les résultats
│   ├── all_results.json
│   ├── best_model.json
│   └── ..._metrics.json
├── models/                       # Contient tous les modèles entraînés
│   ├── RandomForest_50.pkl
│   └── ...
└── reports/                      # Contient les graphiques de comparaison
    ├── accuracy_comparison.png
    ├── all_metrics_comparison.png
    └── performance_heatmap.png
```

## Nouveautés de l'Exercice 3

### 1. `train_experiments.py`

Ce script est le cœur de l'entraînement :

- **Matrice d'expériences** : Définit une liste de modèles à entraîner (RandomForest, SVM) avec différents hyperparamètres.
- **Entraînement en boucle** : Itère sur chaque expérience, entraîne le modèle et calcule un ensemble complet de métriques (accuracy, precision, recall, F1-score).
- **Sauvegarde individuelle** : Chaque modèle entraîné est sauvegardé dans `models/` et ses métriques dans `experiments/`.
- **Résultats agrégés** : Tous les résultats sont compilés dans un unique fichier `experiments/all_results.json`.

### 2. `compare_models.py`

Ce script analyse les résultats de l'entraînement :

- **Agrégation** : Charge `all_results.json` dans un DataFrame pandas.
- **Sélection du meilleur modèle** : Trie les modèles par `accuracy` et identifie le meilleur, en sauvegardant ses informations dans `experiments/best_model.json`.
- **Génération de visualisations** : Crée plusieurs graphiques pour comparer les performances :
    - **`accuracy_comparison.png`** : Bar chart des accuracies.
    - **`all_metrics_comparison.png`** : Bar chart groupé pour toutes les métriques.
    - **`performance_heatmap.png`** : Heatmap pour une vue d'ensemble des performances.
- **Rapport console** : Affiche un tableau comparatif clair dans les logs du workflow.

### 3. `.github/workflows/cml.yml`

Le workflow est entièrement repensé pour ce nouveau pipeline :

- **Exécute `train_experiments.py`** pour entraîner tous les modèles.
- **Exécute `compare_models.py`** pour analyser les résultats et générer les graphiques.
- **Rapport CML complet** : Publie un rapport très détaillé dans la Pull Request, incluant :
    - Les informations du **meilleur modèle**.
    - Les **3 graphiques de comparaison**.
    - Les **résultats détaillés** de toutes les expériences dans une section dépliable (`<details>`).

## Résultat Attendu

Le commentaire CML dans la Pull Request devient un véritable tableau de bord de l'expérimentation, permettant de :

- Voir immédiatement quel est le meilleur modèle et pourquoi.
- Comparer visuellement les performances des différents modèles.
- Accéder aux données brutes de chaque expérience pour une analyse plus approfondie.

## Test Local

```bash
# Installer les dépendances
pip install -r requirements.txt

# Étape 1: Entraîner tous les modèles
python train_experiments.py

# Étape 2: Comparer les modèles et générer les rapports
python compare_models.py

# Vérifier les résultats
cat experiments/best_model.json
ls -l reports/
```

