import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import joblib
import os
from datetime import datetime

# Charger les donnees
iris = load_iris()
X, y = iris.data, iris.target

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Définir la matrice d'expériences
experiments = [
    # RandomForest avec différents n_estimators
    {
        "name": "RandomForest_50",
        "model": RandomForestClassifier(n_estimators=50, random_state=42),
        "params": {"n_estimators": 50}
    },
    {
        "name": "RandomForest_100",
        "model": RandomForestClassifier(n_estimators=100, random_state=42),
        "params": {"n_estimators": 100}
    },
    {
        "name": "RandomForest_200",
        "model": RandomForestClassifier(n_estimators=200, random_state=42),
        "params": {"n_estimators": 200}
    },
    # SVM avec différents kernels
    {
        "name": "SVM_linear",
        "model": SVC(kernel='linear', random_state=42),
        "params": {"kernel": "linear"}
    },
    {
        "name": "SVM_rbf",
        "model": SVC(kernel='rbf', random_state=42),
        "params": {"kernel": "rbf"}
    },
    {
        "name": "SVM_poly",
        "model": SVC(kernel='poly', degree=3, random_state=42),
        "params": {"kernel": "poly", "degree": 3}
    },
]

# Créer le répertoire experiments s'il n'existe pas
os.makedirs('experiments', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Entraîner tous les modèles
results = []

print("=" * 60)
print("Entraînement de la matrice d'expériences")
print("=" * 60)

for exp in experiments:
    print(f"\nEntraînement: {exp['name']}")
    
    # Entraîner le modèle
    model = exp['model']
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Calculer les métriques
    metrics = {
        "name": exp['name'],
        "algorithm": exp['model'].__class__.__name__,
        "params": exp['params'],
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average='weighted')),
        "recall": float(recall_score(y_test, y_pred, average='weighted')),
        "f1_score": float(f1_score(y_test, y_pred, average='weighted')),
        "test_size": len(X_test),
        "timestamp": datetime.now().isoformat()
    }
    
    # Sauvegarder le modèle
    model_path = f'models/{exp["name"]}.pkl'
    joblib.dump(model, model_path)
    
    # Sauvegarder les métriques individuelles
    metrics_path = f'experiments/{exp["name"]}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    results.append(metrics)
    
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  Modèle sauvegardé: {model_path}")

print("\n" + "=" * 60)
print(f"Entraînement terminé: {len(results)} modèles")
print("=" * 60)

# Sauvegarder tous les résultats
with open('experiments/all_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nRésultats sauvegardés dans experiments/all_results.json")

