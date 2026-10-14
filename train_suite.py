"""
Multi-Model Training Script for PhiUSIIL Phishing URL Detection
Trains and exports multiple models for comparative evaluation and ensemble predictions.
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATASET_PATH = "PhiUSIIL_Phishing_URL_Dataset.csv"
PICKLE_DIR = "pickle"

def train_suite():
    print(f"Loading {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH, nrows=50000)

    drop_cols = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title', 'label']
    X = df.drop(columns=drop_cols)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=100, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=25, max_depth=12, random_state=42, n_jobs=-1),
        'DecisionTree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=500, solver='liblinear')
    }

    trained_suite = {}
    print("\nTraining Model Suite...")
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        trained_suite[name] = {
            'model': m,
            'accuracy': round(acc * 100, 2)
        }
        print(f"  > {name:<22}: {acc * 100:.2f}% accuracy")

    os.makedirs(PICKLE_DIR, exist_ok=True)
    suite_path = os.path.join(PICKLE_DIR, "models_suite.pkl")
    with open(suite_path, "wb") as f:
        pickle.dump(trained_suite, f)

    # Keep default model as HistGradientBoosting
    with open(os.path.join(PICKLE_DIR, "model.pkl"), "wb") as f:
        pickle.dump(trained_suite['HistGradientBoosting']['model'], f)

    print(f"\nSuccessfully exported model suite to {suite_path}")

if __name__ == "__main__":
    train_suite()
