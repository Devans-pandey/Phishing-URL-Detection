"""
Multi-Model Training Script for PhiUSIIL Phishing URL Detection
Trains and exports 6 models for comparative evaluation and ensemble predictions:
  Logistic Regression, XGBoost, Decision Tree, SVM, HistGradientBoosting, Random Forest
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

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
        'Logistic Regression':    make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, solver='liblinear', class_weight='balanced')),
        'XGBoost':                XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss'),
        'Decision Tree':          DecisionTreeClassifier(max_depth=10, random_state=42),
        'SVM':                    CalibratedClassifierCV(LinearSVC(dual='auto', max_iter=2000, random_state=42)),
        'HistGradientBoosting':   HistGradientBoostingClassifier(max_iter=100, random_state=42),
        'Random Forest':          RandomForestClassifier(n_estimators=25, max_depth=12, random_state=42, n_jobs=-1),
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

    # Export XGBoost as primary fallback model.pkl
    with open(os.path.join(PICKLE_DIR, "model.pkl"), "wb") as f:
        pickle.dump(trained_suite['XGBoost']['model'], f)

    print(f"\nSuccessfully exported model suite to {suite_path}")

if __name__ == "__main__":
    train_suite()
