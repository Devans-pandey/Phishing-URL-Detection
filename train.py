"""
Model Training Script for PhiUSIIL Phishing URL Dataset (235,795 URLs, 50 Features)
Trains a high-performance HistGradientBoostingClassifier and exports pickle/model.pkl
"""

import os
import time
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

DATASET_PATH = "PhiUSIIL_Phishing_URL_Dataset.csv"
PICKLE_DIR = "pickle"

def train_and_export():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    print(f"Loading {DATASET_PATH}...")
    t0 = time.time()
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(df):,} samples in {time.time() - t0:.2f} seconds.")

    # Generate and export TLD probability mapping for fast inference
    os.makedirs(PICKLE_DIR, exist_ok=True)
    tld_map = df.groupby('TLD')['TLDLegitimateProb'].mean().to_dict()
    tld_map_path = os.path.join(PICKLE_DIR, "tld_map.pkl")
    with open(tld_map_path, "wb") as f:
        pickle.dump(tld_map, f)
    print(f"Saved TLD lookup table ({len(tld_map)} TLDs) to {tld_map_path}")

    # Exclude non-feature identifier / text columns
    drop_cols = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title', 'label']
    X = df.drop(columns=drop_cols)
    y = df['label']
    feature_cols = list(X.columns)

    feature_cols_path = os.path.join(PICKLE_DIR, "feature_cols.pkl")
    with open(feature_cols_path, "wb") as f:
        pickle.dump(feature_cols, f)
    print(f"Features: {len(feature_cols)}")

    print("Splitting dataset (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training HistGradientBoostingClassifier on 235,000+ samples...")
    t1 = time.time()
    clf = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=42
    )
    clf.fit(X_train, y_train)
    print(f"Training completed in {time.time() - t1:.2f} seconds!")

    print("\nEvaluating model on test set...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Phishing (0)", "Legitimate (1)"]))

    model_path = os.path.join(PICKLE_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    print(f"Successfully saved modernized model to {model_path}!")

if __name__ == "__main__":
    train_and_export()
