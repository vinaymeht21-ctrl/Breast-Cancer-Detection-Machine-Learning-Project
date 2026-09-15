"""
train_model.py
---------------
Reproduces the workflow from the original notebook (Untitled.ipynb):

  1. Load data.csv (Breast Cancer Wisconsin diagnostic dataset)
  2. Drop the 'id' and 'Unnamed: 32' columns
  3. Encode diagnosis: M -> 1 (Malignant), B -> 0 (Benign)
  4. Standardize the 30 numeric features with StandardScaler
  5. Train/test split (70/30, random_state=42)
  6. Train a LogisticRegression classifier
  7. Evaluate accuracy / classification report
  8. Persist the fitted model, scaler, and feature metadata to disk
     so the Streamlit app (app.py) can load them instantly without
     retraining on every run.

Run this once before starting the app:

    python train_model.py
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
META_PATH = "feature_meta.json"


def main():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)

    # 2. Clean up columns (mirrors the notebook exactly)
    drop_cols = [c for c in ["Unnamed: 32", "id"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # 3. Encode target
    df["diagnosis"] = [1 if v == "M" else 0 for v in df["diagnosis"]]

    y = df["diagnosis"]
    X = df.drop(columns=["diagnosis"])
    feature_names = X.columns.tolist()

    # 4. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.30, random_state=42
    )

    # 6. Train model
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    # 7. Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Benign", "Malignant"])
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}\n")
    print("Classification report:")
    print(report)
    print("Confusion matrix:")
    print(cm)

    # 8. Persist artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    # Save per-feature stats (min/max/mean/std) from the raw (unscaled) data.
    # The Streamlit app uses these to build sensible slider ranges and to
    # let the user reset to "average patient" values.
    stats = {
        "feature_names": feature_names,
        "min": X.min().round(5).to_dict(),
        "max": X.max().round(5).to_dict(),
        "mean": X.mean().round(5).to_dict(),
        "std": X.std().round(5).to_dict(),
        "accuracy": round(float(acc), 4),
    }
    with open(META_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved scaler -> {SCALER_PATH}")
    print(f"Saved feature metadata -> {META_PATH}")


if __name__ == "__main__":
    main()
