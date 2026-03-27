import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import os
import sys
from pathlib import Path
from datetime import datetime

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    X_train = np.load("data/X_train.npy")
    X_test = np.load("data/X_test.npy")
    y_train = np.load("data/y_train.npy")
    y_test = np.load("data/y_test.npy")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, f1


def save_model(model):
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model_path


if __name__ == "__main__":
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    print("\nTraining baseline model...")
    model, f1 = train_model(X_train, y_train, X_test, y_test)

    model_path = save_model(model)

    print(f"\nBaseline model trained with F1: {f1:.4f}")
    print(f"Model saved to: {model_path}")

    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from scripts.mlflow_helper import log_training_run

    log_training_run(
        run_name="baseline_rf",
        params={
            "script": "train.py",
            "n_estimators": 100,
            "max_depth": 10,
        },
        metrics={"f1_weighted": f1},
    )
