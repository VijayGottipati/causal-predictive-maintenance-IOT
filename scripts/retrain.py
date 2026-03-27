import numpy as np
import joblib
import os
import json
import sys

sys.path.insert(0, ".")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from optimizations.counterfactual import CSOSIntegrator

MODEL_DIR = "models"
DATA_DIR = "data"


def load_csoss_integrator():
    """Load CSOS integrator if available"""
    csos_path = os.path.join(MODEL_DIR, "csos_integrator.pkl")
    if os.path.exists(csos_path):
        try:
            return joblib.load(csos_path)
        except Exception as e:
            print(f"Could not load CSOS integrator: {e}")
            return None
    return None


def retrain_model(use_csos=True):
    print("=== SELF-HEALING: MODEL RETRAINING ===")
    print(f"CSOS enabled: {use_csos}\n")

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    print(f"Original training data: {len(X_train)} samples, {sum(y_train)} failures")

    sample_weights = None

    if use_csos:
        csos_integrator = load_csoss_integrator()

        if csos_integrator is not None:
            print("\nApplying CSOS augmentation...")
            X_train, y_train, sample_weights = csos_integrator.augment_dataset(
                X_train, y_train, n_synthetic_per_failure=15, near_miss_ratio=0.3
            )
            print(f"Augmented training data: {len(X_train)} samples")
        else:
            print("CSOS integrator not found, using standard training")

    print(f"\nTraining on {len(X_train)} samples")
    print(f"Testing on {len(X_test)} samples")

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    if sample_weights is not None:
        print("Training with sample weights (CSOS + IWL integration)...")
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print(f"\nModel trained with F1 (weighted): {f1:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    joblib.dump(model, model_path)

    print(f"\nNew model saved to: {model_path}")

    version_file = os.path.join(MODEL_DIR, "model_version.json")

    csos_status = (
        "CSOS augmented" if use_csos and sample_weights is not None else "Standard"
    )

    version_data = {
        "version": "v3.0",
        "f1_score_weighted": float(f1),
        "f1_score_macro": float(f1_macro),
        "retrain_reason": "drift_detected",
        "csos_enabled": use_csos,
        "training_status": csos_status,
        "training_samples": len(X_train),
    }
    with open(version_file, "w") as f:
        json.dump(version_data, f, indent=2)

    from scripts.mlflow_helper import log_training_run

    log_training_run(
        run_name="retrain_rf",
        params={
            "script": "retrain.py",
            "csos_enabled": use_csos,
            "training_status": csos_status,
            "n_estimators": 150,
        },
        metrics={
            "f1_weighted": float(f1),
            "f1_macro": float(f1_macro),
        },
    )

    return model, f1


if __name__ == "__main__":
    retrain_model(use_csos=True)
    print("\n=== RETRAINING COMPLETE ===")
