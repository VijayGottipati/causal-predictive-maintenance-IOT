"""
Vanilla baseline: scripts/train.py style — smaller RF, no class_weight, threshold 0.5.
No CSOS, IWL, CAQ, PMW, CCI, threshold tuning, or streaming hooks.

Run from project root: python experiments/baseline_no_optimizations/run_experiment.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
THRESHOLD = 0.5


def main() -> None:
    sys.path.insert(0, str(ROOT))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    X_train = np.load(DATA / "X_train.npy")
    X_test = np.load(DATA / "X_test.npy")
    y_train = np.load(DATA / "y_train.npy")
    y_test = np.load(DATA / "y_test.npy")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)

    metrics = {
        "variant": "baseline_no_optimizations",
        "description": "Vanilla RF (train.py recipe), no class_weight, threshold 0.5",
        "threshold": THRESHOLD,
        "n_train": int(len(X_train)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    }

    model_path = ARTIFACTS / "model.pkl"
    joblib.dump(model, model_path)
    metrics["model_path"] = str(model_path.relative_to(ROOT))
    metrics["model_size_bytes"] = os.path.getsize(model_path)

    out = ARTIFACTS / "metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(classification_report(y_test, y_pred))
    print(f"Saved model -> {model_path}")
    print(f"Saved metrics -> {out}")


if __name__ == "__main__":
    main()
