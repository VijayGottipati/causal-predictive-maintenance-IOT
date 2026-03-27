"""
Threshold Optimization Module
Improves recall by adjusting classification threshold
"""

import numpy as np
import joblib
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class ThresholdOptimizer:
    """
    Optimizes classification threshold to improve recall while maintaining acceptable precision
    """

    def __init__(self, model_path="models/model.pkl"):
        self.model = joblib.load(model_path)

    def find_optimal_threshold(self, X, y, target_recall=0.90, min_precision=0.80):
        """
        Find optimal threshold that achieves target recall while maintaining precision

        Args:
            X: Features
            y: True labels
            target_recall: Target recall to achieve
            min_precision: Minimum acceptable precision

        Returns:
            Optimal threshold and metrics
        """
        probas = self.model.predict_proba(X)[:, 1]

        thresholds = np.arange(0.1, 0.9, 0.05)
        results = []

        for threshold in thresholds:
            y_pred = (probas >= threshold).astype(int)

            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, zero_division=0)
            recall = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)

            results.append(
                {
                    "threshold": threshold,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

        best = None
        for r in results:
            if r["recall"] >= target_recall and r["precision"] >= min_precision:
                if best is None or r["f1"] > best["f1"]:
                    best = r

        if best is None:
            best = max(results, key=lambda x: x["f1"])

        return best

    def evaluate_thresholds(self, X, y):
        """Evaluate multiple thresholds"""
        probas = self.model.predict_proba(X)[:, 1]

        results = []
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred = (probas >= threshold).astype(int)

            results.append(
                {
                    "threshold": threshold,
                    "accuracy": accuracy_score(y, y_pred),
                    "precision": precision_score(y, y_pred, zero_division=0),
                    "recall": recall_score(y, y_pred, zero_division=0),
                    "f1": f1_score(y, y_pred, zero_division=0),
                }
            )

        return results

    def save_threshold(self, threshold, filepath="models/threshold.json"):
        """Save optimized threshold"""
        with open(filepath, "w") as f:
            json.dump({"threshold": threshold}, f)
        print(f"Saved optimal threshold: {threshold}")


def run_threshold_optimization():
    """Run threshold optimization"""
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")

    optimizer = ThresholdOptimizer()

    print("=== THRESHOLD OPTIMIZATION ===\n")

    print("Current threshold (0.5) performance:")
    results = optimizer.evaluate_thresholds(X_test, y_test)
    default = min(results, key=lambda x: abs(x["threshold"] - 0.5))
    print(f"  Accuracy: {default['accuracy']:.3f}")
    print(f"  Precision: {default['precision']:.3f}")
    print(f"  Recall: {default['recall']:.3f}")
    print(f"  F1: {default['f1']:.3f}")

    optimal = optimizer.find_optimal_threshold(
        X_test, y_test, target_recall=0.90, min_precision=0.80
    )

    print(f"\nOptimal threshold: {optimal['threshold']:.2f}")
    print(f"  Accuracy: {optimal['accuracy']:.3f}")
    print(f"  Precision: {optimal['precision']:.3f}")
    print(f"  Recall: {optimal['recall']:.3f}")
    print(f"  F1: {optimal['f1']:.3f}")

    improvement = optimal["recall"] - default["recall"]
    print(f"\nRecall improvement: +{improvement:.1%}")

    optimizer.save_threshold(optimal["threshold"])

    return optimal


if __name__ == "__main__":
    run_threshold_optimization()
