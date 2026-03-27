import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple

INTERVENTION_CONFIGS = {
    "reduce_temp": {
        "feature": "Air temperature",
        "change": -0.1,
        "description": "Reduce air temperature by 10%",
    },
    "reduce_torque": {
        "feature": "Torque",
        "change": -0.2,
        "description": "Reduce torque by 20%",
    },
    "reduce_speed": {
        "feature": "Rotational speed",
        "change": -0.15,
        "description": "Reduce rotational speed by 15%",
    },
    "reduce_wear": {
        "feature": "Tool wear",
        "change": -0.3,
        "description": "Reduce tool wear by 30%",
    },
    "reduce_power": {
        "feature": "Power",
        "change": -0.25,
        "description": "Reduce power consumption by 25%",
    },
}


class WhatIfEngine:
    """
    Digital Twin - What-if Simulator

    Tests various interventions to find which can prevent failures
    """

    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler

    def simulate(self, X, intervention_name):
        """
        Apply intervention and predict outcome

        Args:
            X: features (before intervention)
            intervention_name: name of intervention to apply

        Returns:
            predictions after intervention
        """
        if self.model is None:
            return None

        X_intervention = X.copy()

        config = INTERVENTION_CONFIGS.get(intervention_name)
        if config is None:
            return None

        feature_idx = self._get_feature_index(config["feature"])
        if feature_idx is not None:
            X_intervention[:, feature_idx] *= 1 + config["change"]

        proba = self.model.predict_proba(X_intervention)[:, 1]

        return proba

    def _get_feature_index(self, feature_name):
        feature_map = {
            "Type": 0,
            "Air temperature": 1,
            "Process temperature": 2,
            "Rotational speed": 3,
            "Torque": 4,
            "Tool wear": 5,
            "Temp_Diff": 6,
            "Power": 7,
            "Temp_Rate": 8,
            "Wear_Stress": 9,
        }
        return feature_map.get(feature_name)

    def test_all_interventions(self, X):
        """
        Test all possible interventions on a single sample
        """
        results = {}

        for intervention_name in INTERVENTION_CONFIGS.keys():
            proba = self.simulate(X, intervention_name)
            results[intervention_name] = proba

        return results

    def find_optimal_intervention(self, X, threshold=0.5):
        """
        Find intervention that reduces failure probability below threshold
        """
        all_results = self.test_all_interventions(X)

        best_intervention = None
        best_prob = 1.0

        for intervention, prob in all_results.items():
            if isinstance(prob, np.ndarray):
                prob = prob[0]

            if prob < best_prob:
                best_prob = prob
                best_intervention = intervention

        if best_prob < threshold:
            return {
                "intervention": best_intervention,
                "probability": best_prob,
                "reduces_failure": True,
            }
        else:
            return {
                "intervention": None,
                "probability": best_prob,
                "reduces_failure": False,
            }


class ActionabilityClassifier:
    """
    Classifies failure cases as Actionable vs Unavoidable

    Actionable: can be prevented through intervention
    Unavoidable: no intervention works
    """

    def __init__(self, whatif_engine):
        self.whatif_engine = whatif_engine

    def classify(self, X, y_true, threshold=0.5):
        """
        Classify each sample as actionable or unavoidable

        Args:
            X: features
            y_true: actual labels (1 = failure)
            threshold: probability threshold for intervention

        Returns:
            actionability_labels: 1=actionable, 0=unavoidable
            weights: weight for each sample in loss function
        """
        actionability = []

        failure_mask = y_true == 1
        failure_indices = np.where(failure_mask)[0]

        for idx in failure_indices:
            X_sample = X[idx : idx + 1]

            result = self.whatif_engine.find_optimal_intervention(X_sample, threshold)

            if result["reduces_failure"]:
                actionability.append(1)
            else:
                actionability.append(0)

        return np.array(actionability)

    def compute_weights(
        self, X, y, actionability_labels, actionable_weight=1.0, unavoidable_weight=0.3
    ):
        """
        Compute sample weights for intervention-weighted loss

        Higher weight for actionable failures (can be prevented)
        Lower weight for unavoidable failures (no intervention helps)
        """
        weights = np.ones(len(y))

        failure_mask = y == 1
        failure_indices = np.where(failure_mask)[0]

        for i, idx in enumerate(failure_indices):
            if actionability_labels[i] == 1:
                weights[idx] = actionable_weight
            else:
                weights[idx] = unavoidable_weight

        return weights

    def get_actionability_report(self, X, y):
        failure_mask = y == 1
        failure_X = X[failure_mask]
        failure_y = y[failure_mask]

        actionability_labels = self.classify(failure_X, failure_y)

        n_actionable = np.sum(actionability_labels == 1)
        n_unavoidable = np.sum(actionability_labels == 0)
        total_failures = len(failure_y)

        return {
            "total_failures": total_failures,
            "actionable": n_actionable,
            "unavoidable": n_unavoidable,
            "actionable_ratio": n_actionable / total_failures
            if total_failures > 0
            else 0,
            "description": f"{n_actionable}/{total_failures} failures can be prevented through intervention",
        }


class WeightedLossTrainer:
    """
    Trains model with Intervention-Weighted Loss

    Prioritizes learning patterns that can be prevented through intervention
    """

    def __init__(self, model, actionability_classifier):
        self.model = model
        self.actionability_classifier = actionability_classifier

    def fit_with_weights(self, X, y, **kwargs):
        """
        Train model with sample weights based on actionability
        """
        X_actionable = X[y == 1]
        y_actionable = y[y == 1]

        actionability_labels = self.actionability_classifier.classify(
            X_actionable, y_actionable
        )

        weights = self.actionability_classifier.compute_weights(
            X, y, actionability_labels
        )

        self.model.fit(X, y, sample_weight=weights, **kwargs)

        return self.model, weights

    def get_actionability_report(self, X, y):
        """
        Generate report on actionability of failures
        """
        failure_mask = y == 1
        failure_X = X[failure_mask]
        failure_y = y[failure_mask]

        actionability_labels = self.actionability_classifier.classify(
            failure_X, failure_y
        )

        n_actionable = np.sum(actionability_labels == 1)
        n_unavoidable = np.sum(actionability_labels == 0)
        total_failures = len(failure_y)

        return {
            "total_failures": total_failures,
            "actionable": n_actionable,
            "unavoidable": n_unavoidable,
            "actionable_ratio": n_actionable / total_failures
            if total_failures > 0
            else 0,
            "description": f"{n_actionable}/{total_failures} failures can be prevented through intervention",
        }


def run_iwl():
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    model = joblib.load("models/model.pkl")
    scaler = joblib.load("data/scaler.pkl")

    X_train = np.load("data/X_train.npy")
    X_test = np.load("data/X_test.npy")
    y_train = np.load("data/y_train.npy")
    y_test = np.load("data/y_test.npy")

    whatif = WhatIfEngine(model, scaler)
    classifier = ActionabilityClassifier(whatif)

    report = classifier.get_actionability_report(X_train, y_train)

    print("\n=== INTERVENTION-WEIGHTED LOSS (IWL) ===")
    print(f"\n{report['description']}")
    print(f"Actionable failures: {report['actionable']}")
    print(f"Unavoidable failures: {report['unavoidable']}")
    print(f"Actionable ratio: {report['actionable_ratio']:.1%}")

    weights = classifier.compute_weights(
        X_train,
        y_train,
        classifier.classify(X_train[y_train == 1], y_train[y_train == 1]),
    )

    print(f"\nWeight distribution:")
    print(f"  - High weight (actionable): {np.sum(weights > 0.5)} samples")
    print(f"  - Low weight (unavoidable): {np.sum(weights <= 0.5)} samples")

    joblib.dump(classifier, "models/iwl_classifier.pkl")
    print("\nIWL classifier saved to models/iwl_classifier.pkl")

    return classifier, report


if __name__ == "__main__":
    run_iwl()
