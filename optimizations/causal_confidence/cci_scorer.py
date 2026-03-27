import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
import joblib

FEATURE_NAMES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    "Temp_Diff",
    "Power",
    "Temp_Rate",
    "Wear_Stress",
]

CAUSAL_FEATURES = ["Type", "Tool wear", "Wear_Stress"]


class CausalGraph:
    """
    Represents the causal structural causal model (SCM).
    Defines which variables can be intervened upon (root causes).
    """

    def __init__(self):
        self.nodes = FEATURE_NAMES
        self.edges = {
            "Type": [],
            "Air temperature": [],
            "Process temperature": ["Air temperature"],
            "Rotational speed": [],
            "Torque": ["Rotational speed"],
            "Tool wear": [],
            "Temp_Diff": ["Process temperature", "Air temperature"],
            "Power": ["Rotational speed", "Torque"],
            "Temp_Rate": ["Temp_Diff", "Air temperature"],
            "Wear_Stress": ["Tool wear", "Torque"],
        }

    def get_root_causes(self) -> List[str]:
        """Features with no parents - valid intervention targets"""
        root_causes = []
        for node, parents in self.edges.items():
            if len(parents) == 0:
                root_causes.append(node)
        return root_causes

    def get_interventionable_features(self) -> List[str]:
        """Features we can meaningfully intervene on"""
        return self.get_root_causes()


class CausalConfidenceInverter:
    """
    Causal Confidence Inversion (CCI)

    Measures prediction robustness by finding the minimum causal intervention
    needed to flip a prediction.

    Low CCI = Fragile prediction (tiny change flips it)
    High CCI = Robust prediction (major change needed)

    This addresses the "Brittle Model" problem where models are 99% confident
    about predictions that are physically impossible to maintain.
    """

    def __init__(self, model=None, scaler=None, causal_graph=None):
        self.model = model
        self.scaler = scaler
        self.causal_graph = causal_graph or CausalGraph()
        self.interventionable_features = (
            self.causal_graph.get_interventionable_features()
        )

    def compute_confidence(self, X, prediction_threshold=0.5) -> Dict:
        """
        Compute Causal Confidence Inversion score

        Args:
            X: Feature vector (can be 1D or 2D)
            prediction_threshold: Threshold to flip prediction

        Returns:
            dict with CCI score and analysis
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        original_proba = self.model.predict_proba(X)[0, 1]
        original_prediction = int(original_proba >= prediction_threshold)

        if original_prediction == 0:
            return {
                "original_prediction": 0,
                "original_proba": float(original_proba),
                "cci_score": 0.0,
                "interpretation": "No failure predicted - CCI not applicable",
                "is_robust": True,
                "min_intervention": None,
            }

        intervention_result = self._find_minimal_intervention(
            X[0], prediction_threshold
        )

        cci_score = intervention_result["cci_score"]

        interpretation = self._interpret_cci(cci_score)

        return {
            "original_prediction": original_prediction,
            "original_proba": float(original_proba),
            "cci_score": float(cci_score),
            "interpretation": interpretation["text"],
            "is_robust": interpretation["robust"],
            "min_intervention": intervention_result["intervention"],
            "flipped_proba": float(intervention_result["flipped_proba"]),
            "intervention_magnitude": float(intervention_result["magnitude"]),
        }

    def _find_minimal_intervention(self, X, threshold=0.5) -> Dict:
        """
        Find minimum intervention needed to flip prediction
        Uses iterative search - much faster than optimization
        """
        interventionable_indices = [
            FEATURE_NAMES.index(f)
            for f in self.interventionable_features
            if f in FEATURE_NAMES
        ]

        if not interventionable_indices:
            interventionable_indices = [4, 5, 9]

        best_delta = None
        best_magnitude = float("inf")

        for feature_idx in interventionable_indices:
            original_val = X[feature_idx]

            for direction in [-1, 1]:
                for scale in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
                    delta = np.zeros(len(interventionable_indices))
                    idx_in_delta = interventionable_indices.index(feature_idx)
                    delta[idx_in_delta] = direction * scale * abs(original_val)

                    X_modified = X.copy()
                    for i, idx in enumerate(interventionable_indices):
                        X_modified[idx] += delta[i]

                    X_scaled = (
                        self.scaler.transform(X_modified.reshape(1, -1))
                        if self.scaler
                        else X_modified.reshape(1, -1)
                    )
                    proba = self.model.predict_proba(X_scaled)[0, 1]

                    if proba < threshold:
                        magnitude = np.sqrt(np.sum(delta**2))
                        if magnitude < best_magnitude:
                            best_magnitude = magnitude
                            best_delta = delta.copy()
                    break

        if best_delta is None:
            return {
                "cci_score": 10.0,
                "intervention": {},
                "flipped_proba": threshold,
                "magnitude": 999.0,
            }

        cci_score = 1.0 / (best_magnitude + 0.01)

        X_flipped = X.copy()
        for i, idx in enumerate(interventionable_indices):
            X_flipped[idx] += best_delta[i]

        X_flipped_scaled = (
            self.scaler.transform(X_flipped.reshape(1, -1))
            if self.scaler
            else X_flipped.reshape(1, -1)
        )
        flipped_proba = self.model.predict_proba(X_flipped_scaled)[0, 1]

        intervention_dict = {}
        for i, idx in enumerate(interventionable_indices):
            if abs(best_delta[i]) > 0.01:
                intervention_dict[FEATURE_NAMES[idx]] = float(best_delta[i])

        return {
            "cci_score": cci_score,
            "intervention": intervention_dict,
            "flipped_proba": flipped_proba,
            "magnitude": best_magnitude,
        }

    def _interpret_cci(self, cci_score: float) -> Dict:
        """Interpret CCI score into human-readable format"""

        if cci_score < 0.5:
            return {
                "text": "FRAGILE - Tiny intervention flips prediction. Likely overfitting to nuisance variable.",
                "robust": False,
                "level": "Very Low",
            }
        elif cci_score < 1.0:
            return {
                "text": "LOW - Small intervention needed. Prediction may be spurious correlation.",
                "robust": False,
                "level": "Low",
            }
        elif cci_score < 2.0:
            return {
                "text": "MODERATE - Moderate intervention needed. Prediction has some causal basis.",
                "robust": True,
                "level": "Moderate",
            }
        elif cci_score < 5.0:
            return {
                "text": "HIGH - Large intervention required. Prediction is causally robust.",
                "robust": True,
                "level": "High",
            }
        else:
            return {
                "text": "VERY HIGH - Fundamental causal change needed. Deep causal structure. Trust this prediction.",
                "robust": True,
                "level": "Very High",
            }

    def batch_evaluate(self, X, prediction_threshold=0.5) -> List[Dict]:
        """Evaluate CCI for multiple samples"""
        results = []
        for i in range(len(X)):
            result = self.compute_confidence(X[i : i + 1], prediction_threshold)
            results.append(result)
        return results

    def get_robustness_summary(self, X, prediction_threshold=0.5) -> Dict:
        """Get summary statistics of model robustness"""
        results = self.batch_evaluate(X, prediction_threshold)

        cci_scores = [r["cci_score"] for r in results if r["cci_score"] > 0]

        if not cci_scores:
            return {"message": "No failure predictions to evaluate"}

        fragile_count = sum(1 for s in cci_scores if s < 1.0)
        robust_count = len(cci_scores) - fragile_count

        return {
            "total_predictions": len(X),
            "failures_predicted": len(cci_scores),
            "fragile_predictions": fragile_count,
            "robust_predictions": robust_count,
            "robustness_ratio": robust_count / len(cci_scores) if cci_scores else 0,
            "mean_cci": float(np.mean(cci_scores)),
            "median_cci": float(np.median(cci_scores)),
            "interpretation": f"{robust_count}/{len(cci_scores)} failure predictions are causally robust",
        }


class CCIService:
    """
    Service wrapper for CCI - integrates with API and model loading
    """

    def __init__(self, model_path="models/model.pkl", scaler_path="data/scaler.pkl"):
        self.model = None
        self.scaler = None
        self.cci_scorer = None
        self.model_path = model_path
        self.scaler_path = scaler_path

    def initialize(self):
        """Initialize model and CCI scorer"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)

        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)

        if self.model is not None:
            self.cci_scorer = CausalConfidenceInverter(
                model=self.model, scaler=self.scaler
            )

        return self.cci_scorer is not None

    def evaluate_prediction(self, features: np.ndarray) -> Dict:
        """Evaluate a single prediction with CCI"""
        if self.cci_scorer is None:
            self.initialize()

        if self.cci_scorer is None:
            return {"error": "Model not loaded"}

        return self.cci_scorer.compute_confidence(features)


def run_cci_demo():
    """Run CCI demonstration"""
    import joblib

    print("=== CCI - CAUSAL CONFIDENCE INVERSION ===\n")

    model = joblib.load("models/model.pkl")
    scaler = joblib.load("data/scaler.pkl")

    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")

    failure_indices = np.where(y_test == 1)[0]

    print(f"Total test samples: {len(X_test)}")
    print(f"Failure samples: {len(failure_indices)}")

    cci_scorer = CausalConfidenceInverter(model=model, scaler=scaler)

    sample_failures = (
        failure_indices[:5] if len(failure_indices) >= 5 else failure_indices
    )

    print(f"\nEvaluating CCI on {len(sample_failures)} failure predictions:\n")

    robust_count = 0
    fragile_count = 0

    for idx in sample_failures:
        result = cci_scorer.compute_confidence(X_test[idx : idx + 1])

        status = "ROBUST" if result["is_robust"] else "FRAGILE"

        if result["is_robust"]:
            robust_count += 1
        else:
            fragile_count += 1

        print(f"Sample {idx}:")
        print(
            f"  Prediction: {result['original_prediction']}, Prob: {result['original_proba']:.3f}"
        )
        print(f"  CCI Score: {result['cci_score']:.3f}")
        print(f"  Status: {status}")
        print(f"  Interpretation: {result['interpretation']}")
        if result["min_intervention"]:
            print(f"  Min Intervention: {result['min_intervention']}")
        print()

    print("=== Computing robustness summary (this may take a minute) ===")
    summary = cci_scorer.get_robustness_summary(X_test[:100])

    print(f"\nROBUSTNESS SUMMARY:")
    print(f"Total failure predictions: {summary['failures_predicted']}")
    print(f"Robust predictions: {summary['robust_predictions']}")
    print(f"Fragile predictions: {summary['fragile_predictions']}")
    print(f"Robustness ratio: {summary['robustness_ratio']:.1%}")
    print(f"Mean CCI: {summary['mean_cci']:.3f}")
    print(f"Median CCI: {summary['median_cci']:.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(cci_scorer, "models/cci_scorer.pkl")
    print(f"\nCCI scorer saved to models/cci_scorer.pkl")

    return cci_scorer, summary


if __name__ == "__main__":
    run_cci_demo()
