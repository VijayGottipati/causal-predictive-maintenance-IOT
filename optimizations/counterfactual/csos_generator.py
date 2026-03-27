import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier

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


class CausalBoundaryMapper:
    """
    Identifies the 'edge of failure' for each causal feature.

    Uses the causal graph to determine threshold values where
    failure probability transitions from low to high.
    """

    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler
        self.boundaries = {}
        self.feature_ranges = {}

    def fit(self, X, y):
        """
        Learn failure boundaries from data

        Args:
            X: Feature matrix (scaled)
            y: Labels (0=normal, 1=failure)
        """
        failure_data = X[y == 1]
        normal_data = X[y == 0]

        for i, feature in enumerate(FEATURE_NAMES):
            fail_values = failure_data[:, i]
            norm_values = normal_data[:, i]

            fail_mean = np.mean(fail_values)
            fail_std = np.std(fail_values)
            norm_max = np.max(norm_values)

            boundary = (fail_mean + norm_max) / 2

            self.boundaries[feature] = {
                "threshold": float(boundary),
                "fail_mean": float(fail_mean),
                "fail_min": float(np.min(fail_values)),
                "fail_max": float(np.max(fail_values)),
                "norm_max": float(norm_max),
            }

            self.feature_ranges[feature] = {
                "min": float(np.min(X[:, i])),
                "max": float(np.max(X[:, i])),
                "std": float(np.std(X[:, i])),
            }

        return self

    def get_near_boundary_features(self, X_sample, threshold_pct=0.8):
        """
        Identify features within X% of their failure boundary
        """
        near_boundary = []

        for i, feature in enumerate(FEATURE_NAMES):
            if feature not in self.boundaries:
                continue

            boundary = self.boundaries[feature]["threshold"]
            norm_max = self.boundaries[feature]["norm_max"]

            distance_to_boundary = abs(boundary - X_sample[i])
            total_range = abs(boundary - norm_max)

            if total_range > 0:
                pct_to_boundary = 1 - (distance_to_boundary / total_range)

                if pct_to_boundary >= threshold_pct:
                    near_boundary.append(
                        {
                            "feature": feature,
                            "index": i,
                            "current_value": float(X_sample[i]),
                            "boundary": boundary,
                            "distance_to_boundary": float(distance_to_boundary),
                            "pct_to_boundary": float(pct_to_boundary),
                        }
                    )

        return near_boundary

    def calculate_causal_shift(self, X_sample, target_feature, shift_pct=0.3):
        """
        Calculate how much to shift a feature to cross failure boundary
        """
        if target_feature not in self.boundaries:
            return None

        boundary = self.boundaries[target_feature]["threshold"]
        current = X_sample[FEATURE_NAMES.index(target_feature)]

        shift_needed = boundary - current

        return {
            "feature": target_feature,
            "current_value": float(current),
            "boundary": boundary,
            "shift_needed": float(shift_needed),
            "shift_pct": float(shift_needed / (abs(current) + 1e-10) * 100)
            if current != 0
            else 0,
        }

    def get_boundary_summary(self):
        """Get summary of all identified boundaries"""
        summary = {}
        for feature, info in self.boundaries.items():
            summary[feature] = {
                "threshold": info["threshold"],
                "gap_from_normal": info["threshold"] - info["norm_max"],
            }
        return summary


class CounterfactualGenerator:
    """
    Generates synthetic failure samples using counterfactual reasoning.

    Instead of just duplicating existing failures, uses causal knowledge
    to generate 'what-if' failure scenarios from healthy data.
    """

    def __init__(self, boundary_mapper, model=None):
        self.boundary_mapper = boundary_mapper
        self.model = model
        self.synthetic_failures = []
        self.near_misses = []

    def generate_from_healthy(
        self, X_healthy, n_samples=None, failures_per_sample=3, near_miss_prob=0.3
    ):
        """
        Generate synthetic failures from healthy samples

        Args:
            X_healthy: Normal/healthy samples
            n_samples: How many to generate (None = all)
            failures_per_sample: How many failure variations per healthy sample
            near_miss_prob: Probability of generating near-miss vs true failure
        """
        if n_samples is None:
            n_samples = min(1000, len(X_healthy))

        indices = np.random.choice(len(X_healthy), n_samples, replace=False)

        for idx in indices:
            sample = X_healthy[idx]
            near_boundary_features = self.boundary_mapper.get_near_boundary_features(
                sample, threshold_pct=0.6
            )

            if not near_boundary_features:
                near_boundary_features = [
                    {"feature": "Torque", "index": 4, "current_value": sample[4]},
                    {"feature": "Tool wear", "index": 5, "current_value": sample[5]},
                ]

            for _ in range(failures_per_sample):
                if np.random.random() < near_miss_prob:
                    synthetic_sample = self._create_near_miss(
                        sample, near_boundary_features
                    )
                    if synthetic_sample is not None:
                        self.near_misses.append(synthetic_sample)
                else:
                    synthetic_sample = self._create_counterfactual_failure(
                        sample, near_boundary_features
                    )
                    if synthetic_sample is not None:
                        self.synthetic_failures.append(synthetic_sample)

        return {
            "synthetic_failures": np.array(self.synthetic_failures),
            "near_misses": np.array(self.near_misses),
        }

    def _create_counterfactual_failure(self, healthy_sample, near_boundary_features):
        """Create a true failure by pushing features past their boundaries"""
        synthetic = healthy_sample.copy()

        if not near_boundary_features:
            return None

        feature_info = near_boundary_features[
            np.random.randint(len(near_boundary_features))
        ]
        idx = feature_info["index"]

        boundary = self.boundary_mapper.boundaries.get(
            FEATURE_NAMES[idx], {"threshold": feature_info["current_value"] + 1}
        )["threshold"]

        overshoot = np.random.uniform(0.05, 0.2) * (
            abs(boundary - feature_info["current_value"]) + 0.1
        )

        if boundary > feature_info["current_value"]:
            synthetic[idx] = boundary + overshoot
        else:
            synthetic[idx] = boundary - overshoot

        for i, name in enumerate(FEATURE_NAMES):
            if i != idx:
                range_info = self.boundary_mapper.feature_ranges.get(name, {})
                if range_info:
                    synthetic[i] = np.clip(
                        synthetic[i]
                        + np.random.uniform(-0.5, 0.5) * range_info.get("std", 1),
                        range_info.get("min", -10),
                        range_info.get("max", 10),
                    )

        return synthetic

    def _create_near_miss(self, healthy_sample, near_boundary_features):
        """Create a near-miss: high risk but doesn't cross into failure"""
        synthetic = healthy_sample.copy()

        if not near_boundary_features:
            return None

        feature_info = near_boundary_features[
            np.random.randint(len(near_boundary_features))
        ]
        idx = feature_info["index"]

        boundary = self.boundary_mapper.boundaries.get(
            FEATURE_NAMES[idx], {"threshold": feature_info["current_value"] + 1}
        )["threshold"]

        if boundary > feature_info["current_value"]:
            synthetic[idx] = boundary - np.random.uniform(0.01, 0.1) * abs(
                boundary - feature_info["current_value"]
            )
        else:
            synthetic[idx] = boundary + np.random.uniform(0.01, 0.1) * abs(
                boundary - feature_info["current_value"]
            )

        return synthetic

    def validate_synthetic(self, model=None):
        """Validate that synthetic failures are actually predicted as failures"""
        if not self.synthetic_failures:
            return {"valid": True, "message": "No synthetic samples"}

        if model is None:
            model = self.model

        if model is None:
            return {"valid": True, "message": "No model to validate"}

        predictions = model.predict(self.synthetic_failures)

        failure_rate = np.mean(predictions == 1)

        near_miss_preds = (
            model.predict(self.near_misses) if len(self.near_misses) > 0 else []
        )
        near_miss_failure_rate = (
            np.mean(near_miss_preds == 1) if len(near_miss_preds) > 0 else 0
        )

        return {
            "valid": failure_rate > 0.5,
            "synthetic_failure_rate": float(failure_rate),
            "near_miss_failure_rate": float(near_miss_failure_rate),
            "n_synthetic_failures": len(self.synthetic_failures),
            "n_near_misses": len(self.near_misses),
        }


class CSOSIntegrator:
    """
    Integrates CSOS into the training pipeline as progressive enhancement.
    """

    def __init__(self, boundary_mapper=None, generator=None):
        self.boundary_mapper = boundary_mapper or CausalBoundaryMapper()
        self.generator = generator or CounterfactualGenerator(self.boundary_mapper)

    def augment_dataset(
        self,
        X,
        y,
        n_synthetic_per_failure=10,
        near_miss_ratio=0.3,
        validation_split=0.2,
    ):
        """
        Create augmented dataset with synthetic failures

        Args:
            X, y: Original data
            n_synthetic_per_failure: How many synthetic samples per real failure
            near_miss_ratio: Ratio of near-misses in synthetic data

        Returns:
            X_aug, y_aug, sample_weights (for IWL integration)
        """
        normal_mask = y == 0
        failure_mask = y == 1

        X_normal = X[normal_mask]
        X_failure = X[failure_mask]

        self.boundary_mapper.fit(X, y)

        n_healthy_samples = min(len(X_failure) * n_synthetic_per_failure, 1000)

        self.generator.model = None
        result = self.generator.generate_from_healthy(
            X_normal, n_samples=n_healthy_samples, near_miss_prob=near_miss_ratio
        )

        X_synthetic_failures = result["synthetic_failures"]
        X_near_misses = result["near_misses"]

        y_synthetic_failures = np.ones(len(X_synthetic_failures))
        y_near_misses = np.zeros(len(X_near_misses))

        sample_weights = np.ones(
            len(X) + len(X_synthetic_failures) + len(X_near_misses)
        )

        weight_idx = len(X)

        for _ in range(len(X_synthetic_failures)):
            sample_weights[weight_idx] = 1.0
            weight_idx += 1

        for _ in range(len(X_near_misses)):
            sample_weights[weight_idx] = 0.5
            weight_idx += 1

        X_aug = np.vstack([X, X_synthetic_failures, X_near_misses])
        y_aug = np.concatenate([y, y_synthetic_failures, y_near_misses])

        print(f"\n=== CSOS AUGMENTATION RESULTS ===")
        print(f"Original: {len(X)} samples ({sum(y)} failures)")
        print(f"Synthetic failures: {len(X_synthetic_failures)}")
        print(f"Near-misses: {len(X_near_misses)}")
        print(f"Augmented total: {len(X_aug)} samples")

        return X_aug, y_aug, sample_weights

    def get_summary(self):
        """Get CSOS system summary"""
        boundaries = self.boundary_mapper.get_boundary_summary()

        return {
            "boundaries": boundaries,
            "n_synthetic_failures": len(self.generator.synthetic_failures),
            "n_near_misses": len(self.generator.near_misses),
            "technique": "CSOS - Counterfactual Synthetic Over-Sampling",
            "purpose": "Address data scarcity by generating plausible unseen failure scenarios",
        }


def run_csos():
    """Run CSOS demonstration"""
    import joblib

    print("=== CSOS - COUNTERFACTUAL SYNTHETIC OVER-SAMPLING ===\n")

    X = np.load("data/X_train.npy")
    y = np.load("data/y_train.npy")

    try:
        model = joblib.load("models/model.pkl")
    except:
        model = RandomForestClassifier(n_estimators=50, random_state=42)

    print(
        f"Original dataset: {len(X)} samples, {sum(y)} failures ({sum(y) / len(y) * 100:.1f}%)"
    )

    integrator = CSOSIntegrator()

    X_aug, y_aug, weights = integrator.augment_dataset(
        X, y, n_synthetic_per_failure=15, near_miss_ratio=0.3
    )

    summary = integrator.get_summary()

    print(f"\nBoundary thresholds identified:")
    for feature, info in summary["boundaries"].items():
        print(f"  {feature}: {info['threshold']:.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(integrator, "models/csos_integrator.pkl")
    print(f"\nCSOS integrator saved to models/csos_integrator.pkl")

    return integrator, X_aug, y_aug, weights


if __name__ == "__main__":
    run_csos()
