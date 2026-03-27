import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import time


class DriftPredictor:
    """
    Preemptive Model Warming (PMW) - Drift Prediction

    Predicts WHEN data drift will happen BEFORE it happens.
    Uses temporal patterns and statistical tests.
    """

    def __init__(self, window_size=50, threshold=0.7):
        self.window_size = window_size
        self.threshold = threshold
        self.history = []
        self.drift_model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def add_observation(self, X):
        """Add new observation to history"""
        if len(X.shape) > 1:
            X = X.mean(axis=0)

        self.history.append(X)

        if len(self.history) > self.window_size * 2:
            self.history = self.history[-self.window_size * 2 :]

    def compute_drift_features(self) -> np.ndarray:
        """
        Compute features that predict drift:
        - Statistical moments (mean, std, skewness)
        - Trend direction
        - KS statistic vs reference
        """
        if len(self.history) < self.window_size:
            return None

        current_window = np.array(self.history[-self.window_size :])
        reference_window = np.array(
            self.history[-2 * self.window_size : -self.window_size]
        )

        features = []

        for i in range(current_window.shape[1]):
            curr_col = current_window[:, i]
            ref_col = reference_window[:, i]

            ks_stat, p_value = stats.ks_2samp(ref_col, curr_col)

            mean_curr = np.mean(curr_col)
            mean_ref = np.mean(ref_col)
            mean_shift = abs(mean_curr - mean_ref) / (abs(mean_ref) + 1e-10)

            std_curr = np.std(curr_col)
            std_ref = np.std(ref_col)
            std_ratio = std_curr / (std_ref + 1e-10)

            features.extend(
                [
                    ks_stat,
                    p_value,
                    mean_shift,
                    std_ratio,
                    np.mean(curr_col),
                    np.std(curr_col),
                    np.percentile(curr_col, 25),
                    np.percentile(curr_col, 75),
                ]
            )

        return np.array(features).reshape(1, -1)

    def predict_drift_probability(self) -> float:
        """
        Predict probability of drift occurring in next window
        """
        features = self.compute_drift_features()

        if features is None:
            return 0.0

        if not self.is_fitted:
            return self._rule_based_drift(features)

        features_scaled = self.scaler.transform(features)
        prob = self.drift_model.predict_proba(features_scaled)[0, 1]

        return prob

    def _rule_based_drift(self, features) -> float:
        """Simple rule-based drift detection"""
        ks_stat = features[0, 0]
        p_value = features[0, 1]

        if p_value < 0.05 and ks_stat > 0.3:
            return 0.8
        elif p_value < 0.1 or ks_stat > 0.2:
            return 0.5
        return 0.2

    def fit(self, X_history, y_drift_labels):
        """
        Train drift prediction model

        Args:
            X_history: historical feature matrices
            y_drift_labels: 1=drift occurred, 0=no drift
        """
        self.scaler.fit(X_history)
        X_scaled = self.scaler.transform(X_history)

        self.drift_model = RandomForestClassifier(
            n_estimators=50, max_depth=5, class_weight="balanced", random_state=42
        )

        self.drift_model.fit(X_scaled, y_drift_labels)
        self.is_fitted = True

        print(f"Drift prediction model trained on {len(y_drift_labels)} samples")


class ModelManager:
    """
    Manages dual-model system for seamless switching

    Keeps both active and standby models loaded
    """

    def __init__(self, model_path="models/model.pkl"):
        self.model_path = model_path
        self.active_model = None
        self.standby_model = None
        self.model_version = "v1.0"
        self.is_warmed = False

    def load_active_model(self):
        """Load the primary model"""
        self.active_model = joblib.load(self.model_path)
        self.is_warmed = True
        print(f"Active model loaded: {self.model_version}")

    def load_standby_model(self, model_path):
        """Pre-load standby model for fast switching"""
        self.standby_model = joblib.load(model_path)
        print("Standby model pre-loaded")

    def switch_to_standby(self):
        """Switch standby to active (for model updates)"""
        if self.standby_model is None:
            return False, "No standby model available"

        self.active_model = self.standby_model
        self.standby_model = None
        self.is_warmed = True

        return True, f"Switched to new model"

    def warmup(self, X_sample):
        """Warm up model by running dummy predictions"""
        if self.active_model is None:
            return False

        for _ in range(5):
            _ = self.active_model.predict_proba(X_sample)

        self.is_warmed = True
        print("Model warmed up (JIT compiled)")
        return True


class PreemptiveWarmingScheduler:
    """
    Orchestrates preemptive model warming

    Monitors drift prediction and triggers background retraining
    when drift is likely
    """

    def __init__(self, drift_predictor, model_manager):
        self.drift_predictor = drift_predictor
        self.model_manager = model_manager
        self.warming_enabled = True
        self.last_check = time.time()
        self.check_interval = 60

    def check_and_warm(self, X_new) -> Dict:
        """
        Check for drift prediction and trigger warming if needed

        Returns:
            dict with status and actions taken
        """
        result = {
            "drift_probability": 0.0,
            "warming_triggered": False,
            "action": "none",
        }

        self.drift_predictor.add_observation(X_new)

        drift_prob = self.drift_predictor.predict_drift_probability()
        result["drift_probability"] = drift_prob

        if drift_prob > self.drift_predictor.threshold and self.warming_enabled:
            result["warming_triggered"] = True
            result["action"] = "start_background_retrain"
            print(
                f"Drift predicted ({drift_prob:.1%}), triggering background retraining"
            )

        return result

    def get_status(self) -> Dict:
        """Get current system status"""
        return {
            "warmed": self.model_manager.is_warmed,
            "model_version": self.model_manager.model_version,
            "history_size": len(self.drift_predictor.history),
            "warming_enabled": self.warming_enabled,
        }


def generate_drift_data(n_samples=500):
    """Generate synthetic data for drift prediction training"""
    np.random.seed(42)

    X_history = []
    y_labels = []

    for i in range(n_samples // 50):
        base = np.random.randn(50, 10)

        if i > n_samples // 100:
            shifted = base + np.random.uniform(0.5, 2.0, size=(50, 10))
            y_labels.append(1)
        else:
            shifted = base
            y_labels.append(0)

        X_history.append(shifted.mean(axis=0))

    return np.array(X_history), np.array(y_labels)


def run_pmw():
    """Run PMW initialization"""
    X_history, y_labels = generate_drift_data()

    predictor = DriftPredictor(window_size=50, threshold=0.7)
    predictor.fit(X_history, y_labels)

    manager = ModelManager()
    manager.load_active_model()

    scheduler = PreemptiveWarmingScheduler(predictor, manager)

    X_sample = np.random.randn(1, 10)
    result = scheduler.check_and_warm(X_sample)

    print("\n=== PREEMPTIVE MODEL WARMING (PMW) ===")
    print(f"Drift prediction enabled: {scheduler.warming_enabled}")
    print(f"System status: {scheduler.get_status()}")
    print(f"\nTest prediction: {result}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(predictor, "models/drift_predictor.pkl")
    joblib.dump(scheduler, "models/pmw_scheduler.pkl")
    print("\nPMW components saved")

    return predictor, scheduler


if __name__ == "__main__":
    run_pmw()
