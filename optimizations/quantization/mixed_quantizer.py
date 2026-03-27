import numpy as np
import joblib
import os
from typing import Dict, List

PRECISION_LEVELS = {
    0: {"name": "FP32", "bits": 32, "dtype": "float32"},
    1: {"name": "FP16", "bits": 16, "dtype": "float16"},
    2: {"name": "INT8", "bits": 8, "dtype": "int8"},
    3: {"name": "BINARY", "bits": 1, "dtype": "uint8"},
}

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


class PrecisionAllocator:
    def __init__(self, causal_ranks: Dict[str, int] = None):
        self.causal_ranks = causal_ranks or {}
        self.precision_map = {}

    def allocate(self):
        """
        Allocate precision levels based on causal importance

        Causal ranks:
        - 1 (direct cause): FP32 or FP16 (high precision)
        - 2 (indirect): INT8 (medium precision)
        - 3 (no link): INT4 (low precision)
        """
        if not self.causal_ranks:
            self.causal_ranks = {f: 2 for f in FEATURE_NAMES}

        for feature, rank in self.causal_ranks.items():
            if rank == 1:
                self.precision_map[feature] = 1
            elif rank == 2:
                self.precision_map[feature] = 2
            else:
                self.precision_map[feature] = 3

        return self.precision_map

    def get_summary(self):
        summary = {}
        for level_id, level_info in PRECISION_LEVELS.items():
            features = [f for f, p in self.precision_map.items() if p == level_id]
            if features:
                summary[level_info["name"]] = features
        return summary


class CausalQuantizer:
    """
    Causal-Aware Quantization (CAQ)

    Instead of uniform quantization, applies different precision
    based on causal importance of each feature.
    """

    def __init__(self, causal_engine_path="models/causal_engine.pkl"):
        self.causal_engine = None
        self.precision_allocator = None

        try:
            if os.path.exists(causal_engine_path):
                self.causal_engine = joblib.load(causal_engine_path)
                if hasattr(self.causal_engine, "_causal_ranks"):
                    self.precision_allocator = PrecisionAllocator(
                        self.causal_engine._causal_ranks
                    )
                else:
                    self.precision_allocator = PrecisionAllocator()
            else:
                self.precision_allocator = PrecisionAllocator()
        except Exception as e:
            print(f"Could not load causal engine: {e}")
            self.precision_allocator = PrecisionAllocator()

        self.precision_map = self.precision_allocator.allocate()

    def quantize_features(self, X, feature_names=None):
        """
        Apply causal-aware quantization to feature matrix

        Args:
            X: numpy array of features
            feature_names: list of feature names

        Returns:
            Quantized features + metadata
        """
        if feature_names is None:
            feature_names = FEATURE_NAMES

        X_quantized = X.copy()
        metadata = {
            "original_dtype": str(X.dtype),
            "precision_map": self.precision_map,
            "quantization_info": {},
        }

        for i, feature in enumerate(feature_names):
            if feature in self.precision_map:
                precision_level = self.precision_map[feature]
                X_quantized[:, i] = self._quantize_column(X[:, i], precision_level)
                metadata["quantization_info"][feature] = {
                    "precision_level": precision_level,
                    "precision_name": PRECISION_LEVELS[precision_level]["name"],
                    "bits": PRECISION_LEVELS[precision_level]["bits"],
                }

        return X_quantized, metadata

    def _quantize_column(self, column, precision_level):
        """
        Quantize a single column to target precision
        """
        if precision_level == 0:
            return column.astype(np.float32)
        elif precision_level == 1:
            return column.astype(np.float16)
        elif precision_level == 2:
            col_min, col_max = column.min(), column.max()
            if col_max - col_min < 1e-10:
                return np.zeros_like(column, dtype=np.int8)
            scale = 255 / (col_max - col_min)
            quantized = ((column - col_min) * scale).astype(np.int8)
            return quantized
        elif precision_level == 3:
            col_min, col_max = column.min(), column.max()
            if col_max - col_min < 1e-10:
                return np.zeros_like(column, dtype=np.uint8)
            scale = 15 / (col_max - col_min)
            quantized = ((column - col_min) * scale).astype(np.uint8)
            return quantized
        return column

    def get_memory_savings(self, X):
        """
        Calculate memory savings from causal-aware quantization
        vs uniform quantization (all INT8)
        """
        baseline_bits = 8 * X.shape[1]

        caq_bits = 0
        for i, feature in enumerate(FEATURE_NAMES):
            precision = self.precision_map.get(feature, 2)
            caq_bits += PRECISION_LEVELS[precision]["bits"]

        baseline_size = baseline_bits / 8
        caq_size = caq_bits / 8

        savings_pct = (1 - caq_size / baseline_size) * 100

        return {
            "baseline_bytes": baseline_size,
            "caq_bytes": caq_size,
            "savings_bytes": baseline_size - caq_size,
            "savings_percent": savings_pct,
        }

    def visualize_precision_allocation(self):
        """Print visualization of precision allocation"""
        summary = self.precision_allocator.get_summary()

        print("\n=== CAUSAL-AWARE QUANTIZATION (CAQ) ===")
        print("\nPrecision Allocation by Causal Importance:")

        for precision, features in summary.items():
            print(f"\n{precision}:")
            for f in features:
                causal_rank = (
                    self.causal_engine._causal_ranks.get(f, 2)
                    if self.causal_engine
                    else 2
                )
                print(f"  - {f} (causal rank: {causal_rank})")

        savings = self.precision_allocator.allocate()
        mem_info = self.get_memory_savings(np.zeros((1, len(FEATURE_NAMES))))
        print(f"\nEstimated Memory Savings: {mem_info['savings_percent']:.1f}%")

        return summary


def run_caq():
    import numpy as np

    X_train = np.load("data/X_train.npy")
    X_test = np.load("data/X_test.npy")

    quantizer = CausalQuantizer()

    X_train_quantized, metadata = quantizer.quantize_features(X_train)
    X_test_quantized, _ = quantizer.quantize_features(X_test)

    quantizer.visualize_precision_allocation()

    print(f"\nOriginal shape: {X_train.shape}")
    print(f"Quantized shape: {X_train_quantized.shape}")
    print(f"Original dtype: {X_train.dtype}")
    print(f"Quantized dtype: {X_train_quantized.dtype}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(quantizer, "models/caq_quantizer.pkl")
    print("\nCAQ quantizer saved to models/caq_quantizer.pkl")

    return quantizer, metadata


if __name__ == "__main__":
    run_caq()
