"""
Neural Network Model with ONNX Optimization
MLP-based model for better inference performance
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import warnings

warnings.filterwarnings("ignore")

try:
    import onnx
    from onnx import helper, TensorProto
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX not available - using sklearn MLP only")


class NeuralModel:
    """
    Neural network model optimized for inference with ONNX
    """

    def __init__(self, hidden_layers=(64, 32), activation="relu"):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.model = None
        self.scaler = None
        self.threshold = 0.5

    def fit(self, X, y, threshold=0.5):
        """Train the neural network model"""
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layers,
            activation=self.activation,
            solver="adam",
            alpha=0.01,
            batch_size=64,
            learning_rate="adaptive",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            verbose=False,
        )

        self.model.fit(X_scaled, y)
        self.threshold = threshold

        return self

    def predict(self, X, threshold=None):
        """Predict with custom threshold"""
        if threshold is None:
            threshold = self.threshold

        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[:, 1]

        return (proba >= threshold).astype(int)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X, y):
        """Evaluate model performance"""
        y_pred = self.predict(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
        }

        return metrics

    def save(
        self, model_path="models/mlp_model.pkl", config_path="models/mlp_config.json"
    ):
        """Save model and configuration"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, "models/mlp_scaler.pkl")

        config = {
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "threshold": self.threshold,
            "model_path": model_path,
            "scaler_path": "models/mlp_scaler.pkl",
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to {model_path}")
        print(f"Config saved to {config_path}")

    def convert_to_onnx(self, output_path="models/mlp_model.onnx"):
        """Convert model to ONNX format for optimized inference"""
        if not ONNX_AVAILABLE:
            print("ONNX not available")
            return None

        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]

        onnx_model = convert_sklearn(
            self.model, initial_types=initial_type, target_opset=12
        )

        onnx.save(onnx_model, output_path)
        print(f"ONNX model saved to {output_path}")

        return output_path

    def optimize_for_inference(self, X_sample, output_path="models/mlp_model.onnx"):
        """Optimize ONNX model for inference"""
        if not ONNX_AVAILABLE:
            return None

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 2

        try:
            session = ort.InferenceSession(output_path, sess_options)

            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            result = session.run([output_name], {input_name: X_sample})

            print("ONNX inference optimized successfully")
            return session
        except Exception as e:
            print(f"ONNX optimization error: {e}")
            return None


def train_neural_model():
    """Train and evaluate neural network model"""

    print("=== NEURAL NETWORK MODEL TRAINING ===\n")

    X_train = np.load("data/X_train.npy")
    X_test = np.load("data/X_test.npy")
    y_train = np.load("data/y_train.npy")
    y_test = np.load("data/y_test.npy")

    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Positive class ratio: {y_train.mean():.2%}\n")

    neural = NeuralModel(hidden_layers=(128, 64, 32), activation="relu")
    neural.fit(X_train, y_train)

    print("Training complete!\n")

    print("Default threshold (0.5) performance:")
    metrics = neural.evaluate(X_test, y_test)
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1:        {metrics['f1']:.3f}")

    optimal_threshold = 0.35
    neural.threshold = optimal_threshold

    print(f"\nOptimized threshold ({optimal_threshold}) performance:")
    metrics_opt = neural.evaluate(X_test, y_test)
    print(f"  Accuracy:  {metrics_opt['accuracy']:.3f}")
    print(f"  Precision: {metrics_opt['precision']:.3f}")
    print(f"  Recall:    {metrics_opt['recall']:.3f}")
    print(f"  F1:        {metrics_opt['f1']:.3f}")

    neural.save()

    if ONNX_AVAILABLE:
        try:
            print("\nConverting to ONNX...")
            neural.convert_to_onnx()
        except Exception as e:
            print(f"ONNX conversion skipped: {e}")

    print("\n=== COMPARISON: Random Forest vs Neural Network ===\n")

    rf_model = joblib.load("models/model.pkl")
    rf_pred = rf_model.predict(X_test)

    rf_metrics = {
        "accuracy": accuracy_score(y_test, rf_pred),
        "precision": precision_score(y_test, rf_pred, zero_division=0),
        "recall": recall_score(y_test, rf_pred, zero_division=0),
        "f1": f1_score(y_test, rf_pred, zero_division=0),
    }

    print("Random Forest:")
    print(f"  Accuracy:  {rf_metrics['accuracy']:.3f}")
    print(f"  Precision: {rf_metrics['precision']:.3f}")
    print(f"  Recall:    {rf_metrics['recall']:.3f}")
    print(f"  F1:        {rf_metrics['f1']:.3f}")

    print("\nNeural Network (MLP):")
    print(f"  Accuracy:  {metrics_opt['accuracy']:.3f}")
    print(f"  Precision: {metrics_opt['precision']:.3f}")
    print(f"  Recall:    {metrics_opt['recall']:.3f}")
    print(f"  F1:        {metrics_opt['f1']:.3f}")

    return neural, metrics_opt


if __name__ == "__main__":
    train_neural_model()
