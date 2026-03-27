"""
Kafka Consumer for Real-Time Predictions
Consumes sensor data from Kafka and makes predictions
"""

import json
import numpy as np
import joblib
from datetime import datetime
from typing import Optional, Dict, List
import os
from collections import deque

try:
    from kafka import KafkaConsumer
    from kafka.errors import KafkaError

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("Kafka not installed - using mock mode")


class PredictionConsumer:
    """
    Kafka consumer that processes sensor data and makes predictions
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "sensor-data",
        group_id: str = "prediction-group",
        model_path: str = "models/model.pkl",
        threshold: float = 0.35,
    ):

        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.threshold = threshold

        self.model = None
        self.scaler = None
        self.le = None
        self.consumer = None

        self.predictions_buffer = deque(maxlen=1000)
        self.running = False

        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load the ML model"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load("data/scaler.pkl")
            self.le = joblib.load("data/label_encoder.pkl")
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def preprocess(self, reading: dict) -> np.ndarray:
        """Preprocess sensor reading for prediction"""
        try:
            type_encoded = self.le.transform([reading["type"]])[0]
        except:
            type_encoded = 1

        features = np.array(
            [
                [
                    type_encoded,
                    reading["air_temp"],
                    reading["process_temp"],
                    reading["rotational_speed"],
                    reading["torque"],
                    reading["tool_wear"],
                    reading.get(
                        "temp_diff", reading["process_temp"] - reading["air_temp"]
                    ),
                    reading.get(
                        "power", reading["rotational_speed"] * reading["torque"]
                    ),
                    reading.get("temp_rate", 0),
                    reading.get(
                        "wear_stress", reading["tool_wear"] * reading["torque"]
                    ),
                ]
            ]
        )

        features_scaled = self.scaler.transform(features)
        return features_scaled

    def predict(self, reading: dict) -> dict:
        """Make prediction on sensor reading"""
        if self.model is None:
            return {"error": "Model not loaded"}

        features = self.preprocess(reading)
        proba = self.model.predict_proba(features)[0, 1]
        prediction = 1 if proba >= self.threshold else 0

        result = {
            "timestamp": reading.get("timestamp", datetime.now().isoformat()),
            "machine_id": reading.get("machine_id", "UNKNOWN"),
            "prediction": prediction,
            "failure_probability": float(proba),
            "threshold": self.threshold,
            "status": "FAILURE" if prediction == 1 else "OK",
            "drift_detected": reading.get("drift_applied", False),
        }

        self.predictions_buffer.append(result)
        return result

    def connect(self) -> bool:
        """Connect to Kafka"""
        if not KAFKA_AVAILABLE:
            print("Kafka not available")
            return False

        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=True,
                api_version=(2, 8, 0),
            )
            print(f"Connected to Kafka topic: {self.topic}")
            return True
        except Exception as e:
            print(f"Kafka connection error: {e}")
            return False

    def start_consuming(self, max_messages: int = 100) -> List[dict]:
        """Start consuming and predicting"""
        if not self.consumer:
            print("Consumer not connected")
            return []

        self.running = True
        predictions = []

        print(f"Starting prediction consumer (max {max_messages} messages)...\n")

        for i, message in enumerate(self.consumer):
            if not self.running or i >= max_messages:
                break

            try:
                reading = message.value
                result = self.predict(reading)
                predictions.append(result)

                print(
                    f"[{i + 1}] {result['machine_id']} | "
                    f"Prob: {result['failure_probability']:.2f} | "
                    f"{result['status']} | "
                    f"Drift: {result['drift_detected']}"
                )

            except Exception as e:
                print(f"Error processing message: {e}")

        self.running = False
        return predictions

    def stop_consuming(self):
        """Stop consuming"""
        self.running = False
        if self.consumer:
            self.consumer.close()
            print("Consumer stopped")

    def get_statistics(self) -> dict:
        """Get prediction statistics"""
        if not self.predictions_buffer:
            return {"message": "No predictions yet"}

        preds = list(self.predictions_buffer)
        failures = [p for p in preds if p["prediction"] == 1]

        return {
            "total_predictions": len(preds),
            "failures_detected": len(failures),
            "failure_rate": len(failures) / len(preds) if preds else 0,
            "avg_probability": np.mean([p["failure_probability"] for p in preds]),
            "drift_events": sum(1 for p in preds if p["drift_detected"]),
        }


class MockKafkaConsumer:
    """
    Mock consumer for testing without Kafka
    """

    def __init__(self, model_path="models/model.pkl", threshold=0.35):
        self.threshold = threshold
        self.predictions = []

        self.model = joblib.load(model_path)
        self.scaler = joblib.load("data/scaler.pkl")
        self.le = joblib.load("data/label_encoder.pkl")

    def predict(self, reading):
        type_encoded = self.le.transform([reading["type"]])[0]

        features = np.array(
            [
                [
                    type_encoded,
                    reading["air_temp"],
                    reading["process_temp"],
                    reading["rotational_speed"],
                    reading["torque"],
                    reading["tool_wear"],
                    reading["temp_diff"],
                    reading["power"],
                    reading["temp_rate"],
                    reading["wear_stress"],
                ]
            ]
        )

        features_scaled = self.scaler.transform(features)
        proba = self.model.predict_proba(features_scaled)[0, 1]
        prediction = 1 if proba >= self.threshold else 0

        return {
            "timestamp": reading["timestamp"],
            "machine_id": reading.get("machine_id", "MOCK"),
            "prediction": prediction,
            "failure_probability": float(proba),
            "status": "FAILURE" if prediction == 1 else "OK",
        }


def run_kafka_consumer():
    """Run the Kafka consumer demo"""
    print("=== KAFKA PREDICTION CONSUMER ===\n")

    bootstrap_servers = (
        os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
        or os.environ.get("KAFKA_BOOTSTRAP")
        or "localhost:9092"
    )
    topic = os.environ.get("KAFKA_TOPIC", "sensor-data")

    consumer = PredictionConsumer(
        bootstrap_servers=bootstrap_servers, topic=topic, threshold=0.35
    )

    if consumer.connect():
        predictions = consumer.start_consuming(max_messages=20)
        consumer.stop_consuming()

        stats = consumer.get_statistics()
        print(f"\n=== STATISTICS ===")
        for k, v in stats.items():
            print(f"{k}: {v}")
    else:
        print("\nFalling back to mock consumer...")
        from scripts.streaming_pipeline import SensorSimulator

        simulator = SensorSimulator()
        mock = MockKafkaConsumer()

        data = simulator.start_streaming(duration=10, drift_prob=0.3)

        print("\n=== PREDICTIONS ===")
        for reading in data[-10:]:
            result = mock.predict(reading)
            print(
                f"{result['machine_id']} | Prob: {result['failure_probability']:.2f} | {result['status']}"
            )


if __name__ == "__main__":
    run_kafka_consumer()
