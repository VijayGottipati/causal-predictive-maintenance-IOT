"""
Streaming Data Simulation Pipeline
Simulates real-time sensor data ingestion
"""

import numpy as np
import pandas as pd
import json
import time
import random
from datetime import datetime, timedelta
from collections import deque
import os


class SensorSimulator:
    """
    Simulates real-time sensor data for predictive maintenance
    """

    def __init__(self, n_sensors=5, base_interval=1.0):
        self.n_sensors = n_sensors
        self.base_interval = base_interval
        self.data_buffer = deque(maxlen=1000)
        self.running = False

    def generate_sensor_reading(self):
        """Generate a single sensor reading"""
        readings = {
            "timestamp": datetime.now().isoformat(),
            "type": random.choice(["L", "M", "H"]),
            "air_temp": round(random.uniform(295, 305), 1),
            "process_temp": round(random.uniform(305, 315), 1),
            "rotational_speed": random.randint(1300, 1600),
            "torque": round(random.uniform(30, 50), 1),
            "tool_wear": random.randint(0, 25),
        }

        readings["temp_diff"] = round(
            readings["process_temp"] - readings["air_temp"], 1
        )
        readings["power"] = readings["rotational_speed"] * readings["torque"]
        readings["temp_rate"] = round(
            readings["temp_diff"] / (readings["air_temp"] + 1), 3
        )
        readings["wear_stress"] = readings["tool_wear"] * readings["torque"]

        return readings

    def add_drift(self, readings, drift_factor=0.2):
        """Add drift to simulate changing conditions"""
        drifted = readings.copy()

        if random.random() < drift_factor:
            drifted["torque"] *= random.uniform(1.1, 1.3)
            drifted["air_temp"] *= random.uniform(1.02, 1.05)
            drifted["tool_wear"] += random.randint(1, 3)

        return drifted

    def start_streaming(self, duration=60, drift_prob=0.1):
        """Start streaming sensor data"""
        self.running = True
        start_time = time.time()

        print(f"=== STARTING SENSOR STREAM ===")
        print(f"Duration: {duration} seconds")
        print(f"Interval: {self.base_interval}s\n")

        while self.running and (time.time() - start_time) < duration:
            reading = self.generate_sensor_reading()

            if random.random() < drift_prob:
                reading = self.add_drift(reading)

            self.data_buffer.append(reading)

            print(
                f"[{reading['timestamp']}] "
                f"Air: {reading['air_temp']:.1f}K, "
                f"Torque: {reading['torque']:.1f}Nm, "
                f"Speed: {reading['rotational_speed']}rpm"
            )

            time.sleep(self.base_interval)

        self.running = False
        return list(self.data_buffer)

    def stop_streaming(self):
        """Stop streaming"""
        self.running = False

    def get_buffer_summary(self):
        """Get summary of buffered data"""
        if not self.data_buffer:
            return {"message": "No data in buffer"}

        df = pd.DataFrame(list(self.data_buffer))

        return {
            "samples": len(self.data_buffer),
            "time_range": f"{df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}",
            "avg_air_temp": df["air_temp"].mean(),
            "avg_torque": df["torque"].mean(),
            "max_tool_wear": df["tool_wear"].max(),
            "drift_detected": df["torque"].mean() > 42,
        }


class StreamingPredictor:
    """
    Real-time prediction service for streaming data
    """

    def __init__(self, model_path="models/model.pkl", threshold=0.35):
        import joblib

        self.model = joblib.load(model_path)
        self.scaler = joblib.load("data/scaler.pkl")
        self.le = joblib.load("data/label_encoder.pkl")
        self.threshold = threshold
        self.predictions = deque(maxlen=100)

    def preprocess(self, reading):
        """Preprocess sensor reading for prediction"""
        type_encoded = self.le.transform([reading["type"]])[0]

        features = [
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

        features_scaled = self.scaler.transform([features])
        return features_scaled

    def predict(self, reading):
        """Make prediction on sensor reading"""
        features = self.preprocess(reading)

        proba = self.model.predict_proba(features)[0, 1]
        prediction = 1 if proba >= self.threshold else 0

        result = {
            "timestamp": reading["timestamp"],
            "prediction": prediction,
            "probability": float(proba),
            "threshold": self.threshold,
        }

        self.predictions.append(result)
        return result

    def get_prediction_summary(self):
        """Get summary of recent predictions"""
        if not self.predictions:
            return {"message": "No predictions yet"}

        preds = list(self.predictions)
        failures = [p for p in preds if p["prediction"] == 1]

        return {
            "total_predictions": len(preds),
            "failures_detected": len(failures),
            "failure_rate": len(failures) / len(preds) if preds else 0,
            "avg_probability": np.mean([p["probability"] for p in preds]),
        }


def run_streaming_demo():
    """Run streaming data demonstration"""

    print("=== STREAMING DATA SIMULATION ===\n")

    print("1. Starting sensor simulation for 30 seconds...")
    simulator = SensorSimulator(n_sensors=5, base_interval=2.0)
    data = simulator.start_streaming(duration=30, drift_prob=0.3)

    summary = simulator.get_buffer_summary()
    print(f"\nBuffer summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\n2. Running real-time predictions...")
    predictor = StreamingPredictor(threshold=0.35)

    for reading in data[-10:]:
        result = predictor.predict(reading)
        status = "FAILURE" if result["prediction"] == 1 else "OK"
        print(f"  [{result['timestamp']}] {status} (prob: {result['probability']:.2f})")

    pred_summary = predictor.get_prediction_summary()
    print(f"\nPrediction summary:")
    for key, value in pred_summary.items():
        print(f"  {key}: {value}")

    output_file = "data/streaming_predictions.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nData saved to {output_file}")


if __name__ == "__main__":
    run_streaming_demo()
