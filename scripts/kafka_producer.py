"""
Kafka Producer for Sensor Data Streaming
Publishes real-time sensor readings to Kafka topic
"""

import json
import time
import random
import numpy as np
from datetime import datetime
from typing import Optional
import os

try:
    from kafka import KafkaProducer
    from kafka.errors import NoBrokersAvailable

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("Kafka not installed - using simulation mode")


class SensorDataProducer:
    """
    Kafka producer for streaming sensor data
    Simulates real-time sensor readings and publishes to Kafka
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "sensor-data",
        batch_size: int = 10,
        compression_type: str = "gzip",
    ):

        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer: Optional[KafkaProducer] = None
        self.batch_size = batch_size
        self.compression_type = compression_type
        self.running = False

    def connect(self) -> bool:
        """Connect to Kafka broker"""
        if not KAFKA_AVAILABLE:
            print("Kafka library not available")
            return False

        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                batch_size=self.batch_size * 1024,
                compression_type=self.compression_type,
                api_version=(2, 8, 0),
            )
            print(f"Connected to Kafka at {self.bootstrap_servers}")
            return True
        except NoBrokersAvailable:
            print(f"No brokers available at {self.bootstrap_servers}")
            return False
        except Exception as e:
            print(f"Kafka connection error: {e}")
            return False

    def generate_sensor_reading(self, add_drift: bool = False) -> dict:
        """Generate a single sensor reading"""
        reading = {
            "timestamp": datetime.now().isoformat(),
            "type": random.choice(["L", "M", "H"]),
            "air_temp": round(random.uniform(295, 305), 1),
            "process_temp": round(random.uniform(305, 315), 1),
            "rotational_speed": random.randint(1300, 1600),
            "torque": round(random.uniform(30, 50), 1),
            "tool_wear": random.randint(0, 25),
            "machine_id": f"MACHINE-{random.randint(1, 10):03d}",
        }

        reading["temp_diff"] = round(reading["process_temp"] - reading["air_temp"], 1)
        reading["power"] = reading["rotational_speed"] * reading["torque"]
        reading["temp_rate"] = round(
            reading["temp_diff"] / (reading["air_temp"] + 1), 3
        )
        reading["wear_stress"] = reading["tool_wear"] * reading["torque"]

        if add_drift:
            reading["torque"] = round(reading["torque"] * random.uniform(1.1, 1.4), 1)
            reading["air_temp"] = round(
                reading["air_temp"] * random.uniform(1.02, 1.06), 1
            )
            reading["tool_wear"] = min(reading["tool_wear"] + random.randint(1, 4), 30)
            reading["drift_applied"] = True
        else:
            reading["drift_applied"] = False

        return reading

    def publish_reading(self, reading: dict) -> bool:
        """Publish a single reading to Kafka"""
        if not self.producer:
            return False

        try:
            future = self.producer.send(
                self.topic, key=reading.get("machine_id"), value=reading
            )
            future.get(timeout=10)
            return True
        except Exception as e:
            print(f"Publish error: {e}")
            return False

    def start_streaming(
        self, duration: int = 60, interval: float = 1.0, drift_probability: float = 0.2
    ) -> int:
        """
        Start streaming sensor data to Kafka

        Args:
            duration: How long to stream (seconds)
            interval: Time between readings (seconds)
            drift_probability: Probability of adding drift

        Returns:
            Number of messages sent
        """
        if not self.producer:
            print("Producer not connected")
            return 0

        self.running = True
        start_time = time.time()
        messages_sent = 0

        print(f"Starting sensor data stream for {duration} seconds...")
        print(f"Topic: {self.topic}, Interval: {interval}s\n")

        while self.running and (time.time() - start_time) < duration:
            add_drift = random.random() < drift_probability
            reading = self.generate_sensor_reading(add_drift)

            if self.publish_reading(reading):
                messages_sent += 1
                status = "DRIFT" if reading["drift_applied"] else "OK"
                print(
                    f"[{messages_sent}] {reading['machine_id']} | "
                    f"Air: {reading['air_temp']:.1f}K | "
                    f"Torque: {reading['torque']:.1f}Nm | "
                    f"{status}"
                )

            time.sleep(interval)

        self.running = False
        self.producer.flush()
        print(f"\nStream complete. Sent {messages_sent} messages.")
        return messages_sent

    def stop_streaming(self):
        """Stop the streaming"""
        self.running = False
        if self.producer:
            self.producer.flush()
            self.producer.close()
            print("Producer disconnected")

    def close(self):
        """Close the producer"""
        if self.producer:
            self.producer.close()


def run_kafka_producer():
    """Run the Kafka producer demo"""
    print("=== KAFKA SENSOR DATA PRODUCER ===\n")

    bootstrap_servers = (
        os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
        or os.environ.get("KAFKA_BOOTSTRAP")
        or "localhost:9092"
    )
    topic = os.environ.get("KAFKA_TOPIC", "sensor-data")

    producer = SensorDataProducer(bootstrap_servers=bootstrap_servers, topic=topic)

    if producer.connect():
        messages = producer.start_streaming(
            duration=30, interval=1.0, drift_probability=0.3
        )
        producer.close()
    else:
        print("\nFalling back to simulation mode...")
        from scripts.streaming_pipeline import run_streaming_demo

        run_streaming_demo()


if __name__ == "__main__":
    run_kafka_producer()
