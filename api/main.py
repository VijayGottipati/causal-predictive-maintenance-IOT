from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import joblib
import numpy as np
import os
import json
from datetime import datetime
from typing import Optional, List
import threading

from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Predictive Maintenance API",
    description="ML-powered predictive maintenance with streaming and self-healing",
    version="2.1.0",
)

MODEL_PATH = "models/model.pkl"
SCALER_PATH = "data/scaler.pkl"
LE_PATH = "data/label_encoder.pkl"
THRESHOLD_PATH = "models/threshold.json"

model = None
scaler = None
le = None
threshold = 0.35
cci_scorer = None

stream_active = False
stream_thread = None
predictions_buffer = []

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


class PredictionRequest(BaseModel):
    type: str
    air_temp: float
    process_temp: float
    rotational_speed: int
    torque: float
    tool_wear: float


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    cci_score: Optional[float] = None
    cci_interpretation: Optional[str] = None
    is_robust: Optional[bool] = None
    timestamp: str


class StreamStatus(BaseModel):
    streaming: bool
    predictions_count: int
    last_prediction: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    threshold: float
    streaming_active: bool
    timestamp: str
    mlflow_configured: bool = False
    kafka_bootstrap_configured: bool = False


def load_model():
    global model, scaler, le, threshold, cci_scorer

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(LE_PATH)

        if os.path.exists(THRESHOLD_PATH):
            with open(THRESHOLD_PATH) as f:
                threshold_data = json.load(f)
                threshold = threshold_data.get("threshold", 0.35)

        try:
            cci_scorer = joblib.load("models/cci_scorer.pkl")
        except:
            cci_scorer = None

        print(f"Model loaded. Threshold: {threshold}")
        return True
    return False


@app.on_event("startup")
def startup():
    load_model()


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Predictive Maintenance API",
        "version": "2.1.0",
        "features": ["CSOS", "IWL", "CAQ", "PMW", "CCI", "Kafka Streaming"],
        "status": "running" if model else "no model",
    }


def _kafka_env_configured() -> bool:
    return bool(
        os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "").strip()
        or os.environ.get("KAFKA_BOOTSTRAP", "").strip()
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        threshold=threshold,
        streaming_active=stream_active,
        timestamp=datetime.now().isoformat(),
        mlflow_configured=bool(os.environ.get("MLFLOW_TRACKING_URI", "").strip()),
        kafka_bootstrap_configured=_kafka_env_configured(),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    temp_diff = request.process_temp - request.air_temp
    power = request.rotational_speed * request.torque
    temp_rate = temp_diff / (request.air_temp + 1)
    wear_stress = request.tool_wear * request.torque

    try:
        type_encoded = le.transform([request.type])[0]
    except:
        type_encoded = 1

    features = np.array(
        [
            [
                type_encoded,
                request.air_temp,
                request.process_temp,
                request.rotational_speed,
                request.torque,
                request.tool_wear,
                temp_diff,
                power,
                temp_rate,
                wear_stress,
            ]
        ]
    )

    features_scaled = scaler.transform(features)

    proba = model.predict_proba(features_scaled)[0]
    # Use optimized threshold instead of default 0.5
    pred = 1 if proba[1] >= threshold else 0

    cci_score = None
    cci_interpretation = None
    is_robust = None

    if cci_scorer is not None and pred == 1:
        try:
            cci_result = cci_scorer.compute_confidence(features_scaled)
            cci_score = cci_result.get("cci_score")
            cci_interpretation = cci_result.get("interpretation")
            is_robust = cci_result.get("is_robust")
        except:
            pass

    global predictions_buffer
    predictions_buffer.append(
        {
            "timestamp": datetime.now().isoformat(),
            "prediction": int(pred),
            "probability": float(proba[1]),
        }
    )
    if len(predictions_buffer) > 100:
        predictions_buffer = predictions_buffer[-100:]

    return PredictionResponse(
        prediction=int(pred),
        probability=float(proba[1]),
        cci_score=cci_score,
        cci_interpretation=cci_interpretation,
        is_robust=is_robust,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/model/info", tags=["Model"])
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": type(model).__name__,
        "n_features": model.n_features_in_,
        "feature_names": FEATURE_NAMES,
        "classes": model.classes_.tolist(),
        "threshold": threshold,
        "techniques": ["CSOS", "IWL", "CAQ", "PMW", "CCI"],
    }


@app.get("/stream/status", response_model=StreamStatus, tags=["Streaming"])
def stream_status():
    return StreamStatus(
        streaming=stream_active,
        predictions_count=len(predictions_buffer),
        last_prediction=predictions_buffer[-1]["timestamp"]
        if predictions_buffer
        else None,
    )


@app.get("/stream/predictions", tags=["Streaming"])
def get_predictions(limit: int = 20):
    return predictions_buffer[-limit:]


@app.get("/stream/stats", tags=["Streaming"])
def stream_stats():
    if not predictions_buffer:
        return {"message": "No predictions yet"}

    preds = predictions_buffer
    failures = [p for p in preds if p["prediction"] == 1]

    return {
        "total": len(preds),
        "failures": len(failures),
        "failure_rate": len(failures) / len(preds) if preds else 0,
        "avg_probability": np.mean([p["probability"] for p in preds]),
    }


@app.get("/model/robustness", tags=["CCI"])
def model_robustness(limit: int = 50):
    if cci_scorer is None:
        raise HTTPException(status_code=503, detail="CCI scorer not loaded")

    X_test = np.load("data/X_test.npy")
    X_test = X_test[:limit]

    summary = cci_scorer.get_robustness_summary(X_test)

    return summary


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
