import pandas as pd
import numpy as np
import json
import os
from scipy import stats

REFERENCE_DATA_PATH = "data/X_train.npy"
RECENT_DATA_PATH = "data/recent_predictions.json"
DRIFT_THRESHOLD = 0.5
DRIFT_OUTPUT = "drift_detected.json"

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


def load_reference_data():
    X_train = np.load(REFERENCE_DATA_PATH)
    df = pd.DataFrame(X_train, columns=FEATURE_NAMES)
    return df


def load_recent_data():
    if not os.path.exists(RECENT_DATA_PATH):
        return None

    with open(RECENT_DATA_PATH, "r") as f:
        data = json.load(f)

    if not data:
        return None

    df = pd.DataFrame(data)
    return df


def check_drift():
    ref_data = load_reference_data()
    current_data = load_recent_data()

    if current_data is None:
        print("No recent data found for drift detection")
        print("Creating sample recent data for demo...")

        np.random.seed(42)
        current_data = ref_data.copy()

        shift_cols = ["Air temperature", "Process temperature", "Torque"]
        for col in shift_cols:
            if col in current_data.columns:
                current_data[col] = current_data[col] + np.random.uniform(
                    0.5, 2.0, size=len(current_data)
                )

        os.makedirs("data", exist_ok=True)
        current_data.to_json(RECENT_DATA_PATH, orient="records")

        current_data = load_recent_data()

    drift_scores = []
    drift_details = {}

    for col in ref_data.columns:
        if col in current_data.columns:
            ref_vals = ref_data[col].values
            curr_vals = current_data[col].values

            ks_stat, p_value = stats.ks_2samp(ref_vals, curr_vals)

            drift_scores.append(ks_stat)
            drift_details[col] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
            }

    avg_drift_score = np.mean(drift_scores)
    max_drift_score = np.max(drift_scores)

    print(f"\n=== DRIFT DETECTION RESULTS ===")
    print(f"Average Drift Score: {avg_drift_score:.3f}")
    print(f"Max Drift Score: {max_drift_score:.3f}")
    print(f"Threshold: {DRIFT_THRESHOLD}")

    is_drift = bool(avg_drift_score > DRIFT_THRESHOLD)

    result = {
        "drift_detected": is_drift,
        "drift_score": float(avg_drift_score),
        "max_drift_score": float(max_drift_score),
        "threshold": DRIFT_THRESHOLD,
        "reference_size": len(ref_data),
        "current_size": len(current_data),
        "feature_details": drift_details,
    }

    with open(DRIFT_OUTPUT, "w") as f:
        json.dump(result, f, indent=2)

    if is_drift:
        print(
            f"DRIFT DETECTED! Score {avg_drift_score:.3f} exceeds threshold {DRIFT_THRESHOLD}"
        )
        drifted_features = [
            k for k, v in drift_details.items() if v["ks_statistic"] > 0.3
        ]
        print(f"Drifted features: {drifted_features}")
    else:
        print(
            f"No significant drift detected. Score {avg_drift_score:.3f} below threshold {DRIFT_THRESHOLD}"
        )

    return result


if __name__ == "__main__":
    result = check_drift()
