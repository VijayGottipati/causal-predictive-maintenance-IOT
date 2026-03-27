import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os


def preprocess_data(df, fit_scaler=None):
    """
    Preprocess AI4I data for model training
    """
    df = df.copy()

    le = LabelEncoder()
    df["Type"] = le.fit_transform(df["Type"])

    feature_cols = [
        "Type",
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear",
    ]

    X = df[feature_cols].values
    y = df["Machine failure"].values

    if fit_scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = fit_scaler
        X = scaler.transform(X)

    return X, y, scaler, le


def add_engineered_features(df):
    """
    Add derived features that help with prediction
    """
    df = df.copy()

    df["Temp_Diff"] = df["Process temperature"] - df["Air temperature"]

    df["Power"] = df["Rotational speed"] * df["Torque"]

    df["Temp_Rate"] = df["Temp_Diff"] / (df["Air temperature"] + 1)

    df["Wear_Stress"] = df["Tool wear"] * df["Torque"]

    return df


def create_train_test_split(X, y, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def save_preprocessed_data(
    X_train, X_test, y_train, y_test, scaler, le, output_dir="data"
):
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))

    print(f"Preprocessed data saved to {output_dir}")


if __name__ == "__main__":
    df = pd.read_csv("data/ai4i_train.csv")

    df = add_engineered_features(df)

    le = LabelEncoder()
    df["Type"] = le.fit_transform(df["Type"])

    feature_cols = [
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

    X = df[feature_cols].values
    y = df["Machine failure"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    save_preprocessed_data(X_train, X_test, y_train, y_test, scaler, le)

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Positive class ratio: {y.mean():.2%}")
