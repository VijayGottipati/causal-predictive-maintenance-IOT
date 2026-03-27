import os
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo


def download_ai4i_dataset(output_dir="data", subset_size=5000):
    """
    Download AI4I 2020 Predictive Maintenance Dataset from UCI
    Uses subset for faster iteration on free tier
    """
    print("Downloading AI4I 2020 Predictive Maintenance Dataset...")

    ai4i = fetch_ucirepo(id=601)

    X = ai4i.data.features
    y = ai4i.data.targets

    df = pd.concat([X, y], axis=1)

    if subset_size and len(df) > subset_size:
        df = df.sample(n=subset_size, random_state=42)
        print(f"Using subset of {subset_size} rows")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ai4i_train.csv")
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nFeatures: {list(X.columns)}")
    print(f"\nTarget distribution:\n{df['Machine failure'].value_counts()}")

    return df


if __name__ == "__main__":
    download_ai4i_dataset()
