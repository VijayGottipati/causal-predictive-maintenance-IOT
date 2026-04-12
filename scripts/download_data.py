import os
import time
import pandas as pd
import numpy as np
import urllib.request
from ucimlrepo import fetch_ucirepo


def download_ai4i_dataset(output_dir="data", subset_size=5000):
    """
    Download AI4I 2020 Predictive Maintenance Dataset from UCI.
    Includes retry logic and fallback for robust CI/CD execution.
    """
    print("Downloading AI4I 2020 Predictive Maintenance Dataset...")
    
    max_retries = 5
    retry_delay = 2  # seconds
    df = None

    # Try downloading via ucimlrepo with retries
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} via fetch_ucirepo...")
            ai4i = fetch_ucirepo(id=601)
            X = ai4i.data.features
            y = ai4i.data.targets
            df = pd.concat([X, y], axis=1)
            print("Successfully downloaded via fetch_ucirepo.")
            break
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                sleep_time = retry_delay * (2 ** attempt)
                print(f"Retrying in {sleep_time}s...")
                time.sleep(sleep_time)

    # Fallback to direct CSV download if ucimlrepo fails
    if df is None:
        print("Attempting fallback direct CSV download...")
        # UCI direct URL for this dataset
        direct_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
        try:
            df = pd.read_csv(direct_url)
            print("Successfully downloaded via direct CSV URL.")
        except Exception as e:
            print(f"Fallback failed: {e}")
            raise ConnectionError(f"Could not download dataset after {max_retries} attempts and fallback: {e}")

    # Subsampling for exploration/efficiency if requested
    if subset_size and len(df) > subset_size:
        df = df.sample(n=subset_size, random_state=42)
        print(f"Using subset of {subset_size} rows")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ai4i_train.csv")
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    
    # Print some info about the target if it exists
    if "Machine failure" in df.columns:
        print(f"\nTarget distribution:\n{df['Machine failure'].value_counts()}")

    return df


if __name__ == "__main__":
    download_ai4i_dataset()
