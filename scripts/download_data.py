import os
import time
import pandas as pd
import numpy as np
import urllib.request
from ucimlrepo import fetch_ucirepo


def normalize_columns(df):
    """Strip units in brackets from column names for compatibility."""
    df = df.copy()
    # Handle common variants seen in different mirrors
    new_cols = {}
    for col in df.columns:
        # 'Air temperature [K]' -> 'Air temperature'
        clean_col = col.split(' [')[0].strip()
        new_cols[col] = clean_col
    return df.rename(columns=new_cols)


def generate_synthetic_data(num_rows=10000):
    """Generate a synthetic dataset matching the AI4I 2020 schema as a last resort."""
    print("Generating synthetic dataset (fallback)...")
    np.random.seed(42)
    
    data = {
        "UDI": np.arange(1, num_rows + 1),
        "Product ID": [f"L{np.random.randint(47000, 58000)}" for _ in range(num_rows)],
        "Type": np.random.choice(["L", "M", "H"], num_rows),
        "Air temperature": np.random.normal(300, 2, num_rows),
        "Process temperature": np.random.normal(310, 2, num_rows),
        "Rotational speed": np.random.normal(1500, 200, num_rows).astype(int),
        "Torque": np.random.normal(40, 10, num_rows),
        "Tool wear": np.random.randint(0, 250, num_rows),
        "Machine failure": np.random.choice([0, 1], num_rows, p=[0.96, 0.04]),
        "TWF": np.random.choice([0, 1], num_rows, p=[0.99, 0.01]),
        "HDF": np.random.choice([0, 1], num_rows, p=[0.99, 0.01]),
        "PWF": np.random.choice([0, 1], num_rows, p=[0.99, 0.01]),
        "OSF": np.random.choice([0, 1], num_rows, p=[0.99, 0.01]),
        "RNF": np.random.choice([0, 1], num_rows, p=[0.999, 0.001]),
    }
    return pd.DataFrame(data)


def download_ai4i_dataset(output_dir="data", subset_size=5000):
    """
    Download AI4I 2020 Predictive Maintenance Dataset with tiered fallback.
    Sources: 1) UCI API, 2) UCI Direct CSV, 3) GitHub Mirrors, 4) Synthetic Generate.
    """
    print("Initiating robust acquisition for AI4I 2020 dataset...")
    df = None

    # Source List
    sources = [
        {"name": "UCI Repository API", "type": "api", "id": 601},
        {"name": "UCI Direct CSV", "type": "url", "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"},
        {"name": "GitHub Mirror (yamanipa)", "type": "url", "url": "https://raw.githubusercontent.com/yamanipa/Predictive_Maintenance_using_Machine_Learning/master/ai4i2020.csv"},
        {"name": "GitHub Mirror (sharmaroshan)", "type": "url", "url": "https://raw.githubusercontent.com/sharmaroshan/Predictive-Maintenance-Dataset-AI4I-2020/master/ai4i2020.csv"},
    ]

    # Tiered Acquisition
    for source in sources:
        try:
            print(f"Attempting Source: {source['name']}...")
            if source["type"] == "api":
                ai4i = fetch_ucirepo(id=source["id"])
                df = pd.concat([ai4i.data.features, ai4i.data.targets], axis=1)
            else:
                # Use urllib for more control over timeout
                with urllib.request.urlopen(source["url"], timeout=10) as response:
                    df = pd.read_csv(response)
            
            if df is not None and not df.empty:
                print(f"Successfully acquired data from {source['name']}.")
                break
        except Exception as e:
            print(f"Source {source['name']} failed: {e}")
            continue

    # Final Fallback: Synthetic Generation
    if df is None:
        if os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("FORCE_SYNTHETIC") == "1":
            df = generate_synthetic_data()
        else:
            # If not in CI, maybe retry or fail
            print("All network sources failed. Creating synthetic data for continuity...")
            df = generate_synthetic_data()

    # Normalize column names for downstream tasks
    df = normalize_columns(df)

    # Subsampling
    if subset_size and len(df) > subset_size:
        df = df.sample(n=subset_size, random_state=42)
        print(f"Using subset of {subset_size} rows")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ai4i_train.csv")
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    if "Machine failure" in df.columns:
        print(f"\nTarget distribution:\n{df['Machine failure'].value_counts()}")

    return df


if __name__ == "__main__":
    download_ai4i_dataset()
