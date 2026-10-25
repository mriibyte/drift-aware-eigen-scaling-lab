import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification

DATA_DIR = Path(__file__).resolve().parent

def generate_extreme_shift(n_samples=10000, n_features=20):
    chunks = 4
    chunk_size = n_samples // chunks
    X_list, y_list = [], []
    
    X_base, y_base = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=15, 
        weights=[0.85, 0.15], flip_y=0.05, random_state=42
    )
    
    for i in range(chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        
        # Extreme Variance Shift: 1x -> 5x -> 20x -> 50x
        multipliers = [1.0, 5.0, 20.0, 50.0]
        drift_multiplier = multipliers[i]
        
        X_chunk = (X_base[start:end] * drift_multiplier) + (2.0 * i)
        X_list.append(X_chunk)
        y_list.append(y_base[start:end])
        
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    df = pd.DataFrame(X, columns=[f"F_{i}" for i in range(n_features)])
    df["TARGET"] = y
    output = DATA_DIR / "extreme_shift.csv"
    df.to_csv(output, index=False)
    print(f"Extreme drift dataset created: {output}")

if __name__ == "__main__":
    generate_extreme_shift()
