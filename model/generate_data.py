import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification

DATA_DIR = Path(__file__).resolve().parent

def generate_covariate_shift_dataset(n_samples=10000, n_features=20, filename="synthetic_covariate_shift.csv"):
    """
    Generates a binary classification dataset with intentional covariate shift 
    to test drift-aware optimization algorithms.
    """
    # Divide the dataset into 4 temporal chunks to simulate distribution drift
    chunks = 4
    chunk_size = n_samples // chunks
    
    X_list = []
    y_list = []
    
    # Generate the base classification problem (imbalanced to match realistic scenarios)
    X_base, y_base = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=15, 
        n_redundant=5, weights=[0.85, 0.15], flip_y=0.05, random_state=42
    )
    
    # Apply covariate shift (progressively scaling and shifting features)
    for i in range(chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        
        # The drift: features scale and shift significantly across chunks
        drift_multiplier = 1.0 + (0.75 * i)
        drift_offset = 1.5 * i
        
        X_chunk = (X_base[start:end] * drift_multiplier) + drift_offset
        y_chunk = y_base[start:end]
        
        X_list.append(X_chunk)
        y_list.append(y_chunk)
        
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    # Package into a DataFrame
    columns = [f"FEATURE_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df["TARGET"] = y
    
    output = Path(filename)
    if not output.is_absolute():
        output = DATA_DIR / output
    df.to_csv(output, index=False)
    print(f"Dataset generated: {output} with shape {df.shape}")

if __name__ == "__main__":
    generate_covariate_shift_dataset()
