import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent

# 1. Load Data (Keeping sequential order intact for drift)
df = pd.read_csv(DATA_DIR / "synthetic_covariate_shift.csv")
y = df.pop("TARGET").values
X = df.values.astype(float)

# 2. Scale based on the initial distribution only (so drift remains visible)
chunk_size = len(X) // 4
sc = StandardScaler().fit(X[:chunk_size])
X_scaled = np.c_[np.ones(len(X)), sc.transform(X)]

# 3. Model Setup (Mini-batch aware)
lam = 1e-2
sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -250, 250)))

def loss(w, X_b, y_b):
    p = sig(X_b @ w)
    e = 1e-12
    return -np.mean(y_b*np.log(p+e) + (1-y_b)*np.log(1-p+e)) + lam/2 * w @ w

def grad(w, X_b, y_b):
    return X_b.T @ (sig(X_b @ w) - y_b) / len(y_b) + lam * w

def hvp(w, v, X_b, y_b):
    p = sig(X_b @ w)
    return X_b.T @ (p*(1-p) * (X_b @ v)) / len(y_b) + lam * v

# 4. Optimizer Update Rules
def update_gd(w, g, st, lr=0.1):
    return w - lr*g

def update_adam(w, g, t, st, lr=0.01, b1=.9, b2=.999):
    st["m"] = b1*st["m"] + (1-b1)*g
    st["s"] = b2*st["s"] + (1-b2)*g*g
    mh = st["m"]/(1-b1**t)
    sh = st["s"]/(1-b2**t)
    return w - lr*mh/(np.sqrt(sh)+1e-8)

def update_eigen(w, g, X_b, y_b, st, gamma=1.0, safety=0.1, k=5):
    v = st["v"]
    lam_hat = 1e-8
    for _ in range(k):
        u = hvp(w, v, X_b, y_b)
        lam_hat = np.linalg.norm(u) + 1e-8
        v = u / lam_hat
    st["v"] = v
    eta = gamma / ((1 + safety) * lam_hat)
    return w - eta*g

# 5. Live Streaming Execution
if __name__ == "__main__":
    batch_size = 100
    n_batches = len(X) // batch_size
    
    # Initialize states
    w_gd = np.zeros(X_scaled.shape[1])
    w_adam = np.zeros(X_scaled.shape[1])
    w_eigen = np.zeros(X_scaled.shape[1])
    
    st_gd = {}
    st_adam = {"m": 0, "s": 0}
    st_eigen = {"v": np.random.randn(X_scaled.shape[1])}
    
    hist_gd, hist_adam, hist_eigen = [], [], []
    acc_gd, acc_adam, acc_eigen = [], [], []

    # Setup Live Plot with 2 subplots
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    print("Starting live prequential evaluation...")
    
    for t in range(n_batches):
        # Fetch the current sequential mini-batch
        X_b = X_scaled[t*batch_size : (t+1)*batch_size]
        y_b = y[t*batch_size : (t+1)*batch_size]
        
        # 1. PREQUENTIAL EVALUATION (Test before training)
        pred_gd = (sig(X_b @ w_gd) >= 0.5).astype(int)
        pred_adam = (sig(X_b @ w_adam) >= 0.5).astype(int)
        pred_eigen = (sig(X_b @ w_eigen) >= 0.5).astype(int)
        
        # Calculate moving average accuracy (last 5 batches) for smoother lines
        window = 5
        acc_gd.append(np.mean(pred_gd == y_b))
        acc_adam.append(np.mean(pred_adam == y_b))
        acc_eigen.append(np.mean(pred_eigen == y_b))
        
        smooth_gd = pd.Series(acc_gd).rolling(window, min_periods=1).mean()
        smooth_adam = pd.Series(acc_adam).rolling(window, min_periods=1).mean()
        smooth_eigen = pd.Series(acc_eigen).rolling(window, min_periods=1).mean()
        
        # 2. GRADIENT CALCULATION
        g_gd = grad(w_gd, X_b, y_b)
        g_adam = grad(w_adam, X_b, y_b)
        g_eigen = grad(w_eigen, X_b, y_b)
        
        # 3. WEIGHT UPDATES
        w_gd = update_gd(w_gd, g_gd, st_gd)
        w_adam = update_adam(w_adam, g_adam, t+1, st_adam)
        w_eigen = update_eigen(w_eigen, g_eigen, X_b, y_b, st_eigen)
        
        # 4. RECORD POST-UPDATE LOSS
        hist_gd.append(loss(w_gd, X_b, y_b))
        hist_adam.append(loss(w_adam, X_b, y_b))
        hist_eigen.append(loss(w_eigen, X_b, y_b))
        
        # Update the live plot every 5 batches
        if t % 5 == 0 or t == n_batches - 1:
            # Loss Plot
            ax1.clear()
            ax1.plot(hist_gd, label="Standard GD", alpha=0.6)
            ax1.plot(hist_adam, label="Adam", alpha=0.6)
            ax1.plot(hist_eigen, label="Drift-Aware Eigen-Scaling", linewidth=2, color='green')
            ax1.set_title("Live Batch Loss")
            ax1.set_ylabel("Logistic Loss (Log Scale)")
            ax1.set_yscale("log")
            ax1.legend()
            ax1.grid(True, which="both", ls="--", alpha=0.5)
            
            # Accuracy Plot
            ax2.clear()
            ax2.plot(smooth_gd, label="Standard GD", alpha=0.6)
            ax2.plot(smooth_adam, label="Adam", alpha=0.6)
            ax2.plot(smooth_eigen, label="Drift-Aware Eigen-Scaling", linewidth=2, color='green')
            ax2.set_title(f"Live Prequential Accuracy (Rolling Window = {window})")
            ax2.set_xlabel("Mini-batch Iteration")
            ax2.set_ylabel("Accuracy")
            ax2.set_ylim(0.4, 1.0)
            ax2.legend()
            ax2.grid(True, which="both", ls="--", alpha=0.5)
            
            # Mark the covariate shift points on both plots
            chunks = 4
            batches_per_chunk = n_batches // chunks
            for shift_point in range(batches_per_chunk, n_batches, batches_per_chunk):
                ax1.axvline(x=shift_point, color='red', linestyle='--', alpha=0.4)
                ax2.axvline(x=shift_point, color='red', linestyle='--', alpha=0.4)
                if t == n_batches - 1 and shift_point == batches_per_chunk:
                    ax1.text(shift_point+1, ax1.get_ylim()[1]*0.8, 'Shift', color='red', alpha=0.7)
            
            plt.draw()
            plt.pause(0.01)

    plt.ioff()
    
    print("\n--- Final Metrics (Averaged over all batches) ---")
    print(f"Standard GD Mean Accuracy: {np.mean(acc_gd):.4f}")
    print(f"Adam Mean Accuracy:        {np.mean(acc_adam):.4f}")
    print(f"Eigen-Scaling Mean Acc:    {np.mean(acc_eigen):.4f}")
    
    plt.show()
