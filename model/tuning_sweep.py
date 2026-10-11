import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Load Extreme Data
df = pd.read_csv("extreme_shift.csv")
y = df.pop("TARGET").values
X = df.values.astype(float)

chunk_size = len(X) // 4
sc = StandardScaler().fit(X[:chunk_size])
X_scaled = np.c_[np.ones(len(X)), sc.transform(X)]

lam = 1e-2
sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -250, 250)))

def grad(w, X_b, y_b): return X_b.T @ (sig(X_b @ w) - y_b) / len(y_b) + lam * w
def hvp(w, v, X_b, y_b): p = sig(X_b @ w); return X_b.T @ (p*(1-p) * (X_b @ v)) / len(y_b) + lam * v

def evaluate_stream(update_fn, update_kwargs):
    batch_size = 100
    n_batches = len(X) // batch_size
    w = np.zeros(X_scaled.shape[1])
    st = {"m": 0, "s": 0, "v": np.random.randn(X_scaled.shape[1])}
    acc_history = []
    
    for t in range(n_batches):
        X_b = X_scaled[t*batch_size : (t+1)*batch_size]
        y_b = y[t*batch_size : (t+1)*batch_size]
        
        # Prequential accuracy
        pred = (sig(X_b @ w) >= 0.5).astype(int)
        acc_history.append(np.mean(pred == y_b))
        
        # Update
        g = grad(w, X_b, y_b)
        w = update_fn(w, g, X_b, y_b, t+1, st, **update_kwargs)
        
    return np.mean(acc_history)

# Update rules adapted for uniform signature
def update_gd(w, g, X_b, y_b, t, st, lr=0.1): return w - lr*g
def update_adam(w, g, X_b, y_b, t, st, lr=0.01):
    st["m"] = 0.9*st["m"] + 0.1*g; st["s"] = 0.999*st["s"] + 0.001*g*g
    return w - lr*(st["m"]/(1-0.9**t))/(np.sqrt(st["s"]/(1-0.999**t))+1e-8)
def update_eigen(w, g, X_b, y_b, t, st, gamma=1.0):
    v = st["v"]
    for _ in range(5):
        u = hvp(w, v, X_b, y_b)
        lam_hat = np.linalg.norm(u) + 1e-8
        v = u / lam_hat
    st["v"] = v
    return w - (gamma / (1.1 * lam_hat))*g

if __name__ == "__main__":
    print("--- Sweeping GD Learning Rates ---")
    best_gd_acc, best_gd_lr = 0, 0
    for lr in [1.0, 0.5, 0.1, 0.05, 0.01, 0.001]:
        acc = evaluate_stream(update_gd, {"lr": lr})
        print(f"GD (lr={lr}): {acc:.4f}")
        if acc > best_gd_acc: best_gd_acc, best_gd_lr = acc, lr

    print("\n--- Sweeping Adam Learning Rates ---")
    best_adam_acc, best_adam_lr = 0, 0
    for lr in [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]:
        acc = evaluate_stream(update_adam, {"lr": lr})
        print(f"Adam (lr={lr}): {acc:.4f}")
        if acc > best_adam_acc: best_adam_acc, best_adam_lr = acc, lr

    print("\n--- Running Drift-Aware Eigen-Scaling ---")
    eigen_acc = evaluate_stream(update_eigen, {"gamma": 1.0})
    print(f"Eigen-Scaling (untuned): {eigen_acc:.4f}")

    print("\n=== FINAL SHOWDOWN ===")
    print(f"Best Tuned GD (lr={best_gd_lr}):   {best_gd_acc:.4f}")
    print(f"Best Tuned Adam (lr={best_adam_lr}): {best_adam_acc:.4f}")
    print(f"Untuned Eigen-Scaling:        {eigen_acc:.4f}")