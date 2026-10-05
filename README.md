# Drift-aware eigen scaling lab

An educational simulation for a maths project: gradient descent on a strongly convex quadratic whose Hessian changes over time.

The project has two interfaces:

- `drift_aware_eigen_scaling.py` — a Python/Matplotlib dashboard with static plots.
- `docs/index.html` — a dependency-free interactive website. It runs the simulation in the browser and lets visitors tune the curvature drift, target motion, number of iterations, normal GD step, eigen gain, safety margin, and power iterations.

## Mathematical idea

The simulated objective is

```text
f_t(x) = 1/2 (x - x*_t)^T A_t (x - x*_t)
```

where `A_t` is symmetric positive definite but its eigenvalues and eigenvectors drift. Gradient descent is

```text
x_(t+1) = x_t - eta_t grad f_t(x_t)
```

For a fixed SPD quadratic, the step is stable when `0 < eta < 2 / lambda_max(A)`. The drift-aware method estimates `lambda_max(A_t)` with power iteration and uses

```text
eta_t = gamma / ((1 + safety) * lambda_hat_max,t)
```

The online estimate reacts faster when curvature increases. The visualizer compares it with normal fixed-step GD and a frozen eigen-scaled step that becomes stale under drift.

## Run the Python visualisation

```bash
# Termux package names, if needed:
pkg install python-numpy matplotlib

python3 drift_aware_eigen_scaling.py
# writes drift_aware_eigen_scaling.png

python3 drift_aware_eigen_scaling.py --show
```

## Run the website locally

No frontend build is required:

```bash
python3 -m http.server 8000 --directory docs
```

Then open <http://localhost:8000>.

The website is designed to be served from GitHub Pages using the repository's `/docs` folder.
