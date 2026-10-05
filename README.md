# Drift-aware eigen scaling lab

An educational simulation for a maths project: gradient descent on a strongly convex quadratic whose Hessian changes over time.

**Live interactive website:** <https://mriibyte.github.io/drift-aware-eigen-scaling-lab/>

The project has two interfaces:

- `drift_aware_eigen_scaling.py` — a Python/Matplotlib dashboard with static plots.
- `docs/index.html` — an interactive Plotly.js website. It runs the simulation in the browser, renders rotatable/zoomable 3D charts, and lets visitors tune the curvature drift, target motion, number of iterations, normal GD step, eigen gain, safety margin, and power iterations. Slider changes update the simulation automatically.

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

The online estimate reacts faster when curvature increases. The visualizer compares it with normal fixed-step GD and a frozen eigen-scaled step that becomes stale under drift. This is an educational heuristic: power iteration may underestimate the largest eigenvalue, the target is moving, and neither method is guaranteed to outperform the other for all settings.

## Interactive 3D views

- Drag to rotate, scroll/pinch to zoom, hover for numerical values, and click a legend entry to hide a series.
- Sliders update automatically during adjustment. Rendering is throttled and serialized, so the latest values win without overlapping updates. Camera angles persist across parameter changes.
- The parameter plot offers an **initial loss landscape** and an **iteration/time** view. Surface and paths use the same coordinates and `f_0`; the landscape is a fixed reference, not a claim that all iterates belong to one unchanging objective.
- Parameter axes have equal scale. Off-screen divergent path segments are clipped rather than rescaled; their numerical gaps remain in the comparison charts.
- Other 3D charts use categorical series lanes. Objective gap and tracking error heights are `log10` values with a `1e-14` display floor; hover reveals the original number.
- The step-size ceiling is a common `2 / lambda_max` surface across all method lanes. The drift response chart explicitly normalizes each signal by its own peak.
- Equations use native MathML with separate labeled cards and accessible text. Plotly needs WebGL and a network connection to its pinned CDN; a visible message reports loading/rendering failures.

## Run the Python visualisation

```bash
# Termux package names, if needed:
pkg install python-numpy matplotlib

python3 drift_aware_eigen_scaling.py
# writes drift_aware_eigen_scaling.png

python3 drift_aware_eigen_scaling.py --show
```

## Run the website locally

No frontend build is required. Plotly.js is loaded from its versioned public CDN:

```bash
python3 -m http.server 8000 --directory docs
```

Then open <http://localhost:8000>.

The website is served from GitHub Pages using the repository's `/docs` folder.

## Checks

```bash
npm ci
npm test                        # numerical regression and coordinate tests
npx playwright install chromium # on a supported desktop/CI operating system
npm run test:browser             # 3D rendering, live inputs, camera and mobile checks
```

GitHub Actions runs both check suites in Chromium (software WebGL), including desktop/mobile screenshots. Termux can run the Node tests; the browser suite requires a supported Chromium environment. No frontend build step is required.
