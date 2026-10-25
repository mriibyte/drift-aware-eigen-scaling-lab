# Drift-Aware Eigenvalue Scaling Lab

An educational, reproducible simulation study of gradient descent when the objective's curvature changes over time.

**Project links**

- **Interactive maths lab:** <https://mriibyte.github.io/drift-aware-eigen-scaling-lab/>
- **Research paper:** [`paper/drift-aware-eigen-scaling-final-paper.pdf`](paper/drift-aware-eigen-scaling-final-paper.pdf)
- **Lung-health research prototype:** <https://lung-health-research-app.vercel.app/>

> The lung-health prototype is a separate experimental application. It is not a medical device, does not diagnose disease, and must not be used as a substitute for professional medical care.

## What this repository contains

| Area | Contents |
| --- | --- |
| `drift_aware_eigen_scaling.py` | Python/Matplotlib reference implementation of the source-matched 2-D experiment |
| `docs/` | Dependency-light browser visualizer served by GitHub Pages |
| `model/` | Historical streaming-classification scripts and synthetic CSV inputs |
| `paper/` | The project paper and its reproducibility notes |
| `tests/` | JavaScript numerical tests, Python regression tests, and Playwright browser checks |

The browser visualizer and Python prototype intentionally share the same default problem schedule, optimizer constants, and source-matched seed-4 starting point. The default Python run reproduces the paper's headline values:

| Method | Final gap | Status |
| --- | ---: | --- |
| Plain GD | `0.0631940560` | stable for 260 updates |
| Frozen eigen scaling | `3.05895e+13` | implementation threshold crossed at update 42 |
| Drift-aware eigen scaling | `0.0355166284` | stable for 260 updates |

These are controlled synthetic results, not a claim of universal optimizer superiority.

## Mathematical idea

The simulation uses the moving strongly-convex quadratic

```text
f_t(x) = 1/2 (x - x*_t)^T A_t (x - x*_t)
```

where `A_t` is symmetric positive definite, its spectrum and eigenbasis drift, and the minimizer `x*_t` moves. Gradient descent updates are

```text
x_(t+1) = x_t - eta_t grad f_t(x_t)
```

For one fixed SPD quadratic, the classical fixed-objective stability condition is `0 < eta < 2 / lambda_max(A)`. The practical method estimates the largest eigenvalue with a short power iteration, filters it with a drift-responsive tracker, and uses

```text
eta_t = gamma / ((1 + safety) * lambda_hat_max,t)
```

The tracker is a scalar step-size heuristic. It is not a matrix preconditioner and it does not provide a non-stationary convergence theorem. Finite power iteration, eigendirection rotation, estimator lag, and target motion can still produce unsafe or suboptimal steps.

## Run locally

### Python reference report

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python drift_aware_eigen_scaling.py
```

The command writes `drift_aware_eigen_scaling.png` (ignored by Git). Useful options:

```bash
python drift_aware_eigen_scaling.py --steps 350 --seed 7 --save report.png
python drift_aware_eigen_scaling.py --show
```

### Browser visualizer

No frontend build is required. Plotly.js `2.35.2` is vendored in `docs/plotly-2.35.2.min.js`, so the lab and its browser checks do not depend on a third-party CDN being reachable at runtime.

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open <http://127.0.0.1:8000>. The visualizer runs the numerical core in the browser and provides interactive 3-D trajectories, objective-gap comparisons, spectrum tracking, step-size safety, and tracking-error views.

### Historical classification scripts

The scripts in `model/` are a separate, prequential logistic-classification experiment. They preserve the repository's HVP-norm rule and synthetic streams; they are not the same algorithm as the quadratic EMA tracker. Start with:

```bash
python model/generate_data.py
python model/new_data.py
python model/main_optimizer.py
python model/live_extreme.py
```

The classification inputs are synthetic and intentionally ordered to expose distribution drift. Do not interpret their metrics as clinical evidence or as validation of the quadratic tracker.

## Checks

Install JavaScript dependencies and run the narrow checks first:

```bash
npm ci
npm test
npm run test:python
```

For the full browser suite:

```bash
npx playwright install chromium
npm run test:browser
```

The browser tests exercise live slider updates, WebGL chart rendering, camera persistence, mobile layout, and the explicit Plotly-load failure state. The default CI workflow runs the deterministic JavaScript and Python checks; the browser workflow is available as a manual GitHub Actions dispatch because WebGL/browser availability varies across hosted runners.

## Reproducibility and scope

- The paper is included as a frozen artifact in [`paper/`](paper/).
- The paper's expanded multi-regime harness and run-level CSV outputs are not currently committed to this compact public repository; the checked-in Python and browser implementations cover the source-matched interactive lab and the historical classification scripts.
- Synthetic CSV files in `model/` are retained as project inputs. They contain no clinical patient data.
- The implementation uses a numerical divergence threshold of `||x|| > 10^6`; this is an instrumentation rule, not a mathematical definition of divergence.
- The 30-seed intervals and classification results described in the paper should be read with the limitations in the paper's Section 8.

## Research references

The paper contains the project bibliography. A plain-text copy with links and scope notes is available in [`REFERENCES.md`](REFERENCES.md). The most important background references are Polyak's fixed-objective gradient-method analysis, Barzilai–Borwein scalar step sizes, AdaGrad, Adam, randomized numerical linear algebra, and online optimization in dynamic environments.

## License

Source code in this repository is released under the MIT License; see [`LICENSE`](LICENSE). The included paper is an authored research artifact and is separately identified in [`paper/README.md`](paper/README.md). Third-party libraries and the Plotly.js CDN remain under their own licenses.

## Citation

If you use this repository or the paper, cite it using [`CITATION.cff`](CITATION.cff).

## Contributing

Bug reports and reproducibility improvements are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md). Please avoid presenting the synthetic demonstrations as clinical or universal optimization evidence.
