#!/usr/bin/env python3
"""Drift-aware eigen-scaling gradient descent: a visual simulation.

This is a small numerical experiment for a maths project.  The objective is a
2-D, strongly convex quadratic whose Hessian changes smoothly over time:

    f_t(x) = 1/2 (x - x*_t)^T A_t (x - x*_t),
    grad f_t(x) = A_t (x - x*_t).

The eigenvalues and eigenvectors of A_t drift, so a step size that was safe at
t=0 may become unsafe later.  Three methods are compared:

1. Plain GD: a deliberately conservative, fixed step size.
2. Frozen eigen scaling: eta = gain / lambda_max(0), computed only once.
3. Drift-aware eigen scaling: estimate lambda_max(A_t) with a few power
   iterations, react faster to upward spectral drift, and use
   eta_t = gain / ((1 + safety) * lambda_hat_t).

The plot is a dashboard containing trajectories, objective gaps, spectral
tracking, step sizes, and distance to the moving optimum.

Requirements:
    pip install numpy matplotlib

Examples:
    python drift_aware_eigen_scaling.py
    python drift_aware_eigen_scaling.py --steps 350 --show
    python drift_aware_eigen_scaling.py --save my_simulation.png --seed 7
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters shared by the drifting problem and the optimizers."""

    steps: int = 260
    gain: float = 1.78
    safety: float = 0.10
    plain_eta: float = 0.035
    power_iterations: int = 7
    tracker_alpha: float = 0.16
    tracker_alpha_max: float = 0.82
    drift_sensitivity: float = 7.0
    seed: int = 4


class DriftingQuadratic:
    """A two-dimensional SPD quadratic with changing curvature and minimizer."""

    def __init__(self, steps: int):
        self.steps = steps

    def spectrum(self, t: int) -> tuple[float, float]:
        """Return (lambda_min, lambda_max) at time t.

        The high-curvature direction gets both periodic and gradual drift.  The
        gradual term makes the frozen step-size failure visible rather than
        relying on a single abrupt discontinuity.
        """

        phase = 2.0 * np.pi * t / max(self.steps - 1, 1)
        lam_min = 0.65 + 0.18 * np.sin(phase + 0.75)
        lam_max = 8.0 + 13.0 * (0.5 + 0.5 * np.sin(phase - 0.85))
        lam_max += 11.0 * t / max(self.steps - 1, 1)
        return float(lam_min), float(lam_max)

    def rotation(self, t: int) -> float:
        """Angle of the leading eigenvector at time t."""

        phase = 2.0 * np.pi * t / max(self.steps - 1, 1)
        return float(0.18 + 0.70 * np.sin(0.55 * phase) + 0.12 * np.sin(phase))

    def optimum(self, t: int) -> np.ndarray:
        """A slowly moving minimizer, so tracking error is meaningful."""

        phase = 2.0 * np.pi * t / max(self.steps - 1, 1)
        return np.array(
            [1.35 * np.cos(0.72 * phase), 0.88 * np.sin(0.72 * phase + 0.55)]
        )

    def hessian(self, t: int) -> np.ndarray:
        """Construct A_t = Q diag(lambda_min, lambda_max) Q^T."""

        lam_min, lam_max = self.spectrum(t)
        theta = self.rotation(t)
        c, s = np.cos(theta), np.sin(theta)
        # Columns of Q are the low- and high-curvature eigenvectors.
        q = np.array([[c, -s], [s, c]])
        return q @ np.diag([lam_min, lam_max]) @ q.T

    def value(self, x: np.ndarray, t: int) -> float:
        """Return f_t(x), or NaN after a numerical divergence."""

        if not np.all(np.isfinite(x)):
            return np.nan
        delta = x - self.optimum(t)
        return float(0.5 * delta @ self.hessian(t) @ delta)

    def gradient(self, x: np.ndarray, t: int) -> np.ndarray:
        return self.hessian(t) @ (x - self.optimum(t))


class SpectralTracker:
    """Online estimate of the largest Hessian eigenvalue.

    Power iteration only needs Hessian-vector products in a large-scale
    implementation.  Here the 2-D matrix is available explicitly so that the
    educational simulation stays easy to read.
    """

    def __init__(self, config: SimulationConfig, vector: np.ndarray):
        self.config = config
        self.vector = vector / np.linalg.norm(vector)
        self.estimate: Optional[float] = None
        self.previous_raw: Optional[float] = None

    def update(self, hessian: np.ndarray) -> tuple[float, float, float]:
        """Return (raw Rayleigh estimate, filtered estimate, drift score)."""

        for _ in range(self.config.power_iterations):
            candidate = hessian @ self.vector
            norm = np.linalg.norm(candidate)
            if norm == 0.0:
                break
            self.vector = candidate / norm

        raw = float(self.vector @ hessian @ self.vector)
        if self.estimate is None:
            self.estimate = raw
            drift = 0.0
        else:
            assert self.previous_raw is not None
            drift = abs(raw - self.previous_raw) / max(abs(self.previous_raw), 1e-12)
            alpha = np.clip(
                self.config.tracker_alpha
                + self.config.drift_sensitivity * drift,
                self.config.tracker_alpha,
                self.config.tracker_alpha_max,
            )
            filtered = (1.0 - alpha) * self.estimate + alpha * raw
            # Curvature increases are dangerous: never let the filtered
            # estimate lag below the newly observed upward Rayleigh estimate.
            if raw > self.estimate:
                filtered = max(filtered, raw)
            self.estimate = float(filtered)

        self.previous_raw = raw
        return raw, float(self.estimate), float(drift)


@dataclass
class RunResult:
    name: str
    states: np.ndarray
    gaps: np.ndarray
    distances: np.ndarray
    eta: np.ndarray
    raw_lambda: np.ndarray
    tracked_lambda: np.ndarray
    drift_score: np.ndarray
    diverged_at: Optional[int]


def run_optimizer(
    problem: DriftingQuadratic,
    config: SimulationConfig,
    method: str,
    x0: np.ndarray,
) -> RunResult:
    """Run one optimizer and record quantities used by the visual report."""

    n = config.steps
    states = np.full((n + 1, 2), np.nan)
    gaps = np.full(n + 1, np.nan)
    distances = np.full(n + 1, np.nan)
    eta = np.full(n, np.nan)
    raw_lambda = np.full(n, np.nan)
    tracked_lambda = np.full(n, np.nan)
    drift_score = np.full(n, np.nan)

    x = x0.astype(float).copy()
    states[0] = x
    gaps[0] = problem.value(x, 0)
    distances[0] = np.linalg.norm(x - problem.optimum(0))

    initial_lambda = max(problem.spectrum(0))
    tracker: Optional[SpectralTracker] = None
    if method == "drift-aware eigen scaling":
        tracker = SpectralTracker(config, np.array([0.83, 0.41]))

    diverged_at: Optional[int] = None
    for t in range(n):
        hessian = problem.hessian(t)
        true_lambda = max(problem.spectrum(t))

        if method == "plain GD":
            step = config.plain_eta
        elif method == "frozen eigen scaling":
            step = config.gain / initial_lambda
        elif method == "drift-aware eigen scaling":
            assert tracker is not None
            raw, tracked, score = tracker.update(hessian)
            raw_lambda[t] = raw
            tracked_lambda[t] = tracked
            drift_score[t] = score
            step = config.gain / ((1.0 + config.safety) * tracked)
        else:
            raise ValueError(f"unknown optimizer: {method}")

        eta[t] = step
        # Stop recording once a deliberately unsafe method leaves the useful
        # numerical range; this keeps the visualization readable.
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e6:
            if diverged_at is None:
                diverged_at = t
            break

        x = x - step * problem.gradient(x, t)
        states[t + 1] = x
        gaps[t + 1] = problem.value(x, min(t + 1, n - 1))
        distances[t + 1] = np.linalg.norm(x - problem.optimum(min(t + 1, n - 1)))

        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e6:
            diverged_at = t + 1
            break

    if method != "drift-aware eigen scaling":
        # Fill these for convenient plotting and a complete table.
        tracked_lambda[:] = initial_lambda
        raw_lambda[:] = initial_lambda

    display_name = method
    return RunResult(
        name=display_name,
        states=states,
        gaps=gaps,
        distances=distances,
        eta=eta,
        raw_lambda=raw_lambda,
        tracked_lambda=tracked_lambda,
        drift_score=drift_score,
        diverged_at=diverged_at,
    )


def build_figure(
    problem: DriftingQuadratic,
    config: SimulationConfig,
    results: Dict[str, RunResult],
):
    """Create the multi-panel visual explanation."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    colors = {
        "plain GD": "#7c8798",
        "frozen eigen scaling": "#ef8354",
        "drift-aware eigen scaling": "#00a6a6",
    }
    styles = {
        "plain GD": "--",
        "frozen eigen scaling": ":",
        "drift-aware eigen scaling": "-",
    }

    fig, axes = plt.subplots(3, 2, figsize=(15, 16), constrained_layout=False)
    fig.patch.set_facecolor("#f7f9fb")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.07, hspace=0.34, wspace=0.24)
    fig.suptitle(
        "Drift-aware eigen scaling for gradient descent",
        fontsize=21,
        fontweight="bold",
        color="#17324d",
        y=0.965,
    )
    fig.text(
        0.5,
        0.938,
        "A moving quadratic objective: estimate the largest curvature online, then rescale the step before drift causes instability",
        ha="center",
        fontsize=11,
        color="#526477",
    )

    for ax in axes.flat:
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.22, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    # Panel 1: trajectory on the initial objective landscape.
    ax = axes[0, 0]
    grid = np.linspace(-3.5, 3.5, 240)
    xx, yy = np.meshgrid(grid, grid)
    xy = np.stack([xx - problem.optimum(0)[0], yy - problem.optimum(0)[1]], axis=-1)
    a0 = problem.hessian(0)
    zz = 0.5 * np.einsum("...i,ij,...j->...", xy, a0, xy)
    levels = np.geomspace(max(zz.min(), 1e-4), zz.max(), 14)
    ax.contourf(xx, yy, zz, levels=levels, norm=LogNorm(), cmap="Blues", alpha=0.38)
    ax.contour(xx, yy, zz, levels=levels, colors="#6793ad", linewidths=0.45, alpha=0.55)
    for name, result in results.items():
        valid = np.all(np.isfinite(result.states), axis=1)
        ax.plot(
            result.states[valid, 0],
            result.states[valid, 1],
            color=colors[name],
            linestyle=styles[name],
            linewidth=2.2,
            label=name,
        )
        if np.any(valid):
            last = np.flatnonzero(valid)[-1]
            ax.scatter(result.states[0, 0], result.states[0, 1], color=colors[name], s=28, zorder=5)
            ax.scatter(result.states[last, 0], result.states[last, 1], color=colors[name], s=42, marker="X", zorder=5)
    ax.scatter(*problem.optimum(0), s=90, marker="*", color="#17324d", label="initial optimum", zorder=6)
    ax.set_title("Parameter trajectories (initial contours)", fontweight="bold")
    # Keep the educational landscape visible even if an unsafe method shoots
    # far outside it; the divergent path will be clipped at the plot boundary.
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(fontsize=8, loc="upper right", frameon=True)
    ax.text(
        0.03,
        0.04,
        "Contours are $f_0$; the target and Hessian move during the run.",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#526477",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"),
    )

    times = np.arange(config.steps + 1)

    # Panel 2: objective gap.
    ax = axes[0, 1]
    for name, result in results.items():
        ax.semilogy(times, np.maximum(result.gaps, 1e-14), color=colors[name], linestyle=styles[name], linewidth=2.0, label=name)
        if result.diverged_at is not None:
            ax.axvline(result.diverged_at, color=colors[name], alpha=0.22, linewidth=1.3)
    ax.set_title("Instantaneous objective gap $f_t(x_t)$", fontweight="bold")
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel("gap (log scale)")
    ax.legend(fontsize=8, frameon=True)
    ax.text(
        0.03,
        0.06,
        "Lower is better; vertical fade marks numerical divergence.",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#526477",
    )

    # Panel 3: true spectrum and tracker.
    ax = axes[1, 0]
    true_min = np.array([problem.spectrum(t)[0] for t in range(config.steps)])
    true_max = np.array([problem.spectrum(t)[1] for t in range(config.steps)])
    t_steps = np.arange(config.steps)
    ax.plot(t_steps, true_max, color="#17324d", linewidth=2.1, label=r"true $\lambda_{\max}(A_t)$")
    ax.plot(t_steps, true_min, color="#6793ad", linewidth=1.6, label=r"true $\lambda_{\min}(A_t)$")
    frozen = results["frozen eigen scaling"].tracked_lambda
    tracker = results["drift-aware eigen scaling"]
    ax.plot(t_steps, frozen, color="#ef8354", linestyle=":", linewidth=1.8, label="frozen estimate")
    ax.plot(t_steps, tracker.raw_lambda, color="#00a6a6", linestyle=(0, (2, 2)), linewidth=1.1, alpha=0.75, label="raw power estimate")
    ax.plot(t_steps, tracker.tracked_lambda, color="#00a6a6", linewidth=2.3, label="drift-aware estimate")
    ax.set_title("Curvature drift and spectral tracking", fontweight="bold")
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel("eigenvalue")
    ax.legend(fontsize=8, ncol=2, frameon=True)

    # Panel 4: step sizes and safe upper bound 2/lambda_max.
    ax = axes[1, 1]
    safe_limit = 2.0 / true_max
    ax.plot(t_steps, safe_limit, color="#17324d", linewidth=1.8, label=r"stability limit $2/\lambda_{\max}$")
    for name, result in results.items():
        ax.plot(t_steps, result.eta, color=colors[name], linestyle=styles[name], linewidth=2.0, label=name)
    ax.set_title("Step size adapts before curvature makes GD unsafe", fontweight="bold")
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel(r"step size $\eta_t$")
    ax.legend(fontsize=8, frameon=True)
    ax.text(
        0.03,
        0.06,
        r"For a fixed SPD quadratic, $0 < \eta < 2/\lambda_{\max}$ is stable.",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#526477",
    )

    # Panel 5: distance to the moving minimizer.
    ax = axes[2, 0]
    for name, result in results.items():
        ax.semilogy(times, np.maximum(result.distances, 1e-14), color=colors[name], linestyle=styles[name], linewidth=2.0, label=name)
    ax.set_title(r"Tracking error $\|x_t - x_t^*\|_2$", fontweight="bold")
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel("distance (log scale)")
    ax.legend(fontsize=8, frameon=True)

    # Panel 6: drift score + compact mathematical explanation.
    ax = axes[2, 1]
    ax2 = ax.twinx()
    ax2.plot(t_steps, tracker.drift_score, color="#8c5fb2", linewidth=1.8, label="relative spectral drift")
    ax2.set_ylabel("drift score", color="#8c5fb2")
    ax2.tick_params(axis="y", colors="#8c5fb2")
    ax2.spines["top"].set_visible(False)
    ax.plot(t_steps, tracker.tracked_lambda, color="#00a6a6", linewidth=2.0, label=r"$\hat{\lambda}_{\max,t}$")
    ax.set_title("Why the tracker reacts faster during drift", fontweight="bold")
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel(r"tracked $\lambda_{\max}$", color="#00a6a6")
    ax.tick_params(axis="y", colors="#00a6a6")
    ax.text(
        0.04,
        0.91,
        "MATHEMATICAL IDEA",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color="#17324d",
    )
    ax.text(
        0.04,
        0.74,
        r"$x_{t+1}=x_t-\eta_t\nabla f_t(x_t)$" + "\n"
        r"$\eta_t=\frac{\gamma}{(1+s)\hat\lambda_{\max,t}}$" + "\n"
        r"$\hat\lambda$ uses power iteration + EMA" + "\n"
        "with faster response when drift spikes.",
        transform=ax.transAxes,
        fontsize=10,
        color="#31485d",
        linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef7f7", edgecolor="#b8dddd"),
    )
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8, loc="lower right", frameon=True)

    # Footer summary table.
    summary_lines = []
    for name, result in results.items():
        valid = np.isfinite(result.gaps)
        final_gap = result.gaps[np.flatnonzero(valid)[-1]] if np.any(valid) else np.nan
        status = f"diverged at t={result.diverged_at}" if result.diverged_at is not None else "stable"
        summary_lines.append(f"{name}: final gap={final_gap:.3g} | {status}")
    fig.text(
        0.5,
        0.018,
        "    •    ".join(summary_lines),
        ha="center",
        fontsize=9,
        color="#526477",
        family="monospace",
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=260, help="number of simulated updates (default: 260)")
    parser.add_argument("--seed", type=int, default=4, help="random seed for the initial point (default: 4)")
    parser.add_argument("--save", default="drift_aware_eigen_scaling.png", help="output image path")
    parser.add_argument("--show", action="store_true", help="open an interactive matplotlib window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 10:
        raise SystemExit("--steps must be at least 10")

    config = SimulationConfig(steps=args.steps, seed=args.seed)
    rng = np.random.default_rng(config.seed)
    x0 = np.array([-2.35, 2.05]) + 0.18 * rng.normal(size=2)
    problem = DriftingQuadratic(config.steps)

    methods = ("plain GD", "frozen eigen scaling", "drift-aware eigen scaling")
    results = {method: run_optimizer(problem, config, method, x0) for method in methods}

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    figure = build_figure(problem, config, results)
    figure.savefig(args.save, dpi=170, bbox_inches="tight")
    print(f"Saved visual report to {os.path.abspath(args.save)}")
    for name, result in results.items():
        valid = np.isfinite(result.gaps)
        final_gap = result.gaps[np.flatnonzero(valid)[-1]] if np.any(valid) else np.nan
        status = f"diverged at t={result.diverged_at}" if result.diverged_at is not None else "stable"
        print(f"  {name:28s} final gap={final_gap:.6g}  {status}")

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
