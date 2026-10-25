import unittest

import numpy as np

from drift_aware_eigen_scaling import (
    DriftingQuadratic,
    SimulationConfig,
    run_optimizer,
)


class PythonSimulationTests(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig()
        rng = np.random.default_rng(self.config.seed)
        self.start = np.array([-2.35, 2.05]) + 0.18 * rng.normal(size=2)
        self.problem = DriftingQuadratic(self.config.steps)

    def test_hessian_is_symmetric_positive_definite(self):
        for t in (0, 1, 60, 150, 259):
            eigenvalues = np.linalg.eigvalsh(self.problem.hessian(t))
            self.assertGreater(eigenvalues[0], 0.0)
            self.assertGreater(eigenvalues[1], eigenvalues[0])

    def test_source_matched_default_results(self):
        plain = run_optimizer(self.problem, self.config, "plain GD", self.start)
        frozen = run_optimizer(self.problem, self.config, "frozen eigen scaling", self.start)
        drift = run_optimizer(self.problem, self.config, "drift-aware eigen scaling", self.start)

        self.assertIsNone(plain.diverged_at)
        self.assertEqual(frozen.diverged_at, 42)
        self.assertIsNone(drift.diverged_at)
        self.assertAlmostEqual(plain.gaps[np.flatnonzero(np.isfinite(plain.gaps))[-1]], 0.06319405600107264, places=12)
        self.assertAlmostEqual(drift.gaps[np.flatnonzero(np.isfinite(drift.gaps))[-1]], 0.035516628350514536, places=12)


if __name__ == "__main__":
    unittest.main()
