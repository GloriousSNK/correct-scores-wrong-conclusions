from __future__ import annotations

import unittest

import numpy as np

from bench.identifiability import equivalence_log_residual, equivalent_transform
from bench.simulator import PendulumParams, dynamics


class IdentifiabilityTests(unittest.TestCase):
    def test_equivalent_transform_preserves_derivative(self) -> None:
        params = PendulumParams.make(
            k=3, L=[0.8, 1.2, 0.9], m=[0.5, 1.8, 1.0], g=1.62, damping=0.1
        )
        transformed = equivalent_transform(params, mass_scale=2.7, length_scale=1.6)
        state = np.array([0.2, -0.7, 1.1, 0.4, -0.3, 0.8])
        np.testing.assert_allclose(dynamics(state, params), dynamics(state, transformed),
                                   rtol=1e-12, atol=1e-12)

    def test_equivalent_parameters_have_zero_residual(self) -> None:
        truth = {"g": 3.71, "L": [1.4, 0.7], "m": [2.0, 0.6], "damping": 0.05}
        params = PendulumParams.make(k=2, **truth)
        transformed = equivalent_transform(params, mass_scale=0.4, length_scale=1.8)
        predicted = {
            "g": transformed.g, "L": transformed.L.tolist(),
            "m": transformed.m.tolist(), "damping": transformed.damping,
        }
        score = equivalence_log_residual(predicted, truth, 2)
        self.assertLess(score["log_rmse"], 1e-12)

    def test_non_equivalent_parameters_have_positive_residual(self) -> None:
        truth = {"g": 3.71, "L": [1.4, 0.7], "m": [2.0, 0.6], "damping": 0.05}
        predicted = {"g": 3.71, "L": [1.4, 1.1], "m": [2.0, 0.6], "damping": 0.05}
        score = equivalence_log_residual(predicted, truth, 2)
        self.assertGreater(score["log_rmse"], 0.05)


if __name__ == "__main__":
    unittest.main()
