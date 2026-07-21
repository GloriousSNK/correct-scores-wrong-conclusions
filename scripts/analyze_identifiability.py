"""Verify structural parameter invariances of the pendulum simulator.

Angle trajectories cannot identify every absolute simulator constant.  The
equations are unchanged by two continuous transformations:

1. ``m -> a*m`` and ``damping -> a*damping``;
2. ``L -> b*L``, ``g -> b*g``, and ``damping -> b**2*damping``.

For a single link this reduces to the identifiable combinations ``g/L`` and
``damping/(m*L**2)``.  This script checks the invariances numerically and writes
a machine-readable record for the manuscript analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.identifiability import equivalent_transform
from bench.simulator import PendulumParams, dynamics, integrate


def single_link_groups(params: PendulumParams) -> dict[str, float]:
    if params.k != 1:
        raise ValueError("single_link_groups requires k=1")
    return {
        "gravity_over_length": float(params.g / params.L[0]),
        "normalized_damping": float(params.damping / (params.m[0] * params.L[0] ** 2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/identifiability/invariance_check.json")
    parser.add_argument("--seed", type=int, default=770022)
    parser.add_argument("--trials", type=int, default=32)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    records = []
    worst_derivative = 0.0
    worst_trajectory = 0.0
    for trial in range(args.trials):
        k = int(rng.integers(1, 4))
        params = PendulumParams.make(
            k=k,
            L=rng.uniform(0.5, 1.5, size=k),
            m=rng.uniform(0.5, 2.0, size=k),
            g=float(rng.uniform(1.6, 12.0)),
            damping=float(rng.uniform(0.01, 0.1)),
        )
        mass_scale = float(rng.uniform(0.25, 4.0))
        length_scale = float(rng.uniform(0.5, 2.0))
        equivalent = equivalent_transform(params, mass_scale=mass_scale, length_scale=length_scale)
        state0 = np.concatenate([
            rng.uniform(-2.5, 2.5, size=k),
            rng.uniform(-1.0, 1.0, size=k),
        ])

        derivative_error = float(np.max(np.abs(dynamics(state0, params) - dynamics(state0, equivalent))))
        times, states = integrate(state0, params, t_end=1.0, dt=1e-3, method="rk4", record_every=10)
        times_eq, states_eq = integrate(
            state0, equivalent, t_end=1.0, dt=1e-3, method="rk4", record_every=10
        )
        if not np.array_equal(times, times_eq):
            raise RuntimeError("Equivalent integrations returned different time grids")
        trajectory_error = float(np.max(np.abs(states - states_eq)))
        worst_derivative = max(worst_derivative, derivative_error)
        worst_trajectory = max(worst_trajectory, trajectory_error)
        records.append({
            "trial": trial,
            "k": k,
            "mass_scale": mass_scale,
            "length_scale": length_scale,
            "max_derivative_difference": derivative_error,
            "max_trajectory_difference": trajectory_error,
        })

    one = PendulumParams.make(k=1, L=[0.8], m=[1.7], g=3.2, damping=0.07)
    one_eq = equivalent_transform(one, mass_scale=2.3, length_scale=1.4)
    output = {
        "seed": args.seed,
        "trials": args.trials,
        "invariances": [
            {"m": "a*m", "damping": "a*damping"},
            {"L": "b*L", "g": "b*g", "damping": "b^2*damping"},
        ],
        "single_link_identifiable_groups": single_link_groups(one),
        "single_link_transformed_groups": single_link_groups(one_eq),
        "worst_max_derivative_difference": worst_derivative,
        "worst_max_trajectory_difference": worst_trajectory,
        "records": records,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: output[key] for key in (
        "trials", "worst_max_derivative_difference", "worst_max_trajectory_difference",
        "single_link_identifiable_groups", "single_link_transformed_groups",
    )}, indent=2))


if __name__ == "__main__":
    main()
