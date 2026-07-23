"""
In-sample evaluation of the six stable learned models on their own nine
training trajectories (one per (k, regime) cell), at the paper's comparison
horizons {1, 10} s, using the same inference path as the held-out harness
(scipy solve_ivp RK45, rtol 1e-6, atol 1e-8).

This produces the in-sample column of the in-sample-versus-held-out table:
the held-out column is the mean over the shared held-out grid at the same
horizons, taken from the main summary records.

    python scripts/eval_insample.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bench.models.base import PredictionRequest  # noqa: E402
from bench.models.learned import HNNPredictor, NeuralODEPredictor  # noqa: E402
from bench.schema import EvalCell  # noqa: E402
from bench.simulator import PendulumParams  # noqa: E402

DATASET = os.path.join(ROOT, "results", "dataset")
CKPT = os.path.join(ROOT, "results", "learned_models")
REGIMES = ["normal", "changed_disclosed", "changed_hidden"]
HORIZONS = [1.0, 10.0]

MODELS = {
    "neural-ode": (NeuralODEPredictor, "neural_ode.pt"),
    "neural-ode-rollout": (NeuralODEPredictor, "neural_ode_rollout.pt"),
    "neural-ode-rollout-mixed": (NeuralODEPredictor, "neural_ode_rollout_mixed.pt"),
    "hnn": (HNNPredictor, "hnn.pt"),
    "hnn-rollout": (HNNPredictor, "hnn_rollout.pt"),
    "hnn-rollout-mixed": (HNNPredictor, "hnn_rollout_mixed.pt"),
}


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def load_traj(k, regime):
    path = os.path.join(DATASET, f"k{k}_{regime}_0000.json")
    with open(path) as f:
        d = json.load(f)
    t = np.asarray(d["time"])
    i0 = int(np.argmin(np.abs(t)))          # t = 0 (files include context)
    dt = float(t[1] - t[0])
    theta = np.asarray(d["theta"])
    omega = np.asarray(d["omega"])
    return d["constants"], theta, omega, i0, dt


def main():
    rows = []
    for name, (cls, fname) in MODELS.items():
        predictor = cls(name, os.path.join(CKPT, fname))
        for k in (1, 2, 3):
            for regime in REGIMES:
                const, theta, omega, i0, dt = load_traj(k, regime)
                params = PendulumParams.make(
                    k=k, L=const["L"], m=const["m"],
                    g=const["g"], damping=const["damping"])
                state0 = np.concatenate([theta[i0], omega[i0]])
                for T in HORIZONS:
                    cell = EvalCell(name, k, regime, "coords", T, "direct",
                                    f"k{k}_{regime}_0000")
                    req = PredictionRequest(cell=cell, params=params,
                                            disclosed_params=params,
                                            state0=state0, horizon=T)
                    res = predictor.predict(req)
                    true = theta[i0 + int(round(T / dt))]
                    err = (np.pi / 2 if not res.success else
                           float(np.mean(np.abs(wrap(np.asarray(res.pred_theta)
                                                     - true)))))
                    rows.append((name, k, regime, T, err, res.success))

    import pandas as pd
    df = pd.DataFrame(rows, columns=["model", "k", "regime", "horizon",
                                     "err", "answered"])
    out = os.path.join(ROOT, "results", "summary_llm",
                       "insample_training_trajectories.csv")
    df.to_csv(out, index=False)

    print("In-sample wrapped mean angle error (rad), nine training "
          "trajectories, harness inference path:")
    for name in MODELS:
        d = df[df.model == name]
        by_h = d.groupby("horizon").err.mean()
        print(f"  {name:26s} 1s={by_h[1.0]:.3f} 10s={by_h[10.0]:.3f} "
              f"mean={d.err.mean():.3f}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
