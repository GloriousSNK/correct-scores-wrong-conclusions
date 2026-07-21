"""Evaluate validated SINDy and NVAR baselines on balanced held-out seeds."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pysindy as ps
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.metrics import all_pointwise_metrics
from bench.runner import cell_checkpoint_path, load_checkpoint, save_checkpoint
from bench.schema import EvalCell, Prediction, Trajectory
from bench.simulator import PendulumParams
from scripts.run_eval import load_trajectory


def state_at(trajectory: Trajectory, horizon: float) -> np.ndarray:
    index = int(np.argmin(np.abs(trajectory.times - horizon)))
    return trajectory.states[index].copy()


def load_manifest(directory: Path) -> list[Trajectory]:
    entries = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return [load_trajectory(entry["file"]) for entry in entries]


def finite_difference(states: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(states, dt, axis=0, edge_order=2)


@dataclass
class SINDYModel:
    model: ps.SINDy
    dt: float
    threshold: float
    validation_mse: float

    def forecast(self, state0: np.ndarray, horizon: float) -> np.ndarray:
        steps = max(1, int(round(horizon / self.dt)))
        state = np.asarray(state0, dtype=float).copy()

        def derivative(value: np.ndarray) -> np.ndarray:
            return np.asarray(self.model.predict(value[None, :])[0], dtype=float)

        for _ in range(steps):
            k1 = derivative(state)
            k2 = derivative(state + 0.5 * self.dt * k1)
            k3 = derivative(state + 0.5 * self.dt * k2)
            k4 = derivative(state + self.dt * k3)
            state = state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if not np.all(np.isfinite(state)) or np.max(np.abs(state)) > 1e6:
                raise FloatingPointError("SINDy rollout diverged")
        return state


def fit_sindy(trajectory: Trajectory, *, stride: int = 2) -> SINDYModel:
    states = trajectory.states[::stride]
    dt = float(np.median(np.diff(trajectory.times[::stride])))
    derivatives = finite_difference(states, dt)
    split = max(20, int(0.8 * len(states)))
    library = ps.PolynomialLibrary(degree=2, include_bias=True) + ps.FourierLibrary(n_frequencies=1)
    thresholds = (1e-4, 1e-3, 1e-2, 5e-2, 1e-1)
    best = None
    for threshold in thresholds:
        candidate = ps.SINDy(
            optimizer=ps.STLSQ(threshold=threshold, alpha=1e-5, max_iter=100,
                               normalize_columns=True),
            feature_library=library,
        )
        candidate.fit(states[:split], t=dt, x_dot=derivatives[:split])
        predicted = candidate.predict(states[split:])
        mse = float(np.mean((predicted - derivatives[split:]) ** 2))
        if best is None or mse < best[0]:
            best = (mse, threshold)
    assert best is not None
    final = ps.SINDy(
        optimizer=ps.STLSQ(threshold=best[1], alpha=1e-5, max_iter=100,
                           normalize_columns=True),
        feature_library=library,
    )
    final.fit(states, t=dt, x_dot=derivatives)
    return SINDYModel(final, dt, best[1], best[0])


@dataclass
class NVARModel:
    polynomial: PolynomialFeatures
    ridge: Ridge
    delays: int
    dt: float
    alpha: float
    validation_mse: float

    def forecast(self, history: np.ndarray, horizon: float) -> np.ndarray:
        window = [row.copy() for row in history[-self.delays:]]
        steps = max(1, int(round(horizon / self.dt)))
        for _ in range(steps):
            features = np.concatenate(window)[None, :]
            delta = self.ridge.predict(self.polynomial.transform(features))[0]
            next_state = window[-1] + delta
            if not np.all(np.isfinite(next_state)) or np.max(np.abs(next_state)) > 1e6:
                raise FloatingPointError("NVAR rollout diverged")
            window = window[1:] + [next_state]
        return np.asarray(window[-1], dtype=float)


def nvar_rows(states: np.ndarray, delays: int) -> tuple[np.ndarray, np.ndarray]:
    inputs = []
    targets = []
    for index in range(delays - 1, len(states) - 1):
        inputs.append(states[index - delays + 1:index + 1].reshape(-1))
        targets.append(states[index + 1] - states[index])
    return np.asarray(inputs), np.asarray(targets)


def fit_nvar(trajectory: Trajectory, *, delays: int = 3) -> NVARModel:
    dt = float(np.median(np.diff(trajectory.times)))
    inputs, targets = nvar_rows(trajectory.states, delays)
    split = max(20, int(0.8 * len(inputs)))
    polynomial = PolynomialFeatures(degree=2, include_bias=True)
    transformed_train = polynomial.fit_transform(inputs[:split])
    transformed_validation = polynomial.transform(inputs[split:])
    best = None
    for alpha in (1e-8, 1e-6, 1e-4, 1e-2, 1.0):
        candidate = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr")
        candidate.fit(transformed_train, targets[:split])
        mse = float(np.mean((candidate.predict(transformed_validation) - targets[split:]) ** 2))
        if best is None or mse < best[0]:
            best = (mse, alpha)
    assert best is not None
    transformed = polynomial.fit_transform(inputs)
    final = Ridge(alpha=best[1], fit_intercept=False, solver="lsqr")
    final.fit(transformed, targets)
    return NVARModel(polynomial, final, delays, dt, best[1], best[0])


def true_params(trajectory: Trajectory) -> PendulumParams:
    constants = trajectory.constants
    return PendulumParams.make(
        k=trajectory.k, L=constants["L"], m=constants["m"],
        g=constants["g"], damping=constants["damping"],
    )


def summarize(root: Path) -> None:
    rows = []
    for path in root.glob("checkpoints/seed_*/*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        cell = record["cell"]
        error = (float(record["metrics"]["angle_error_mean"])
                 if record.get("success") else float(np.pi / 2))
        rows.append({
            "seed": int(path.parent.name.split("_")[-1]),
            "model": cell["model_name"], "horizon": float(cell["horizon"]),
            "k": int(cell["k"]), "regime": cell["regime"],
            "movement_id": cell["movement_id"], "error": error,
            "answered": bool(record.get("success")),
        })
    frame = pd.DataFrame(rows)
    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    frame.to_csv(summary / "cells.csv", index=False)
    per_seed = (frame.groupby(["seed", "model"], as_index=False)
                .agg(error=("error", "mean"), answered=("answered", "mean"), cells=("error", "size")))
    per_seed.to_csv(summary / "per_seed.csv", index=False)
    pooled = (per_seed.groupby("model", as_index=False)
              .agg(seeds=("seed", "nunique"), error_mean=("error", "mean"),
                   error_sd=("error", "std"), answered_mean=("answered", "mean")))
    pooled.to_csv(summary / "pooled.csv", index=False)
    print(pooled.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", default="results/dataset")
    parser.add_argument("--held-root", default="results/balanced_seed_study/datasets")
    parser.add_argument("--output-root", default="results/balanced_seed_study/scientific_baselines")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[20260620, 20260621, 20260622, 20260623, 20260624])
    parser.add_argument("--horizons", type=float, nargs="+", default=[1.0, 10.0])
    parser.add_argument("--models", nargs="+", choices=["sindy", "nvar"], default=["sindy", "nvar"])
    parser.add_argument("--trajectory-limit", type=int, default=0,
                        help="limit trajectories per seed for smoke testing")
    args = parser.parse_args()

    training = load_manifest(Path(args.training_dir))
    by_cell = {(trajectory.k, trajectory.regime): trajectory for trajectory in training}
    if len(by_cell) != 9:
        raise RuntimeError(f"Expected one training trajectory per cell; found {len(by_cell)}")

    fitted = {}
    metadata = {}
    for key, trajectory in sorted(by_cell.items()):
        metadata[str(key)] = {}
        if "sindy" in args.models:
            model = fit_sindy(trajectory)
            fitted[("sindy", *key)] = model
            metadata[str(key)]["sindy"] = {
                "threshold": model.threshold, "validation_derivative_mse": model.validation_mse,
                "dt": model.dt,
            }
        if "nvar" in args.models:
            model = fit_nvar(trajectory)
            fitted[("nvar", *key)] = model
            metadata[str(key)]["nvar"] = {
                "alpha": model.alpha, "validation_increment_mse": model.validation_mse,
                "dt": model.dt, "delays": model.delays,
            }

    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "fit_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for seed in args.seeds:
        trajectories = load_manifest(Path(args.held_root) / f"seed_{seed}")
        if args.trajectory_limit:
            trajectories = trajectories[:args.trajectory_limit]
        checkpoint_dir = output / "checkpoints" / f"seed_{seed}"
        for trajectory in trajectories:
            zero = int(np.argmin(np.abs(trajectory.times)))
            history = trajectory.states[:zero + 1]
            for model_name in args.models:
                model = fitted[(model_name, trajectory.k, trajectory.regime)]
                for horizon in args.horizons:
                    cell = EvalCell(
                        model_name=model_name, k=trajectory.k, regime=trajectory.regime,
                        modality="coords", horizon=horizon, prompting="no_cot",
                        movement_id=trajectory.movement_id,
                    )
                    path = cell_checkpoint_path(cell, str(checkpoint_dir))
                    if load_checkpoint(path) is not None:
                        continue
                    started = time.perf_counter()
                    try:
                        predicted = (model.forecast(state_at(trajectory, 0.0), horizon)
                                     if model_name == "sindy" else model.forecast(history, horizon))
                        metrics = all_pointwise_metrics(predicted, state_at(trajectory, horizon),
                                                        true_params(trajectory))
                        prediction = Prediction(
                            cell=cell, pred_theta=predicted[:trajectory.k].tolist(),
                            pred_omega=predicted[trajectory.k:].tolist(),
                            latency_s=time.perf_counter() - started, metrics=metrics,
                        )
                    except Exception as error:
                        prediction = Prediction(
                            cell=cell, pred_theta=[], pred_omega=[], success=False,
                            latency_s=time.perf_counter() - started,
                            error=f"{type(error).__name__}: {error}",
                        )
                    save_checkpoint(path, prediction)
    summarize(output)


if __name__ == "__main__":
    main()
