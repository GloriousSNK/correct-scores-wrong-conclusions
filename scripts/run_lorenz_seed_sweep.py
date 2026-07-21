"""Resumable local-only Lorenz-63 evaluation over disjoint held-out seeds.

The learned Neural ODE trains only on ``train_seed``. Numerical baselines and
Chronos-2 see the same pre-t=0 history as the learned model. Results are stored
per cell, so rerunning the command resumes completed work.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pysindy as ps
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.lorenz import LorenzParams, dynamics, integrate, normalized_mae, sample_trajectory


DEFAULT_MODELS = [
    "rk4", "euler", "persistence", "sindy", "nvar", "neural-ode",
    "chronos2-uni", "chronos2-multi",
]


def _dataset(path: Path, *, seed: int, trajectories: int, context: float, forecast: float, dt: float) -> dict:
    if path.exists():
        with np.load(path) as data:
            loaded = {key: data[key] for key in data.files}
        expected = {"seed": seed, "trajectories": trajectories, "context": context,
                    "forecast": forecast, "dt": dt}
        actual = {key: loaded[key].item() for key in expected if key in loaded}
        if actual != expected:
            raise ValueError(f"Cached dataset metadata mismatch at {path}: {actual} != {expected}")
        return loaded
    rng = np.random.default_rng(seed)
    generated = [sample_trajectory(rng, context_seconds=context, forecast_seconds=forecast, dt=dt)
                 for _ in range(trajectories)]
    times = generated[0][0]
    states = np.stack([states for _, states in generated])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, times=times, states=states, seed=seed,
                        trajectories=trajectories, context=context, forecast=forecast, dt=dt)
    return {"times": times, "states": states, "seed": np.array(seed),
            "trajectories": np.array(trajectories), "context": np.array(context),
            "forecast": np.array(forecast), "dt": np.array(dt)}


class _NeuralODE:
    def __init__(self, checkpoint: Path, train_states: np.ndarray, *, seed: int, epochs: int, device: str):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(3, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 3)).to(self.device)
        if checkpoint.exists():
            self.net.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
            self.net.eval()
            return

        samples = train_states.reshape(-1, 3).astype(np.float32)
        targets = np.stack([dynamics(state) for state in samples]).astype(np.float32)
        x = torch.tensor(samples, device=self.device)
        y = torch.tensor(targets, device=self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        batch = 1024
        for _ in range(epochs):
            order = torch.randperm(len(x), device=self.device)
            for start in range(0, len(x), batch):
                idx = order[start:start + batch]
                loss = torch.nn.functional.mse_loss(self.net(x[idx]), y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), checkpoint)
        self.net.eval()

    def predict(self, state0: np.ndarray, horizon: float, dt: float) -> np.ndarray:
        torch = self.torch
        steps = int(round(horizon / dt))
        state = torch.tensor(state0, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for _ in range(steps):
                k1 = self.net(state)
                k2 = self.net(state + 0.5 * dt * k1)
                k3 = self.net(state + 0.5 * dt * k2)
                k4 = self.net(state + dt * k3)
                state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return state.detach().cpu().numpy()


class _Chronos2:
    def __init__(self, *, multivariate: bool, device: str):
        import torch
        from chronos import Chronos2Pipeline

        target = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=target)
        self.multivariate = multivariate
        self.quantiles = list(getattr(self.pipeline, "quantiles", [0.5]))
        self.qidx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2

    def predict(self, history: np.ndarray, steps: int) -> np.ndarray:
        context = np.asarray(history, dtype=np.float32).T
        if self.multivariate:
            output = self.pipeline.predict([context], prediction_length=steps)
            array = np.asarray(output[0].detach().cpu())
            return array[:, self.qidx, -1]
        output = self.pipeline.predict([channel for channel in context], prediction_length=steps)
        return np.array([np.asarray(item.detach().cpu())[0, self.qidx, -1] for item in output])


class _SINDy:
    def __init__(self, train_states: np.ndarray, dt: float):
        samples = train_states.reshape(-1, 3)
        targets = np.stack([dynamics(state) for state in samples])
        split = int(0.8 * len(samples))
        library = ps.PolynomialLibrary(degree=2, include_bias=True)
        best = None
        for threshold in (1e-5, 1e-4, 1e-3, 1e-2, 5e-2):
            candidate = ps.SINDy(
                optimizer=ps.STLSQ(threshold=threshold, alpha=1e-8, max_iter=100,
                                   normalize_columns=True),
                feature_library=library,
            )
            candidate.fit(samples[:split], t=dt, x_dot=targets[:split])
            mse = float(np.mean((candidate.predict(samples[split:]) - targets[split:]) ** 2))
            if best is None or mse < best[0]:
                best = (mse, threshold)
        self.validation_mse, self.threshold = best
        self.model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=self.threshold, alpha=1e-8, max_iter=100,
                               normalize_columns=True),
            feature_library=library,
        )
        self.model.fit(samples, t=dt, x_dot=targets)

    def predict(self, state0: np.ndarray, horizon: float, dt: float) -> np.ndarray:
        times = np.linspace(0.0, horizon, int(round(horizon / dt)) + 1)
        prediction = np.asarray(self.model.simulate(state0, times)[-1], dtype=float)
        if not np.all(np.isfinite(prediction)):
            raise FloatingPointError("SINDy rollout diverged")
        return prediction


def _nvar_rows(states: np.ndarray, delays: int) -> tuple[np.ndarray, np.ndarray]:
    inputs, targets = [], []
    for trajectory in states:
        for index in range(delays - 1, len(trajectory) - 1):
            inputs.append(trajectory[index - delays + 1:index + 1].reshape(-1))
            targets.append(trajectory[index + 1] - trajectory[index])
    return np.asarray(inputs), np.asarray(targets)


class _NVAR:
    def __init__(self, train_states: np.ndarray, dt: float, delays: int = 3):
        self.dt = dt
        self.delays = delays
        inputs, targets = _nvar_rows(train_states, delays)
        split = int(0.8 * len(inputs))
        self.polynomial = PolynomialFeatures(degree=2, include_bias=True)
        train_features = self.polynomial.fit_transform(inputs[:split])
        validation_features = self.polynomial.transform(inputs[split:])
        best = None
        for alpha in (1e-8, 1e-6, 1e-4, 1e-2, 1.0):
            candidate = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr")
            candidate.fit(train_features, targets[:split])
            mse = float(np.mean((candidate.predict(validation_features) - targets[split:]) ** 2))
            if best is None or mse < best[0]:
                best = (mse, alpha)
        self.validation_mse, self.alpha = best
        self.ridge = Ridge(alpha=self.alpha, fit_intercept=False, solver="lsqr")
        self.ridge.fit(self.polynomial.fit_transform(inputs), targets)

    def predict(self, history: np.ndarray, steps: int) -> np.ndarray:
        window = [row.copy() for row in history[-self.delays:]]
        for _ in range(steps):
            features = self.polynomial.transform(np.concatenate(window)[None, :])
            next_state = window[-1] + self.ridge.predict(features)[0]
            if not np.all(np.isfinite(next_state)) or np.max(np.abs(next_state)) > 1e6:
                raise FloatingPointError("NVAR rollout diverged")
            window = window[1:] + [next_state]
        return np.asarray(window[-1])


def _write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)


def _summarize(root: Path, seeds: set[int] | None = None,
               output_name: str = "summary") -> None:
    rows = []
    for path in root.glob("checkpoints/seed_*/*.json"):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    if not rows:
        return
    frame = pd.DataFrame(rows)
    if seeds is not None:
        frame = frame[frame["seed"].isin(seeds)]
    per_seed = (frame.groupby(["seed", "model", "horizon"], dropna=False)
                .agg(nmae=("nmae", "mean"), mae=("mae", "mean"), rmse=("rmse", "mean"),
                     success=("success", "mean"), cells=("success", "size"))
                .reset_index())
    pooled = (per_seed.groupby(["model", "horizon"], dropna=False)
              .agg(seeds=("seed", "nunique"), nmae_mean=("nmae", "mean"),
                   nmae_sd=("nmae", "std"), mae_mean=("mae", "mean"),
                   rmse_mean=("rmse", "mean"), success_rate=("success", "mean"))
              .reset_index())
    pooled["nmae_se"] = pooled["nmae_sd"] / np.sqrt(pooled["seeds"])
    output = root / output_name
    output.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output / "per_seed.csv", index=False)
    pooled.to_csv(output / "pooled.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run local Lorenz-63 seed robustness experiments")
    ap.add_argument("--root", default="results/lorenz_seed_sweep")
    ap.add_argument("--train-seed", type=int, default=42)
    ap.add_argument("--held-seeds", type=int, nargs="+", default=[20260620, 20260621, 20260622, 20260623, 20260624])
    ap.add_argument("--train-trajectories", type=int, default=20)
    ap.add_argument("--held-trajectories", type=int, default=5)
    ap.add_argument("--context-seconds", type=float, default=5.0)
    ap.add_argument("--forecast-seconds", type=float, default=2.0)
    ap.add_argument("--horizons", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--summary-name", default="summary")
    args = ap.parse_args()

    if args.summarize_only:
        _summarize(Path(args.root), set(args.held_seeds), args.summary_name)
        return

    models = list(dict.fromkeys(args.models))
    unknown = sorted(set(models) - set(DEFAULT_MODELS))
    if unknown:
        raise SystemExit(f"Unknown local Lorenz models: {', '.join(unknown)}")
    if max(args.horizons) > args.forecast_seconds:
        raise SystemExit("Every horizon must be no larger than --forecast-seconds")

    root = Path(args.root)
    train = _dataset(root / "datasets" / f"seed_{args.train_seed}.npz", seed=args.train_seed,
                     trajectories=args.train_trajectories, context=args.context_seconds,
                     forecast=args.forecast_seconds, dt=args.dt)
    zero = int(np.argmin(np.abs(train["times"])))
    scale = np.std(train["states"][:, :zero + 1, :].reshape(-1, 3), axis=0)
    root.mkdir(parents=True, exist_ok=True)
    run_config_path = root / "run_config.json"
    previous_config = (json.loads(run_config_path.read_text(encoding="utf-8"))
                       if run_config_path.exists() else {})
    run_config = {
        "system": "Lorenz-63",
        "parameters": {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        "ground_truth": {"integrator": "RK4", "dt": args.dt},
        "train_seed": args.train_seed,
        "held_seeds": sorted(set(previous_config.get("held_seeds", [])) | set(args.held_seeds)),
        "train_trajectories": args.train_trajectories,
        "held_trajectories_per_seed": args.held_trajectories,
        "context_seconds": args.context_seconds,
        "forecast_seconds": args.forecast_seconds,
        "horizons": args.horizons,
        "models": list(dict.fromkeys(previous_config.get("models", []) + models)),
        "chronos_model_id": "amazon/chronos-2",
        "neural_ode": {"hidden_width": 128, "hidden_layers": 2,
                       "epochs": args.epochs, "training_target": "analytic state derivative"},
        "primary_metric": "mean coordinate MAE divided by training-history coordinate standard deviation",
        "normalization_scale_xyz": scale.tolist(),
        "failure_policy": "store null errors and report success rate separately",
    }
    run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    neural = (_NeuralODE(root / "models" / f"neural_ode_seed_{args.train_seed}.pt", train["states"], seed=args.train_seed,
                         epochs=args.epochs, device=args.device) if "neural-ode" in models else None)
    sindy = _SINDy(train["states"], args.dt) if "sindy" in models else None
    nvar = _NVAR(train["states"], args.dt) if "nvar" in models else None
    if sindy is not None:
        run_config["sindy"] = {
            "library": "polynomial degree 2", "optimizer": "STLSQ",
            "threshold": sindy.threshold, "validation_derivative_mse": sindy.validation_mse,
            "training_target": "analytic state derivative",
        }
    if nvar is not None:
        run_config["nvar"] = {
            "delays": nvar.delays, "polynomial_degree": 2, "ridge_alpha": nvar.alpha,
            "validation_increment_mse": nvar.validation_mse,
        }
    run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    chronos: dict[str, _Chronos2 | Exception] = {}

    for seed in args.held_seeds:
        held = _dataset(root / "datasets" / f"seed_{seed}.npz", seed=seed,
                        trajectories=args.held_trajectories, context=args.context_seconds,
                        forecast=args.forecast_seconds, dt=args.dt)
        t0 = int(np.argmin(np.abs(held["times"])))
        for trajectory_index, states in enumerate(held["states"]):
            state0 = states[t0]
            history = states[:t0 + 1]
            for horizon in args.horizons:
                steps = int(round(horizon / args.dt))
                target = states[t0 + steps]
                for model in models:
                    path = root / "checkpoints" / f"seed_{seed}" / f"{model}__traj{trajectory_index:03d}__h{horizon:g}.json"
                    if path.exists():
                        continue
                    try:
                        if model == "rk4":
                            _, rollout = integrate(state0, t_end=horizon, dt=args.dt, method="rk4")
                            prediction = rollout[-1]
                        elif model == "euler":
                            _, rollout = integrate(state0, t_end=horizon, dt=args.dt, method="euler")
                            prediction = rollout[-1]
                        elif model == "persistence":
                            prediction = state0
                        elif model == "neural-ode":
                            prediction = neural.predict(state0, horizon, args.dt)
                        elif model == "sindy":
                            prediction = sindy.predict(state0, horizon, args.dt)
                        elif model == "nvar":
                            prediction = nvar.predict(history, steps)
                        else:
                            if model not in chronos:
                                try:
                                    chronos[model] = _Chronos2(
                                        multivariate=model.endswith("multi"), device=args.device)
                                except Exception as load_error:
                                    chronos[model] = load_error
                            predictor = chronos[model]
                            if isinstance(predictor, Exception):
                                raise predictor
                            prediction = predictor.predict(history, steps)
                        prediction = np.asarray(prediction, dtype=float)
                        absolute_error = np.abs(prediction - target)
                        value = normalized_mae(prediction, target, scale)
                        record = {
                            "system": "lorenz63", "train_seed": args.train_seed,
                            "seed": seed, "trajectory": trajectory_index, "model": model,
                            "horizon": horizon, "dt": args.dt,
                            "parameters": {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
                            "state0": state0.tolist(), "target": target.tolist(),
                            "prediction": prediction.tolist(),
                            "absolute_error_xyz": absolute_error.tolist(),
                            "mae": float(np.mean(absolute_error)),
                            "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
                            "normalization_scale_xyz": scale.tolist(),
                            "nmae": value, "success": bool(math.isfinite(value)), "error": None,
                        }
                    except Exception as exc:
                        record = {
                            "system": "lorenz63", "train_seed": args.train_seed,
                            "seed": seed, "trajectory": trajectory_index, "model": model,
                            "horizon": horizon, "dt": args.dt,
                            "parameters": {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
                            "state0": state0.tolist(), "target": target.tolist(),
                            "prediction": None, "absolute_error_xyz": None,
                            "mae": None, "rmse": None,
                            "normalization_scale_xyz": scale.tolist(),
                            "nmae": None, "success": False, "error": repr(exc),
                        }
                    _write_record(path, record)
        _summarize(root, set(args.held_seeds), args.summary_name)
    _summarize(root, set(args.held_seeds), args.summary_name)


if __name__ == "__main__":
    main()
