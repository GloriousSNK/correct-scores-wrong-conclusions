"""Evaluate and aggregate the nested learned-model training-budget study."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def write_config(base_path: Path, destination: Path, *, budget: int, init_seed: int,
                 model_dir: Path) -> None:
    config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    config["modalities"] = ["coords"]
    config["prompting"] = ["no_cot"]
    config["horizons_seconds"] = [1.0, 10.0]
    config["models"] = [
        {
            "name": f"neural-ode-b{budget}-i{init_seed}", "kind": "learned",
            "variant": "neural_ode", "checkpoint": str(model_dir / "neural_ode.pt"),
        },
        {
            "name": f"hnn-b{budget}-i{init_seed}", "kind": "learned",
            "variant": "hnn", "checkpoint": str(model_dir / "hnn.pt"),
        },
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def aggregate(root: Path, *, bootstrap_samples: int, held_seeds: set[int] | None = None) -> None:
    rows = []
    if held_seeds is None:
        paths = root.glob("checkpoints/budget_*/init_*/seed_*/*.json")
    else:
        paths = (path for seed in held_seeds
                 for path in root.glob(f"checkpoints/budget_*/init_*/seed_{seed}/*.json"))
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        cell = record["cell"]
        model = "neural_ode" if cell["model_name"].startswith("neural-ode") else "hnn"
        held_seed = int(path.parent.name.split("_")[-1])
        if held_seeds is not None and held_seed not in held_seeds:
            continue
        rows.append({
            "budget": int(path.parents[2].name.split("_")[-1]),
            "init_seed": int(path.parents[1].name.split("_")[-1]),
            "held_seed": held_seed,
            "model": model, "training_scope": "normal" if model == "hnn" else "all",
            "horizon": float(cell["horizon"]),
            "k": int(cell["k"]), "regime": cell["regime"],
            "movement_id": cell["movement_id"], "answered": bool(record.get("success")),
            "error": (float(record["metrics"]["angle_error_mean"])
                      if record.get("success") else float(np.pi / 2)),
        })
    frame = pd.DataFrame(rows)
    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    frame.to_csv(summary / "cells.csv", index=False)
    per_run = (frame.groupby(["budget", "init_seed", "held_seed", "model", "training_scope"], as_index=False)
               .agg(error=("error", "mean"), answered=("answered", "mean"), cells=("error", "size")))
    per_run.to_csv(summary / "per_run.csv", index=False)

    rng = np.random.default_rng(770024)
    output = []
    for (budget, model, training_scope), group in per_run.groupby(["budget", "model", "training_scope"]):
        matrix = group.pivot(index="init_seed", columns="held_seed", values="error")
        if matrix.isna().any().any():
            continue
        values = matrix.to_numpy(dtype=float)
        draws = []
        for _ in range(bootstrap_samples):
            init_indices = rng.integers(0, values.shape[0], size=values.shape[0])
            seed_indices = rng.integers(0, values.shape[1], size=values.shape[1])
            draws.append(float(values[np.ix_(init_indices, seed_indices)].mean()))
        low, high = np.quantile(draws, [0.025, 0.975])
        output.append({
            "budget_per_cell": budget,
            "total_trajectories": (3 if training_scope == "normal" else 9) * budget,
            "model": model, "training_scope": training_scope,
            "initializations": values.shape[0], "held_seeds": values.shape[1],
            "error_mean": float(values.mean()),
            "init_mean_sd": float(values.mean(axis=1).std(ddof=1)),
            "held_seed_mean_sd": float(values.mean(axis=0).std(ddof=1)),
            "hierarchical_bootstrap_low": float(low),
            "hierarchical_bootstrap_high": float(high),
            "answered_mean": float(group["answered"].mean()),
        })
    pooled = pd.DataFrame(output).sort_values(["model", "budget_per_cell"])
    pooled.to_csv(summary / "pooled.csv", index=False)
    print(pooled.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--root", default="results/training_budget_study")
    parser.add_argument("--held-root", default="results/balanced_seed_study/datasets")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--initialization-seeds", type=int, nargs="+", default=[101, 102, 103])
    parser.add_argument("--held-seeds", type=int, nargs="+",
                        default=[20260620, 20260621, 20260622, 20260623, 20260624])
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    root = Path(args.root)
    if not args.aggregate_only:
        for budget in args.budgets:
            for init_seed in args.initialization_seeds:
                model_dir = root / "models" / f"budget_{budget}" / f"init_{init_seed}"
                required = [model_dir / "neural_ode.pt", model_dir / "hnn.pt"]
                if not all(path.exists() for path in required):
                    raise SystemExit(f"Training run incomplete: {model_dir}")
                config = root / "configs" / f"budget_{budget}_init_{init_seed}.yaml"
                write_config(Path(args.config), config, budget=budget, init_seed=init_seed,
                             model_dir=model_dir)
                for held_seed in args.held_seeds:
                    dataset = Path(args.held_root) / f"seed_{held_seed}"
                    checkpoints = (root / "checkpoints" / f"budget_{budget}" /
                                   f"init_{init_seed}" / f"seed_{held_seed}")
                    node_command = [
                        sys.executable, "scripts/run_eval.py", "--config", str(config),
                        "--dataset-dir", str(dataset), "--checkpoint-dir", str(checkpoints),
                        "--models", f"neural-ode-b{budget}-i{init_seed}",
                        "--horizons", "1", "10",
                        "--withhold-hidden-constants-for-learned",
                    ]
                    print("+", " ".join(node_command), flush=True)
                    subprocess.run(node_command, check=True)
                    normal_dataset = (Path("results/balanced_seed_study/datasets_normal") /
                                      f"seed_{held_seed}")
                    hnn_command = [
                        sys.executable, "scripts/run_eval.py", "--config", str(config),
                        "--dataset-dir", str(normal_dataset), "--checkpoint-dir", str(checkpoints),
                        "--models", f"hnn-b{budget}-i{init_seed}", "--horizons", "1", "10",
                    ]
                    print("+", " ".join(hnn_command), flush=True)
                    subprocess.run(hnn_command, check=True)
    aggregate(root, bootstrap_samples=args.bootstrap_samples,
              held_seeds=set(args.held_seeds))


if __name__ == "__main__":
    main()
