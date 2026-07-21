"""Train Neural ODE and HNN models over nested trajectory budgets.

All budgets are manifest views into one seed-42 dataset.  The sample cap keeps
the number of optimizer examples fixed while trajectory diversity changes.
Runs are resumable at the checkpoint-directory level.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def trajectory_index(movement_id: str) -> int:
    match = re.search(r"_(\d+)$", movement_id)
    if match is None:
        raise ValueError(movement_id)
    return int(match.group(1))


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def prepare_datasets(root: Path, config: str, budgets: list[int]) -> None:
    full = root / "datasets" / "full_seed42"
    manifest = full / "manifest.json"
    maximum = max(budgets)
    if not manifest.exists():
        run([
            sys.executable, "scripts/generate_dataset.py", "--config", config,
            "--seed", "42", "--output-dir", str(full),
            "--id-prefix", "train42__", "--trajectories-per-cell", str(maximum),
            "--resume",
        ])
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    expected = 3 * 3 * maximum
    if len(entries) != expected:
        raise RuntimeError(f"Expected {expected} full-dataset trajectories; found {len(entries)}")
    for budget in budgets:
        selected = [entry for entry in entries if trajectory_index(entry["movement_id"]) < budget]
        destination = root / "datasets" / f"budget_{budget}"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "manifest.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
        normal = [entry for entry in selected if entry["regime"] == "normal"]
        normal_destination = root / "datasets" / f"budget_{budget}_normal"
        normal_destination.mkdir(parents=True, exist_ok=True)
        (normal_destination / "manifest.json").write_text(json.dumps(normal, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--root", default="results/training_budget_study")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--initialization-seeds", type=int, nargs="+", default=[101, 102, 103])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    root = Path(args.root)
    prepare_datasets(root, args.config, args.budgets)
    run_config = {
        "budgets_trajectories_per_cell": args.budgets,
        "total_trajectories": {
            "neural_ode": [9 * value for value in args.budgets],
            "hnn": [3 * value for value in args.budgets],
        },
        "initialization_seeds": args.initialization_seeds,
        "epochs": args.epochs,
        "max_samples_per_run": args.max_samples,
        "batch_size": args.batch_size,
        "models": {"neural_ode": "all regimes", "hnn": "undamped normal regime only"},
        "hnn_max_samples_per_run": 20000,
        "trajectory_seed": 42,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    for budget in args.budgets:
        dataset = root / "datasets" / f"budget_{budget}"
        for seed in args.initialization_seeds:
            output = root / "models" / f"budget_{budget}" / f"init_{seed}"
            complete = output / "training_metadata.json"
            checkpoints = {"neural_ode": output / "neural_ode.pt", "hnn": output / "hnn.pt"}
            if complete.exists() and all(path.exists() for path in checkpoints.values()):
                print(f"Reusing complete run {output}")
                continue
            output.mkdir(parents=True, exist_ok=True)
            model_settings = {
                "neural_ode": (dataset, args.max_samples),
                "hnn": (root / "datasets" / f"budget_{budget}_normal", 20000),
            }
            for model, checkpoint in checkpoints.items():
                if checkpoint.exists():
                    print(f"Reusing checkpoint {checkpoint}")
                    continue
                model_dataset, model_max_samples = model_settings[model]
                run([
                    sys.executable, "scripts/train_learned.py",
                    "--dataset-dir", str(model_dataset), "--output-dir", str(output),
                    "--models", model, "--epochs", str(args.epochs),
                    "--max-samples", str(model_max_samples), "--batch-size", str(args.batch_size),
                    "--seed", str(seed),
                ])


if __name__ == "__main__":
    main()
