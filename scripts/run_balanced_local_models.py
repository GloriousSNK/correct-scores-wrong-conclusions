"""Evaluate selected local models on the prepared balanced held-out seeds."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


LOCAL_KINDS = {"numerical", "learned", "ts_local"}
DEFAULT_MODELS = [
    "rk4", "symplectic", "euler", "neural-ode-rollout-mixed",
    "hnn-rollout", "hnn-rollout-mixed", "chronos2-uni", "chronos2-multi",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--root", default="results/balanced_seed_study")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260622, 20260623, 20260624])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--horizons", type=float, nargs="+", default=[1.0, 10.0])
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    available = {model["name"]: model for model in config["models"]}
    invalid = [name for name in args.models
               if name not in available or available[name].get("kind") not in LOCAL_KINDS]
    if invalid:
        raise SystemExit(f"Non-local or unknown models: {', '.join(invalid)}")

    root = Path(args.root)
    for seed in args.seeds:
        dataset = root / "datasets" / f"seed_{seed}"
        if not (dataset / "manifest.json").exists():
            raise SystemExit(f"Missing prepared dataset: {dataset}")
        checkpoints = root / "checkpoints" / f"seed_{seed}"
        command = [
            sys.executable, "scripts/run_eval.py", "--config", args.config,
            "--dataset-dir", str(dataset), "--checkpoint-dir", str(checkpoints),
            "--models", *args.models, "--horizons", *[str(value) for value in args.horizons],
            "--withhold-hidden-constants-for-learned",
        ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
