"""Run local models on multiple held-out double-pendulum seeds.

The original learned-model training data remain untouched at ``results/dataset``
(seed 42). Each sweep seed creates a separate test set and checkpoint directory.
Only local model kinds are selected: numerical, learned, and ts_local. Azure and
other API-backed models are deliberately excluded.

Example:
    python scripts/run_seed_sweep.py --seeds 20260620 20260621 20260622 20260623 20260624
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


LOCAL_KINDS = {"numerical", "learned", "ts_local"}


def _local_model_names(config_path: Path) -> list[str]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return [model["name"] for model in config["models"]
            if model.get("kind") in LOCAL_KINDS]


def _run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate local models over held-out dataset seeds")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[20260620, 20260621, 20260622, 20260623, 20260624])
    ap.add_argument("--root", default="results/seed_sweep",
                    help="root directory for generated held-out datasets and checkpoints")
    ap.add_argument("--trajectories-per-cell", type=int, default=5,
                    help="held-out trajectories per (k, regime, seed); default: 5")
    ap.add_argument("--models", nargs="*", default=None,
                    help="optional subset of local model names")
    ap.add_argument("--horizons", type=float, nargs="*", default=None,
                    help="optional evaluation horizons passed to run_eval.py")
    ap.add_argument("--smoke", action="store_true",
                    help="generate and evaluate one small cell per seed")
    ap.add_argument("--no-hidden-prior", action="store_true",
                    help="do not apply the paper's neutral-prior setting to learned hidden-regime cells")
    args = ap.parse_args()

    config_path = Path(args.config)
    available = _local_model_names(config_path)
    selected = args.models if args.models is not None else available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise SystemExit(f"Only local models are allowed in this sweep; unsupported: {', '.join(unknown)}")

    root = Path(args.root)
    for seed in args.seeds:
        dataset_dir = root / "datasets" / f"seed_{seed}"
        checkpoint_dir = root / "checkpoints" / f"seed_{seed}"
        generator = [sys.executable, "scripts/generate_dataset.py", "--config", str(config_path),
                     "--seed", str(seed), "--output-dir", str(dataset_dir),
                     "--id-prefix", f"seed_{seed}__",
                     "--trajectories-per-cell", str(args.trajectories_per_cell)]
        if args.smoke:
            generator.extend(["--smoke", "--systems", "1", "--regimes", "normal"])
        _run(generator)

        evaluator = [sys.executable, "scripts/run_eval.py", "--config", str(config_path),
                     "--dataset-dir", str(dataset_dir), "--checkpoint-dir", str(checkpoint_dir),
                     "--models", *selected]
        if args.smoke:
            evaluator.append("--smoke")
        if args.horizons is not None:
            evaluator.extend(["--horizons", *[str(value) for value in args.horizons]])
        if not args.no_hidden_prior:
            evaluator.append("--withhold-hidden-constants-for-learned")
        _run(evaluator)


if __name__ == "__main__":
    main()
