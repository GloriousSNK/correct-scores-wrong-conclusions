"""Aggregate balanced pendulum checkpoints with seed-level uncertainty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = [
    "rk4", "symplectic", "euler", "chronos2-uni", "chronos2-multi",
    "hnn-rollout", "neural-ode-rollout-mixed", "hnn-rollout-mixed", "lnn",
    "sindy", "nvar",
]


def bootstrap_seed_mean(values: np.ndarray, *, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def read_records(root: Path) -> list[dict]:
    rows = []
    for path in root.glob("seed_*/*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            cell = record["cell"]
        except Exception:
            continue
        if cell["model_name"] not in MODELS or float(cell["horizon"]) not in (1.0, 10.0):
            continue
        rows.append({
            "seed": int(path.parent.name.split("_")[-1]),
            "model": cell["model_name"], "k": int(cell["k"]),
            "regime": cell["regime"], "horizon": float(cell["horizon"]),
            "movement_id": cell["movement_id"],
            "answered": bool(record.get("success")),
            "error": (float(record["metrics"]["angle_error_mean"])
                      if record.get("success") else float(np.pi / 2)),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="results/balanced_seed_study/checkpoints")
    parser.add_argument("--scientific-root",
                        default="results/balanced_seed_study/scientific_baselines_final/checkpoints")
    parser.add_argument("--output", default="results/balanced_seed_study/summary")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    rows = read_records(Path(args.checkpoint_root))
    if Path(args.scientific_root).exists():
        rows.extend(read_records(Path(args.scientific_root)))
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "cells.csv", index=False)
    per_seed = (frame.groupby(["seed", "model"], as_index=False)
                .agg(error=("error", "mean"), answered=("answered", "mean"), cells=("error", "size")))
    per_seed.to_csv(output / "per_seed.csv", index=False)

    rng = np.random.default_rng(770023)
    pooled_rows = []
    for model in MODELS:
        values = per_seed.loc[per_seed["model"] == model, "error"].to_numpy(dtype=float)
        if len(values) == 0:
            continue
        low, high = bootstrap_seed_mean(values, samples=args.bootstrap_samples, rng=rng)
        pooled_rows.append({
            "model": model, "seeds": len(values), "error_mean": float(values.mean()),
            "error_sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
            "seed_bootstrap_low": low, "seed_bootstrap_high": high,
            "answered_mean": float(per_seed.loc[per_seed["model"] == model, "answered"].mean()),
        })
    pooled = pd.DataFrame(pooled_rows).sort_values("error_mean")
    pooled.to_csv(output / "pooled.csv", index=False)
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
