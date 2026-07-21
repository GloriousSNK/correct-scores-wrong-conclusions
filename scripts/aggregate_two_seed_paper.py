"""Build the manuscript's two-seed pendulum replication table from raw checkpoints."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


MODELS = [
    "rk4", "symplectic", "euler", "neural-ode-rollout-mixed",
    "hnn-rollout", "hnn-rollout-mixed", "lnn",
    "chronos2-uni", "chronos2-multi",
]
LEARNED = {"neural-ode-rollout-mixed", "hnn-rollout", "hnn-rollout-mixed", "lnn"}
HORIZONS = {1.0, 10.0}
FAILURE_PENALTY = math.pi / 2.0


def _load_record(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _row(record: dict, seed: int) -> dict | None:
    cell = record.get("cell", {})
    model = cell.get("model_name")
    horizon = float(cell.get("horizon", -1))
    if model not in MODELS or horizon not in HORIZONS:
        return None
    error = record.get("metrics", {}).get("angle_error_mean")
    answered = (bool(record.get("success")) and isinstance(error, (float, int))
                and math.isfinite(float(error)))
    return {
        "seed": seed, "model": model, "k": int(cell["k"]),
        "regime": cell["regime"], "horizon": horizon,
        "answered": answered,
        "error": float(error) if answered else FAILURE_PENALTY,
    }


def _original_seed(root: Path) -> list[dict]:
    base = root / "checkpoints_boot"
    hidden = root / "checkpoints_boot_hidden_prior"
    timeseries = root / "checkpoints_tsboot"
    rows = []
    for model in MODELS:
        source = timeseries if model.startswith("chronos2-") else base
        records = [_load_record(path) for path in source.glob(f"{model}__*.json")]
        if model in LEARNED:
            records = [record for record in records
                       if record and record.get("cell", {}).get("regime") != "changed_hidden"]
            hidden_records = [_load_record(path) for path in hidden.glob(f"{model}__*.json")]
            records.extend(record for record in hidden_records
                           if record and record.get("cell", {}).get("regime") == "changed_hidden")
        rows.extend(row for record in records if record and (row := _row(record, 20260620)))
    return rows


def _new_seed(root: Path) -> list[dict]:
    checkpoint_dir = root / "seed_sweep_two_seed_focused" / "checkpoints" / "seed_20260621"
    rows = []
    for path in checkpoint_dir.glob("*.json"):
        record = _load_record(path)
        if record and (row := _row(record, 20260621)):
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-dir", default="results/two_seed_summary")
    args = parser.parse_args()

    results = Path(args.results_root)
    rows = _original_seed(results) + _new_seed(results)
    expected = {20260620: 360, 20260621: 90}
    for seed, cells in expected.items():
        for model in MODELS:
            actual = sum(row["seed"] == seed and row["model"] == model for row in rows)
            if actual > cells:
                raise SystemExit(f"Expected at most {cells} cells for {model}, seed {seed}; found {actual}")
            rows.extend({"seed": seed, "model": model, "k": -1, "regime": "missing",
                         "horizon": -1.0, "answered": False, "error": FAILURE_PENALTY}
                        for _ in range(cells - actual))
    frame = pd.DataFrame(rows)

    per_seed = (frame.groupby(["seed", "model"], sort=False)
                .agg(error=("error", "mean"), answered=("answered", "mean"), cells=("error", "size"))
                .reset_index())
    pooled = (per_seed.groupby("model", sort=False)
              .agg(seeds=("seed", "nunique"), error_mean=("error", "mean"),
                   error_sd=("error", "std"), answered_mean=("answered", "mean"))
              .reset_index())
    pivot = per_seed.pivot(index="model", columns="seed", values="error").reset_index()
    pooled = pooled.merge(pivot, on="model")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "cells.csv", index=False)
    per_seed.to_csv(output / "per_seed.csv", index=False)
    pooled.to_csv(output / "pooled.csv", index=False)
    print(per_seed.to_string(index=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
