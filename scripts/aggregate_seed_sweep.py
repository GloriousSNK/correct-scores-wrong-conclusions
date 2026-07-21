"""Summarize local-model robustness across held-out dataset seeds.

Each generated trajectory carries its seed in the movement ID. This script applies
the paper's reliability rule (failed or non-finite predictions score pi/2), first
averages trajectories within each seed, then summarizes those seed-level means.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd


SEED_RE = re.compile(r"^seed_(\d+)__")
FAILURE_PENALTY = math.pi / 2.0


def _record(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as f:
            result = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    cell = result.get("cell", {})
    match = SEED_RE.match(str(cell.get("movement_id", "")))
    if match is None:
        return None
    error = result.get("metrics", {}).get("angle_error_mean")
    success = bool(result.get("success")) and isinstance(error, (float, int)) and math.isfinite(error)
    return {
        "seed": int(match.group(1)),
        "model": cell.get("model_name"),
        "k": cell.get("k"),
        "regime": cell.get("regime"),
        "horizon": cell.get("horizon"),
        "modality": cell.get("modality"),
        "prompting": cell.get("prompting"),
        "answered": success,
        "reliability_adjusted_angle_error": float(error) if success else FAILURE_PENALTY,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate held-out local-model seed sweeps")
    ap.add_argument("--root", default="results/seed_sweep")
    args = ap.parse_args()

    root = Path(args.root)
    rows = [record for path in root.glob("checkpoints/seed_*/*.json") if (record := _record(path))]
    if not rows:
        raise SystemExit(f"No seed-sweep checkpoint records found below {root}")

    raw = pd.DataFrame(rows)
    group = ["seed", "model", "k", "regime", "horizon", "modality", "prompting"]
    per_seed = (raw.groupby(group, dropna=False)
                .agg(error=("reliability_adjusted_angle_error", "mean"),
                     answered=("answered", "mean"),
                     cells=("answered", "size"))
                .reset_index())
    summary_group = [column for column in group if column != "seed"]
    pooled = (per_seed.groupby(summary_group, dropna=False)
              .agg(seeds=("seed", "nunique"),
                   error_mean=("error", "mean"),
                   error_sd=("error", "std"),
                   answered_mean=("answered", "mean"),
                   cells_per_seed=("cells", "mean"))
              .reset_index())
    pooled["error_se"] = pooled["error_sd"] / pooled["seeds"].pow(0.5)

    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output / "per_seed.csv", index=False)
    pooled.to_csv(output / "pooled.csv", index=False)
    print(f"Wrote {output / 'per_seed.csv'} ({len(per_seed)} rows)")
    print(f"Wrote {output / 'pooled.csv'} ({len(pooled)} rows)")


if __name__ == "__main__":
    main()
