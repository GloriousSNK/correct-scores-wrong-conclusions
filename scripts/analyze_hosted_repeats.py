"""Measure run-to-run variation in the stratified hosted-model repeat study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def key(record: dict) -> tuple:
    return (record["model"], record["movement_id"], float(record["horizon"]), record["prompting"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="results/hosted_repeat_plan/plan.json")
    parser.add_argument("--original-dir", default="results/llm_main")
    parser.add_argument("--repeat-dir", default="results/llm_repeat/main")
    parser.add_argument("--output-dir", default="results/summary_llm_repeat")
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    selected = {key(record) for record in plan["selected_cells"]}
    rows = []
    for path in Path(args.original_dir).glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if key(record) in selected:
            rows.append({**record, "replicate": 0})
    for path in Path(args.repeat_dir).glob("*.json"):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    frame = pd.DataFrame([{
        "model": record["model"], "movement_id": record["movement_id"],
        "k": int(record["k"]), "regime": record["regime"],
        "horizon": float(record["horizon"]), "replicate": int(record.get("replicate", 0)),
        "error": float(record.get("angle_error_mean", np.pi / 2)),
        "answered": bool(record.get("success")),
    } for record in rows])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "cells.csv", index=False)
    per_cell = (frame.groupby(["model", "movement_id", "k", "regime", "horizon"], as_index=False)
                .agg(error_mean=("error", "mean"), error_sd=("error", "std"),
                     error_range=("error", lambda values: float(values.max() - values.min())),
                     answered=("answered", "mean"), repeats=("replicate", "nunique")))
    per_cell.to_csv(output / "per_cell.csv", index=False)
    summary = (per_cell.groupby("model", as_index=False)
               .agg(cells=("movement_id", "size"), mean_within_cell_sd=("error_sd", "mean"),
                    median_within_cell_sd=("error_sd", "median"),
                    mean_range=("error_range", "mean"), answered=("answered", "mean")))
    summary.to_csv(output / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
