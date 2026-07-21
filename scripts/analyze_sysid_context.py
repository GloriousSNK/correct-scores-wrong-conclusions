"""Aggregate the observed-trajectory disclosure and identifiable-parameter study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="results/llm_sysid_context")
    parser.add_argument("--output-dir", default="results/summary_llm_sysid_context")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    rows = []
    for path in Path(args.checkpoint_dir).glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        identifiable = record.get("identifiable_parameter_error") or {}
        rows.append({
            "model": record["model"], "movement_id": record["movement_id"],
            "k": int(record["k"]), "set_id": record["set_id"],
            "horizon": float(record["horizon"]), "disclosure": record["disclosure"],
            "forecast_error": float(record.get("angle_error_mean", np.pi / 2)),
            "answered": bool(record.get("success")),
            "identifiable_log_rmse": identifiable.get("log_rmse"),
        })
    frame = pd.DataFrame(rows)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "cells.csv", index=False)

    index = ["model", "movement_id", "k", "set_id", "horizon"]
    paired = frame.pivot(index=index, columns="disclosure", values="forecast_error").reset_index()
    paired["disclosure_gap"] = paired["hidden"] - paired["disclosed"]
    paired.to_csv(output / "paired.csv", index=False)

    rng = np.random.default_rng(770026)
    summaries = []
    for model, group in paired.groupby("model"):
        trajectory_means = group.groupby("movement_id")["disclosure_gap"].mean().to_numpy()
        draws = rng.choice(
            trajectory_means,
            size=(args.bootstrap_samples, len(trajectory_means)), replace=True,
        ).mean(axis=1)
        low, high = np.quantile(draws, [0.025, 0.975])
        parameter = frame[(frame["model"] == model) & (frame["disclosure"] == "hidden")]
        summaries.append({
            "model": model, "paired_cells": len(group),
            "forecast_disclosure_gap": float(group["disclosure_gap"].mean()),
            "trajectory_bootstrap_low": float(low), "trajectory_bootstrap_high": float(high),
            "hidden_answered": float(parameter["answered"].mean()),
            "identifiable_parameter_log_rmse": float(parameter["identifiable_log_rmse"].mean()),
            "identifiable_parameter_scored_fraction": float(parameter["identifiable_log_rmse"].notna().mean()),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
