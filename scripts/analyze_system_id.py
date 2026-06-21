"""
System-identification analysis over existing eval checkpoints.

Motivation (see RESEARCH_IDEA_EVALUATION.md, "the comparison is partly foregone"):
the genuinely novel, non-foregone cell in this benchmark is the `changed_hidden`
regime — physical constants (g, L, m, damping) are *not* disclosed, so a model
must infer them in-context from the initial state to predict accurately. This is
real in-context system identification.

The clean contrast is `changed_hidden` vs `changed_disclosed`: both use the same
non-standard constants, but only one discloses them. The "system-ID gap" =
error(hidden) - error(disclosed) measures how much a model is hurt by having to
infer the constants rather than being told. A small gap => the model recovers the
constants from context; a large gap => it cannot.

Usage:
    python scripts/analyze_system_id.py
    python scripts/analyze_system_id.py --checkpoint-dir results/checkpoints
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_checkpoints(ckpt_dir: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(ckpt_dir, "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        cell = d.get("cell", {})
        metrics = d.get("metrics", {}) or {}
        rows.append({
            "model":     cell.get("model_name"),
            "k":         cell.get("k"),
            "regime":    cell.get("regime"),
            "modality":  cell.get("modality"),
            "horizon":   cell.get("horizon"),
            "prompting": cell.get("prompting"),
            "success":   d.get("success"),
            "angle_error_mean": metrics.get("angle_error_mean"),
        })
    if not rows:
        raise SystemExit(f"No checkpoints found in {ckpt_dir}/. Run eval first.")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="System-identification (hidden-constants) analysis")
    ap.add_argument("--checkpoint-dir", default="results/checkpoints")
    ap.add_argument("--summary-dir", default="results/summary")
    ap.add_argument("--max-horizon", type=float, default=1.0,
                    help="ignore horizons beyond this (chaos invalidates long-horizon "
                         "pointwise error; default 1.0s keeps the discriminating range)")
    args = ap.parse_args()

    df = load_checkpoints(args.checkpoint_dir)
    os.makedirs(args.summary_dir, exist_ok=True)

    df = df[df["success"] == True].copy()  # noqa: E712
    df = df.dropna(subset=["angle_error_mean"])
    # Restrict to the horizons where pointwise error is still meaningful.
    df = df[df["horizon"] <= args.max_horizon]

    # Mean angle error per (model, regime) within the discriminating horizon range.
    grp = (df.groupby(["model", "regime"])["angle_error_mean"]
             .mean().reset_index())
    pivot = grp.pivot(index="model", columns="regime", values="angle_error_mean")

    # Build the system-ID gap where both regimes are present.
    if {"changed_hidden", "changed_disclosed"}.issubset(pivot.columns):
        pivot["sysid_gap (hidden-disclosed)"] = (
            pivot["changed_hidden"] - pivot["changed_disclosed"])
    if {"changed_hidden", "normal"}.issubset(pivot.columns):
        pivot["hidden_vs_normal"] = pivot["changed_hidden"] - pivot["normal"]

    sort_col = ("sysid_gap (hidden-disclosed)"
                if "sysid_gap (hidden-disclosed)" in pivot.columns
                else pivot.columns[0])
    pivot = pivot.sort_values(sort_col)

    print(f"\n=== Mean angle error (rad) by regime, horizons <= {args.max_horizon}s ===")
    print("sysid_gap = error(hidden) - error(disclosed): smaller (or negative) means the")
    print("model better infers the unstated constants from context.\n")
    print(pivot.round(4).to_string())

    out_path = os.path.join(args.summary_dir, "system_id.csv")
    pivot.to_csv(out_path)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
