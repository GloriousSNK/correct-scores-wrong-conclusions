"""
Chaos-appropriate analysis over existing eval checkpoints: predictability horizon.

Motivation (see RESEARCH_IDEA_EVALUATION.md): for chaotic systems, pointwise
state error past ~1 Lyapunov time saturates to "essentially random" for every
model, so the raw long-horizon angle-error leaderboard does not discriminate
model skill. The chaos-appropriate summary is the *predictability horizon*: the
longest horizon at which a model still tracks the true trajectory below an error
threshold.

This script computes, per model, the predictability horizon from the existing
endpoint checkpoints (no re-running needed) and sweeps the threshold so rankings
can be checked for sensitivity (the eval doc flags hand-set thresholds as a
validity risk).

Usage:
    python scripts/analyze_divergence.py
    python scripts/analyze_divergence.py --checkpoint-dir results/checkpoints \
        --thresholds 0.1 0.25 0.5 1.0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
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
            "model":       cell.get("model_name"),
            "k":           cell.get("k"),
            "regime":      cell.get("regime"),
            "modality":    cell.get("modality"),
            "horizon":     cell.get("horizon"),
            "prompting":   cell.get("prompting"),
            "movement_id": cell.get("movement_id"),
            "success":     d.get("success"),
            "angle_error_mean": metrics.get("angle_error_mean"),
        })
    if not rows:
        raise SystemExit(f"No checkpoints found in {ckpt_dir}/. Run eval first.")
    return pd.DataFrame(rows)


def predictability_horizon(per_horizon: pd.Series, threshold: float) -> float:
    """Longest horizon the model tracks *contiguously* from the start.

    `per_horizon` is indexed by horizon (seconds). We walk horizons in ascending
    order and stop at the first one whose mean angle error exceeds the threshold,
    returning the last horizon that passed (0.0 if even the shortest fails).

    The contiguous rule is deliberate: these systems are chaotic, so once a
    trajectory has diverged the pointwise error saturates and can dip back below
    a threshold at a later horizon purely by angle-wrap coincidence. That later
    dip is not "predictability" — only an unbroken run of below-threshold
    horizons from t=0 is.
    """
    per_horizon = per_horizon.sort_index()
    last_ok = 0.0
    for h, err in per_horizon.items():
        if err <= threshold:
            last_ok = float(h)
        else:
            break
    return last_ok


def main():
    ap = argparse.ArgumentParser(description="Predictability-horizon analysis")
    ap.add_argument("--checkpoint-dir", default="results/checkpoints")
    ap.add_argument("--summary-dir", default="results/summary")
    ap.add_argument("--thresholds", type=float, nargs="*",
                    default=[0.1, 0.25, 0.5, 1.0],
                    help="angle-error thresholds (rad) to sweep")
    ap.add_argument("--coords-only", action="store_true",
                    help="restrict to coords modality + no_cot (fair cross-family view)")
    args = ap.parse_args()

    df = load_checkpoints(args.checkpoint_dir)
    os.makedirs(args.summary_dir, exist_ok=True)

    # Only successful predictions carry a meaningful angle error.
    df = df[df["success"] == True].copy()  # noqa: E712
    df = df.dropna(subset=["angle_error_mean", "horizon"])

    if args.coords_only:
        df = df[(df["modality"] == "coords") & (df["prompting"] == "no_cot")]

    # Mean angle error per (model, horizon), averaged over k / regime / trajectories.
    grp = (df.groupby(["model", "horizon"])["angle_error_mean"]
             .mean().reset_index())
    pivot = grp.pivot(index="model", columns="horizon", values="angle_error_mean")
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    print("\n=== Mean angle error (rad) by horizon ===")
    print(pivot.round(4).to_string())

    # Predictability horizon per model under each threshold.
    out = {}
    for thr in args.thresholds:
        out[f"pred_horizon@{thr}rad"] = pivot.apply(
            lambda row: predictability_horizon(row.dropna(), thr), axis=1)
    sens = pd.DataFrame(out)
    # Rank by the middle threshold (longest horizon = best).
    mid = f"pred_horizon@{args.thresholds[len(args.thresholds)//2]}rad"
    sens = sens.sort_values(mid, ascending=False)

    print(f"\n=== Predictability horizon (s) — longest horizon below threshold ===")
    print("(higher = tracks the true trajectory for longer; threshold sensitivity shown)")
    print(sens.to_string())

    out_path = os.path.join(args.summary_dir, "predictability_horizon.csv")
    sens.to_csv(out_path)
    print(f"\nWrote {out_path}")

    pivot_path = os.path.join(args.summary_dir, "angle_error_by_horizon.csv")
    pivot.to_csv(pivot_path)
    print(f"Wrote {pivot_path}")


if __name__ == "__main__":
    main()
