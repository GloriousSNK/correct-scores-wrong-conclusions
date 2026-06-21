"""
Per-model confidence intervals + key paired contrasts for the n=20 boot eval.

Gives every model a mean angle error with a 95% bootstrap CI (overall and per
horizon), and runs the headline paired comparisons (same trajectories) with a
Wilcoxon signed-rank p-value, bootstrap CI on the mean difference, and Holm
correction across the family of contrasts. Turns the round-2 point-estimate
leaderboard into one with uncertainty.

Usage:
    python scripts/analyze_boot_cis.py --checkpoint-dir results/checkpoints_boot
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

# Headline contrasts to test (A vs B; positive diff = B better, i.e. lower error).
DEFAULT_PAIRS = [
    ("neural-ode-rollout", "neural-ode-rollout-mixed"),   # does mixed-k beat single-k?
    ("symplectic", "neural-ode-rollout-mixed"),           # best non-numerical vs symplectic
    ("hnn-rollout", "hnn-rollout-mixed"),                 # does mixed-k help HNN?
    ("neural-ode", "neural-ode-rollout"),                 # rollout vs derivative MSE
    ("chronos2-uni", "neural-ode-rollout-mixed"),         # learned vs time-series FM
]


def load(ckpt_dir: str) -> pd.DataFrame:
    rows = []
    for p in glob.glob(os.path.join(ckpt_dir, "*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        c = d.get("cell", {}); m = d.get("metrics", {}) or {}
        rows.append({"model": c.get("model_name"), "k": c.get("k"),
                     "regime": c.get("regime"), "horizon": c.get("horizon"),
                     "movement_id": c.get("movement_id"), "success": d.get("success"),
                     "angle_error_mean": pd.to_numeric(m.get("angle_error_mean"),
                                                       errors="coerce")})
    if not rows:
        raise SystemExit(f"No checkpoints in {ckpt_dir}/.")
    df = pd.DataFrame(rows)
    return df[(df.success == True) & df.angle_error_mean.notna()].copy()  # noqa: E712


def mean_ci(x, n_boot=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(x)


def holm(pvals):
    order = np.argsort(pvals); adj = np.empty(len(pvals)); m = len(pvals); prev = 0.0
    for rank, i in enumerate(order):
        v = min(1.0, (m - rank) * pvals[i]); prev = max(prev, v); adj[i] = prev
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default="results/checkpoints_boot")
    ap.add_argument("--summary-dir", default="results/summary_boot")
    args = ap.parse_args()
    df = load(args.checkpoint_dir)
    os.makedirs(args.summary_dir, exist_ok=True)

    print("\n=== Per-model mean angle error, 95% bootstrap CI (overall) ===")
    rows = []
    for m in sorted(df.model.unique()):
        sub = df[df.model == m]
        mn, lo, hi, n = mean_ci(sub.angle_error_mean.to_numpy())
        rows.append({"model": m, "mean": mn, "ci_lo": lo, "ci_hi": hi, "n": n})
    cis = pd.DataFrame(rows).sort_values("mean")
    for _, r in cis.iterrows():
        print(f"  {r['model']:26s} {r['mean']:.3f}  95%CI[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  n={int(r['n'])}")
    cis.to_csv(os.path.join(args.summary_dir, "boot_model_cis.csv"), index=False)

    print("\n=== Headline paired contrasts (diff = A - B; + => B better/lower error) ===")
    from scipy.stats import wilcoxon
    keys = ["k", "regime", "horizon", "movement_id"]
    res = []
    for a, b in DEFAULT_PAIRS:
        da, db = df[df.model == a], df[df.model == b]
        if len(da) == 0 or len(db) == 0:
            print(f"  {a} vs {b}: missing (skip)"); continue
        mg = da.merge(db, on=keys, suffixes=("_a", "_b"))
        diff = (mg.angle_error_mean_a - mg.angle_error_mean_b).to_numpy()
        md, lo, hi, n = mean_ci(diff)
        p = float(wilcoxon(diff, alternative="two-sided").pvalue) if np.any(diff != 0) else 1.0
        res.append({"A": a, "B": b, "mean_diff_A_minus_B": md, "ci_lo": lo, "ci_hi": hi,
                    "wilcoxon_p": p, "n": n})
    if res:
        ps = [r["wilcoxon_p"] for r in res]
        padj = holm(np.array(ps))
        for r, pa in zip(res, padj):
            r["p_holm"] = float(pa)
            sig = "SIG" if pa < 0.05 else "n.s."
            better = r["B"] if r["mean_diff_A_minus_B"] > 0 else r["A"]
            print(f"  {r['A']:24s} vs {r['B']:24s} n={r['n']:3d}  diff={r['mean_diff_A_minus_B']:+.4f} "
                  f"95%CI[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  p_holm={pa:.4g} ({sig}; favors {better})")
        pd.DataFrame(res).to_csv(os.path.join(args.summary_dir, "boot_paired_contrasts.csv"), index=False)
    print(f"\nWrote CIs + contrasts to {args.summary_dir}/")


if __name__ == "__main__":
    main()
