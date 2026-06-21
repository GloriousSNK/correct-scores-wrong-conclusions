"""
Paired significance test for the time-series univariate-vs-multivariate study.

Both modes forecast the SAME trajectories, so we compare them paired on
(k, regime, horizon, movement_id): diff = err(uni) - err(multi). A positive diff
means the multivariate model is better (lower error) on that trajectory. We report,
overall and per stratum (k, horizon): the paired mean difference with a 95%
bootstrap CI and a Wilcoxon signed-rank p-value, plus each model's mean error with
a 95% CI. This turns the round-2 point estimates into claims with uncertainty.

Usage:
    python scripts/analyze_ts_significance.py --checkpoint-dir results/checkpoints_tsboot
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


def load(ckpt_dir: str) -> pd.DataFrame:
    rows = []
    for p in glob.glob(os.path.join(ckpt_dir, "*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        c = d.get("cell", {})
        m = d.get("metrics", {}) or {}
        rows.append({
            "model": c.get("model_name"), "k": c.get("k"), "regime": c.get("regime"),
            "horizon": c.get("horizon"), "movement_id": c.get("movement_id"),
            "success": d.get("success"),
            "angle_error_mean": pd.to_numeric(m.get("angle_error_mean"), errors="coerce"),
        })
    if not rows:
        raise SystemExit(f"No checkpoints in {ckpt_dir}/.")
    df = pd.DataFrame(rows)
    return df[(df.success == True) & df.angle_error_mean.notna()].copy()  # noqa: E712


def mean_ci(x: np.ndarray, n_boot: int = 10000, seed: int = 0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), len(x)


def paired(uni: pd.DataFrame, multi: pd.DataFrame, label: str):
    keys = ["k", "regime", "horizon", "movement_id"]
    m = uni.merge(multi, on=keys, suffixes=("_uni", "_multi"))
    if len(m) == 0:
        print(f"  {label}: no paired cells"); return None
    diff = (m.angle_error_mean_uni - m.angle_error_mean_multi).to_numpy()
    md, lo, hi, n = mean_ci(diff)
    # Wilcoxon signed-rank (two-sided); guard the all-zero / tiny-n cases.
    p = float("nan")
    try:
        from scipy.stats import wilcoxon
        if np.any(diff != 0):
            p = float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        pass
    better = "multi" if md > 0 else "uni"
    sig = "significant" if (p == p and p < 0.05) else "n.s."
    print(f"  {label:18s} n={n:3d}  mean(uni-multi)={md:+.4f}  "
          f"95%CI[{lo:+.4f},{hi:+.4f}]  p={p:.4g} ({sig}; favors {better})")
    return {"stratum": label, "n": n, "mean_diff_uni_minus_multi": md,
            "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": p, "favors": better}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default="results/checkpoints_tsboot")
    ap.add_argument("--summary-dir", default="results/summary_tsboot")
    ap.add_argument("--uni", default="chronos2-uni")
    ap.add_argument("--multi", default="chronos2-multi")
    args = ap.parse_args()

    df = load(args.checkpoint_dir)
    os.makedirs(args.summary_dir, exist_ok=True)
    uni = df[df.model == args.uni]
    multi = df[df.model == args.multi]
    if len(uni) == 0 or len(multi) == 0:
        raise SystemExit(f"Missing models. Found: {sorted(df.model.unique())}")

    print(f"\n=== Per-model mean angle error (95% bootstrap CI) ===")
    for name, sub in [(args.uni, uni), (args.multi, multi)]:
        mn, lo, hi, n = mean_ci(sub.angle_error_mean.to_numpy())
        print(f"  {name:16s} mean={mn:.4f}  95%CI[{lo:.4f},{hi:.4f}]  n={n}")

    print(f"\n=== Paired uni-vs-multi (diff = err_uni - err_multi; + => multi better) ===")
    out = [paired(uni, multi, "OVERALL")]
    print("  -- by k --")
    for k in sorted(df.k.dropna().unique()):
        out.append(paired(uni[uni.k == k], multi[multi.k == k], f"k={int(k)}"))
    print("  -- by horizon --")
    for h in sorted(df.horizon.dropna().unique()):
        out.append(paired(uni[uni.horizon == h], multi[multi.horizon == h], f"h={h}"))

    res = pd.DataFrame([r for r in out if r])
    path = os.path.join(args.summary_dir, "ts_uni_vs_multi_significance.csv")
    res.to_csv(path, index=False)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
