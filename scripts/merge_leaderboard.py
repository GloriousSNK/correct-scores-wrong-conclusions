"""Unified cross-family leaderboard: merge the out-of-sample local results
(numerical / learned / time-series) with the Part A LLM heldout results.

Both families were evaluated on the SAME out-of-sample trajectories
(movement_id) at horizons {1, 10}s, so they merge by (movement_id, horizon).

- Local models  : results/summary_boot/results_long.csv (12 models).
- LLMs          : Part A checkpoints in results/llm_main/. We use the no_cot
                  forecast as the canonical single-shot number (the local models
                  have no prompting axis; CoT remains a separate ablation).
- Reliability-adjusted: a failed cell scored at pi/2 rad (random-guess), matching
  the Part A/B convention, so every model is scored on the identical cell grid.

Output (results/summary_llm/):
  unified_leaderboard.csv / .md  - per model: family, reliability-adjusted mean
  angle error + 95% bootstrap CI, success rate, n, over the common cell grid.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

PI_2 = math.pi / 2.0
N_BOOT = 10000
RNG = np.random.default_rng(770021)
HORIZONS = [1.0, 10.0]

FAMILY = {
    "rk4": "numerical", "symplectic": "numerical", "euler": "numerical",
    "hnn": "learned", "hnn-rollout": "learned", "hnn-rollout-mixed": "learned",
    "neural-ode": "learned", "neural-ode-rollout": "learned",
    "neural-ode-rollout-mixed": "learned", "lnn": "learned",
    "chronos2-uni": "time-series", "chronos2-multi": "time-series",
    "kimi-k2.6": "llm", "deepseek-v4-pro": "llm",
    "grok-4-1-fast-reasoning": "llm",
}


def load_local(path):
    df = pd.read_csv(path)
    df = df[df["horizon"].isin(HORIZONS) & (df["modality"] == "coords")].copy()
    err = df["angle_error_mean"].to_numpy(dtype=float)
    ok = df["success"].astype(bool).to_numpy()
    adj = np.where(ok & np.isfinite(err), err, PI_2)
    return pd.DataFrame({
        "model": df["model"].to_numpy(), "movement_id": df["movement_id"].to_numpy(),
        "horizon": df["horizon"].to_numpy(), "regime": df["regime"].to_numpy(),
        "k": df["k"].to_numpy(), "success": ok.astype(int),
        "angle_error_mean": adj,
    })


def load_llm(ckpt_dir):
    rows = []
    for p in glob.glob(os.path.join(ckpt_dir, "*.json")):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if r.get("prompting") != "no_cot":
            continue
        ok = bool(r.get("success"))
        ae = r.get("angle_error_mean")
        err = float(ae) if (ok and ae is not None and np.isfinite(ae)) else PI_2
        rows.append({"model": r.get("model"), "movement_id": r.get("movement_id"),
                     "horizon": float(r.get("horizon")), "regime": r.get("regime"),
                     "k": r.get("k"), "success": int(ok),
                     "angle_error_mean": err})
    return pd.DataFrame(rows)


def boot_ci(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan")
    idx = RNG.integers(0, len(x), size=(N_BOOT, len(x)))
    bm = x[idx].mean(axis=1)
    return float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def leaderboard(df, grid_cells):
    """Score every model on the SAME canonical cell grid. A cell a model never
    produced (e.g. lnn's dropped NaN cells) is a failure -> pi/2, so abstention
    cannot inflate a model's mean."""
    grid = sorted(grid_cells)
    rows = []
    for m, sub in df.groupby("model"):
        by_cell = sub.drop_duplicates("cell").set_index("cell")
        err = by_cell["angle_error_mean"].reindex(grid)
        ok = by_cell["success"].reindex(grid).fillna(0)
        n_missing = int(err.isna().sum())
        err = err.fillna(PI_2)
        x = err.to_numpy(float)
        lo, hi = boot_ci(x)
        rows.append({"model": m, "family": FAMILY.get(m, "?"),
                     "angle_error_mean": float(x.mean()),
                     "ci_lo": lo, "ci_hi": hi,
                     "success_rate": float(ok.mean()),
                     "n": len(grid), "n_missing": n_missing})
    out = pd.DataFrame(rows).sort_values("angle_error_mean").reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def _df_to_md(df):
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default="results/summary_boot/results_long.csv")
    ap.add_argument("--llm-dir", default="results/llm_main")
    ap.add_argument("--out-dir", default="results/summary_llm")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    loc = load_local(args.local)
    llm = load_llm(args.llm_dir)
    both = pd.concat([loc, llm], ignore_index=True)
    both["cell"] = both["movement_id"].astype(str) + "@" + both["horizon"].astype(str)

    # canonical grid = every (movement_id, horizon) the LLMs covered (the full
    # 180 traj x {1,10} = 360 out-of-sample cells). Models missing a cell are
    # scored pi/2 there, so no model benefits from dropping hard cells.
    n_models = both["model"].nunique()
    grid = set(llm["movement_id"].astype(str) + "@" + llm["horizon"].astype(str))
    print(f"{n_models} models; canonical grid = {len(grid)} cells "
          f"(180 traj x {{1,10}}s). Missing cells scored pi/2.")

    lb = leaderboard(both, grid)
    lb.to_csv(os.path.join(args.out_dir, "unified_leaderboard.csv"), index=False)

    lines = ["# Unified cross-family leaderboard (out-of-sample)\n",
             f"- Canonical cell grid: **{len(grid)}** cells "
             f"(movement_id x horizon in {{1,10}}s), all **{n_models} models** "
             "scored on the SAME grid.",
             "- Reliability-adjusted mean angle error (rad); failed OR missing "
             "cell = pi/2. 95% bootstrap CI (B=10000). LLMs scored on no_cot "
             "forecasts. `n_missing` = cells the model never produced.\n",
             _df_to_md(lb)]
    open(os.path.join(args.out_dir, "unified_leaderboard.md"), "w").write(
        "\n".join(lines) + "\n")
    print(f"Wrote unified_leaderboard.csv/.md to {args.out_dir}/")
    print(lb.to_string(index=False))


if __name__ == "__main__":
    main()
