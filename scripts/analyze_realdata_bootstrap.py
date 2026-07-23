"""
Paired bootstrap confidence intervals for the hardware check, from the
per-cell records written by eval_realdata_myers.py.

Reports two CIs for each comparison and horizon: the iid bootstrap over
evaluation states used in the paper, and a moving-block bootstrap within
each held-out trial (block of 10 consecutive states, about 4 s) that
respects the temporal dependence of states sampled along one trajectory.
The conclusions agree: the RK4/persistence difference at 1.0 s straddles
zero under both, and the damped-model gain at 1.0 s excludes zero under
both.

    python scripts/analyze_realdata_bootstrap.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "summary_llm", "realdata_myers.csv")
TRIALS = ["Trial2", "Trial3"]
COMPARISONS = [
    ("physics-rk4", "persistence", "rk4 - persistence"),
    ("physics-rk4-damped", "physics-rk4", "damped - rk4"),
]
N_BOOT = 10_000
BLOCK = 10  # consecutive evaluation states (~4 s at the trial spacing)


def series(df, model, horizon):
    """Per-trial error series in evaluation-state (time) order."""
    return {tr: df[(df.trial == tr) & (df.model == model)
                   & (df.horizon == horizon)].sort_index().err.values
            for tr in TRIALS}


def iid_ci(diffs, rng):
    alld = np.concatenate(list(diffs.values()))
    means = [np.mean(rng.choice(alld, len(alld))) for _ in range(N_BOOT)]
    return np.percentile(means, [2.5, 97.5])


def block_ci(diffs, rng):
    means = []
    for _ in range(N_BOOT):
        chunks = []
        for d in diffs.values():
            n = len(d)
            n_blocks = int(np.ceil(n / BLOCK))
            starts = rng.integers(0, n - BLOCK + 1, n_blocks)
            chunks.append(np.concatenate([d[s:s + BLOCK]
                                          for s in starts])[:n])
        means.append(np.mean(np.concatenate(chunks)))
    return np.percentile(means, [2.5, 97.5])


def main():
    df = pd.read_csv(CSV)
    rng = np.random.default_rng(0)
    for T in sorted(df.horizon.unique()):
        print(f"--- horizon {T} s ---")
        for a, b, name in COMPARISONS:
            sa, sb = series(df, a, T), series(df, b, T)
            diffs = {tr: sa[tr] - sb[tr] for tr in TRIALS}
            point = np.mean(np.concatenate(list(diffs.values())))
            lo_i, hi_i = iid_ci(diffs, rng)
            lo_b, hi_b = block_ci(diffs, rng)
            print(f"  {name:20s} {point:+.3f}  "
                  f"iid [{lo_i:+.3f}, {hi_i:+.3f}]  "
                  f"block [{lo_b:+.3f}, {hi_b:+.3f}]")

    # Block-length sensitivity for the borderline comparison (damped - rk4
    # at 1.0 s): the significance should not hinge on the choice of block.
    global BLOCK
    print("--- block-length sensitivity, damped - rk4 at 1.0 s ---")
    sa, sb = series(df, "physics-rk4-damped", 1.0), series(df, "physics-rk4", 1.0)
    diffs = {tr: sa[tr] - sb[tr] for tr in TRIALS}
    default_block = BLOCK
    for BLOCK in (5, 10, 15, 20, 30):
        lo, hi = block_ci(diffs, np.random.default_rng(0))
        verdict = "excludes 0" if hi < 0 else "INCLUDES 0"
        print(f"  block={BLOCK:2d} ({BLOCK * 0.4:4.1f} s): "
              f"[{lo:+.3f}, {hi:+.3f}]  {verdict}")
    BLOCK = default_block


if __name__ == "__main__":
    main()
