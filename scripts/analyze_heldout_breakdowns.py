"""
Per-k and per-regime breakdowns + cluster-bootstrap (by-trajectory) CIs on the
shared out-of-sample held-out grid (horizons {1,10}s, coords/no_cot).

Addresses reviewer items: #11 (per-k), #12 (per-regime), #5 (cluster bootstrap by
trajectory rather than treating correlated cells as i.i.d.).

Sources: results/checkpoints_boot/*.json (numerical/learned/Chronos) and
results/summary_llm/llm_main_results_long.csv (LLMs). Reliability-adjusted: any
failed/NaN cell scored at pi/2.

    python scripts/analyze_heldout_breakdowns.py
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np, pandas as pd

PI2 = np.pi / 2
HORIZONS = {1.0, 10.0}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load() -> pd.DataFrame:
    rows = []
    # local (numerical / learned / chronos)
    for p in glob.glob(os.path.join(ROOT, "results/checkpoints_boot/*.json")):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        c = d.get("cell", {})
        if c.get("modality") not in (None, "coords"): continue
        if c.get("prompting") not in (None, "no_cot"): continue
        if float(c.get("horizon", -1)) not in HORIZONS: continue
        e = (d.get("metrics") or {}).get("angle_error_mean")
        ok = bool(d.get("success")) and e is not None and np.isfinite(e)
        rows.append((c.get("model_name"), int(c["k"]), c["regime"], float(c["horizon"]),
                     c["movement_id"], e if ok else PI2))
    # LLMs
    csv = os.path.join(ROOT, "results/summary_llm/llm_main_results_long.csv")
    if os.path.exists(csv):
        L = pd.read_csv(csv)
        L = L[(L.prompting == "no_cot") & (L.horizon.isin(HORIZONS))]
        for _, r in L.iterrows():
            e = pd.to_numeric(r.angle_error_mean, errors="coerce")
            ok = bool(r.success) and np.isfinite(e)
            rows.append((r.model, int(r.k), r.regime, float(r.horizon),
                         r.movement_id, e if ok else PI2))
    return pd.DataFrame(rows, columns=["model", "k", "regime", "horizon", "mid", "err"])


def cluster_ci(sub: pd.DataFrame, B=2000, seed=0):
    """Bootstrap resampling whole trajectories (movement_ids), not cells."""
    mids = sub.mid.unique()
    by = {m: sub.err[sub.mid == m].to_numpy() for m in mids}
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    for b in range(B):
        pick = rng.choice(mids, size=len(mids), replace=True)
        boots[b] = np.concatenate([by[m] for m in pick]).mean()
    return float(sub.err.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def fill_full_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Every model is scored on the SAME 360-cell grid; cells a model never
    produced (e.g. LNN's NaN/diverged k=3) are reliability-adjusted to pi/2."""
    grid = (df[df.model == "rk4"][["mid", "horizon", "k", "regime"]]
            .drop_duplicates())
    assert len(grid) == 360, f"expected 360-cell grid, got {len(grid)}"
    out = []
    for m in df.model.unique():
        sub = df[df.model == m][["mid", "horizon", "err"]].drop_duplicates(["mid", "horizon"])
        merged = grid.merge(sub, on=["mid", "horizon"], how="left")
        merged["err"] = merged["err"].fillna(PI2)
        merged["model"] = m
        out.append(merged)
    return pd.concat(out, ignore_index=True)


def main():
    df = fill_full_grid(load())
    order = df.groupby("model").err.mean().sort_values().index.tolist()

    print("\n=== Overall, cluster-bootstrap CI by trajectory (#5) ===")
    for m in order:
        mn, lo, hi = cluster_ci(df[df.model == m])
        print(f"  {m:26s} {mn:.3f}  95%CI[{lo:.3f},{hi:.3f}]")

    print("\n=== Per-k mean angle error (#11) ===")
    pk = df.pivot_table("err", "model", "k", "mean").reindex(order)
    print(pk.round(3).to_string())

    print("\n=== Per-regime mean angle error (#12) ===")
    pr = df.pivot_table("err", "model", "regime", "mean").reindex(order)
    print(pr.round(3).to_string())

    out = os.path.join(ROOT, "results/summary_llm")
    pk.to_csv(os.path.join(out, "heldout_per_k.csv"))
    pr.to_csv(os.path.join(out, "heldout_per_regime.csv"))
    print(f"\nWrote heldout_per_k.csv + heldout_per_regime.csv to {out}")


if __name__ == "__main__":
    main()
