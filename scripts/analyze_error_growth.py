"""
Error-growth curves: reliability-adjusted mean angle error (rad) per (model, horizon)
on the shared out-of-sample held-out set. Non-LLM models span the full ladder
{0.01,1,10,60}s; LLMs were evaluated at {1,10}s only. Reviewer item #19.

Missing/failed cells scored at pi/2; each (model,horizon) cell is a mean over the full
180-trajectory grid for that horizon. Emits both a table and pgfplots coordinate lines.

    python scripts/analyze_error_growth.py
"""
from __future__ import annotations
import glob, json, os
import numpy as np, pandas as pd

PI2 = np.pi / 2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load() -> pd.DataFrame:
    rows = []
    for p in glob.glob(os.path.join(ROOT, "results/checkpoints_boot/*.json")):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        c = d.get("cell", {})
        if c.get("modality") not in (None, "coords"): continue
        if c.get("prompting") not in (None, "no_cot"): continue
        e = (d.get("metrics") or {}).get("angle_error_mean")
        ok = bool(d.get("success")) and e is not None and np.isfinite(e)
        rows.append((c.get("model_name"), float(c["horizon"]), c["movement_id"],
                     e if ok else PI2))
    csv = os.path.join(ROOT, "results/summary_llm/llm_main_results_long.csv")
    if os.path.exists(csv):
        L = pd.read_csv(csv)
        L = L[L.prompting == "no_cot"]
        for _, r in L.iterrows():
            e = pd.to_numeric(r.angle_error_mean, errors="coerce")
            ok = bool(r.success) and np.isfinite(e)
            rows.append((r.model, float(r.horizon), r.movement_id, e if ok else PI2))
    return pd.DataFrame(rows, columns=["model", "horizon", "mid", "err"])


def main():
    df = load()
    # full grid per horizon = the set of movement_ids rk4 produced at that horizon (180)
    rk4 = df[df.model == "rk4"]
    grid = {h: set(rk4.mid[rk4.horizon == h]) for h in sorted(rk4.horizon.unique())}
    recs = {}
    for m in df.model.unique():
        sub = df[df.model == m]
        for h, mids in grid.items():
            present = sub[sub.horizon == h].drop_duplicates("mid").set_index("mid").err
            if len(present) == 0 and m in LLMS and h not in (1.0, 10.0):
                recs[(m, h)] = np.nan  # LLMs not evaluated off {1,10}
                continue
            vals = [present.get(mid, PI2) for mid in mids]
            recs[(m, h)] = float(np.mean(vals))
    tab = pd.Series(recs).unstack()
    order = tab[[1.0, 10.0]].mean(axis=1).sort_values().index
    tab = tab.reindex(order)
    pd.set_option("display.width", 120)
    print(tab.round(3).to_string())
    tab.to_csv(os.path.join(ROOT, "results/summary_llm/heldout_error_growth.csv"))

    print("\n=== pgfplots coordinates (skip NaN) ===")
    for m in order:
        pts = " ".join(f"({h:g},{tab.loc[m,h]:.3f})" for h in tab.columns
                       if pd.notna(tab.loc[m, h]))
        print(f"% {m}\n\\addplot coordinates {{{pts}}};")


LLMS = {"kimi-k2.6", "grok-4-1-fast-reasoning", "deepseek-v4-pro"}

if __name__ == "__main__":
    main()
