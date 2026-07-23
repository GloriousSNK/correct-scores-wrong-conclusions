"""
Analyses of the released-but-previously-unreported evaluation axes, from the
saved per-cell records only (no API calls):

  1. Chain-of-thought versus direct prompting on the main held-out grid
     (results/llm_main, 2,160 records): paired per (movement, horizon),
     trajectory-level bootstrap. Non-finite predictions marked success are
     scored as failures at pi/2, matching the harness rule.
  2. Input modality on the one-trajectory modality grid (results/checkpoints):
     coordinates versus rendered image versus both, paired per
     (model, k, regime, horizon, prompting).
  3. Temperature-0 repeatability: a second independent run of one model's
     216-cell grid (results/checkpoints_bon/run2) against the original
     (results/checkpoints), paired per cell.

    python scripts/analyze_prompting_modality_repeat.py
"""
from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PI2 = np.pi / 2
LLMS = ["kimi-k2.6", "grok-4-1-fast-reasoning", "deepseek-v4-pro"]
N_BOOT = 10_000


def valid_err(success, e):
    ok = success and e is not None and math.isfinite(e)
    return (e if ok else PI2), ok


def boot_ci(values, rng, n=N_BOOT):
    means = [np.mean(rng.choice(values, len(values))) for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def cot_analysis():
    print("=== 1. Chain-of-thought vs direct (main grid, 2,160 records) ===")
    rows, answered = defaultdict(dict), defaultdict(lambda: [0, 0])
    for f in glob.glob(os.path.join(ROOT, "results", "llm_main", "*.json")):
        d = json.load(open(f))
        err, ok = valid_err(d["success"], d["angle_error_mean"])
        rows[(d["model"], d["movement_id"], d["horizon"])][d["prompting"]] = err
        answered[(d["model"], d["prompting"])][0] += int(ok)
        answered[(d["model"], d["prompting"])][1] += 1
    rng = np.random.default_rng(0)
    for model in LLMS:
        by_mov = defaultdict(list)
        direct, cot = [], []
        for (m, mov, _h), p in rows.items():
            if m != model or len(p) < 2:
                continue
            by_mov[mov].append(p["cot"] - p["no_cot"])
            direct.append(p["no_cot"])
            cot.append(p["cot"])
        movs = [np.mean(v) for v in by_mov.values()]
        lo, hi = boot_ci(movs, rng)
        ad, ac = answered[(model, "no_cot")], answered[(model, "cot")]
        print(f"  {model:26s} direct {np.mean(direct):.3f}  cot {np.mean(cot):.3f}  "
              f"diff {np.mean(movs):+.3f} [{lo:+.3f},{hi:+.3f}]  "
              f"answered {ad[0]}/{ad[1]} vs {ac[0]}/{ac[1]}")


def cell_err(path):
    r = json.load(open(path))
    m = r.get("metrics") or {}
    return valid_err(r.get("success"), m.get("angle_error_mean"))[0]


def modality_analysis():
    print("=== 2. Modality (one-trajectory grid, coords vs images vs both) ===")
    rows = defaultdict(dict)
    for f in glob.glob(os.path.join(ROOT, "results", "checkpoints", "*.json")):
        r = json.load(open(f))
        c = r["cell"]
        if c["model_name"] not in LLMS:
            continue
        m = r.get("metrics") or {}
        err = valid_err(r.get("success"), m.get("angle_error_mean"))[0]
        key = (c["model_name"], c["k"], c["regime"], c["horizon"], c["prompting"])
        rows[key][c["modality"]] = err
    rng = np.random.default_rng(0)
    for model in LLMS:
        co, di, dic = [], [], []
        for k, p in rows.items():
            if k[0] != model or not all(x in p for x in
                                        ("coords", "images", "images_coords")):
                continue
            co.append(p["coords"])
            di.append(p["images"] - p["coords"])
            dic.append(p["images_coords"] - p["coords"])
        li, hi_ = boot_ci(di, rng)
        lic, hic = boot_ci(dic, rng)
        print(f"  {model:26s} n={len(co)}  coords {np.mean(co):.3f}  "
              f"images-coords {np.mean(di):+.3f} [{li:+.3f},{hi_:+.3f}]  "
              f"both-coords {np.mean(dic):+.3f} [{lic:+.3f},{hic:+.3f}]")


def repeat_analysis():
    print("=== 3. Temperature-0 repeatability (kimi 216-cell grid, run 1 vs 2) ===")
    e1, e2 = [], []
    run2 = os.path.join(ROOT, "results", "checkpoints_bon", "run2")
    for f2 in glob.glob(os.path.join(run2, "*.json")):
        f1 = os.path.join(ROOT, "results", "checkpoints", os.path.basename(f2))
        if os.path.exists(f1):
            e1.append(cell_err(f1))
            e2.append(cell_err(f2))
    e1, e2 = np.array(e1), np.array(e2)
    d = e2 - e1
    rng = np.random.default_rng(0)
    lo, hi = boot_ci(d, rng)
    same = int((np.abs(d) < 1e-9).sum())
    print(f"  n={len(d)} paired cells; identical {same}/{len(d)}; "
          f"mean |diff| {np.abs(d).mean():.3f}; Pearson r "
          f"{np.corrcoef(e1, e2)[0, 1]:.3f}")
    print(f"  aggregate run2-run1 {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")


if __name__ == "__main__":
    cot_analysis()
    modality_analysis()
    repeat_analysis()
