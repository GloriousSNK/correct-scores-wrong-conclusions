"""
Export a compact, machine-portable held-out evaluation set for the LLM side.

The local learned models are TRAINED on results/dataset (seed 42). To compare any
model fairly they must be tested on trajectories disjoint from that training set.
This exports the boot dataset (results/dataset_boot, seed 20260620) as a single small
JSON containing, per trajectory: the initial (t=0) state the model is given, the
ground-truth target state at each eval horizon, and the physical constants. The Mac
LLM run should evaluate on THIS set so both machines test out-of-sample on identical
trajectories.

Usage:
    python scripts/export_heldout.py --dataset-dir results/dataset_boot \
        --out results/heldout_llm_eval_set.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HORIZONS = [0.01, 1.0, 10.0, 60.0]


def nearest(times: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(times - t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="results/dataset_boot")
    ap.add_argument("--out", default="results/heldout_llm_eval_set.json")
    ap.add_argument("--horizons", type=float, nargs="*", default=HORIZONS)
    args = ap.parse_args()

    with open(os.path.join(args.dataset_dir, "manifest.json")) as f:
        manifest = json.load(f)

    regimes = {}
    trajs = []
    for entry in manifest:
        path = entry["file"]
        if not os.path.exists(path):
            path = os.path.join(args.dataset_dir, f"{entry['movement_id']}.json")
        with open(path) as f:
            d = json.load(f)
        k = int(d["number_of_pendulums"])
        times = np.array(d["time"])
        theta = np.array(d["theta"])
        omega = np.array(d["omega"])
        c = d["constants"]
        regimes.setdefault(entry["regime"], c)

        i0 = nearest(times, 0.0)
        targets = {}
        for h in args.horizons:
            ih = nearest(times, h)
            targets[str(h)] = {"theta": theta[ih].tolist(), "omega": omega[ih].tolist()}

        trajs.append({
            "movement_id": entry["movement_id"], "k": k, "regime": entry["regime"],
            "constants": {"g": c["g"], "L": c["L"], "m": c["m"], "damping": c["damping"]},
            "initial_state": {"theta": theta[i0].tolist(), "omega": omega[i0].tolist()},
            "targets": targets,
        })

    out = {
        "description": "Held-out k-pendulum test set (disjoint from learned-model "
                       "training, seed 42). Evaluate LLMs here so all models are "
                       "tested out-of-sample on identical trajectories.",
        "seed": 20260620,
        "horizons": args.horizons,
        "metric": "mean over links of |wrap(theta_pred - theta_true)|, wrapped to [-pi,pi]; "
                  "reliability-adjusted: failed/unparseable cells scored at pi/2.",
        "regime_constants": regimes,
        "n_trajectories": len(trajs),
        "trajectories": trajs,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"Wrote {args.out}: {len(trajs)} trajectories, {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
