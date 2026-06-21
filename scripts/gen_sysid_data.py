"""Part B data generation: controlled system-identification trajectories.

K=5 non-standard constant sets (sampled away from Earth-standard to limit
memorised priors). For each set x k in {1,2,3}: 10 trajectories, ground truth
RK4 @ dt=1e-3, targets recorded at t in {1,10}s. Same trajectories are later
evaluated under DISCLOSED and HIDDEN conditions (matched -> confound-free).

B3 difficulty control: per cell we also record a coarse-RK4 (dt=0.01) error vs
the dt=1e-3 ground truth, a proxy for the intrinsic predictability of that set.

Additive: writes ONLY to results/dataset_llm_sysid/. Reports its seed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.simulator import PendulumParams, integrate

SEED = 770021
HORIZONS = [1.0, 10.0]
N_TRAJ = 10
KS = [1, 2, 3]
N_SETS = 5
GT_DT = 1e-3
COARSE_DT = 0.01


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def sample_constant_sets(rng):
    sets = []
    for s in range(N_SETS):
        g = float(rng.uniform(1.6, 12.0))
        L = rng.uniform(0.5, 1.5, size=3).tolist()
        m = rng.uniform(0.5, 2.0, size=3).tolist()
        damping = float(rng.uniform(0.0, 0.1))
        sets.append({"set_id": f"set{s}", "g": round(g, 4),
                     "L": [round(x, 4) for x in L],
                     "m": [round(x, 4) for x in m],
                     "damping": round(damping, 4)})
    return sets


def state_at(times, states, t):
    idx = int(np.argmin(np.abs(times - t)))
    return states[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/dataset_llm_sysid")
    args = ap.parse_args()
    if os.path.exists(args.out) and os.listdir(args.out):
        sys.exit(f"REFUSING: {args.out} exists and is non-empty (additive only).")
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(SEED)
    sets = sample_constant_sets(rng)
    t_end = max(HORIZONS)

    trajectories = []
    for cs in sets:
        for k in KS:
            L = np.array(cs["L"][:k]); m = np.array(cs["m"][:k])
            p = PendulumParams.make(k=k, L=L, m=m, g=cs["g"], damping=cs["damping"])
            p_coarse = p
            for j in range(N_TRAJ):
                theta0 = rng.uniform(-3.0, 3.0, size=k)
                omega0 = rng.uniform(-1.0, 1.0, size=k)
                s0 = np.concatenate([theta0, omega0])
                times, states = integrate(s0, p, t_end=t_end, dt=GT_DT,
                                          method="rk4", t_start=0.0,
                                          record_every=max(1, int(round(0.01 / GT_DT))))
                tc, sc = integrate(s0, p_coarse, t_end=t_end, dt=COARSE_DT,
                                   method="rk4", t_start=0.0, record_every=1)
                targets = {}
                difficulty = {}
                for h in HORIZONS:
                    gt = state_at(times, states, h)
                    coarse = state_at(tc, sc, h)
                    targets[str(h)] = {"theta": gt[:k].tolist(),
                                       "omega": gt[k:].tolist()}
                    difficulty[str(h)] = float(
                        np.abs(wrap(coarse[:k] - gt[:k])).mean())
                trajectories.append({
                    "movement_id": f"{cs['set_id']}_k{k}_{j:04d}",
                    "set_id": cs["set_id"], "k": k,
                    "constants": {"g": cs["g"], "L": cs["L"][:k],
                                  "m": cs["m"][:k], "damping": cs["damping"]},
                    "initial_state": {"theta": theta0.tolist(),
                                      "omega": omega0.tolist()},
                    "targets": targets,
                    "rk4_coarse_difficulty": difficulty,
                })

    out = {"seed": SEED, "horizons": HORIZONS, "n_traj_per_set_k": N_TRAJ,
           "ks": KS, "ground_truth": "RK4 dt=1e-3",
           "difficulty_baseline": "RK4 dt=0.01 angle error vs ground truth",
           "constant_sets": sets, "n_trajectories": len(trajectories),
           "trajectories": trajectories}
    path = os.path.join(args.out, "sysid_eval_set.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"Wrote {len(trajectories)} trajectories ({N_SETS} sets x {len(KS)} k "
          f"x {N_TRAJ}) to {path} (seed={SEED})")
    for cs in sets:
        print(f"  {cs['set_id']}: g={cs['g']} L={cs['L']} m={cs['m']} "
              f"damping={cs['damping']}")


if __name__ == "__main__":
    main()
