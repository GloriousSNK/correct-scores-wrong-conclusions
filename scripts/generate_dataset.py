from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.simulator import (PendulumParams, integrate, bob_positions_batch,
                             energy_batch)
from bench.schema import Trajectory
from bench.export import write_trajectory_json, write_trajectory_csv


def _resolve_regime(regime_cfg: dict, k: int) -> dict:
    L = regime_cfg["L"]
    m = regime_cfg["m"]
    if len(L) < k:
        L = list(L) + [L[-1]] * (k - len(L))
    if len(m) < k:
        m = list(m) + [m[-1]] * (k - len(m))
    return {"g": float(regime_cfg["g"]),
            "L": L[:k], "m": m[:k],
            "damping": float(regime_cfg["damping"])}


def _generate_one(k: int, regime_name: str, regime_cfg: dict, idx: int,
                  rng: np.random.Generator, ic_cfg: dict,
                  ground_truth_dt: float, total_seconds: float,
                  pre_context_seconds: float) -> Trajectory:
    consts = _resolve_regime(regime_cfg, k)
    p = PendulumParams.make(k=k, L=consts["L"], m=consts["m"],
                            g=consts["g"], damping=consts["damping"])

    theta_warmup = rng.uniform(ic_cfg["theta_low"], ic_cfg["theta_high"], size=k)
    omega_warmup = rng.uniform(ic_cfg["omega_low"], ic_cfg["omega_high"], size=k)
    state_warmup = np.concatenate([theta_warmup, omega_warmup])

    record_every = max(1, int(round(0.01 / ground_truth_dt)))
    t_start = -float(pre_context_seconds)
    times, states = integrate(
        state_warmup, p, t_end=total_seconds, dt=ground_truth_dt,
        method="rk4", t_start=t_start, record_every=record_every,
    )

    idx_t0 = int(np.argmin(np.abs(times - 0.0)))
    state_at_zero = states[idx_t0]
    theta0 = state_at_zero[:k].tolist()
    omega0 = state_at_zero[k:].tolist()

    xy = bob_positions_batch(states[:, :k], p)
    energies = energy_batch(states, p)

    return Trajectory(
        movement_id=f"k{k}_{regime_name}_{idx:04d}",
        k=k, regime=regime_name,
        times=times, states=states, xy=xy,
        ke=energies["KE"], pe=energies["PE"], e_total=energies["E"],
        constants={"g": p.g, "L": p.L.tolist(), "m": p.m.tolist(),
                   "damping": p.damping},
        initial_conditions={"theta0": theta0, "omega0": omega0,
                            "pre_context_seconds": float(pre_context_seconds)},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=None,
                    help="override dataset.seed; useful for disjoint held-out sets")
    ap.add_argument("--output-dir", default=None,
                    help="override paths.dataset_dir without modifying the config")
    ap.add_argument("--id-prefix", default="",
                    help="prefix trajectory IDs to prevent collisions across datasets")
    ap.add_argument("--systems", type=int, nargs="*", default=None,
                    help="override dataset.systems")
    ap.add_argument("--regimes", nargs="*", default=None,
                    help="restrict generation to named regimes")
    ap.add_argument("--trajectories-per-cell", type=int, default=None,
                    help="override dataset.trajectories_per_cell")
    ap.add_argument("--resume", action="store_true",
                    help="reuse valid trajectory JSON files while rebuilding the manifest")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_dir = args.output_dir or cfg["paths"]["dataset_dir"]
    os.makedirs(dataset_dir, exist_ok=True)

    ds = cfg["dataset"]
    regimes = cfg["regimes"]
    systems = args.systems if args.systems is not None else ds["systems"]
    if args.regimes is not None:
        unknown = sorted(set(args.regimes) - set(regimes))
        if unknown:
            raise ValueError(f"Unknown regimes: {', '.join(unknown)}")
        regimes = {name: regimes[name] for name in args.regimes}
    ic_cfg = cfg["initial_conditions"]
    n_per = 1 if args.smoke else int(args.trajectories_per_cell or ds["trajectories_per_cell"])
    if n_per < 1:
        raise ValueError("--trajectories-per-cell must be positive")
    seed = int(ds["seed"] if args.seed is None else args.seed)
    rng = np.random.default_rng(seed)
    pre_context_seconds = float(ds.get("pre_context_seconds", 5.0))

    manifest = []
    total = len(systems) * len(regimes) * n_per
    pbar = tqdm(total=total, desc="generating")
    for k in systems:
        for regime_name, regime_cfg in regimes.items():
            for idx in range(n_per):
                base_id = f"k{k}_{regime_name}_{idx:04d}"
                movement_id = f"{args.id_prefix}{base_id}" if args.id_prefix else base_id
                json_path = os.path.join(dataset_dir, f"{movement_id}.json")
                if args.resume and os.path.exists(json_path):
                    # Preserve the exact RNG stream used by _generate_one.
                    rng.uniform(ic_cfg["theta_low"], ic_cfg["theta_high"], size=k)
                    rng.uniform(ic_cfg["omega_low"], ic_cfg["omega_high"], size=k)
                    try:
                        with open(json_path, "r", encoding="utf-8") as handle:
                            existing = json.load(handle)
                        if (existing.get("movement_id") != movement_id
                                or int(existing.get("number_of_pendulums", -1)) != k
                                or existing.get("regime") != regime_name):
                            raise ValueError("trajectory metadata mismatch")
                    except Exception as error:
                        raise RuntimeError(f"Invalid resumable trajectory {json_path}: {error}") from error
                    manifest.append({
                        "movement_id": movement_id, "k": k,
                        "regime": regime_name, "file": json_path, "seed": seed,
                    })
                    pbar.update(1)
                    continue
                traj = _generate_one(
                    k=k, regime_name=regime_name, regime_cfg=regime_cfg,
                    idx=idx, rng=rng, ic_cfg=ic_cfg,
                    ground_truth_dt=float(ds["ground_truth_dt"]),
                    total_seconds=float(ds["total_seconds"]),
                    pre_context_seconds=pre_context_seconds,
                )
                if args.id_prefix:
                    traj.movement_id = f"{args.id_prefix}{traj.movement_id}"
                json_path = os.path.join(dataset_dir, f"{traj.movement_id}.json")
                write_trajectory_json(traj, json_path)
                if args.csv:
                    csv_path = os.path.join(dataset_dir, f"{traj.movement_id}.csv")
                    write_trajectory_csv(traj, csv_path)
                manifest.append({
                    "movement_id": traj.movement_id, "k": k,
                    "regime": regime_name, "file": json_path, "seed": seed,
                })
                pbar.update(1)
    pbar.close()

    with open(os.path.join(dataset_dir, "manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(manifest)} trajectories (seed {seed}) to {dataset_dir}/")


if __name__ == "__main__":
    main()
