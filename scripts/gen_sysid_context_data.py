"""Generate matched disclosure trajectories with an observed pre-forecast window."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.simulator import PendulumParams, integrate
from scripts.gen_sysid_data import sample_constant_sets


SEED = 770025
HORIZONS = [1.0, 10.0]
KS = [1, 2, 3]


def state_at(times: np.ndarray, states: np.ndarray, time_value: float) -> np.ndarray:
    return states[int(np.argmin(np.abs(times - time_value)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/dataset_llm_sysid_context")
    parser.add_argument("--trajectories-per-set-k", type=int, default=5)
    parser.add_argument("--context-seconds", type=float, default=1.0)
    parser.add_argument("--context-step", type=float, default=0.1)
    args = parser.parse_args()

    output = Path(args.out)
    target = output / "sysid_context_eval_set.json"
    if target.exists():
        raise SystemExit(f"REFUSING: {target} already exists")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    constant_sets = sample_constant_sets(rng)
    trajectories = []
    observation_times = np.arange(-args.context_seconds, 1e-12, args.context_step)

    for constants in constant_sets:
        for k in KS:
            params = PendulumParams.make(
                k=k, L=constants["L"][:k], m=constants["m"][:k],
                g=constants["g"], damping=constants["damping"],
            )
            for index in range(args.trajectories_per_set_k):
                state_start = np.concatenate([
                    rng.uniform(-3.0, 3.0, size=k), rng.uniform(-1.0, 1.0, size=k),
                ])
                times, states = integrate(
                    state_start, params, t_start=-args.context_seconds,
                    t_end=max(HORIZONS), dt=1e-3, method="rk4", record_every=10,
                )
                history = []
                for observed_time in observation_times:
                    state = state_at(times, states, float(observed_time))
                    history.append({
                        "t": round(float(observed_time), 6),
                        "theta": state[:k].tolist(), "omega": state[k:].tolist(),
                    })
                state0 = state_at(times, states, 0.0)
                targets = {}
                for horizon in HORIZONS:
                    state = state_at(times, states, horizon)
                    targets[str(horizon)] = {
                        "theta": state[:k].tolist(), "omega": state[k:].tolist(),
                    }
                trajectories.append({
                    "movement_id": f"context_{constants['set_id']}_k{k}_{index:04d}",
                    "set_id": constants["set_id"], "k": k,
                    "constants": {"g": constants["g"], "L": constants["L"][:k],
                                  "m": constants["m"][:k],
                                  "damping": constants["damping"]},
                    "history": history,
                    "initial_state": {"theta": state0[:k].tolist(),
                                      "omega": state0[k:].tolist()},
                    "targets": targets,
                })

    payload = {
        "seed": SEED, "horizons": HORIZONS, "ks": KS,
        "context_seconds": args.context_seconds, "context_step": args.context_step,
        "trajectories_per_set_k": args.trajectories_per_set_k,
        "constant_sets": constant_sets, "n_trajectories": len(trajectories),
        "ground_truth": "RK4 dt=1e-3; observations every 0.1 s",
        "parameter_scoring": "log residual after fitting mass and length symmetry scales",
        "trajectories": trajectories,
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(trajectories)} trajectories to {target}")


if __name__ == "__main__":
    main()
