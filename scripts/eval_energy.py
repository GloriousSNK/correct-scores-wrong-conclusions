"""
Axis-1 fair-metric evaluation: score the learned models on the axis HNN/LNN are
designed for -- physical energy conservation and rollout stability -- rather than
pointwise angle error.

For each held-out initial condition in the ENERGY-CONSERVING (normal, undamped)
regime, we roll each trained model forward to T=10s and measure:
  * relative energy drift  = mean_t |E(t) - E(0)| / |E(0)|   (0 = perfect conservation)
  * stability              = fraction of rollouts that stay finite/bounded

Numerical integrators (rk4, symplectic/leapfrog, euler) are included as references:
symplectic is the gold standard for conservation, euler is the anti-example.

Uses the committed checkpoints in results/learned_models/ -- no retraining.

    python scripts/eval_energy.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from scipy.integrate import solve_ivp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bench.simulator import PendulumParams, integrate, total_energy  # noqa: E402
from bench.models.learned import (  # noqa: E402
    ODEFunc, HamiltonianNet, LagrangianNet,
    _ode_rhs_node, _ode_rhs_hnn, _ode_rhs_lnn, _pad_state, _unpad_state, _state_mask,
)

T_END, DT = 10.0, 0.1
BOUND = 1e3

LEARNED = {
    "neural-ode":               ("neural_ode.pt",               ODEFunc,        "node"),
    "neural-ode-rollout":       ("neural_ode_rollout.pt",       ODEFunc,        "node"),
    "neural-ode-rollout-mixed": ("neural_ode_rollout_mixed.pt", ODEFunc,        "node"),
    "hnn":                      ("hnn.pt",                      HamiltonianNet, "hnn"),
    "hnn-rollout":              ("hnn_rollout.pt",              HamiltonianNet, "hnn"),
    "hnn-rollout-mixed":        ("hnn_rollout_mixed.pt",        HamiltonianNet, "hnn"),
    "lnn":                      ("lnn.pt",                      LagrangianNet,  "lnn"),
}
NUMERICAL = {"rk4": "rk4", "symplectic": "leapfrog", "euler": "euler"}


def encode_params(k, c):
    L = np.zeros(3); L[:k] = c["L"]
    m = np.zeros(3); m[:k] = c["m"]
    return np.array([k, c["g"], *L, *m, c.get("damping", 0.0)], dtype=np.float32)


def load_net(fname, cls):
    import torch
    net = cls(hidden=256, layers=3)
    ck = torch.load(os.path.join(ROOT, "results/learned_models", fname),
                    map_location="cpu", weights_only=True)
    net.load_state_dict(ck); net.eval()
    return net


def energy_drift(state_cols, p, k):
    """state_cols: (2k or 6, n_t) padded states -> relative energy drift."""
    E = np.array([total_energy(_unpad_state(state_cols[:, i], k) if state_cols.shape[0] == 6
                               else state_cols[:, i], p)
                  for i in range(state_cols.shape[1])])
    if not np.all(np.isfinite(E)):
        return None
    E0 = E[0]
    return float(np.mean(np.abs(E - E0)) / (abs(E0) + 1e-9))


def rollout_learned(net, kind, s0_6, params9, k):
    """Fixed-step RK4 rollout of the learned vector field (bounded cost)."""
    mask = _state_mask(k)
    if kind == "node":
        f = lambda y: _ode_rhs_node(0.0, y, net, params9, mask)
    elif kind == "hnn":
        f = lambda y: _ode_rhs_hnn(0.0, y, net, params9, k)
    else:
        f = lambda y: _ode_rhs_lnn(0.0, y, net, params9, k)
    dt = 0.02
    n = int(round(T_END / dt))
    rec = int(round(DT / dt))
    y = s0_6.astype(float).copy()
    traj = [y.copy()]
    for i in range(n):
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > BOUND:
            return None
        try:
            k1 = f(y); k2 = f(y + 0.5 * dt * k1); k3 = f(y + 0.5 * dt * k2); k4 = f(y + dt * k3)
        except Exception:
            return None
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if (i + 1) % rec == 0:
            traj.append(y.copy())
    arr = np.asarray(traj).T
    if not np.all(np.isfinite(arr)) or np.max(np.abs(arr)) > BOUND:
        return None
    return arr


def main():
    spec = json.load(open(os.path.join(ROOT, "results/heldout_llm_eval_set.json")))
    cells = [c for c in spec["trajectories"] if c["regime"] == "normal"]
    print(f"Energy-conservation test on the normal (undamped) regime: {len(cells)} ICs, "
          f"rollout to {T_END}s.\n")

    rows = []
    # numerical references
    for name, method in NUMERICAL.items():
        drifts = []
        for c in cells:
            k = c["k"]; p = PendulumParams.make(k=k, **{kk: c["constants"][kk] for kk in ("L", "m", "g", "damping")})
            s0 = np.array(c["initial_state"]["theta"] + c["initial_state"]["omega"], float)
            _, states = integrate(s0, p, t_end=T_END, dt=0.01, method=method)
            d = energy_drift(states.T, p, k)
            if d is not None:
                drifts.append(d)
        rows.append((name, "numerical", np.median(drifts), len(drifts) / len(cells)))

    # learned models
    for name, (fname, cls, kind) in LEARNED.items():
        print(f"  running {name} ...", flush=True)
        net = load_net(fname, cls)
        drifts, n_bounded = [], 0
        for c in cells:
            k = c["k"]; p = PendulumParams.make(k=k, **{kk: c["constants"][kk] for kk in ("L", "m", "g", "damping")})
            s0 = np.array(c["initial_state"]["theta"] + c["initial_state"]["omega"], float)
            s0_6 = _pad_state(s0, k)
            params9 = encode_params(k, c["constants"])
            y = rollout_learned(net, kind, s0_6, params9, k)
            if y is None:
                continue
            n_bounded += 1
            d = energy_drift(y, p, k)
            if d is not None:
                drifts.append(d)
        med = np.median(drifts) if drifts else float("nan")
        rows.append((name, kind, med, n_bounded / len(cells)))
        print(f"  done {name:26s} stable={n_bounded}/{len(cells)}  med rel-drift={med:.3f}")

    print("\n=== Energy conservation + stability (normal regime, 10s rollout) ===")
    print(f"{'model':26s} {'family':10s} {'rel-energy-drift':>16s} {'stable':>8s}")
    for name, fam, med, stab in rows:
        print(f"{name:26s} {fam:10s} {med:16.3f} {stab:8.0%}")

    import pandas as pd
    pd.DataFrame(rows, columns=["model", "family", "rel_energy_drift", "stable_frac"]).to_csv(
        os.path.join(ROOT, "results/summary_llm/energy_stability.csv"), index=False)
    print("\nWrote results/summary_llm/energy_stability.csv")


if __name__ == "__main__":
    main()
