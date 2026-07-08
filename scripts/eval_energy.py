"""
Axis-1 fair-metric evaluation: score the learned models on physical energy
conservation and rollout stability (the axis HNN/LNN are designed for) rather than
pointwise angle error, on the energy-conserving (normal, undamped) regime.

Hardening vs. reviewer objections:
  * Integrator is not the cause of drift: HNN is additionally rolled out with a
    SYMPLECTIC integrator (symplectic Euler); if its true-energy drift matches the
    RK4 value, the drift comes from the learned Hamiltonian, not the integrator.
  * Robust metric: energy drift is normalized by a FIXED energy scale E* = g*sum(m*L)
    (not by E0, which can be ~0), so it does not blow up.

Uses the committed checkpoints in results/learned_models/ -- no retraining.

    python scripts/eval_energy.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bench.simulator import PendulumParams, integrate, total_energy  # noqa: E402
from bench.models.learned import (  # noqa: E402
    ODEFunc, HamiltonianNet, LagrangianNet,
    _ode_rhs_node, _ode_rhs_hnn, _ode_rhs_lnn, _pad_state, _unpad_state, _state_mask,
)

T_END, DT_ROLL, DT_REC, BOUND = 10.0, 0.02, 0.1, 1e3

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


def e_scale(c):
    return float(c["g"] * np.sum(np.asarray(c["L"]) * np.asarray(c["m"])) + 1e-9)


def load_net(fname, cls):
    import torch
    net = cls(hidden=256, layers=3)
    ck = torch.load(os.path.join(ROOT, "results/learned_models", fname),
                    map_location="cpu", weights_only=True)
    net.load_state_dict(ck); net.eval()
    return net


def drift_norm(state_cols, p, k, escale):
    """Mean |E(t)-E(0)| normalized by a fixed energy scale E* (E0-independent)."""
    E = np.array([total_energy(_unpad_state(state_cols[:, i], k) if state_cols.shape[0] == 6
                               else state_cols[:, i], p)
                  for i in range(state_cols.shape[1])])
    if not np.all(np.isfinite(E)):
        return None
    return float(np.mean(np.abs(E - E[0])) / escale)


def _rhs(kind, net, params9, k):
    mask = _state_mask(k)
    if kind == "node":
        return lambda y: _ode_rhs_node(0.0, y, net, params9, mask)
    if kind == "hnn":
        return lambda y: _ode_rhs_hnn(0.0, y, net, params9, k)
    return lambda y: _ode_rhs_lnn(0.0, y, net, params9, k)


def rollout_rk4(f, s0_6):
    n, rec = int(round(T_END / DT_ROLL)), int(round(DT_REC / DT_ROLL))
    y = s0_6.astype(float).copy(); traj = [y.copy()]
    for i in range(n):
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > BOUND:
            return None
        try:
            k1 = f(y); k2 = f(y + 0.5 * DT_ROLL * k1); k3 = f(y + 0.5 * DT_ROLL * k2); k4 = f(y + DT_ROLL * k3)
        except Exception:
            return None
        y = y + (DT_ROLL / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if (i + 1) % rec == 0:
            traj.append(y.copy())
    arr = np.asarray(traj).T
    return arr if np.all(np.isfinite(arr)) and np.max(np.abs(arr)) <= BOUND else None


def rollout_hnn_symplectic(net, params9, k, s0_6):
    """Symplectic (semi-implicit) Euler for the HNN Hamilton's equations:
    domega/dt = -dH/dtheta, dtheta/dt = dH/domega. Update omega, then theta."""
    mask = _state_mask(k)
    n, rec = int(round(T_END / DT_ROLL)), int(round(DT_REC / DT_ROLL))
    theta = s0_6[:3].astype(float).copy(); omega = s0_6[3:].astype(float).copy()
    traj = [np.concatenate([theta, omega])]
    for i in range(n):
        y = np.concatenate([theta, omega])
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > BOUND:
            return None
        try:
            dy = _ode_rhs_hnn(0.0, y, net, params9, k)      # dy[3:] = -dH/dtheta
            omega = omega + DT_ROLL * dy[3:] * mask[3:]      # omega_{n+1}
            y2 = np.concatenate([theta, omega])
            dy2 = _ode_rhs_hnn(0.0, y2, net, params9, k)     # dy2[:3] = dH/domega at omega_{n+1}
            theta = theta + DT_ROLL * dy2[:3] * mask[:3]     # theta_{n+1}
        except Exception:
            return None
        if (i + 1) % rec == 0:
            traj.append(np.concatenate([theta, omega]))
    arr = np.asarray(traj).T
    return arr if np.all(np.isfinite(arr)) and np.max(np.abs(arr)) <= BOUND else None


def main():
    spec = json.load(open(os.path.join(ROOT, "results/heldout_llm_eval_set.json")))
    cells = [c for c in spec["trajectories"] if c["regime"] == "normal"]
    print(f"Energy test on the normal (undamped) regime: {len(cells)} ICs, {T_END}s rollout, "
          f"drift normalized by E*=g*sum(m*L).\n")
    rows = []

    for name, method in NUMERICAL.items():
        d = []
        for c in cells:
            k = c["k"]; p = PendulumParams.make(k=k, **{q: c["constants"][q] for q in ("L", "m", "g", "damping")})
            s0 = np.array(c["initial_state"]["theta"] + c["initial_state"]["omega"], float)
            _, states = integrate(s0, p, t_end=T_END, dt=0.01, method=method)
            nd = drift_norm(states.T, p, k, e_scale(c["constants"]))
            if nd is not None:
                d.append(nd)
        rows.append([name, "numerical", float(np.median(d)), np.nan, 1.0])

    for name, (fname, cls, kind) in LEARNED.items():
        print(f"  running {name} ...", flush=True)
        net = load_net(fname, cls)
        d_rk4, d_symp, n_bounded = [], [], 0
        for c in cells:
            k = c["k"]; p = PendulumParams.make(k=k, **{q: c["constants"][q] for q in ("L", "m", "g", "damping")})
            s0 = np.array(c["initial_state"]["theta"] + c["initial_state"]["omega"], float)
            s0_6 = _pad_state(s0, k); params9 = encode_params(k, c["constants"]); es = e_scale(c["constants"])
            y = rollout_rk4(_rhs(kind, net, params9, k), s0_6)
            if y is not None:
                n_bounded += 1
                nd = drift_norm(y, p, k, es)
                if nd is not None:
                    d_rk4.append(nd)
            if kind == "hnn":
                ys = rollout_hnn_symplectic(net, params9, k, s0_6)
                if ys is not None:
                    nds = drift_norm(ys, p, k, es)
                    if nds is not None:
                        d_symp.append(nds)
        med = float(np.median(d_rk4)) if d_rk4 else float("nan")
        meds = float(np.median(d_symp)) if d_symp else np.nan
        rows.append([name, kind, med, meds, n_bounded / len(cells)])
        extra = f"  symplectic={meds:.3f}" if kind == "hnn" else ""
        print(f"  done {name:26s} stable={n_bounded}/{len(cells)}  drift(RK4)={med:.3f}{extra}", flush=True)

    print("\n=== Energy drift (E*-normalized) + stability, normal regime, 10s ===")
    print(f"{'model':26s} {'family':10s} {'drift(RK4)':>11s} {'drift(sympl)':>13s} {'stable':>8s}")
    for name, fam, med, meds, stab in rows:
        s = f"{meds:13.3f}" if not np.isnan(meds) else f"{'--':>13s}"
        print(f"{name:26s} {fam:10s} {med:11.3f} {s} {stab:8.0%}")

    import pandas as pd
    pd.DataFrame(rows, columns=["model", "family", "drift_rk4_norm", "drift_symplectic_norm",
                                "stable_frac"]).to_csv(
        os.path.join(ROOT, "results/summary_llm/energy_stability.csv"), index=False)
    print("\nWrote results/summary_llm/energy_stability.csv")


if __name__ == "__main__":
    main()
