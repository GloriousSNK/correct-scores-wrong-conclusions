"""
Trivial / floor baselines on the shared out-of-sample held-out set, scored with the
same reliability-adjusted mean-angle-error metric as the cross-family leaderboard.

Baselines
  persistence      : theta_T = theta_0            (state frozen)
  constant-velocity: theta_T = theta_0 + omega_0 * T
  linearized       : integrate the small-angle linearization about the downward
                     equilibrium (numerical Jacobian of the true dynamics; matrix
                     exponential). Uses the cell's true constants.
  nearest-neighbor : analog forecast -- find the closest state in the matching
                     (k, regime) training trajectory and return where it went after T.
                     This is a memorization floor built from exactly the data the
                     learned models were trained on.

Scoring: error = mean_links |wrap(theta_pred - theta_true)|, wrapped to [-pi,pi];
unanswered/NaN cells are scored at pi/2. Leaderboard horizons are {1,10}s.

    python scripts/eval_baselines.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from scipy.linalg import expm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bench.simulator import PendulumParams, dynamics  # noqa: E402

PI2 = np.pi / 2
DT = 0.01
LEADER_HORIZONS = [1.0, 10.0]


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def angle_err(pred_theta, true_theta):
    return float(np.mean(np.abs(wrap(np.asarray(pred_theta) - np.asarray(true_theta)))))


def params_of(c, k):
    return PendulumParams.make(k=k, L=c["L"], m=c["m"], g=c["g"], damping=c.get("damping", 0.0))


def linear_system(p):
    """2k x 2k matrix of the dynamics linearized about theta=0, omega=0."""
    n = 2 * p.k
    x0 = np.zeros(n)
    J = np.zeros((n, n))
    eps = 1e-6
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        J[:, i] = (dynamics(x0 + dx, p) - dynamics(x0 - dx, p)) / (2 * eps)
    return J


# --- nearest-neighbor training database: one long trajectory per (k, regime) ---
_NN_CACHE = {}
def nn_db(k, regime):
    key = (k, regime)
    if key in _NN_CACHE:
        return _NN_CACHE[key]
    path = os.path.join(ROOT, f"results/dataset/k{k}_{regime}_0000.json")
    if not os.path.exists(path):
        _NN_CACHE[key] = None; return None
    d = json.load(open(path))
    th = np.asarray(d["theta"], dtype=float)   # (T, k)
    om = np.asarray(d["omega"], dtype=float)    # (T, k)
    state = np.concatenate([th, om], axis=1)    # (T, 2k)
    _NN_CACHE[key] = (state, th)
    return _NN_CACHE[key]


def predict(name, theta0, omega0, T, p, k, regime):
    """Return predicted theta at horizon T, or None to abstain."""
    if name == "persistence":
        return theta0
    if name == "constant-velocity":
        return theta0 + omega0 * T
    if name == "linearized":
        J = linear_system(p)
        try:
            pred = expm(J * T) @ np.concatenate([theta0, omega0])
        except Exception:
            return None
        if not np.all(np.isfinite(pred)):
            return None
        return pred[:k]
    if name == "nearest-neighbor":
        db = nn_db(k, regime)
        if db is None:
            return None
        state, th = db
        q = np.concatenate([theta0, omega0])
        # angle-aware distance: wrap the theta part of the difference
        diff = state - q
        diff[:, :k] = wrap(diff[:, :k])
        step = int(round(T / DT))
        n = state.shape[0]
        valid = n - step
        if valid <= 0:
            return None
        d = np.sum(diff[:valid] ** 2, axis=1)
        j = int(np.argmin(d))
        return th[j + step]
    raise ValueError(name)


def main():
    spec = json.load(open(os.path.join(ROOT, "results/heldout_llm_eval_set.json")))
    cells = spec["trajectories"]
    names = ["persistence", "constant-velocity", "linearized", "nearest-neighbor"]
    horizons = [0.01, 1.0, 10.0, 60.0]

    rows = []
    for c in cells:
        k = c["k"]; regime = c["regime"]
        p = params_of(c["constants"], k)
        theta0 = np.asarray(c["initial_state"]["theta"], float)
        omega0 = np.asarray(c["initial_state"]["omega"], float)
        for T in horizons:
            true_theta = c["targets"][str(T)]["theta"]
            for name in names:
                pred = predict(name, theta0, omega0, T, p, k, regime)
                err = PI2 if pred is None else angle_err(pred, true_theta)
                rows.append((name, k, regime, T, err))

    import pandas as pd
    df = pd.DataFrame(rows, columns=["model", "k", "regime", "horizon", "err"])
    lead = df[df.horizon.isin(LEADER_HORIZONS)]

    print("\n=== Baselines: reliability-adjusted mean angle error (rad) ===")
    print("Leaderboard slice = horizons {1,10}s (360 cells), same as Table (leaderboard).\n")
    overall = lead.groupby("model").err.mean().sort_values()
    for m, v in overall.items():
        print(f"  {m:20s} {v:.3f}")
    print("\n  (leaderboard anchors: rk4 0.059, kimi 0.538, best-learned 0.844, random pi/2 = 1.571)")

    print("\n=== By horizon ===")
    print(df.pivot_table("err", "model", "horizon", "mean").round(3).to_string())
    print("\n=== By k (horizons {1,10}) ===")
    print(lead.pivot_table("err", "model", "k", "mean").round(3).to_string())

    out = os.path.join(ROOT, "results/summary_llm/baselines.csv")
    lead.groupby("model").err.mean().round(4).to_csv(out, header=["err_h1_h10"])
    df.to_csv(os.path.join(ROOT, "results/summary_llm/baselines_long.csv"), index=False)
    print(f"\nWrote {out} and baselines_long.csv")


if __name__ == "__main__":
    main()
