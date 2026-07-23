"""
Real-data check: evaluate the audit's local methods on measured double-pendulum
trajectories from Myers et al.'s open hardware dataset (Mendeley Data,
doi:10.17632/z4hvxjgtbz.2; video tracking at 500 Hz, three free-drop trials).

Here the roles invert relative to the synthetic benchmark: the point-mass RK4
model with the published constants is no longer ground truth but an IMPERFECT
scientific model of the hardware (point-mass simplification, unmodeled bearing
friction, tracked-angle noise), scored against real observations.

Design
  Trial 1  : calibration only (nearest-neighbor database, SINDy fit, damping fit).
  Trials 2-3: held-out. 60 evaluation states per trial, evenly spaced over the
              active window; horizons {0.1, 0.5, 1.0} s; wrapped mean angle
              error over both links; failures scored at pi/2.

The tracker zeroes its angle at the first frame, so the gravity-referenced
offset is unknown a priori. Every trial ends hanging at rest with tracked
angles at exact multiples of 360 degrees, which fixes the offset at zero
independently of any model; the 20 ms physics-tracking selection below agrees
and serves as a cross-check.

    python scripts/eval_realdata_myers.py [--skip-chronos]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.signal import savgol_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bench.simulator import PendulumParams, integrate  # noqa: E402

DATA = os.path.join(ROOT, "results", "realdata_myers", "Video_Tracking_Data")
# Published simplified point-mass parameters (Myers et al., Table 4).
L = [0.172, 0.143]
M = [0.311, 0.111]
G = 9.81
DT_DATA = 0.002          # 500 Hz tracking
HORIZONS = [0.1, 0.5, 1.0]
STATES_PER_TRIAL = 60
ACTIVE_START = 0.2       # seconds after the detected drop
ACTIVE_END = 24.0
SG_WINDOW = 21           # 42 ms Savitzky-Golay window for velocities
PI2 = np.pi / 2


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def load_trial(name):
    t, p1 = np.load(os.path.join(DATA, name, "DPmean_data_RB0.npy"))
    _, p2 = np.load(os.path.join(DATA, name, "DPmean_data_RB1.npy"))
    th = np.deg2rad(np.stack([p1, p2]))          # unwrapped, tracker frame
    om = savgol_filter(th, SG_WINDOW, 3, deriv=1, delta=DT_DATA, axis=1)
    # The video begins with the pendulum being positioned by hand; the free
    # drop is the first time either link exceeds 4 rad/s, which manual
    # positioning never reaches and free fall passes almost immediately.
    release = np.argmax(np.max(np.abs(om), axis=0) > 4.0) * DT_DATA
    return {"name": name, "t": t, "th": th, "om": om, "release": release}


def eval_indices(trial):
    lo = trial["release"] + ACTIVE_START
    hi = min(trial["release"] + ACTIVE_END, trial["t"][-1] - max(HORIZONS) - 0.1)
    times = np.linspace(lo, hi, STATES_PER_TRIAL)
    return (times / DT_DATA).astype(int)


def state_at(trial, i, offset):
    return (np.array([trial["th"][0, i] + offset[0], trial["th"][1, i] + offset[1],
                      trial["om"][0, i], trial["om"][1, i]]),
            None)


def physics_forecast(state, T, damping=0.0):
    p = PendulumParams.make(k=2, L=L, m=M, g=G, damping=damping)
    _, states = integrate(state, p, t_end=T, dt=1e-3, method="rk4")
    out = states[-1][:2]
    return out if np.all(np.isfinite(out)) else None


def pick_offsets(trial):
    """Choose the gravity-referenced offset per link by 20 ms physics tracking."""
    idx = eval_indices(trial)[::4]
    best = None
    for c1 in (0.0, np.pi):
        for c2 in (0.0, np.pi):
            errs = []
            for i in idx:
                s, _ = state_at(trial, i, (c1, c2))
                pred = physics_forecast(s, 0.02)
                true = np.array([trial["th"][0, i + 10] + c1,
                                 trial["th"][1, i + 10] + c2])
                if pred is not None:
                    errs.append(np.mean(np.abs(wrap(pred - true))))
            m = float(np.mean(errs))
            if best is None or m < best[0]:
                best = (m, (c1, c2))
    return best[1], best[0]


def fit_damping(trial, offset):
    """Single viscous coefficient fit on the calibration trial (0.5 s horizon)."""
    idx = eval_indices(trial)[::3]
    best = None
    for d in [0.0, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]:
        errs = []
        for i in idx:
            s, _ = state_at(trial, i, offset)
            pred = physics_forecast(s, 0.5, damping=d)
            true = np.array([trial["th"][0, i + 250] + offset[0],
                             trial["th"][1, i + 250] + offset[1]])
            if pred is not None:
                errs.append(np.mean(np.abs(wrap(pred - true))))
        m = float(np.mean(errs))
        if best is None or m < best[0]:
            best = (m, d)
    return best[1]


def build_nn_db(trial, offset):
    sl = slice(int((trial["release"] + 0.5) / DT_DATA),
               int((trial["release"] + ACTIVE_END) / DT_DATA))
    th = trial["th"][:, sl] + np.array(offset)[:, None]
    om = trial["om"][:, sl]
    feats = np.stack([np.sin(th[0]), np.cos(th[0]), np.sin(th[1]), np.cos(th[1]),
                      om[0], om[1]])
    scale = feats.std(axis=1, keepdims=True) + 1e-9
    return {"feats": feats / scale, "scale": scale, "th": th, "sl_start": sl.start}


def nn_forecast(db, state, T):
    q = np.stack([np.sin(state[0]), np.cos(state[0]), np.sin(state[1]),
                  np.cos(state[1]), state[2], state[3]])[:, None] / db["scale"]
    step = int(round(T / DT_DATA))
    valid = db["feats"].shape[1] - step
    if valid <= 0:
        return None
    d = np.sum((db["feats"][:, :valid] - q) ** 2, axis=0)
    j = int(np.argmin(d))
    return db["th"][:, j + step]


def fit_sindy(trial, offset):
    import pysindy as ps
    sl = slice(int((trial["release"] + 0.5) / DT_DATA),
               int((trial["release"] + ACTIVE_END) / DT_DATA))
    th = trial["th"][:, sl] + np.array(offset)[:, None]
    om = trial["om"][:, sl]
    X = np.stack([th[0], th[1], om[0], om[1]], axis=1)
    Xdot = savgol_filter(X, SG_WINDOW, 3, deriv=1, delta=DT_DATA, axis=0)
    lib = ps.GeneralizedLibrary([ps.PolynomialLibrary(degree=2),
                                 ps.FourierLibrary(n_frequencies=1)])
    split = int(0.8 * len(X))
    best = None
    for thr in (1e-3, 1e-2, 5e-2, 1e-1):
        cand = ps.SINDy(optimizer=ps.STLSQ(threshold=thr, alpha=1e-6,
                                           normalize_columns=True),
                        feature_library=lib)
        try:
            cand.fit(X[:split], t=DT_DATA, x_dot=Xdot[:split])
            mse = float(np.mean((cand.predict(X[split:]) - Xdot[split:]) ** 2))
        except Exception:
            continue
        if best is None or mse < best[0]:
            best = (mse, cand)
    return None if best is None else best[1]


def sindy_forecast(model, state, T):
    y = state.astype(float).copy()
    n = int(round(T / 1e-3))
    for _ in range(n):
        try:
            dy = model.predict(y[None, :])[0]
        except Exception:
            return None
        y = y + 1e-3 * dy
        if not np.all(np.isfinite(y)) or np.max(np.abs(y[2:])) > 1e3:
            return None
    return y[:2]


class Chronos2:
    def __init__(self):
        from chronos import BaseChronosPipeline
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-2", device_map=device)

    def forecast(self, trial, i, offset, T):
        import torch
        dt_c = 0.02                                   # 50 Hz context
        n_ctx = int(5.0 / dt_c)
        stride = int(round(dt_c / DT_DATA))
        lo = i - n_ctx * stride
        if lo < 0:
            return None
        th = trial["th"][:, lo:i + 1:stride] + np.array(offset)[:, None]
        om = trial["om"][:, lo:i + 1:stride]
        ctx = torch.tensor(np.stack([th[0], th[1], om[0], om[1]]),
                           dtype=torch.float32)
        steps = int(round(T / dt_c))
        try:
            q = self.pipe.predict_quantiles([ctx], prediction_length=steps,
                                            quantile_levels=[0.5])[0]
            out = q[0][:, -1, 0].detach().cpu().numpy()
        except Exception:
            return None
        return out[:2] if np.all(np.isfinite(out[:2])) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-chronos", action="store_true")
    args = ap.parse_args()

    trials = {n: load_trial(n) for n in ["Trial1", "Trial2", "Trial3"]}
    offsets, cal_errs = {}, {}
    for n, tr in trials.items():
        offsets[n], cal_errs[n] = pick_offsets(tr)
        print(f"{n}: release={tr['release']:.2f}s offset=({offsets[n][0]:.2f},"
              f"{offsets[n][1]:.2f}) 20ms physics err={cal_errs[n]:.4f} rad")

    cal = trials["Trial1"]
    damping = fit_damping(cal, offsets["Trial1"])
    print(f"fitted damping (Trial1, 0.5s): {damping}")
    nn_db = build_nn_db(cal, offsets["Trial1"])
    sindy = fit_sindy(cal, offsets["Trial1"])
    print("SINDy fit:", "ok" if sindy is not None else "failed")
    chronos = None if args.skip_chronos else Chronos2()

    rows = []
    for n in ["Trial2", "Trial3"]:
        tr = trials[n]
        off = offsets[n]
        for i in eval_indices(tr):
            s, _ = state_at(tr, i, off)
            for T in HORIZONS:
                j = i + int(round(T / DT_DATA))
                true = np.array([tr["th"][0, j] + off[0], tr["th"][1, j] + off[1]])
                preds = {
                    "physics-rk4": physics_forecast(s, T),
                    "physics-rk4-damped": physics_forecast(s, T, damping=damping),
                    "persistence": s[:2],
                    "constant-velocity": s[:2] + s[2:] * T,
                    "nearest-neighbor": nn_forecast(nn_db, s, T),
                    "sindy-real": None if sindy is None else sindy_forecast(sindy, s, T),
                }
                if chronos is not None:
                    preds["chronos2"] = chronos.forecast(tr, i, off, T)
                for model, pred in preds.items():
                    err = PI2 if pred is None else \
                        float(np.mean(np.abs(wrap(np.asarray(pred) - true))))
                    rows.append((model, n, T, err, pred is not None))

    import pandas as pd
    df = pd.DataFrame(rows, columns=["model", "trial", "horizon", "err", "answered"])
    out = os.path.join(ROOT, "results", "summary_llm", "realdata_myers.csv")
    df.to_csv(out, index=False)

    print("\n=== Real-data check: mean wrapped angle error (rad), trials 2-3, "
          f"{2 * STATES_PER_TRIAL} states ===")
    piv = df.pivot_table("err", "model", "horizon", "mean").round(3)
    ans = df.groupby("model").answered.mean().round(3)
    print(piv.assign(answered=ans).to_string())
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
