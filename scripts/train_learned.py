"""
Train Neural ODE, HNN, and LNN on pendulum trajectory data.

Usage (local):
    python scripts/train_learned.py \
        --dataset-dir results/dataset \
        --output-dir  results/learned_models \
        --epochs 500

Usage (Azure ML — called by scripts/azure_ml_job.py):
    python scripts/train_learned.py \
        --dataset-dir $AZURE_ML_INPUT_dataset \
        --output-dir  $AZUREML_BI_output \
        --epochs 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.models.learned import (
    ODEFunc, HamiltonianNet, LagrangianNet,
    STATE_DIM, PARAM_DIM,
    _pad_state, _encode_params, _state_mask,
)
from bench.simulator import PendulumParams


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_manifest(dataset_dir: str) -> list[dict]:
    path = os.path.join(dataset_dir, "manifest.json")
    with open(path) as f:
        return json.load(f)


def _load_trajectory(entry: dict) -> dict:
    with open(entry["file"]) as f:
        return json.load(f)


def build_training_samples(dataset_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (states, derivatives, params) arrays, each row = one training sample.

    states:      (N, 6)  padded state at time t
    derivatives: (N, 6)  d_state/dt estimated by central finite differences
    params:      (N, 9)  encoded physical parameters
    """
    manifest = _load_manifest(dataset_dir)
    all_states, all_derivs, all_params = [], [], []

    for entry in manifest:
        traj   = _load_trajectory(entry)
        k      = int(traj["number_of_pendulums"])
        c      = traj["constants"]
        p      = PendulumParams.make(k=k, L=c["L"], m=c["m"],
                                     g=c["g"], damping=c["damping"])

        theta  = np.array(traj["theta"])   # (T, k)
        omega  = np.array(traj["omega"])   # (T, k)
        times  = np.array(traj["time"])    # (T,)
        T      = len(times)

        # Build a fake PendulumParams-like struct just to call _encode_params
        class _Req:
            params = p
        params9 = np.zeros(PARAM_DIM, dtype=np.float32)
        params9[0] = k
        params9[1] = p.g
        params9[2:2+k] = p.L
        params9[5:5+k] = p.m
        params9[8]     = p.damping

        # Central finite differences for derivative estimation (skip endpoints)
        for i in range(1, T - 1):
            dt_fwd = times[i+1] - times[i]
            dt_bwd = times[i]   - times[i-1]
            dt_avg = (dt_fwd + dt_bwd) / 2.0
            if dt_avg < 1e-10:
                continue

            state_raw = np.concatenate([theta[i], omega[i]])
            state6    = _pad_state(state_raw, k)

            d_theta = (theta[i+1] - theta[i-1]) / (dt_fwd + dt_bwd)
            d_omega = (omega[i+1] - omega[i-1]) / (dt_fwd + dt_bwd)
            deriv6  = np.zeros(STATE_DIM, dtype=np.float32)
            deriv6[:k]    = d_theta
            deriv6[3:3+k] = d_omega

            all_states.append(state6)
            all_derivs.append(deriv6)
            all_params.append(params9)

    return (np.array(all_states,  dtype=np.float32),
            np.array(all_derivs,  dtype=np.float32),
            np.array(all_params,  dtype=np.float32))


def build_rollout_samples(dataset_dir: str, rollout_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (states_0, states_k, params) for k-step rollout training.

    states_0: (N, 6)  padded state at time t
    states_k: (N, 6)  padded state at time t + rollout_steps * dt
    params:   (N, 9)  encoded physical parameters
    """
    manifest = _load_manifest(dataset_dir)
    all_s0, all_sk, all_params = [], [], []

    for entry in manifest:
        traj  = _load_trajectory(entry)
        k     = int(traj["number_of_pendulums"])
        c     = traj["constants"]
        p     = PendulumParams.make(k=k, L=c["L"], m=c["m"],
                                    g=c["g"], damping=c["damping"])

        theta = np.array(traj["theta"])   # (T, k)
        omega = np.array(traj["omega"])   # (T, k)
        T     = len(theta)

        params9 = np.zeros(PARAM_DIM, dtype=np.float32)
        params9[0] = k
        params9[1] = p.g
        params9[2:2+k] = p.L
        params9[5:5+k] = p.m
        params9[8]     = p.damping

        for i in range(T - rollout_steps):
            s0 = _pad_state(np.concatenate([theta[i],                 omega[i]]),                 k)
            sk = _pad_state(np.concatenate([theta[i + rollout_steps], omega[i + rollout_steps]]), k)
            all_s0.append(s0)
            all_sk.append(sk)
            all_params.append(params9)

    return (np.array(all_s0,    dtype=np.float32),
            np.array(all_sk,    dtype=np.float32),
            np.array(all_params, dtype=np.float32))


# ── Training helpers ──────────────────────────────────────────────────────────

def _make_loader(states, derivs, params, batch_size: int, device):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    ds = TensorDataset(
        torch.tensor(states).to(device),
        torch.tensor(derivs).to(device),
        torch.tensor(params).to(device),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)


def _train_node(states, derivs, params, *, epochs, lr, batch_size, hidden, layers, device):
    """Train Neural ODE via derivative matching (MSE on d_state/dt)."""
    import torch
    import torch.nn as nn

    model = ODEFunc(hidden=hidden, layers=layers).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = _make_loader(states, derivs, params, batch_size, device)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for s_b, d_b, p_b in loader:
            pred = model(s_b, p_b)
            loss = nn.functional.mse_loss(pred, d_b)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(s_b)
        sched.step()
        if epoch % 50 == 0 or epoch == 1:
            print(f"  [NODE] epoch {epoch:4d}/{epochs}  loss={total_loss/len(states):.6f}")

    return model


def _train_node_rollout(states_0, states_k, params, *,
                        rollout_steps, dt, epochs, lr, batch_size,
                        hidden, layers, device, grad_clip):
    """Train Neural ODE via k-step Euler rollout loss (MSE on state at t+k)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    model = ODEFunc(hidden=hidden, layers=layers).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ds = TensorDataset(
        torch.tensor(states_0).to(device),
        torch.tensor(states_k).to(device),
        torch.tensor(params).to(device),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    dt_t = torch.tensor(dt, dtype=torch.float32, device=device)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for s0_b, sk_b, p_b in loader:
            s = s0_b
            for _ in range(rollout_steps):
                s = s + dt_t * model(s, p_b)
            loss = nn.functional.mse_loss(s, sk_b)
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total_loss += loss.item() * len(s0_b)
        sched.step()
        if epoch % 50 == 0 or epoch == 1:
            print(f"  [NODE-rollout] epoch {epoch:4d}/{epochs}  loss={total_loss/len(states_0):.6f}")

    return model


def _hnn_loss(model, s_b, d_b, p_b):
    """
    HNN loss: Hamilton's equations must reproduce the observed derivative.
    dtheta/dt = ∂H/∂omega, domega/dt = -∂H/∂theta
    """
    import torch
    s_b = s_b.requires_grad_(True)
    H   = model(s_b, p_b)
    # Sum over batch before autograd so we get per-sample gradients
    dH  = torch.autograd.grad(H.sum(), s_b, create_graph=True)[0]
    pred_d = torch.zeros_like(s_b)
    pred_d[:, :3] =  dH[:, 3:]   # dtheta/dt = ∂H/∂omega
    pred_d[:, 3:] = -dH[:, :3]   # domega/dt = -∂H/∂theta
    import torch.nn as nn
    return nn.functional.mse_loss(pred_d, d_b)


def _train_hnn(states, derivs, params, *, epochs, lr, batch_size, hidden, layers, device):
    import torch
    model = HamiltonianNet(hidden=hidden, layers=layers).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = _make_loader(states, derivs, params, batch_size, device)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for s_b, d_b, p_b in loader:
            loss = _hnn_loss(model, s_b, d_b, p_b)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(s_b)
        sched.step()
        if epoch % 50 == 0 or epoch == 1:
            print(f"  [HNN]  epoch {epoch:4d}/{epochs}  loss={total_loss/len(states):.6f}")

    return model


def _train_hnn_rollout(states_0, states_k, params, *,
                       rollout_steps, dt, epochs, lr, batch_size,
                       hidden, layers, device, grad_clip):
    """
    Train HNN via k-step Euler rollout loss.

    Each integration step derives ds/dt from Hamilton's equations via autograd
    through H, so create_graph=True is required at every step to keep gradients
    flowing back to model parameters. This is more expensive than NODE rollout
    but is the correct way to train an HNN end-to-end on trajectory error.
    Uses a smaller default batch size than NODE rollout due to the deeper graph.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    model = HamiltonianNet(hidden=hidden, layers=layers).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ds = TensorDataset(
        torch.tensor(states_0).to(device),
        torch.tensor(states_k).to(device),
        torch.tensor(params).to(device),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    dt_t = torch.tensor(dt, dtype=torch.float32, device=device)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for s0_b, sk_b, p_b in loader:
            # Detach from data loader graph; leaf tensor with grad for inner autograd
            s = s0_b.detach().requires_grad_(True)
            for _ in range(rollout_steps):
                H    = model(s, p_b)
                dH   = torch.autograd.grad(H.sum(), s, create_graph=True)[0]
                dsdt = torch.cat([dH[:, 3:], -dH[:, :3]], dim=1)
                s    = s + dt_t * dsdt   # non-leaf; retains grad for next step
            loss = nn.functional.mse_loss(s, sk_b)
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total_loss += loss.item() * len(s0_b)
        sched.step()
        if epoch % 50 == 0 or epoch == 1:
            print(f"  [HNN-rollout] epoch {epoch:4d}/{epochs}  loss={total_loss/len(states_0):.6f}")

    return model


def _lnn_loss(model, s_b, d_b, p_b):
    """
    LNN loss: Euler-Lagrange equations must reproduce observed (qdot, qddot).
    qdot  is in d_b[:, :3]
    qddot is in d_b[:, 3:]

    q and qdot must be detached leaf tensors so that autograd can differentiate
    through the full Euler-Lagrange computation back to model parameters.
    create_graph=True is required throughout so torch.linalg.solve(M, rhs)
    retains a gradient path to the model weights.
    """
    import torch
    import torch.nn as nn

    batch = s_b.shape[0]
    # Detach first to make leaf tensors; then enable grad for E-L differentiation
    q     = s_b[:, :3].detach().requires_grad_(True)
    qdot  = s_b[:, 3:].detach().requires_grad_(True)
    state6 = torch.cat([q, qdot], dim=1)
    L      = model(state6, p_b)            # (B,)

    dLdq    = torch.autograd.grad(L.sum(), q,    create_graph=True)[0]  # (B, 3)
    dLdqdot = torch.autograd.grad(L.sum(), qdot, create_graph=True)[0]  # (B, 3)

    # Mass matrix: M_ij = ∂²L/∂qdot_i ∂qdot_j — keep create_graph=True so M
    # retains grad_fn through linalg.solve back to model parameters
    M = torch.zeros(batch, 3, 3, device=s_b.device)
    for i in range(3):
        row = torch.autograd.grad(dLdqdot[:, i].sum(), qdot,
                                  retain_graph=True, create_graph=True)[0]
        M[:, i, :] = row

    # Cross term: Σ_j (∂²L/∂qdot_i ∂q_j) * qdot_j
    cross = torch.zeros(batch, 3, device=s_b.device)
    for i in range(3):
        row = torch.autograd.grad(dLdqdot[:, i].sum(), q,
                                  retain_graph=True, create_graph=True)[0]
        cross[:, i] = (row * qdot).sum(dim=1)

    M = M + 1e-4 * torch.eye(3, device=s_b.device).unsqueeze(0)
    rhs        = dLdq - cross                       # (B, 3) — has grad_fn
    qddot_pred = torch.linalg.solve(M, rhs)         # (B, 3) — has grad_fn

    pred_d = torch.cat([qdot, qddot_pred], dim=1)  # (B, 6)
    return nn.functional.mse_loss(pred_d, d_b)


def _train_lnn(states, derivs, params, *, epochs, lr, batch_size, hidden, layers, device):
    import torch
    model = LagrangianNet(hidden=hidden, layers=layers).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = _make_loader(states, derivs, params, batch_size, device)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for s_b, d_b, p_b in loader:
            loss = _lnn_loss(model, s_b, d_b, p_b)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(s_b)
        sched.step()
        if epoch % 50 == 0 or epoch == 1:
            print(f"  [LNN]  epoch {epoch:4d}/{epochs}  loss={total_loss/len(states):.6f}")

    return model


# ── Mixed-k rollout training ──────────────────────────────────────────────────

def _build_rollout_by_k(dataset_dir: str, ks: list[int], max_samples):
    """Build rollout (states_0, states_k, params) windows for several k values."""
    by_k = {}
    for k in ks:
        s0, sk, p = build_rollout_samples(dataset_dir, k)
        if max_samples is not None and max_samples < len(s0):
            rng = np.random.default_rng(42)
            idx = rng.choice(len(s0), size=max_samples, replace=False)
            s0, sk, p = s0[idx], sk[idx], p[idx]
        by_k[k] = (s0, sk, p)
        print(f"  k={k}: {len(s0):,} rollout windows")
    return by_k


def _train_node_rollout_mixed(by_k, *, ks, dt, epochs, lr, batch_size,
                              hidden, layers, device, grad_clip):
    """Neural ODE trained on several rollout-window lengths simultaneously.

    Each epoch iterates every k's loader, unrolling that many Euler steps, so the
    model is optimised for short, medium, and long horizons at once rather than a
    single window length.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    model = ODEFunc(hidden=hidden, layers=layers).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dt_t  = torch.tensor(dt, dtype=torch.float32, device=device)

    loaders = {}
    for k, (s0, sk, p) in by_k.items():
        ds = TensorDataset(torch.tensor(s0).to(device),
                           torch.tensor(sk).to(device),
                           torch.tensor(p).to(device))
        loaders[k] = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0; n = 0
        for k in ks:
            for s0_b, sk_b, p_b in loaders[k]:
                s = s0_b
                for _ in range(k):
                    s = s + dt_t * model(s, p_b)
                loss = nn.functional.mse_loss(s, sk_b)
                opt.zero_grad(); loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                total_loss += loss.item() * len(s0_b); n += len(s0_b)
        sched.step()
        if epoch % 50 == 0 or epoch == 1:
            print(f"  [NODE-mixed] epoch {epoch:4d}/{epochs}  loss={total_loss/n:.6f}")

    return model


def _train_hnn_rollout_mixed(by_k, *, ks, dt, epochs, lr, batch_size,
                             hidden, layers, device, grad_clip):
    """HNN trained on several rollout-window lengths simultaneously (Hamilton's
    equations via autograd at every step)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    model = HamiltonianNet(hidden=hidden, layers=layers).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dt_t  = torch.tensor(dt, dtype=torch.float32, device=device)

    loaders = {}
    for k, (s0, sk, p) in by_k.items():
        ds = TensorDataset(torch.tensor(s0).to(device),
                           torch.tensor(sk).to(device),
                           torch.tensor(p).to(device))
        loaders[k] = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0; n = 0
        for k in ks:
            for s0_b, sk_b, p_b in loaders[k]:
                s = s0_b.detach().requires_grad_(True)
                for _ in range(k):
                    H    = model(s, p_b)
                    dH   = torch.autograd.grad(H.sum(), s, create_graph=True)[0]
                    dsdt = torch.cat([dH[:, 3:], -dH[:, :3]], dim=1)
                    s    = s + dt_t * dsdt
                loss = nn.functional.mse_loss(s, sk_b)
                opt.zero_grad(); loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                total_loss += loss.item() * len(s0_b); n += len(s0_b)
        sched.step()
        if epoch % 50 == 0 or epoch == 1:
            print(f"  [HNN-mixed] epoch {epoch:4d}/{epochs}  loss={total_loss/n:.6f}")

    return model


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train Neural ODE, HNN, LNN on pendulum data")
    ap.add_argument("--dataset-dir", default="results/dataset",
                    help="Directory containing manifest.json and trajectory files")
    ap.add_argument("--output-dir",  default="results/learned_models",
                    help="Directory where trained .pt checkpoints are saved")
    ap.add_argument("--epochs",      type=int, default=500)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--batch-size",  type=int, default=512)
    ap.add_argument("--hidden",      type=int, default=256)
    ap.add_argument("--layers",      type=int, default=3)
    ap.add_argument("--models",      nargs="*",
                    default=["neural_ode", "hnn", "lnn"],
                    help="Which models to train (subset of: neural_ode hnn lnn)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Subsample training data to this many rows (useful for slow models like LNN)")
    ap.add_argument("--rollout-steps", type=int, default=0,
                    help="Euler rollout steps for Neural ODE (0 = derivative matching, default)")
    ap.add_argument("--rollout-dt",    type=float, default=0.01,
                    help="Timestep for rollout integration (default 0.01s)")
    ap.add_argument("--grad-clip",     type=float, default=1.0,
                    help="Gradient clipping max norm for rollout training (0 = disabled)")
    ap.add_argument("--rollout-ks",    type=int, nargs="*", default=None,
                    help="Mixed-k rollout training: optimise several window lengths at "
                         "once, e.g. --rollout-ks 10 50. Overrides --rollout-steps; "
                         "saves to *_rollout_mixed.pt. (neural_ode / hnn only)")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        sys.exit("torch is not installed. Run: pip install torch")

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading trajectory data …")
    t0 = time.perf_counter()
    states, derivs, params = build_training_samples(args.dataset_dir)
    print(f"  {len(states):,} training samples loaded in {time.perf_counter()-t0:.1f}s")

    if args.max_samples is not None and args.max_samples < len(states):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(states), size=args.max_samples, replace=False)
        states, derivs, params = states[idx], derivs[idx], params[idx]
        print(f"  Subsampled to {len(states):,} samples")

    train_kw = dict(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        hidden=args.hidden, layers=args.layers, device=device,
    )

    mixed = args.rollout_ks is not None and len(args.rollout_ks) > 0
    by_k = None
    if mixed:
        print(f"\nBuilding mixed-k rollout samples for k={args.rollout_ks} …")
        by_k = _build_rollout_by_k(args.dataset_dir, args.rollout_ks, args.max_samples)

    if "neural_ode" in args.models:
        if mixed:
            print(f"\nTraining Neural ODE (mixed rollout, ks={args.rollout_ks}) …")
            model = _train_node_rollout_mixed(
                by_k, ks=args.rollout_ks, dt=args.rollout_dt,
                grad_clip=args.grad_clip, **train_kw)
            out = os.path.join(args.output_dir, "neural_ode_rollout_mixed.pt")
        elif args.rollout_steps > 0:
            print(f"\nTraining Neural ODE (rollout, k={args.rollout_steps}, dt={args.rollout_dt}) …")
            print("  Building rollout training samples …")
            t0 = time.perf_counter()
            states_0, states_k, params_r = build_rollout_samples(
                args.dataset_dir, args.rollout_steps)
            print(f"  {len(states_0):,} rollout windows in {time.perf_counter()-t0:.1f}s")
            if args.max_samples is not None and args.max_samples < len(states_0):
                rng = np.random.default_rng(42)
                idx = rng.choice(len(states_0), size=args.max_samples, replace=False)
                states_0, states_k, params_r = states_0[idx], states_k[idx], params_r[idx]
                print(f"  Subsampled to {len(states_0):,} samples")
            model = _train_node_rollout(
                states_0, states_k, params_r,
                rollout_steps=args.rollout_steps,
                dt=args.rollout_dt,
                grad_clip=args.grad_clip,
                **train_kw,
            )
            out = os.path.join(args.output_dir, "neural_ode_rollout.pt")
        else:
            print("\nTraining Neural ODE …")
            model = _train_node(states, derivs, params, **train_kw)
            out = os.path.join(args.output_dir, "neural_ode.pt")
        torch.save(model.state_dict(), out)
        print(f"  Saved → {out}")

    if "hnn" in args.models:
        if mixed:
            print(f"\nTraining HNN (mixed rollout, ks={args.rollout_ks}) …")
            # HNN rollout graph is deep — halve batch size to avoid OOM
            hnn_kw = {**train_kw, "batch_size": train_kw["batch_size"] // 2}
            model = _train_hnn_rollout_mixed(
                by_k, ks=args.rollout_ks, dt=args.rollout_dt,
                grad_clip=args.grad_clip, **hnn_kw)
            out = os.path.join(args.output_dir, "hnn_rollout_mixed.pt")
        elif args.rollout_steps > 0:
            print(f"\nTraining HNN (rollout, k={args.rollout_steps}, dt={args.rollout_dt}) …")
            if "states_0" not in dir():
                print("  Building rollout training samples …")
                t0 = time.perf_counter()
                states_0, states_k, params_r = build_rollout_samples(
                    args.dataset_dir, args.rollout_steps)
                print(f"  {len(states_0):,} rollout windows in {time.perf_counter()-t0:.1f}s")
                if args.max_samples is not None and args.max_samples < len(states_0):
                    rng = np.random.default_rng(42)
                    idx = rng.choice(len(states_0), size=args.max_samples, replace=False)
                    states_0, states_k, params_r = states_0[idx], states_k[idx], params_r[idx]
            # HNN rollout graph is deep — halve batch size to avoid OOM
            hnn_kw = {**train_kw, "batch_size": train_kw["batch_size"] // 2}
            model = _train_hnn_rollout(
                states_0, states_k, params_r,
                rollout_steps=args.rollout_steps,
                dt=args.rollout_dt,
                grad_clip=args.grad_clip,
                **hnn_kw,
            )
            out = os.path.join(args.output_dir, "hnn_rollout.pt")
        else:
            print("\nTraining HNN …")
            model = _train_hnn(states, derivs, params, **train_kw)
            out = os.path.join(args.output_dir, "hnn.pt")
        torch.save(model.state_dict(), out)
        print(f"  Saved → {out}")

    if "lnn" in args.models:
        print("\nTraining LNN …")
        model = _train_lnn(states, derivs, params, **train_kw)
        out = os.path.join(args.output_dir, "lnn.pt")
        torch.save(model.state_dict(), out)
        print(f"  Saved → {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
