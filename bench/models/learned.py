from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from .base import PredictionRequest, PredictionResult

# Fixed dimensions: always pad state/params to max k=3
STATE_DIM = 6   # [theta_1..3, omega_1..3], inactive dims stay 0
PARAM_DIM = 9   # [k, g, L_1, L_2, L_3, m_1, m_2, m_3, damping]


def _learned_device():
    """Device for learned-model inference. Override with DP_LEARNED_DEVICE=cpu/cuda."""
    import torch
    requested = os.environ.get("DP_LEARNED_DEVICE", "auto").strip().lower()
    if requested in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return _learned_device()


def _encode_params(req: PredictionRequest) -> np.ndarray:
    """Pack PendulumParams into a fixed-length float vector."""
    p = req.params
    k = p.k
    L = np.zeros(3); L[:k] = p.L
    m = np.zeros(3); m[:k] = p.m
    return np.array([k, p.g, *L, *m, p.damping], dtype=np.float32)


def _pad_state(state: np.ndarray, k: int) -> np.ndarray:
    """Pad a 2k state vector to length 6."""
    out = np.zeros(STATE_DIM, dtype=np.float32)
    out[:k]     = state[:k]    # theta
    out[3:3+k]  = state[k:2*k] # omega
    return out


def _unpad_state(state6: np.ndarray, k: int) -> np.ndarray:
    """Extract 2k state from padded 6-vector."""
    return np.concatenate([state6[:k], state6[3:3+k]])


def _state_mask(k: int) -> np.ndarray:
    mask = np.zeros(STATE_DIM, dtype=np.float32)
    mask[:k]    = 1.0
    mask[3:3+k] = 1.0
    return mask


# â”€â”€ PyTorch network definitions (also used by train_learned.py) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _build_mlp(in_dim: int, out_dim: int, hidden: int = 256, layers: int = 3):
    """Tanh MLP used by all three model types."""
    try:
        import torch.nn as nn
    except ImportError:
        raise RuntimeError("torch is required. Run `pip install torch`.")
    mods = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.Tanh()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class ODEFunc:
    """Neural ODE: MLP maps (state, params) â†’ d_state/dt."""

    def __init__(self, hidden: int = 256, layers: int = 3):
        self.net = _build_mlp(STATE_DIM + PARAM_DIM, STATE_DIM, hidden, layers)

    def __call__(self, state6, params9):
        import torch
        x = torch.cat([state6, params9], dim=-1)
        return self.net(x)

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)

    def parameters(self):
        return self.net.parameters()

    def eval(self):
        self.net.eval()
        return self

    def to(self, device):
        self.net.to(device)
        return self


class HamiltonianNet:
    """HNN: MLP maps (state, params) â†’ scalar H; dynamics from autograd.
    Uses (theta, omega) as quasi-canonical coords (omega â‰ˆ generalised momentum).
    """

    def __init__(self, hidden: int = 256, layers: int = 3):
        self.net = _build_mlp(STATE_DIM + PARAM_DIM, 1, hidden, layers)

    def __call__(self, state6, params9):
        import torch
        x = torch.cat([state6, params9], dim=-1)
        return self.net(x).squeeze(-1)

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)

    def parameters(self):
        return self.net.parameters()

    def eval(self):
        self.net.eval()
        return self

    def to(self, device):
        self.net.to(device)
        return self


class LagrangianNet:
    """LNN: MLP maps (q, qdot, params) â†’ scalar L; dynamics via Euler-Lagrange."""

    def __init__(self, hidden: int = 256, layers: int = 3):
        # input: q (3) + qdot (3) + params (9) = 15, but we reuse STATE_DIM+PARAM_DIM
        self.net = _build_mlp(STATE_DIM + PARAM_DIM, 1, hidden, layers)

    def __call__(self, state6, params9):
        """state6 = [q_padded, qdot_padded] i.e. [theta, omega]."""
        import torch
        x = torch.cat([state6, params9], dim=-1)
        return self.net(x).squeeze(-1)

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)

    def parameters(self):
        return self.net.parameters()

    def eval(self):
        self.net.eval()
        return self

    def to(self, device):
        self.net.to(device)
        return self


# â”€â”€ scipy-based ODE integration helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _ode_rhs_node(t, y, model, params9_np, mask):
    """RHS for Neural ODE: call the network, mask inactive dims."""
    import torch
    device = _model_device(model)
    with torch.no_grad():
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
        p_t = torch.as_tensor(params9_np, dtype=torch.float32, device=device)
        dy = model(y_t, p_t).detach().cpu().numpy()
    return dy * mask


def _ode_rhs_hnn(t, y, model, params9_np, k):
    """RHS for HNN: Hamilton's equations via autograd."""
    import torch
    device = _model_device(model)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device).requires_grad_(True)
    p_t = torch.as_tensor(params9_np, dtype=torch.float32, device=device)
    H = model(y_t, p_t)
    dH = torch.autograd.grad(H, y_t)[0].detach().cpu().numpy()
    # dtheta/dt = dH/domega, domega/dt = -dH/dtheta
    dy = np.zeros_like(y)
    dy[:3] = dH[3:]
    dy[3:] = -dH[:3]
    mask = _state_mask(k)
    return dy * mask


def _ode_rhs_lnn(t, y, model, params9_np, k):
    """RHS for LNN: Euler-Lagrange equations via autograd."""
    import torch
    device = _model_device(model)
    q = torch.as_tensor(y[:3], dtype=torch.float32, device=device).requires_grad_(True)
    qdot = torch.as_tensor(y[3:], dtype=torch.float32, device=device).requires_grad_(True)
    state6 = torch.cat([q, qdot])
    p_t = torch.as_tensor(params9_np, dtype=torch.float32, device=device)

    L = model(state6, p_t)

    dLdq = torch.autograd.grad(L, q, create_graph=True)[0]
    dLdqdot = torch.autograd.grad(L, qdot, create_graph=True)[0]

    M = torch.zeros(3, 3, device=device)
    for i in range(3):
        row = torch.autograd.grad(dLdqdot[i], qdot,
                                  retain_graph=True, create_graph=False)[0]
        M[i] = row

    cross = torch.zeros(3, device=device)
    for i in range(3):
        row = torch.autograd.grad(dLdqdot[i], q,
                                  retain_graph=True, create_graph=False)[0]
        cross[i] = (row * qdot.detach()).sum()

    M = M + 1e-4 * torch.eye(3, device=device)
    rhs = dLdq.detach() - cross
    qddot = torch.linalg.solve(M, rhs.unsqueeze(1)).squeeze(1)

    dy = np.zeros(6, dtype=np.float32)
    dy[:3] = qdot.detach().cpu().numpy()
    dy[3:] = qddot.detach().cpu().numpy()
    mask = _state_mask(k)
    return dy * mask


def _integrate(rhs_fn, state0_6, horizon: float, rtol: float = 1e-6, atol: float = 1e-8,
               max_step: float = np.inf) -> np.ndarray:
    from scipy.integrate import solve_ivp
    sol = solve_ivp(rhs_fn, [0.0, horizon], state0_6,
                    method="RK45", rtol=rtol, atol=atol,
                    max_step=max_step, dense_output=False)
    return sol.y[:, -1].astype(float)


# â”€â”€ Predictor classes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class _LearnedBase:
    is_async = False
    uses_modality = False
    uses_prompting = False
    uses_history = False

    def _load(self, checkpoint: str, net_cls, **net_kw):
        import torch
        device = _learned_device()
        net = net_cls(**net_kw).to(device)
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        net.load_state_dict(ckpt)
        net.eval()
        return net

    async def apredict(self, req: PredictionRequest) -> PredictionResult:
        return self.predict(req)


class NeuralODEPredictor(_LearnedBase):

    def __init__(self, name: str, checkpoint: str,
                 hidden: int = 256, layers: int = 3):
        self.name = name
        self._ckpt = checkpoint
        self._hidden = hidden
        self._layers = layers
        self._net = None

    def _ensure_loaded(self):
        if self._net is None:
            self._net = self._load(self._ckpt, ODEFunc,
                                   hidden=self._hidden, layers=self._layers)

    def predict(self, req: PredictionRequest) -> PredictionResult:
        self._ensure_loaded()
        k      = req.params.k
        s0     = _pad_state(req.state0, k)
        params = _encode_params(req)
        mask   = _state_mask(k)
        net    = self._net

        rhs = lambda t, y: _ode_rhs_node(t, y, net, params, mask)
        t0 = time.perf_counter()
        try:
            final = _integrate(rhs, s0, req.horizon)
            latency = time.perf_counter() - t0
            result = _unpad_state(final, k)
            return PredictionResult(
                pred_theta=result[:k].tolist(),
                pred_omega=result[k:].tolist(),
                latency_s=latency, success=True,
            )
        except Exception as e:
            return PredictionResult(
                pred_theta=[float("nan")] * k,
                pred_omega=[float("nan")] * k,
                success=False, error=repr(e),
            )


class HNNPredictor(_LearnedBase):

    def __init__(self, name: str, checkpoint: str,
                 hidden: int = 256, layers: int = 3):
        self.name = name
        self._ckpt = checkpoint
        self._hidden = hidden
        self._layers = layers
        self._net = None

    def _ensure_loaded(self):
        if self._net is None:
            self._net = self._load(self._ckpt, HamiltonianNet,
                                   hidden=self._hidden, layers=self._layers)

    def predict(self, req: PredictionRequest) -> PredictionResult:
        self._ensure_loaded()
        k      = req.params.k
        s0     = _pad_state(req.state0, k)
        params = _encode_params(req)
        net    = self._net

        rhs = lambda t, y: _ode_rhs_hnn(t, y, net, params, k)
        t0 = time.perf_counter()
        try:
            final = _integrate(rhs, s0, req.horizon)
            latency = time.perf_counter() - t0
            result = _unpad_state(final, k)
            return PredictionResult(
                pred_theta=result[:k].tolist(),
                pred_omega=result[k:].tolist(),
                latency_s=latency, success=True,
            )
        except Exception as e:
            return PredictionResult(
                pred_theta=[float("nan")] * k,
                pred_omega=[float("nan")] * k,
                success=False, error=repr(e),
            )


class LNNPredictor(_LearnedBase):

    def __init__(self, name: str, checkpoint: str,
                 hidden: int = 256, layers: int = 3):
        self.name = name
        self._ckpt = checkpoint
        self._hidden = hidden
        self._layers = layers
        self._net = None

    def _ensure_loaded(self):
        if self._net is None:
            self._net = self._load(self._ckpt, LagrangianNet,
                                   hidden=self._hidden, layers=self._layers)

    def predict(self, req: PredictionRequest) -> PredictionResult:
        self._ensure_loaded()
        k      = req.params.k
        s0     = _pad_state(req.state0, k)
        params = _encode_params(req)
        net    = self._net

        rhs = lambda t, y: _ode_rhs_lnn(t, y, net, params, k)
        t0 = time.perf_counter()
        try:
            # Use coarser tolerances and a fixed max_step to bound RHS evaluations â€”
            # LNN's autograd-based RHS is expensive; tight tolerances over long horizons
            # would require thousands of steps each costing 6 autograd calls.
            final = _integrate(rhs, s0, req.horizon,
                               rtol=1e-3, atol=1e-4, max_step=0.5)
            latency = time.perf_counter() - t0
            result = _unpad_state(final, k)
            return PredictionResult(
                pred_theta=result[:k].tolist(),
                pred_omega=result[k:].tolist(),
                latency_s=latency, success=True,
            )
        except Exception as e:
            return PredictionResult(
                pred_theta=[float("nan")] * k,
                pred_omega=[float("nan")] * k,
                success=False, error=repr(e),
            )


# Legacy stub kept for backwards compatibility with any existing serialised cells
class LearnedPredictor(_LearnedBase):
    def __init__(self, name: str, variant: str, checkpoint: str | None = None):
        self.name = name
        self.variant = variant
        self.checkpoint = checkpoint

    def predict(self, req: PredictionRequest) -> PredictionResult:
        return PredictionResult(
            pred_theta=[float("nan")] * req.cell.k,
            pred_omega=[float("nan")] * req.cell.k,
            success=False,
            error=f"use NeuralODEPredictor/HNNPredictor/LNNPredictor instead of LearnedPredictor",
        )
