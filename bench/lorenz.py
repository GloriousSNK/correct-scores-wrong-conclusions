"""Lorenz-63 simulation utilities for the cross-system benchmark."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LorenzParams:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0


def dynamics(state: np.ndarray, params: LorenzParams = LorenzParams()) -> np.ndarray:
    x, y, z = np.asarray(state, dtype=float)
    return np.array([
        params.sigma * (y - x),
        x * (params.rho - z) - y,
        x * y - params.beta * z,
    ])


def step_euler(state: np.ndarray, dt: float, params: LorenzParams = LorenzParams()) -> np.ndarray:
    return np.asarray(state, dtype=float) + dt * dynamics(state, params)


def step_rk4(state: np.ndarray, dt: float, params: LorenzParams = LorenzParams()) -> np.ndarray:
    state = np.asarray(state, dtype=float)
    k1 = dynamics(state, params)
    k2 = dynamics(state + 0.5 * dt * k1, params)
    k3 = dynamics(state + 0.5 * dt * k2, params)
    k4 = dynamics(state + dt * k3, params)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(state0: np.ndarray, *, t_end: float, dt: float,
              params: LorenzParams = LorenzParams(), method: str = "rk4") -> tuple[np.ndarray, np.ndarray]:
    """Integrate a Lorenz trajectory, including the initial state at time zero."""
    if method not in {"rk4", "euler"}:
        raise ValueError(f"unknown integration method {method!r}")
    step = step_rk4 if method == "rk4" else step_euler
    n_steps = int(round(t_end / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    states = np.empty((n_steps + 1, 3), dtype=float)
    states[0] = state0
    for index in range(n_steps):
        states[index + 1] = step(states[index], dt, params)
    return times, states


def sample_trajectory(rng: np.random.Generator, *, warmup_seconds: float = 20.0,
                      context_seconds: float = 5.0, forecast_seconds: float = 2.0,
                      dt: float = 0.01, params: LorenzParams = LorenzParams()) -> tuple[np.ndarray, np.ndarray]:
    """Draw an initial condition, settle it onto the attractor, then return t<=0 history and future."""
    initial = rng.uniform(low=(-15.0, -15.0, 5.0), high=(15.0, 15.0, 35.0))
    _, warm = integrate(initial, t_end=warmup_seconds, dt=dt, params=params)
    times, states = integrate(warm[-1], t_end=context_seconds + forecast_seconds, dt=dt, params=params)
    return times - context_seconds, states


def normalized_mae(prediction: np.ndarray, target: np.ndarray, scale: np.ndarray) -> float:
    """Mean absolute coordinate error normalized by training-set coordinate scale."""
    scale = np.maximum(np.asarray(scale, dtype=float), 1e-8)
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(target)) / scale))
