"""Identifiability helpers for the pendulum parameterization."""
from __future__ import annotations

import numpy as np

from .simulator import PendulumParams


def equivalent_transform(params: PendulumParams, *, mass_scale: float,
                         length_scale: float) -> PendulumParams:
    """Apply a parameter transformation that leaves angular dynamics unchanged."""
    return PendulumParams.make(
        k=params.k,
        L=length_scale * params.L,
        m=mass_scale * params.m,
        g=length_scale * params.g,
        damping=mass_scale * length_scale**2 * params.damping,
    )


def equivalence_log_residual(predicted: dict, truth: dict, k: int) -> dict:
    """Score positive parameter estimates after optimizing over both symmetries.

    The residual is in log-parameter space.  A value of zero means that the
    estimate belongs to the same observational equivalence class as the truth.
    """
    predicted_values = {
        "g": float(predicted["g"]),
        "L": np.asarray(predicted["L"][:k], dtype=float),
        "m": np.asarray(predicted["m"][:k], dtype=float),
        "damping": float(predicted["damping"]),
    }
    true_values = {
        "g": float(truth["g"]),
        "L": np.asarray(truth["L"][:k], dtype=float),
        "m": np.asarray(truth["m"][:k], dtype=float),
        "damping": float(truth["damping"]),
    }
    flat_pred = np.concatenate([
        predicted_values["L"], [predicted_values["g"]], predicted_values["m"],
        [predicted_values["damping"]],
    ])
    flat_true = np.concatenate([
        true_values["L"], [true_values["g"]], true_values["m"], [true_values["damping"]],
    ])
    if np.any(flat_pred <= 0) or np.any(flat_true <= 0) or not np.all(np.isfinite(flat_pred)):
        raise ValueError("equivalence scoring requires finite positive parameters")

    # Rows correspond to L, g, m, and damping. Columns are log mass and length scales.
    design = np.vstack([
        np.tile([0.0, 1.0], (k, 1)),
        [0.0, 1.0],
        np.tile([1.0, 0.0], (k, 1)),
        [1.0, 2.0],
    ])
    log_ratio = np.log(flat_pred) - np.log(flat_true)
    scales, *_ = np.linalg.lstsq(design, log_ratio, rcond=None)
    residual = log_ratio - design @ scales
    return {
        "log_rmse": float(np.sqrt(np.mean(residual**2))),
        "log_mae": float(np.mean(np.abs(residual))),
        "fitted_mass_scale": float(np.exp(scales[0])),
        "fitted_length_scale": float(np.exp(scales[1])),
        "residuals": residual.tolist(),
    }
