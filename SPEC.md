# Can LLMs Predict Physics?

**Evaluating Language and Vision Models via Physical State Forecasting**

## Overview

This project evaluates whether modern Large Language Models (LLMs), Vision-Language
Models (VLMs), time-series foundation models, and learned dynamics models can
predict the future state of physical systems — using k-pendulum dynamics as the
benchmark. Numerical integrators serve as the ground-truth oracle and as
baselines.

## Models Under Test

### Large Language Models (3)
- **gpt-5.5 (latest model)**
- **kimi-k2.6**
- **grok-4-1-fast-reasoning**

### Vision-Language Models (3)
- **qwen-qwen3-vl-32b-thinking**
- **grok-4-1-fast-reasoning**
- **gpt-5.5 (latest model)**

### Time-Series Foundation Models (2)
- **nieche-chronos-bolt-base-fine-tuned-v2**
- **chengsenwang-chattime-1-7b-chat**
<!-- Other time series are: Moirai and TimeFM -->

### Numerical Baselines (3)
- **Euler Integration**
- **Runge–Kutta (RK4)**
- **Symplectic Integrator** (e.g. leapfrog / Verlet)

### Learned Dynamics Models
- **Neural ODE**
- **Hamiltonian Neural Network (HNN)**
- **Lagrangian Neural Network (LNN)**

**Final tally:** 3 LLMs · 3 VLMs · 2 time-series · 3 numerical baselines, plus
the three learned dynamics models for comparison.

## Simulator Requirements

The pendulum simulator must support:

- **Single pendulum** (1-DoF)
- **Double pendulum** (2-DoF)
- **K-pendulum** (general n-link chain)

Configurable per run:
- Gravity `g`
- Rod lengths `L_i`
- Bob masses `m_i`
- Damping coefficient `b`
- Initial angles `θ_i(0)`
- Initial angular velocities `ω_i(0)`

## Data Export

Per simulated **movement** (one trajectory), export:

| Field | Description |
|---|---|
| `movement_id` | Unique identifier for the trajectory |
| `number_of_pendulums` | `k` (1, 2, or 3 for this study) |
| `time` | Time vector (or per-sample timestamp) |
| `initial_conditions` | `θ_i(0)`, `ω_i(0)` for each link |
| `constants` | `g`, `L_i`, `m_i`, `b` |
| `x`, `y` coordinates | Per bob, per time step |
| `potential_energy` | `PE(t)` |
| `total_energy` | `E(t) = KE + PE` |
| `theta` values | `θ_i(t)` |
| `omega` values | `ω_i(t)` |

**Formats:** CSV **or** JSON (both supported; JSON preferred for nested
structure, CSV for tabular consumers).

## Experimental Design

### Systems
1. **1 pendulum**
2. **2 pendulums** (double pendulum)
3. **3 pendulums** (triple pendulum)

### Input Modality
- **Coordinates only** — text/numeric state
- **Images only** — rendered frames
- **Images + Coordinates** — multimodal

### Forecast Horizons
- **10 ms** (next-step)
- **1 s** (short)
- **10 s** (medium)
- **1 min** (long — well past Lyapunov time for chaotic regimes)

### Constant Regimes
- **Normal** — standard Earth-like constants
- **Changed–disclosed** — constants altered; model is told the new values
- **Changed–hidden** — constants altered; model is NOT told (tests inference)

### Prompting Strategy
- **CoT** (chain-of-thought)
- **No CoT** (direct answer)

## Metrics

For each predicted trajectory vs. ground truth:

1. **Coordinate error** — pointwise (x, y) MSE / MAE per bob
2. **Average error across all n pendulums** — aggregated coordinate error
3. **Angle error** — smallest signed angular difference (handles wrap-around)
4. **Direction and velocity** — sign-of-`ω` accuracy and `|ω|` error
5. **ΔPE / ΔKE** — divergence in potential and kinetic energy
6. **ΔE_total** — total-energy drift (key for symplectic comparison)
7. **Long-run deviance** — accumulated trajectory deviation over the horizon
8. **Time to divergence** — first time the prediction error exceeds a
   threshold (e.g. half a rod length, or 1 rad on θ)

## Open Questions / TBDs

- Final selection of 3 LLMs and 3 VLMs on Azure.
- Pick 2 of {Chronos, TimesFM, Moirai} for the time-series slot.
- Threshold definitions for "time to divergence" — fix per-metric or
  per-system?
- Image rendering spec for the vision modality (resolution, frame rate,
  color scheme, presence of trails / phase plot).
- Training data policy for the learned models (Neural ODE / HNN / LNN):
  in-distribution only, or include changed-constant regimes?
- Sample size: how many trajectories per (system × modality × horizon ×
  regime × prompting) cell?
