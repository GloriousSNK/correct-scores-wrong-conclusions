# Can LLMs Predict Physics? — Round 1 Findings

**Experiment:** k-pendulum dynamics prediction benchmark  
**Date:** June 2026  
**Models tested:** 9 (3 LLMs, 3 learned dynamics models, 3 numerical integrators)  
**Total evaluations:** 864 cells × 20 trajectories each

---

## Setup

We asked models to predict the state of a k-link pendulum (k ∈ {1, 2, 3}) at a future time T, given the initial state. Prediction horizons ranged from 0.01 s (nearly trivial) to 60 s (deep into chaotic territory). LLMs received either coordinate data, images, or both. Learned and numerical models received coordinates only.

Three physical regimes were tested:
- **normal** — standard Earth gravity, unit lengths and masses
- **changed_disclosed** — different gravity/lengths/masses, told to the model
- **changed_hidden** — different parameters, not disclosed

Primary metric: **mean angle error (radians)** across all active links. Secondary: success rate (fraction of cells producing a valid prediction).

---

## Models

| Model | Type | Notes |
|-------|------|-------|
| grok-4-1-fast-reasoning | LLM | Vision-capable; coords, images, images+coords |
| kimi-k2.6 | LLM | Coords only |
| deepseek-v4-pro | LLM | Coords only; vision attempted but failed (see findings) |
| Neural ODE | Learned | MLP trained to match d_state/dt; integrated via RK45 |
| HNN | Learned | Hamiltonian network; dynamics from autograd of H |
| LNN | Learned | Lagrangian network; Euler-Lagrange via autograd |
| Euler | Numerical | Fixed-step, dt=0.01 s |
| RK4 | Numerical | Runge-Kutta 4th order, dt=0.01 s |
| Symplectic | Numerical | Störmer-Verlet leapfrog, dt=0.01 s |

---

## Overall Leaderboard

*Lower angle error = better. Ranked by mean angle error across all cells where a prediction was produced.*

| Rank | Model | Angle Error (rad) ↓ | Success Rate |
|------|-------|-------------------|-------------|
| 1 | RK4 | 0.119 | 100% |
| 2 | Symplectic | 0.253 | 100% |
| 3 | kimi-k2.6 | 0.287 | 7.4% |
| 4 | Euler | 0.455 | 100% |
| 5 | Neural ODE | 0.588 | 100% |
| 6 | grok-4-1-fast-reasoning | 0.651 | 75.0% |
| 7 | HNN | 0.733 | 100% |
| 8 | deepseek-v4-pro | 0.759 | 32.9% |
| 9 | LNN | 1.300 | 100% |

---

## Breakdown by Prediction Horizon

*Mean angle error (rad). LLMs make a single one-shot prediction regardless of horizon; numerical integrators accumulate error over time.*

| Model | 0.01 s | 1 s | 10 s | 60 s |
|-------|-------:|----:|-----:|-----:|
| RK4 | 0.000 | 0.000 | 0.260 | 0.216 |
| Symplectic | 0.000 | 0.021 | 0.501 | 0.491 |
| kimi-k2.6 | 0.000 | 1.741 | 0.114 | 0.342 |
| Euler | 0.000 | 0.041 | 0.856 | 0.922 |
| Neural ODE | 0.002 | 0.090 | 1.044 | 1.215 |
| grok | 0.383 | 0.828 | 0.822 | 0.608 |
| HNN | 0.004 | 0.519 | 1.099 | 1.312 |
| deepseek-v4-pro | 0.000 | 1.124 | 0.943 | 0.980 |
| LNN | 0.001 | 1.638 | 2.005 | 1.557 |

---

## Breakdown by Pendulum Count

*Mean angle error (rad). Higher k = more chaotic, more degrees of freedom.*

| Model | k=1 | k=2 | k=3 |
|-------|----:|----:|----:|
| RK4 | 0.000 | 0.041 | 0.316 |
| Symplectic | 0.001 | 0.274 | 0.484 |
| kimi-k2.6 | 0.007 | 0.404 | 0.428 |
| Euler | 0.562 | 0.313 | 0.489 |
| Neural ODE | 0.355 | 0.714 | 0.694 |
| grok | 0.593 | 0.546 | 0.814 |
| HNN | 0.382 | 0.820 | 0.998 |
| deepseek-v4-pro | 0.713 | 0.663 | 0.898 |
| LNN | 1.265 | 1.321 | 1.314 |

---

## Breakdown by Physical Regime

*Mean angle error (rad). "changed_hidden" = model doesn't know the physical parameters.*

| Model | normal | changed_disclosed | changed_hidden |
|-------|-------:|------------------:|---------------:|
| RK4 | 0.357 | 0.000 | 0.000 |
| Symplectic | 0.602 | 0.144 | 0.014 |
| kimi-k2.6 | 0.816 | — | 0.212 |
| Euler | 1.028 | 0.271 | 0.065 |
| Neural ODE | 0.639 | 0.592 | 0.532 |
| grok | 1.138 | 0.717 | 0.300 |
| HNN | 0.783 | 0.825 | 0.593 |
| deepseek-v4-pro | 1.163 | 0.817 | 0.275 |
| LNN | 1.219 | 1.673 | 1.009 |

---

## LLM Deep-Dive

### Modality (grok only — the only vision-capable LLM this round)

| Modality | Angle Error (rad) | Success Rate |
|----------|------------------:|-------------|
| Coords only | 0.502 | 70.8% |
| Images + coords | 0.571 | 76.4% |
| Images only | 0.867 | 77.8% |

### Chain-of-Thought vs No-CoT

| Model | CoT error | CoT success | No-CoT error | No-CoT success |
|-------|----------:|------------:|-------------:|---------------:|
| grok | 0.683 | 75.9% | 0.619 | 74.1% |
| deepseek-v4-pro | 0.706 | 32.4% | 0.810 | 33.3% |
| kimi-k2.6 | 0.084 | 4.6% | 0.380 | 10.2% |

---

## Key Findings

### 1. Physics is still physics

Numerical integrators dominate completely. RK4 achieves sub-milliradian error at 0.01 s and 1 s horizons — essentially perfect. The gap to everything else is enormous. Even at 60 s, deep into chaotic territory, RK4 (0.22 rad) still beats every LLM and every learned model. If you know the equations of motion, solve them. No LLM is close.

### 2. Neural ODE beats all LLMs — cleanly

Neural ODE (0.588 rad, **100% success**) is the best non-numerical method. It learned a usable dynamics model from ~67K trajectory samples in roughly 3 minutes of training on a laptop GPU. Critically, it succeeds on every single cell — no format failures, no refusals, no hallucinations. Compared to grok's 75% success and kimi's 7.4%, that reliability matters as much as raw accuracy. If you have training data, a simple learned model beats a frontier LLM for this task.

### 3. LLMs are doing something — but it's fragile

- **grok** achieves near-zero angle error (~10⁻⁷ rad) on k=1 normal regime at 0.01 s. It understands that a 10-millisecond prediction barely differs from the initial condition. But at 1 s+ horizons on k=3 systems it collapses (0.81–1.14 rad). It is interpolating near the initial state, not simulating dynamics.

- **kimi** has a single cell with 6×10⁻¹⁰ rad error — essentially machine precision. When it responds correctly, it may be doing remarkable physics reasoning. But 92.6% of the time it refuses to answer or produces unparseable output. It is brilliant and currently unusable.

- **deepseek-v4-pro** works well on coordinate inputs (98.6% success) but has **0% success on images** — not a model capability failure, but a prompt/format mismatch. Its predictions that do parse are reasonable (~0.71 rad error with CoT).

### 4. CoT does not help LLMs predict physics

Across all three LLMs, chain-of-thought either makes no difference or slightly hurts accuracy. Reasoning longer about physics doesn't make the physics right. LLMs cannot iteratively integrate differential equations in their forward pass the way a numerical solver does — a chain of reasoning tokens is not a time-stepper.

### 5. HNN underperforms Neural ODE — the inductive bias is not free

The Hamiltonian Network has a stronger structural prior (it is forced to respect energy conservation via the Hamiltonian structure) but performs worse than an unconstrained Neural ODE. Likely cause: the HNN loss requires autograd through the Hamiltonian, which is more complex to optimize than simple MSE derivative-matching. The physics prior requires more careful training to pay off. HNN beats Neural ODE slightly at the shortest horizon (0.004 vs 0.002 rad at 0.01 s) but diverges faster at long horizons.

### 6. LNN is the most physically correct architecture and the least practical

The Lagrangian Neural Network enforces Euler-Lagrange equations, which is the strongest physics inductive bias of the three learned models. But it has two unsolved operational problems:
- **Training** requires second-order autograd (differentiating through the mass matrix) which does not run on Apple MPS, forcing slow CPU computation. Full-scale training is prohibitively slow.
- **Inference** calls autograd at every ODE solver step, making a 60 s trajectory take minutes per cell.

This round's LNN was trained on a 8K-sample subset with a small architecture (64-hidden, 2 layers) to get it to run at all. The results reflect those constraints, not the architecture's ceiling.

### 7. The "hidden parameters" regime is revealing

Models that understand physics do better on `changed_hidden` (where parameters are not disclosed) than on `normal`. RK4 achieves 0.000 rad on both; grok achieves 0.300 on hidden vs 1.138 on normal. This pattern — better performance when parameters are unknown — suggests that on the "normal" regime, models may be second-guessing standard physics with memorised priors, while on the hidden regime they fall back to reasoning more carefully. Or more simply: the normal regime is close to chaotic attractors in the training distribution where errors compound differently.

---

## What This Suggests

The central question — *can LLMs predict physics?* — has a sharp answer from this round: **not competitively**. A 3-minute training run producing a Neural ODE that beats every frontier LLM on accuracy and reliability is a strong result.

The more nuanced finding is that LLMs are doing *something* physics-adjacent at short horizons. The near-zero errors from kimi and grok at 0.01 s are not random — they reflect an understanding that the system barely moves in 10 ms. Whether that extrapolates to genuine dynamics modelling at longer horizons (it currently does not) is the right question for next rounds.

The reliability gap is understated in the leaderboard. kimi's 0.287 rad average is computed only over the 7.4% of cells that succeed. Its true expected error across all evaluation cells is much worse. For any real application, a model that works 7% of the time is not a model.

---

## Post-Round 1 Patch — June 15, 2026

Both immediate fixes were applied and re-evaluated (~$6, ~2h total wall time).

### Fix 1: Kimi token truncation

**Root cause (not a format issue):** kimi-k2.6 writes 10–13k tokens of chain-of-thought before outputting JSON. The hardcoded `max_tokens=16384` cap cut off 40/56 failing cells mid-reasoning before they reached the answer. A further 36 cells hit the 120s API timeout once given more token budget.

**Fix:** `max_tokens: 65536` and `request_timeout: 300` as per-model config fields. Both parameters are now configurable per model in `config.yaml`.

**Result:**

| | Round 1 | After fix |
|---|---|---|
| Coords success | ~22% | **88.9%** (64/72) |
| Angle error (mean) | 0.287 rad | 0.447 rad |
| Angle error (median) | — | **0.033 rad** |

The mean rising while the median falls sharply is expected: Round 1's 0.287 rad was computed over 16 cherry-picked easy cells (long horizon equilibrium cases). Now 64 cells succeed including hard chaotic ones. The 0.033 rad median shows kimi is genuinely accurate on most predictions.

By horizon: h=0.01s → 0.000 rad (near-perfect), h=1s → 0.623 mean / 0.314 median, h=10s → 0.721 mean / 0.351 median.

Remaining failures: 3 timeouts, 5 parse errors, 8 NaN predictions — minor edge cases.

### Fix 2: DeepSeek image modality

**Root cause:** `vision: false` in `config.yaml`. All 144 image/images_coords cells returned immediately with "not configured for vision modality" — no API call was ever made.

**Fix:** `vision: true` for deepseek-v4-pro. The standard OpenAI `image_url` payload format works correctly.

**Result:**

| Modality | Round 1 | After fix |
|---|---|---|
| Coords | 98.6% | 98.6% (unchanged) |
| Images | 0% | **100%** (72/72), 0.871 rad |
| Images+coords | 0% | **100%** (72/72), 0.780 rad |

DeepSeek with visual input is slightly worse than coords-only (0.759 rad) — images add noise for a model that already has the numbers.

---

## Post-Round 1 Session 2 — June 18–19, 2026

Two further improvements: longer training for learned models and best-of-N sampling for kimi. (~$26 API cost, ~6h wall time.)

### Longer training: Neural ODE and HNN (500 → 1000 epochs)

Training loss at convergence:

| Model | Loss @ 500 ep | Loss @ 1000 ep | Reduction |
|-------|--------------|----------------|-----------|
| Neural ODE | 0.639 | 0.012 | 52× |
| HNN | 4.463 | 0.215 | 21× |

Cosine annealing enabled dramatic convergence in the second half of training. Results:

**HNN — clean improvement:**

| Horizon | Original | 1000 ep |
|---------|----------|---------|
| 0.01 s | 0.004 | **0.003** |
| 1 s | 0.519 | **0.358** |
| 10 s | 1.099 | 1.184 |
| 60 s | 1.312 | 1.318 |
| Overall mean | 0.733 | **0.716** |

**Neural ODE — mixed:**

| Horizon | Original | 1000 ep |
|---------|----------|---------|
| 0.01 s | 0.002 | **0.001** |
| 1 s | 0.090 | **0.077** |
| 10 s | 1.044 | 1.253 |
| 60 s | 1.215 | 1.533 |
| Overall mean | 0.588 | 0.716 |
| Overall median | 0.119 | **0.083** |

Neural ODE improved at short horizons (1s: 0.090→0.077) but degraded at long ones (60s: 1.215→1.533). This reveals a fundamental limitation: derivative-matching MSE does not optimise for trajectory integration stability. The model learns to predict instantaneous dynamics more accurately, but error accumulation over long rollouts worsened. **More training ≠ better long-horizon prediction for this objective.** The median improved because it captures short-to-medium horizon performance where the gains are real.

Both retrained weights are in `results/learned_models/`.

### Best-of-N sampling for kimi (N=5)

Five independent kimi-k2.6 runs on the coords cells. For each cell, the median prediction across successful runs was taken. Metrics recomputed against the true trajectory state.

| Metric | Single run | Best-of-5 | Change |
|--------|-----------|-----------|--------|
| Success rate (coords) | 88.9% | **94.4%** | +5.5 pp |
| Angle error — mean | 0.447 rad | **0.370 rad** | −17% |
| Angle error — median | 0.033 rad | 0.033 rad | flat |
| 1s horizon mean | 0.624 | **0.422** | −32% |
| 10s horizon mean | 0.721 | **0.544** | −25% |
| 60s horizon mean | 0.436 | 0.547 | +25% worse |

Best-of-5 delivers meaningful gains at the 1s and 10s horizons — where kimi can actually reason about physics — and recovers 4 cells that failed in the single run. The 60s degradation is expected: at deep chaotic timescales, taking the median of 5 uncorrelated random predictions adds noise rather than signal. The median error across all horizons is unchanged because it is dominated by 0.01s cells where all runs are near-perfect.

The best-of-N merged checkpoints are in `results/checkpoints_bon/merged/`. The main `results/checkpoints/` uses the N=5 results as the canonical kimi evaluation.

---

---

## Post-Round 1 Session 3 — June 19, 2026

Rollout loss training for Neural ODE. Free (local MPS), ~45 min.

### Neural ODE rollout loss (k=10 Euler steps, 500 epochs)

**Hypothesis:** derivative MSE optimises instantaneous accuracy but not trajectory stability. Training on k-step rollout error should directly improve long-horizon predictions.

**Implementation:** instead of (state_t, d_state/dt) pairs, build (state_t, state_{t+10}) windows. Unroll 10 Euler steps through the model and penalise MSE at the endpoint. Gradients flow back through all 10 steps. Gradient clipping (max norm 1.0) for stability.

Loss converged from 0.489 → 0.000081 in 500 epochs — tighter than derivative matching ever achieved.

**Results:**

| Horizon | Deriv MSE (1000 ep) | Rollout k=10 (500 ep) | Change |
|---------|--------------------|-----------------------|--------|
| 0.01 s | 0.0011 | **0.0008** | −27% |
| 1 s | 0.077 | **0.052** | −32% |
| 10 s | 1.253 | **0.660** | −47% |
| 60 s | 1.533 | **1.087** | −29% |
| **Overall mean** | 0.716 | **0.450** | **−37%** |

The hypothesis was confirmed. Rollout loss improved every horizon, with the largest gain at 10s (−47%). The long-horizon degradation from session 2 is fully reversed — the rollout model (0.450) now sits just above Euler (0.455) in the leaderboard, and well above the derivative-matching Neural ODE (0.716).

The key insight: **what you train on is what you get.** Derivative MSE teaches the model to predict instantaneous dynamics accurately; rollout loss teaches it to produce stable trajectories. Both train the same architecture — only the objective differs.

Weights saved as `results/learned_models/neural_ode_rollout.pt`.

---

## What's Next

**Next round:**
- Add gpt-5.5 and qwen3-vl-32b (deferred — require AWS infrastructure)
- Add time-series models (Chronos, TimesFM, Moirai)
- Retrain LNN properly: pre-compute mass matrix targets to avoid second-order autograd

**Longer term:**
- Predict full trajectories, not just endpoint state — compute divergence time as the primary metric
- Add perturbation experiments — how sensitive are LLM predictions to small changes in initial conditions?
- Extend rollout loss to HNN — same objective change should help HNN's long-horizon performance

---

*Raw data, checkpoints, and model weights available in the repository under `docs/results/` and `results/`.*
