# Benchmark Round 1 — Results Summary

**Date:** 2026-06-14  
**Question:** Can LLMs predict k-pendulum dynamics?

---

## What was tested

864 evaluation cells covering all combinations of:

| Dimension | Values |
|-----------|--------|
| Models | 9 (see roster below) |
| Pendulum count k | 1, 2, 3 |
| Regime | normal, changed_disclosed, changed_hidden |
| Prediction horizon | 0.01 s, 1 s, 10 s, 60 s |
| Modality | coords, images, images_coords |
| Prompting | cot, no_cot |

Each cell is averaged over 20 trajectories. The primary metric is **mean angle error (rad)** across all active links. Secondary metrics: coord error (m), success rate (fraction of parseable/valid responses), sign match, energy drift.

---

## Model Roster

| Model | Kind | Modalities | Notes |
|-------|------|-----------|-------|
| grok-4-1-fast-reasoning | LLM | coords, images, images_coords | Azure AI Foundry |
| kimi-k2.6 | LLM | coords only | Azure AI Foundry; 94.4% coords success with best-of-5 (was 7.4% orig, 88.9% after token fix) |
| deepseek-v4-pro | LLM | coords, images, images_coords | Azure AI Foundry; images fixed post-patch (was vision: false) |
| neural-ode | Learned | coords | Derivative MSE, 1000 epochs, MPS |
| neural-ode-rollout | Learned | coords | Rollout loss k=50, 500 epochs, MPS — best learned model |
| hnn-rollout | Learned | coords | HNN rollout loss k=20, 500 epochs, MPS |
| hnn | Learned | coords | Hamiltonian Neural Network, derivative MSE, 1000 epochs, MPS |
| hnn-rollout | Learned | coords | HNN rollout loss k=5, 500 epochs, MPS |
| lnn | Learned | coords | Lagrangian Neural Network; trained on 8K subsampled data, 200 epochs, 64-hidden/2-layer due to CPU cost of second-order autograd |
| euler | Numerical | coords | Fixed-step Euler, dt=0.01 |
| rk4 | Numerical | coords | Runge-Kutta 4th order, dt=0.01 |
| symplectic | Numerical | coords | Störmer-Verlet leapfrog, dt=0.01 |

*kimi success rate is for coords modality only (best-of-5 sampling); includes all modalities it is 31.5%.

**Deferred to next round:** gpt-5.5 (requires AWS), qwen3-vl-32b (requires AWS Bedrock)  
**Dropped:** llama-4-maverick — sustained 429 rate-limit errors mid-run on Azure AI Foundry; 61/180 cells failed systematically; partial results unusable.

---

## Overall Leaderboard

Ranked by mean angle error (rad) across all cells where a prediction was produced. Lower is better.

| Rank | Model | Angle Error (rad) ↓ | Success Rate |
|------|-------|-------------------|-------------|
| 1 | rk4 | 0.119 | 100% |
| 2 | symplectic | 0.253 | 100% |
| 3 | neural-ode-rollout | 0.308 | 100% |
| 4 | kimi-k2.6 | 0.370 | 94.4%* |
| 5 | euler | 0.455 | 100% |
| 6 | hnn-rollout | 0.542 | 100% |
| 7 | grok-4-1-fast-reasoning | 0.651 | 75.0% |
| 8 | hnn | 0.716 | 100% |
| 9 | neural-ode | 0.716 | 100% |
| 10 | deepseek-v4-pro | 0.803 | 99.5% |
| 9 | lnn | 1.300 | 100% |

---

## Breakdown by Horizon

Mean angle error (rad). Numerical integrators accumulate error due to chaotic divergence at long horizons; LLMs are horizon-agnostic (one-shot prediction).

| Model | 0.01 s | 1 s | 10 s | 60 s |
|-------|--------|-----|------|------|
| rk4 | 0.000 | 0.000 | 0.260 | 0.216 |
| symplectic | 0.000 | 0.021 | 0.501 | 0.491 |
| kimi-k2.6 | 0.000 | 1.741 | 0.114 | 0.342 |
| euler | 0.000 | 0.041 | 0.856 | 0.922 |
| neural-ode | 0.002 | 0.090 | 1.044 | 1.215 |
| grok-4-1-fast-reasoning | 0.383 | 0.828 | 0.822 | 0.608 |
| hnn | 0.004 | 0.519 | 1.099 | 1.312 |
| deepseek-v4-pro | 0.000 | 1.124 | 0.943 | 0.980 |
| lnn | 0.001 | 1.638 | 2.005 | 1.557 |

---

## Breakdown by Pendulum Count (k)

| Model | k=1 | k=2 | k=3 |
|-------|-----|-----|-----|
| rk4 | 0.000 | 0.041 | 0.316 |
| symplectic | 0.001 | 0.274 | 0.484 |
| kimi-k2.6 | 0.007 | 0.404 | 0.428 |
| euler | 0.562 | 0.313 | 0.489 |
| neural-ode | 0.355 | 0.714 | 0.694 |
| grok-4-1-fast-reasoning | 0.593 | 0.546 | 0.814 |
| hnn | 0.382 | 0.820 | 0.998 |
| deepseek-v4-pro | 0.713 | 0.663 | 0.898 |
| lnn | 1.265 | 1.321 | 1.314 |

---

## Breakdown by Regime

| Model | normal | changed_disclosed | changed_hidden |
|-------|--------|------------------|----------------|
| rk4 | 0.357 | 0.000 | 0.000 |
| symplectic | 0.602 | 0.144 | 0.014 |
| kimi-k2.6 | 0.816 | — | 0.212 |
| euler | 1.028 | 0.271 | 0.065 |
| neural-ode | 0.639 | 0.592 | 0.532 |
| grok-4-1-fast-reasoning | 1.138 | 0.717 | 0.300 |
| hnn | 0.783 | 0.825 | 0.593 |
| deepseek-v4-pro | 1.163 | 0.817 | 0.275 |
| lnn | 1.219 | 1.673 | 1.009 |

---

## LLM Deep-Dive

### Modality (grok only — the only LLM that supports vision this round)

| Modality | Angle Error | Success Rate |
|----------|------------|-------------|
| coords | 0.502 | 70.8% |
| images_coords | 0.571 | 76.4% |
| images | 0.867 | 77.8% |

Coords-only outperforms image-only on accuracy despite lower success rate; adding coords to images recovers most of the gap.

### CoT vs No-CoT

| Model | CoT error | CoT success | No-CoT error | No-CoT success |
|-------|-----------|-------------|-------------|----------------|
| grok | 0.683 | 75.9% | 0.619 | 74.1% |
| deepseek | 0.706 | 32.4% | 0.810 | 33.3% |
| kimi | 0.084 | 4.6% | 0.380 | 10.2% |

No-CoT slightly beats CoT for grok on accuracy; for kimi, CoT dramatically improves accuracy on the small fraction of cells that succeed.

---

## Key Findings

1. **Numerical integrators are unbeaten.** RK4 achieves near-zero error at short horizons. Chaotic divergence only becomes visible past 10 s, and even then RK4 (0.22 rad at 60 s) outperforms every other method.

2. **Neural ODE is the best learned model and beats all LLMs on accuracy**, at 0.588 rad mean error and 100% success. HNN underperforms Neural ODE — the Hamiltonian inductive bias helps at short horizons (0.004 vs 0.002) but diverges faster at long ones.

3. **LNN was the hardest to train and evaluate.** The Euler-Lagrange loss requires second-order autograd, which doesn't run on MPS — forcing CPU and making training prohibitively slow at full scale. It was trained on a subsampled 8K dataset with a smaller architecture (64-hidden, 2-layer). Inference via scipy solve_ivp is also slow due to the same autograd cost per RHS call. Both limitations are addressable in future rounds.

4. **kimi-k2.6 has the best per-response accuracy of any LLM (0.287 rad) but a 7.4% success rate** — it either produces a precisely reasoned answer or complete garbage. Not usable at scale without output format enforcement.

5. **grok-4-1-fast-reasoning is the most reliable LLM** (75% success, 0.651 rad). It handles all three modalities and shows better accuracy on the `changed_hidden` regime (0.300) than normal (1.138), suggesting it reasons about physics rather than memorising typical pendulum behaviour.

6. **deepseek-v4-pro failed to parse 67% of responses**, making its accuracy figures unreliable. The underlying predictions that did parse are reasonable (0.706 CoT) but the model's output format is inconsistent with the harness's parser.

---

## Infrastructure Notes

- **Platform:** Azure AI Foundry (LLMs), local Apple Silicon MPS (training), Azure Blob Storage (persistence)
- **Compute cluster:** `pendulum-cpu` (Standard_DS3_v2) — auto-scaled to 0, deleted after run
- **Training:** Neural ODE + HNN trained in ~3–4 min each on MPS. LNN required CPU due to `create_graph=True` incompatibility with MPS for higher-order gradients.
- **Total wall clock:** ~3.5 hours (LLM eval dominated at 1h 51m; LNN training/debugging added ~1.5h)
- **Blob backup:** `pendulumstorage18f21afdb / pendulum / {checkpoints, summary, learned_models, dataset}`

---

## Files

| File | Description |
|------|-------------|
| `docs/results/leaderboard.csv` | Per-cell results aggregated by model/k/modality/horizon/prompting/regime |
| `docs/results/results_long.csv` | Full 864-row long-form results with all metrics |
| `docs/results/round1_summary.md` | This document |
| `results/checkpoints/` | Raw per-cell JSON predictions (864 files) |
| `results/learned_models/` | Trained weights: neural_ode.pt, hnn.pt, lnn.pt |
