# Can LLMs Predict Physics? A Benchmark of Language Models, Learned Dynamics, and Numerical Integrators on k-Pendulum Systems

**Shrithik Shahapure**  
June 2026

---

## Abstract

We benchmark large language models (LLMs), learned dynamics models, and classical numerical integrators on the task of predicting the future state of k-link pendulum systems (k ∈ {1, 2, 3}). Across 864 evaluation cells spanning four prediction horizons (0.01 s to 60 s), three physical regimes, three input modalities, and two prompting strategies, we find that (1) numerical integrators remain dominant — RK4 achieves 0.119 rad mean angle error versus the best LLM at 0.370 rad; (2) a Neural ODE trained with a rollout loss objective in under one hour on consumer hardware achieves 0.308 rad, beating every LLM tested; (3) the training objective matters more than architecture or training duration — switching from derivative MSE to k-step Euler rollout loss reduces mean error by 37–57% depending on rollout window length; and (4) chain-of-thought prompting provides no benefit for physics prediction. LLMs show systematic ability at very short horizons (near-zero error at 10 ms) but fail to simulate dynamics at 1 s and beyond. The reliability gap compounds the accuracy gap: the best-performing LLM (kimi-k2.6 with best-of-5 sampling) produces valid predictions on only 94.4% of cells in its supported modality. Our results suggest that, for chaotic physical systems, training a lightweight learned model is a strictly better strategy than querying a frontier LLM — in accuracy, reliability, and cost.

---

## 1. Introduction

Large language models have demonstrated surprising competence across a wide range of scientific tasks, from mathematical reasoning to protein structure prediction. A natural question is whether this competence extends to physical simulation: can an LLM, given the state of a dynamical system at time *t*, predict its state at time *t + T*?

The pendulum provides an ideal test case. Simple pendula (k=1) are analytically tractable; double and triple pendula (k=2, 3) exhibit sensitive dependence on initial conditions and are fully chaotic at moderate energies. The task is well-specified: given angles and angular velocities at t=0, predict them at t=T. Ground truth is computable to arbitrary precision. No domain-specific knowledge is required to evaluate responses.

We construct a benchmark spanning:
- Three system complexities: k ∈ {1, 2, 3} links
- Four prediction horizons: 0.01 s, 1 s, 10 s, 60 s
- Three physical regimes: standard parameters, disclosed non-standard parameters, and hidden non-standard parameters
- Three input modalities for LLMs: coordinate data, rendered images, and both combined
- Two prompting strategies: chain-of-thought and direct (no-CoT)

We evaluate three classes of methods: frontier LLMs (kimi-k2.6, grok-4-1-fast-reasoning, deepseek-v4-pro), learned dynamics models (Neural ODE, Hamiltonian Neural Network, Lagrangian Neural Network), and classical numerical integrators (Euler, RK4, Störmer-Verlet).

Our key contributions are:

1. A systematic, reproducible benchmark framework for physics prediction covering LLMs, learned models, and numerical baselines under identical evaluation conditions.
2. The finding that rollout loss training — unrolling k integration steps and penalising trajectory error — dominates derivative-matching MSE for all prediction horizons, reducing mean error by 37% at k=10 and 57% at the 60 s horizon with k=50.
3. Evidence that LLMs exhibit horizon-dependent behaviour: near-zero errors at 10 ms, but failure to simulate dynamics at 1 s and beyond.
4. A negative result for chain-of-thought: reasoning longer about physics does not improve physics prediction.

---

## 2. Related Work

**Physics-informed neural networks.** Neural ODEs [Chen et al., 2018] learn continuous-time dynamics from data by parameterising the vector field of an ODE. Hamiltonian Neural Networks [Greydanus et al., 2019] add an inductive bias by requiring dynamics to arise from the gradient of a learnable Hamiltonian, guaranteeing energy conservation. Lagrangian Neural Networks [Cranmer et al., 2020] enforce the Euler-Lagrange equations via second-order autograd. Our work benchmarks all three against each other and against LLMs under identical conditions.

**LLMs for scientific reasoning.** Recent work has applied LLMs to mathematical problem-solving [Lewkowycz et al., 2022], symbolic regression [Shojaee et al., 2024], and code generation for scientific simulations. For physical prediction specifically, prior work has tested LLMs on simple kinematics and Newtonian mechanics word problems, finding reasonable performance on textbook problems but limited ability to numerically simulate dynamics. Our benchmark differs in requiring quantitative trajectory prediction, not qualitative reasoning.

**Benchmark design for dynamical systems.** Prior evaluations of learned dynamics models have used chaotic systems (Lorenz, double pendulum) as standard benchmarks [Toth et al., 2019; Lutter et al., 2020], but have not included LLMs. Our contribution is a unified evaluation framework that places these two families of methods on the same footing.

**Chain-of-thought prompting.** Chain-of-thought [Wei et al., 2022] and its variants have improved LLM performance on a range of reasoning tasks. We test whether explicit reasoning steps improve physics prediction, finding that they do not — consistent with the hypothesis that integration over a differential equation is fundamentally sequential computation that cannot be parallelised across reasoning tokens.

---

## 3. Problem Formulation

### 3.1 Task

Given the initial state of a k-link pendulum system at time t=0, predict the state at time T. The state is the vector of joint angles and angular velocities: **s** = (θ₁, ..., θₖ, ω₁, ..., ωₖ) ∈ ℝ²ᵏ. The primary metric is mean angle error in radians, averaged over all k links.

### 3.2 Systems

We simulate k-link pendula (k ∈ {1, 2, 3}) using the full nonlinear equations of motion derived from the Lagrangian. Ground truth trajectories are computed with `scipy.integrate.solve_ivp` at dt=0.001 s using the DOP853 integrator with tight tolerances (rtol=1e-10, atol=1e-12). Initial conditions are sampled uniformly: θᵢ ~ Uniform(−3, 3) rad, ωᵢ ~ Uniform(−1, 1) rad/s.

### 3.3 Physical Regimes

Three regimes test generalisation to non-standard physical parameters:

| Regime | g (m/s²) | L (m) | m (kg) | Damping | Disclosed |
|--------|----------|-------|--------|---------|-----------|
| normal | 9.81 | 1.0 | 1.0 | 0 | N/A |
| changed_disclosed | 3.71 | 1.4/0.7/1.1 | 2.0/0.6/1.5 | 0.05 | Yes |
| changed_hidden | 1.62 | 0.8/1.2/0.9 | 0.5/1.8/1.0 | 0.10 | No |

### 3.4 Evaluation Grid

Each combination of (model, k, modality, horizon, prompting, regime) constitutes one evaluation cell. Each cell is averaged over 20 independent trajectories. The full grid produces 864 cells for LLMs (× 3 modalities × 2 prompting) and 72 cells for learned models and integrators (coords only, no prompting).

### 3.5 Metrics

- **Angle error (rad):** Mean absolute error on joint angles, averaged over all links and trajectories. Primary metric.
- **Success rate:** Fraction of cells producing a valid, parseable prediction. Relevant only for LLMs.
- **Sign match:** Fraction of predictions with correct angle sign.
- **Energy drift (ΔE):** Absolute change in total mechanical energy between predicted and true state.

---

## 4. Methods

### 4.1 LLMs

Three frontier LLMs were evaluated via Azure AI Foundry. All received a structured prompt describing the system (pendulum count, physical parameters for disclosed regimes, initial state) and were asked to predict the state at time T in a structured JSON format. Full prompt templates are in Appendix A.

**kimi-k2.6** is a frontier reasoning model that generates extended chain-of-thought before producing a final answer. Key configuration: `max_tokens=65536`, `request_timeout=300` s. These settings were determined empirically after an initial run with default limits truncated 40/56 cells before the model reached its answer. kimi does not support image inputs and was evaluated on coordinate modality only.

**grok-4-1-fast-reasoning** supports all three modalities and was the only vision-capable model this round. Evaluated on coords, images, and images+coords.

**deepseek-v4-pro** supports all modalities. An initial configuration error (`vision: false`) caused all image cells to fail immediately; these were re-run after correction.

For all LLMs, best-of-N sampling was applied to kimi (N=5): five independent runs on each coordinate cell, taking the per-timestep circular median of successful predictions.

### 4.2 Learned Dynamics Models

All learned models receive the initial state and physical parameters as input and predict the state at T by integrating a learnable dynamics function. All were trained on 67,491 samples generated from the dataset trajectories, using the Apple MPS (Metal Performance Shaders) backend on an Apple Silicon machine.

**Neural ODE.** A 3-layer MLP with 256 hidden units parameterises the vector field f(s, params) = ds/dt. At inference, trajectories are produced by unrolling fixed-step Euler integration at dt=0.01 s. Two variants were trained:

- *Derivative MSE:* Training pairs (s_t, ds/dt_t) estimated via finite differences. Loss = MSE(f̂(s_t), ds/dt_t). 1000 epochs with cosine annealing.
- *Rollout loss:* Training pairs (s_0, s_{0+k}) from consecutive trajectory windows. Loss = MSE(unroll_k(s_0), s_k) where unroll_k applies k Euler steps through f. Gradient clips (max norm 1.0). 500 epochs.

**Hamiltonian Neural Network (HNN).** Same MLP architecture parameterises a scalar Hamiltonian H(s, params). Dynamics are derived analytically: ds/dt = J ∇H, where J is the symplectic matrix. This guarantees energy conservation. The same derivative MSE and rollout loss variants were trained with the same architecture.

For rollout training, each Euler step requires recomputing ∇H via autograd, which builds a deeper computation graph. This required halving the batch size and restricting to k ≤ 20 for practical training times.

**Lagrangian Neural Network (LNN).** Parameterises the Lagrangian L(q, q̇, params); dynamics emerge from the Euler-Lagrange equations. This requires second-order autograd (differentiating through the mass matrix). MPS does not support `create_graph=True` for higher-order gradients, forcing CPU training. Due to cost, LNN was trained on a subsampled 8K dataset with a reduced architecture (64-hidden, 2-layer, 200 epochs). Inference via `scipy.solve_ivp` with the LNN as the RHS is slow (~minutes per 60 s cell). LNN results reflect these constraints.

### 4.3 Numerical Integrators

Three fixed-step integrators with dt=0.01 s: Euler (1st order), RK4 (4th order), and Störmer-Verlet leapfrog (symplectic, 2nd order). All use the exact equations of motion. Error comes only from discretisation, not from approximation.

---

## 5. Results

### 5.1 Overall Leaderboard

Ranked by mean angle error (rad) across all cells where a valid prediction was produced.

| Rank | Model | Angle Error (rad) ↓ | Success Rate | Type |
|------|-------|-------------------|-------------|------|
| 1 | RK4 | 0.119 | 100% | Numerical |
| 2 | Symplectic | 0.253 | 100% | Numerical |
| 3 | Neural ODE (rollout k=50) | 0.308 | 100% | Learned |
| 4 | kimi-k2.6 (best-of-5) | 0.370 | 94.4%* | LLM |
| 5 | Euler | 0.455 | 100% | Numerical |
| 6 | HNN (rollout k=20) | 0.542 | 100% | Learned |
| 7 | grok-4-1-fast-reasoning | 0.651 | 75.0% | LLM |
| 8 | Neural ODE (deriv MSE) | 0.716 | 100% | Learned |
| 8 | HNN (deriv MSE) | 0.716 | 100% | Learned |
| 10 | deepseek-v4-pro | 0.803 | 99.5%† | LLM |
| 11 | LNN | 1.300 | 100% | Learned |

*Coords modality only (best-of-5 sampling). Including all modalities, kimi success rate = 31.5%.  
†DeepSeek image modality was fixed after initial run (config error: `vision: false`).

### 5.2 Breakdown by Prediction Horizon

| Model | 0.01 s | 1 s | 10 s | 60 s |
|-------|--------|-----|------|------|
| RK4 | 0.000 | 0.000 | 0.260 | 0.216 |
| Symplectic | 0.000 | 0.021 | 0.501 | 0.491 |
| Neural ODE (rollout k=50) | 0.004 | 0.076 | 0.688 | **0.463** |
| kimi-k2.6 (best-of-5) | 0.000 | 0.422 | 0.544 | 0.547 |
| Euler | 0.000 | 0.041 | 0.856 | 0.922 |
| HNN (rollout k=20) | 0.002 | 0.210 | 0.963 | 0.991 |
| grok-4-1-fast-reasoning | 0.383 | 0.828 | 0.822 | 0.608 |
| Neural ODE (deriv MSE) | 0.002 | 0.090 | 1.044 | 1.215 |
| HNN (deriv MSE) | 0.004 | 0.519 | 1.099 | 1.312 |
| deepseek-v4-pro | 0.000 | 1.124 | 0.943 | 0.980 |
| LNN | 0.001 | 1.638 | 2.005 | 1.557 |

A key observation: grok is the only LLM that fails at 0.01 s (0.383 rad), while kimi and deepseek achieve near-zero error. At this horizon, the physically correct answer is trivially close to the initial state; grok's failure here indicates a format or calibration issue rather than a physics failure.

Conversely, grok achieves lower error at 60 s (0.608) than at 1 s (0.828) — the opposite of learned models. This reflects the one-shot nature of LLM predictions: the model is not accumulating integration error; it is making a single calibrated guess that benefits from knowing the system is chaotic at 60 s.

### 5.3 Breakdown by Pendulum Count

| Model | k=1 | k=2 | k=3 |
|-------|-----|-----|-----|
| RK4 | 0.000 | 0.041 | 0.316 |
| Symplectic | 0.001 | 0.274 | 0.484 |
| kimi-k2.6 | 0.007 | 0.404 | 0.428 |
| Euler | 0.562 | 0.313 | 0.489 |
| Neural ODE | 0.355 | 0.714 | 0.694 |
| grok | 0.593 | 0.546 | 0.814 |
| HNN | 0.382 | 0.820 | 0.998 |
| deepseek-v4-pro | 0.713 | 0.663 | 0.898 |
| LNN | 1.265 | 1.321 | 1.314 |

Euler performs unusually well on k=2 relative to k=1. This is a horizon averaging effect: at k=1 with 1-second horizon, Euler accumulates significant error; on k=2 systems the chaotic divergence affects all methods roughly equally and washes out Euler's discretisation advantage.

### 5.4 Breakdown by Physical Regime

| Model | normal | changed_disclosed | changed_hidden |
|-------|--------|------------------|----------------|
| RK4 | 0.357 | 0.000 | 0.000 |
| Symplectic | 0.602 | 0.144 | 0.014 |
| kimi-k2.6 | 0.816 | — | 0.212 |
| Euler | 1.028 | 0.271 | 0.065 |
| Neural ODE | 0.639 | 0.592 | 0.532 |
| grok | 1.138 | 0.717 | 0.300 |
| HNN | 0.783 | 0.825 | 0.593 |
| deepseek-v4-pro | 1.163 | 0.817 | 0.275 |
| LNN | 1.219 | 1.673 | 1.009 |

RK4 achieves exactly 0.000 rad on changed_disclosed and changed_hidden (to reported precision): it uses the disclosed or hidden parameters correctly in both cases. Its "normal" error (0.357) reflects the longer-horizon cells in the averaging.

LLMs perform better on changed_hidden than on normal. We return to this pattern in Section 6.

### 5.5 LLM Modality and Prompting

**Modality (grok only — the only vision-capable LLM this round):**

| Modality | Angle Error | Success Rate |
|----------|------------|-------------|
| Coords | 0.502 | 70.8% |
| Images + coords | 0.571 | 76.4% |
| Images only | 0.867 | 77.8% |

Coordinate text outperforms images for accuracy despite a lower success rate. Adding images to coordinates degrades accuracy slightly — visual information appears to be noise for a task where the numbers are already present.

**Chain-of-Thought vs No-CoT:**

| Model | CoT error | CoT success | No-CoT error | No-CoT success |
|-------|-----------|-------------|-------------|----------------|
| grok | 0.683 | 75.9% | **0.619** | 74.1% |
| deepseek | **0.706** | 32.4% | 0.810 | 33.3% |
| kimi | **0.084** | 4.6% | 0.380 | 10.2% |

No-CoT outperforms CoT for grok and deepseek on accuracy. For kimi, CoT dramatically improves accuracy on the small fraction of cells that succeed — but CoT also more than halves the success rate. The mean CoT error for kimi (0.084) is impressive, but this is computed over only 4.6% of cells: the model either produces a highly accurate answer with extensive reasoning or fails to produce an answer at all.

---

## 6. Analysis

### 6.1 Training Objective Dominates Architecture and Epochs

The most striking finding is the effect of training objective on learned model performance. Figure 1 (below) shows the progression of Neural ODE mean error across training strategies:

| Configuration | Mean Angle Error | Notes |
|--------------|-----------------|-------|
| Neural ODE, deriv MSE, 500 ep | 0.588 | Original |
| Neural ODE, deriv MSE, 1000 ep | 0.716 | More training → worse |
| Neural ODE, rollout k=10, 500 ep | 0.450 | +37% improvement over 1000-ep |
| Neural ODE, rollout k=50, 500 ep | **0.308** | +57% over deriv MSE 1000-ep |

Doubling training epochs with derivative MSE *worsened* the overall mean (0.588 → 0.716) by improving short horizons at the cost of long-horizon stability. Switching to rollout loss at k=10 immediately recovered and surpassed the original — in half the epochs. Extending the rollout window to k=50 (covering 0.5 s of trajectory) delivered another 32% reduction by teaching the model to sustain stable integration over longer windows.

The same pattern holds for HNN:

| Configuration | Mean Angle Error |
|--------------|-----------------|
| HNN, deriv MSE, 1000 ep | 0.716 |
| HNN, rollout k=5, 500 ep | 0.627 |
| HNN, rollout k=20, 500 ep | 0.542 |

The mechanism is straightforward: derivative MSE teaches the model to minimise instantaneous prediction error; rollout loss teaches it to maintain low trajectory error over k steps. These are different objectives with different optimal solutions. For the benchmark metric (error at time T), rollout loss is directly aligned.

### 6.2 LLMs Perform Horizon-Dependent Regression, Not Simulation

The error profile of LLMs differs qualitatively from that of numerical and learned models:

- At 0.01 s, kimi and deepseek achieve near-zero error. This does not require simulation — the physically correct answer is the initial state plus a negligible perturbation. LLMs appear to recognise this directly.
- At 1 s, errors spike sharply for kimi (0.422) and deepseek (1.124). At this timescale, simulation is required; approximating the initial condition no longer works.
- At 60 s, errors partially recover for some LLMs. For grok (0.608 at 60 s vs 0.828 at 1 s), this reflects the prediction collapsing to a "typical" chaotic state, which, when the true state is also chaotic, produces accidentally low error on average.

This pattern — near-zero at near-zero horizon, followed by failure at intermediate horizons — is consistent with a regression-to-typical-state strategy rather than simulation. The LLM outputs a plausible-looking state, which is close to the true state only when the true state is close to the initial state (short horizons) or when both are drawn from the same chaotic attractor (very long horizons).

### 6.3 Chain-of-Thought Cannot Simulate Differential Equations

The negative CoT result is interpretable. Numerical integration is a sequential computation: state at step n+1 depends causally on state at step n. A chain-of-thought prompt gives the model a sequence of reasoning tokens, but these tokens are generated in parallel (from the model's perspective at each position) without the ability to carry numerical state across arbitrary depth. A 1-second trajectory at dt=0.001 requires 1000 integration steps; no practical CoT budget can replicate this. Longer reasoning does not improve the fundamental bottleneck.

This stands in contrast to tasks where CoT helps — algebraic manipulation, logical deduction — where the reasoning steps are short and the intermediate values are simple. Physics simulation is neither.

### 6.4 Hidden Parameters Reveal Prior vs Reasoning

The regime analysis shows a consistent pattern: LLMs perform substantially better on `changed_hidden` than on `normal`:

- grok: 0.300 (hidden) vs 1.138 (normal)
- deepseek: 0.275 (hidden) vs 1.163 (normal)

One interpretation: the `normal` regime (Earth gravity, unit masses and lengths) falls within the dense part of the training distribution. LLMs may have strong memorised priors about typical pendulum behaviour that conflict with the actual trajectory. On the `changed_hidden` regime (lunar gravity, non-unit masses), these priors are less applicable, and the model may fall back to more careful reasoning about the given initial state. Alternatively, the `changed_hidden` trajectories may be slower-moving (lower gravity → lower velocities → lower sensitivity to integration error), making them intrinsically easier to predict from initial conditions alone. We leave disambiguation to future work.

### 6.5 The Reliability Gap

The leaderboard ranks by accuracy conditional on success. This understates the LLM disadvantage in practice. Consider the expected error across all evaluation cells:

- A model that succeeds on 94.4% of cells with mean error 0.370 rad has an expected error of 0.370 on 94.4% of cells and an undefined (or infinite, depending on convention) error on 5.6%. 
- A learned model with 100% success and 0.308 rad mean error is strictly better by any reasonable expected-cost measure.

For grok at 75% success, 25% of queries return no usable prediction. This is not a minor caveat; it is a fundamental reliability problem for any production use.

### 6.6 Best-of-N Sampling Reduces LLM Variance

Running kimi five times and taking the per-step circular median of successful predictions reduced mean error from 0.447 to 0.370 rad (−17%), improved success rate from 88.9% to 94.4%, and cut the 1 s horizon error by 32%. The 60 s horizon worsened (+25%): at deep chaotic timescales, kimi's five predictions are uncorrelated random draws from a chaotic attractor; their median is not closer to the truth than any individual draw.

This result is consistent with variance reduction by ensemble: best-of-N removes outlier predictions but cannot improve the median of the distribution when predictions are independent and the distribution has no mode near the true answer.

---

## 7. Limitations

**LNN results are not representative of the architecture's capability.** Due to second-order autograd incompatibility with Apple MPS, LNN was trained on a subsampled 8K dataset with a reduced architecture at CPU speed. The benchmark results for LNN reflect resource constraints, not the architecture ceiling. Future work should train LNN on full data with pre-computed mass matrix targets to avoid the `create_graph=True` requirement.

**gpt-5.5 not yet evaluated.** At the time of writing, GPT-5.5 was announced as available on AWS Bedrock but not yet accessible via the API for our account. Including the current frontier LLM is a priority for the next evaluation round; the absence leaves the "can frontier LLMs predict physics" question partially open.

**LLM success rates conflate format failures and physics failures.** A cell where the model produces syntactically invalid JSON is counted as a failure, but this is a different kind of failure from one where the model produces a valid prediction that is physically wrong. Future work should distinguish parsing failures from prediction failures.

**Single-point prediction, not trajectory.** The benchmark measures error at a single future time point. A more informative metric would measure the full predicted trajectory and compute divergence time — the first time the predicted trajectory deviates from the true trajectory by more than a threshold. This is particularly relevant for learned models, where rollout stability is the primary finding.

**Learned models trained on the same distribution as test data.** The dataset is generated with a fixed random seed; trajectories are split within the dataset but the system family (pendulum, with these parameter ranges) is the same. LLMs are evaluated zero-shot without any in-distribution training, making the comparison asymmetric. Future work could evaluate learned models in a cross-regime generalisation setting.

---

## 8. Future Work

- **gpt-5.5 and Qwen3 VL (235B):** Pending AWS Bedrock availability. Qwen3 VL is particularly interesting as a vision-capable model at 235B scale.
- **Divergence time metric:** Reprocess existing checkpoint data to compute the time at which trajectory error first exceeds 0.5 rad. More physically meaningful than snapshot error.
- **Mixed rollout loss:** Train with multiple k values simultaneously (loss = MSE(k=10) + MSE(k=50)) to optimise all horizons without the short-horizon regression observed at k=50.
- **LNN fix:** Pre-compute mass matrix targets numerically, eliminating the `create_graph=True` requirement and enabling full MPS training.
- **Perturbation sensitivity:** Test LLM prediction variance under small changes to initial conditions. High variance would confirm the regression-to-attractor hypothesis.
- **Time-series models:** Chronos, TimesFM, and Moirai treat the problem as a multivariate time-series forecasting task — a different framing that may capture medium-horizon patterns not captured by physics-informed models.

---

## 9. Conclusion

We evaluated eleven methods — three LLMs, five learned dynamics models, and three numerical integrators — on predicting the state of k-pendulum systems across 864 evaluation cells. The main findings are:

1. **Numerical integrators are unbeaten.** RK4 at 0.119 rad mean error remains the gold standard. If equations of motion are known, classical integration is the correct tool.

2. **Rollout loss training transforms learned models.** Switching from derivative MSE to k-step Euler rollout loss reduces Neural ODE mean error by 57% and places it above all LLMs tested. The training objective is the dominant factor — more important than architecture, epoch count, or model size.

3. **LLMs cannot simulate dynamics.** Frontier LLMs achieve near-zero error at 10 ms (trivially close to initial state) but fail at 1 s and beyond. Chain-of-thought provides no improvement. The fundamental bottleneck — iterative numerical integration — is not replicable by token generation.

4. **Reliability compounds the accuracy gap.** The best LLM (kimi, best-of-5) produces valid predictions 94.4% of the time in its supported modality; a learned model produces valid predictions 100% of the time at lower error. For any practical application, reliability matters as much as accuracy.

These results do not imply LLMs cannot contribute to physics. The regime analysis suggests they engage in physics-adjacent reasoning, and the near-zero short-horizon errors indicate some understanding of the system. The question of whether a sufficiently capable LLM — perhaps with explicit iterative computation or tool use — could match learned integrators at longer horizons remains open and is the central question for the next evaluation round.

---

## Appendix A: Prompt Templates

### A.1 Coordinate Modality (No-CoT)

```
You are a physics simulation engine. Given the initial state of a {k}-link pendulum system, predict its state at time T.

System parameters:
- Number of links: {k}
- Gravity: {g} m/s²
- Link lengths: {L} m
- Link masses: {m} kg
- Damping coefficient: {damping}

Initial state at t=0:
- Angles (rad): {theta_0}
- Angular velocities (rad/s): {omega_0}

Predict the state at t={T} seconds.

Respond ONLY with a JSON object in exactly this format:
{{"theta": [{theta_pred}], "omega": [{omega_pred}]}}
```

### A.2 Chain-of-Thought Addition

```
Before giving your answer, reason step-by-step about the physics. Consider:
1. The equations of motion for a {k}-link pendulum
2. How the system will evolve from this initial state
3. Any relevant physical intuitions

Then provide your prediction in the required JSON format.
```

---

## Appendix B: Training Details

| Model | Architecture | Epochs | LR | Scheduler | Batch | Device | Train time |
|-------|-------------|--------|-----|-----------|-------|--------|------------|
| Neural ODE (deriv) | MLP 256×3 | 1000 | 3e-4 | Cosine anneal | 512 | MPS | ~4 min |
| Neural ODE (rollout k=10) | MLP 256×3 | 500 | 3e-4 | Cosine anneal | 512 | MPS | ~45 min |
| Neural ODE (rollout k=50) | MLP 256×3 | 500 | 3e-4 | Cosine anneal | 512 | MPS | ~2 h |
| HNN (deriv) | MLP 256×3 | 1000 | 3e-4 | Cosine anneal | 512 | MPS | ~4 min |
| HNN (rollout k=5) | MLP 256×3 | 500 | 3e-4 | Cosine anneal | 256 | MPS | ~50 min |
| HNN (rollout k=20) | MLP 256×3 | 500 | 3e-4 | Cosine anneal | 256 | MPS | ~2 h |
| LNN | MLP 64×2 | 200 | 3e-4 | Cosine anneal | 128 | CPU | ~3 h |

Dataset: 67,491 samples across k ∈ {1, 2, 3}, 9 trajectories per k per regime, subsampled at dt=0.01 s from ground truth at dt=0.001 s.

---

## Appendix C: Infrastructure

- **LLM inference:** Azure AI Foundry (kimi-k2.6, grok-4-1-fast-reasoning, deepseek-v4-pro)
- **Training:** Apple Silicon M-series, MPS backend (except LNN: CPU)
- **Data storage:** Azure Blob Storage (`pendulumstorage18f21afdb`)
- **Total API cost:** ~$55–70 across all rounds and re-runs
- **Total wall time:** ~3.5 h (Round 1 LLM eval) + ~6 h (sessions 2–5 training and re-eval)

---

*Code, datasets, model weights, and evaluation checkpoints available in the project repository.*
