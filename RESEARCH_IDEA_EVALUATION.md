# Neutral Evaluation — "Can LLMs Predict Physics?" (Double Pendulum Benchmark)

*Evaluation of the research idea itself (not the code). Date: 2026-06-12.*

## What the idea actually is

A controlled benchmark asking whether frontier LLMs, VLMs, time-series foundation
models, and learned-dynamics models can forecast the future state of k-pendulum
systems, with numerical integrators as both ground-truth oracle and baseline. The
design is a clean factorial sweep:

> model family × k ∈ {1,2,3} × modality (coords / image / both) × horizon (10ms → 60s)
> × constant-regime (normal / disclosed / hidden) × prompting (CoT / no-CoT).

**Framing issue (named up front):** the title asks "Can LLMs predict physics?" but
what the harness *measures* is in-context numerical integration and system
identification from a given initial state. Those are narrower, more specific
capabilities than "understanding physics." That gap between the headline question and
the operationalization is the single biggest thing to fix, and it cascades into the
metric and novelty issues below.

## Novelty: incremental, not new territory

The space is already populated on every axis this project touches:

- **Chaotic-system forecasting with foundation models** is active. "Zero-shot
  forecasting of chaotic systems" (arXiv 2409.15771) already evaluates Chronos-style
  models on chaotic attractors — directly overlapping the time-series slot.
  Multi-pendulum prediction with LSTM/GRU/VRNN was published in 2025 (arXiv 2504.13453).
- **LLM physics reasoning / world-model probing** has multiple 2025–2026 benchmarks:
  UGPhysics, PhysicsMind (sim+real mechanics for VLMs and video world models),
  FeynmanBench.
- **HNN / LNN / Neural-ODE on pendulums** is canonical benchmark material
  (Greydanus 2019, Cranmer 2020).
- **LLMs as zero-shot forecasters** (Gruver et al., NeurIPS 2023) is the conceptual
  ancestor of the coords-modality LLM test.

What is *not* already done, as far as I can see, is the specific **unification**:
putting all these model families on one controlled mechanical substrate with identical
inputs and metrics. That cross-family head-to-head, plus the modality and prompting
axes, is a legitimate but modest contribution — workshop / empirical-study tier, not a
flagship novel idea.

## The central scientific problem: chaos vs. your metrics

This is the most serious issue, and it's a validity problem, not a polish problem.

The double and triple pendulum are chaotic: positive Lyapunov exponent, so *any*
perturbation grows exponentially. Past the Lyapunov time (the spec explicitly puts the
10s and 60s horizons "well past" it), **pointwise state prediction is impossible in
principle** — even the true dynamics with a floating-point-sized perturbation diverges
to a decorrelated state. So metrics 1–4 (coordinate MSE, angle error, ω sign/magnitude)
at 10s and 60s will saturate to "essentially random" for *every* model, including a
near-perfect integrator started from a slightly rounded state. Those long-horizon cells
can't discriminate models; they measure the attractor's mixing time, not model skill.

Consequences:

- The experiment only really has discriminating power at **10ms and ~1s** — and at
  10ms the task degenerates to "apply one integration step," which is arithmetic, not
  physics insight. The smoke results confirm this: even Euler is near-perfect at 10ms;
  the spread only opens at 1s.
- The genuinely *correct* metrics for chaos are the ones currently underweighted:
  **time-to-divergence / predictability horizon** (metric 8) and
  **distributional / attractor-climate** measures (does the long-run trajectory have
  the right energy distribution, the right attractor geometry?). Pointwise long-horizon
  error is the wrong tool. The benchmark would be much stronger if it pivoted its
  headline metric to predictability horizon and added a distributional comparison, and
  explicitly stopped reporting pointwise MSE beyond ~1 Lyapunov time.

## The comparison is partly foregone

The numerical integrator is the data-generating process, so RK4 winning is guaranteed
and uninformative — it's grading the answer key against itself. The interesting science
is in the *gaps and the off-diagonal cells*, not the leaderboard ranking. Specifically:

- **Changed-hidden regime** is the best idea in the whole spec. Inferring unstated
  constants (g, L, m, damping) from a short trajectory is genuine in-context system
  identification — a real, under-measured capability where the answer isn't obvious.
  Make this the centerpiece rather than one cell among many.
- **Modality** (can a VLM read state off a rendered frame precisely enough to
  integrate?) is interesting but near-foregone: pixel-estimated angles can't reach RK4
  precision, so image-only will look terrible. Fine *if* framed as "vision-based state
  estimation," not as a fair forecasting contest.
- **CoT vs no-CoT on numeric integration** is a clean, publishable sub-question on its
  own.

## Other threats to validity

- **Contamination:** pendulum equations of motion are in every training corpus. A model
  reciting the known double-pendulum Lagrangian isn't evidence of a "world model." Hard
  to disentangle recall from reasoning.
- **Learned-dynamics baselines are the load-bearing comparison and they're stubs.** The
  scientifically interesting claim ("LLMs vs. models that actually learned the
  dynamics") requires trained Neural ODE / HNN / LNN with careful train/test splits
  across regimes. That's most of the real work, and conclusions depend heavily on
  getting it fair.
- **Threshold-driven conclusions:** "time to divergence" with a hand-set 0.5 rad /
  half-rod threshold will move rankings around; needs sensitivity analysis.
- **Sample size under chaos:** 20 trajectories/cell is thin for distributional claims;
  chaotic spread demands more for stable estimates.

## Strengths (real ones)

- Genuinely controlled, with a true oracle and deterministic, reproducible data
  generation.
- Physically principled energy-drift metrics (ΔE, symplectic comparison) — the *right*
  lens and not chaos-fragile.
- The cross-family, single-substrate design is a clean experimental backbone.
- The hidden-constants probe is a sharp, fresh question.
- Strong instrumentation/engineering (resumable, factorial, multi-metric) — easy to
  extend into a better-framed study.

## Verdict

**As written: a competent benchmark, a modest research idea.** It is well-engineered
and asks a reasonable question, but (a) the novelty is incremental against a crowded
2024–2026 literature, (b) the marquee comparison is partially foregone, and (c) the
long-horizon pointwise metrics are invalidated by the chaos the system is chosen for.
On a 1–10 research-significance scale I'd place the *current framing* around a **5** —
solid empirical-study / workshop tier, not a top-venue novel contribution.

**It can be elevated to a 7–8** with three changes, none requiring a new codebase:

1. **Reframe the question** from "predict physics" to "in-context numerical integration
   and system identification of chaotic mechanics," and make the **hidden-constants
   regime** the centerpiece.
2. **Fix the metrics for chaos:** lead with predictability-horizon / time-to-divergence
   and add distributional / attractor-climate measures; stop reporting pointwise error
   beyond ~1 Lyapunov time.
3. **Actually train the learned-dynamics baselines** (HNN / LNN / Neural ODE) — that's
   where the interesting, non-obvious comparison lives.

Do those, and *"what's the predictability-horizon gap between frontier LLMs and models
that learned the physics, and can LLMs infer unstated physical constants in-context?"*
is a genuinely worthwhile paper. The "who has lowest 60s MSE" framing is not.

## Sources

- [Zero-shot forecasting of chaotic systems](https://arxiv.org/pdf/2409.15771)
- [Using ML and Neural Networks to Predict Chaos in Multi-Pendulum Systems (2025)](https://arxiv.org/abs/2504.13453)
- [Chaos as an interpretable benchmark for forecasting (dysts)](https://openreview.net/pdf?id=enYjtbjYJrf)
- [PhysicsMind: mechanics benchmarking for VLMs and world models](https://arxiv.org/html/2601.16007v1)
- [UGPhysics: undergraduate physics reasoning benchmark](https://arxiv.org/abs/2502.00334)
- [FeynmanBench: multimodal diagrammatic physics reasoning](https://arxiv.org/html/2604.03893)
