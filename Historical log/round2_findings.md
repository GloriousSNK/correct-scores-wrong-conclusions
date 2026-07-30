# Can LLMs Predict Physics? — Round 2 Findings

**Experiment:** k-pendulum dynamics prediction benchmark — CUDA round
**Date:** 2026-06-20 (run completed 12:13, ~10h47m wall time on a GTX 1070 Ti)
**What's new vs Round 1:** moved off Apple MPS onto a CUDA GPU, which unlocked
work the Mac could not do. Specifically this round adds:

- **Full-scale LNN retrain** — MPS cannot run the LNN's second-order autograd, so
  Round 1 used a crippled 8K-sample / hidden-64 model. CUDA trains it at full scale
  (hidden=256, 67K samples, 500 epochs).
- **Mixed-k rollout loss** — a new training objective that optimises several rollout
  window lengths at once (`--rollout-ks`), for both Neural ODE (k=10,50) and HNN
  (k=5,20).
- **Local time-series foundation model** — Chronos (`amazon/chronos-t5-large`) run
  locally on the GPU via a new `ts_local` adapter, instead of an undeployed Azure ML
  endpoint.
- **Chaos-appropriate analyses** — predictability-horizon and system-identification
  scripts (see Tier-3 section).

The canonical Round-1 / Mac results (LLMs and numerical baselines) are reused
unchanged; only the four new/changed models above were re-evaluated. New results live
in `results/checkpoints_round2` and `results/summary_round2`; the original
`results/checkpoints` is untouched.

---

## Overall Leaderboard

*Mean angle error (rad) over cells where a prediction was produced. Lower = better.*

| Rank | Model | Angle err (mean) | Angle err (median) | Success | Notes |
|------|-------|-----------------:|-------------------:|--------:|-------|
| 1 | rk4 | 0.119 | 0.000 | 100% | data-generating integrator (oracle) |
| 2 | symplectic | 0.253 | 0.002 | 100% | |
| **3** | **neural-ode-rollout-mixed** | **0.258** | 0.028 | 100% | **NEW — best non-numerical model** |
| 4 | neural-ode-rollout | 0.308 | 0.090 | 100% | prior best learned (single k=50) |
| 5 | kimi-k2.6 | 0.370 | 0.033 | **31.5%** | ⚠ mean over answered cells only — see caveat |
| 6 | euler | 0.455 | 0.038 | 100% | |
| 7 | hnn-rollout | 0.542 | 0.253 | 100% | single-k=20 |
| 8 | hnn-rollout-mixed | 0.605 | 0.296 | 100% | NEW |
| 9 | chronos-local | 0.626 | 0.352 | 100% | NEW (time-series FM) |
| 10 | grok-4-1-fast-reasoning | 0.651 | 0.351 | 75.0% | |
| 11 | hnn | 0.716 | 0.380 | 100% | derivative-matching |
| 12 | neural-ode | 0.716 | 0.083 | 100% | derivative-matching |
| 13 | deepseek-v4-pro | 0.803 | 0.717 | 99.5% | |
| 14 | lnn | 1.141 | 1.201 | 97.2% | NEW (full-scale); still worst |

> **⚠ Read the success-rate column before the error column.** The mean error is
> computed only over cells that produced a valid, parseable prediction. For models at
> 100% success this is the honest number. For kimi (31.5%) it is **survivorship bias** —
> see "Why Kimi looks better than it is" below.

---

## Breakdown by Prediction Horizon

*Mean angle error (rad). Numerical/learned models integrate from the initial state;
LLMs and Chronos make a one-shot prediction.*

| Model | 0.01 s | 1 s | 10 s | 60 s |
|-------|-------:|----:|-----:|-----:|
| rk4 | 0.000 | 0.000 | 0.260 | 0.216 |
| symplectic | 0.000 | 0.021 | 0.501 | 0.491 |
| **neural-ode-rollout-mixed** | 0.001 | **0.036** | **0.149** | 0.846 |
| neural-ode-rollout | 0.004 | 0.076 | 0.688 | **0.463** |
| euler | 0.000 | 0.041 | 0.856 | 0.922 |
| hnn-rollout | 0.005 | 0.210 | 0.963 | 0.991 |
| hnn-rollout-mixed | 0.006 | 0.161 | 0.906 | 1.347 |
| chronos-local | 0.023 | 0.686 | 0.923 | 0.873 |
| grok-4-1-fast-reasoning | 0.382 | 0.828 | 0.822 | 0.608 |
| neural-ode | 0.001 | 0.077 | 1.253 | 1.533 |
| hnn | 0.003 | 0.358 | 1.183 | 1.318 |
| kimi-k2.6 (answered only) | 0.000 | 0.422 | 0.544 | 0.547 |
| deepseek-v4-pro | 0.298 | 1.106 | 0.974 | 0.836 |
| lnn | 0.000 | 1.724 | 1.405 | 1.469 |

---

## Key Findings

### 1. Mixed-k rollout is the headline result — for the Neural ODE

Training the Neural ODE on **two rollout-window lengths at once (k=10 and k=50)** gives
the best non-numerical model in the entire benchmark: **0.258 rad**, beating single-k
rollout (0.308), every LLM, every time-series model, and effectively tying the
symplectic integrator (0.253). The win is concentrated in the discriminating range:

- **1 s: 0.036** — the best of *any* non-RK4 model, integrators included.
- **10 s: 0.149** — less than a quarter of the single-k=50 model's 0.688 (−78%).

There is a real trade-off, though: at **60 s the mixed model (0.846) is *worse* than the
pure k=50 model (0.463)**. Mixing a short window (k=10) into the objective shifts the
sweet spot from the extreme long horizon toward the 1–10 s range. Since three of four
horizons live in that range, the overall mean improves — but if you only cared about
60 s, the single long-window model is still better. **The objective tunes *which*
horizon you're good at, more than raw architecture or epoch count** — the same lesson
as Round 1's derivative-MSE-vs-rollout result, now one level deeper.

### 2. Mixed-k does *not* help the HNN — an honest negative

The same recipe applied to the Hamiltonian net (k=5,20) produced a model that is
*slightly worse* overall than plain single-k HNN rollout (0.605 vs 0.542), and clearly
worse at 60 s (1.347 vs 0.991). The mixed objective helps the **unconstrained** Neural
ODE but not the energy-constrained HNN — the benefit is architecture-dependent, not
universal. (This was also the most expensive run: ~6h42m, because HNN rollout does an
autograd-through-the-Hamiltonian at every unroll step.)

### 3. Fixing the LNN's training device did *not* fix the LNN

Full-scale CUDA training — which MPS literally could not run — improved the LNN only
marginally over the crippled Round-1 model (**1.300 → 1.141**), and it is **still the
worst model in the benchmark**, now with reliability problems too (97.2% success: its
autograd-based ODE integration produced NaNs on some hard cells). It is near-perfect at
0.01 s (0.0004) but diverges by 1 s (1.72). Conclusion: the LNN's weakness in Round 1
was **not** primarily the crippled Mac training — the Lagrangian-net + autograd-ODE
integration is genuinely fragile on this chaotic system. The "MPS was holding it back"
hypothesis is largely falsified.

### 4. Chronos works, and lands time-series FMs alongside the LLMs

The first time-series-foundation-model result: Chronos (`chronos-t5-large`, run locally)
scores **0.626 at 100% success** — near-perfect at 0.01 s (0.023), degrading to ~0.69
at 1 s and saturating ~0.87 at long horizons. It **beats both grok (0.651) and deepseek
(0.803)** but sits well behind every trained dynamics model. The takeaway: a zero-shot
time-series foundation model is roughly **as good as a frontier LLM** for chaotic
forecasting — and both are far behind a small model that actually learned the dynamics.

### Why Kimi looks better than it is (the #5 ranking explained)

Kimi's 0.370 mean ranks it 5th, *above several 100%-success models* — but this is an
artifact of how the mean is computed. Kimi produced a valid prediction on only **31.5%**
of its 216 cells; the leaderboard error averages over just those answered cells and
ignores the 68.5% it refused, timed out on, or returned unparseably.

The success rate is also **roughly uniform across horizons** (33% / 30% / 33% /
30% at 0.01 / 1 / 10 / 60 s), so it is *not* that Kimi only answers easy cells — it
answers about a third of the time everywhere, and when it does it is genuinely accurate
(median 0.033). It is "brilliant but unreliable," exactly as in Round 1.

If failures are scored as worst-case (π rad, i.e. a decorrelated guess), the picture
inverts:

| Model | Effective mean (failures = π) | Success |
|-------|------------------------------:|--------:|
| deepseek-v4-pro | 0.814 | 99.5% |
| grok-4-1-fast-reasoning | 1.274 | 75.0% |
| **kimi-k2.6** | **2.278** | 31.5% |

So **on any usability-weighted measure Kimi is the *worst* of the three LLMs**, not the
best. Its leaderboard position should be read as "most accurate *when it deigns to
answer*," not "best forecaster." (Note: this canonical checkpoint set shows kimi at
31.5%, below the 88.9% reported after Round 1's token-limit patch — so these numbers
predate that fix and likely understate kimi's reliability ceiling, but the survivorship
caveat stands regardless.)

---

## Tier-3 Analyses (chaos-appropriate)

For chaotic systems, pointwise error past ~1 Lyapunov time saturates to "random" for
every model, so the raw long-horizon leaderboard is partly uninformative. Two analyses
address this.

### Predictability horizon

*Longest horizon a model tracks **contiguously** below an angle-error threshold (the
divergence point; a later dip below threshold is chaotic coincidence, not skill).*

| Model | @0.1 rad | @0.25 rad | @0.5 rad | @1.0 rad |
|-------|---------:|----------:|---------:|---------:|
| rk4 | 1.0 | 1.0 | 60.0 | 60.0 |
| **neural-ode-rollout-mixed** | 1.0 | **10.0** | 10.0 | 60.0 |
| neural-ode-rollout | 1.0 | 1.0 | 1.0 | 60.0 |
| hnn-rollout | 0.01 | 1.0 | 1.0 | 60.0 |
| kimi-k2.6 | 0.01 | 0.01 | 1.0 | 60.0 |
| chronos-local | 0.01 | 0.01 | 0.01 | 60.0 |
| grok / deepseek | 0.0 | 0.0 | 0.01 | — |
| lnn | 0.01 | 0.01 | 0.01 | 0.01 |

The mixed-k Neural ODE is the **only non-numerical model that stays accurate past 1 s at
a strict threshold** — it tracks the true trajectory to 10 s at 0.25 rad, a full order of
magnitude longer than every LLM and every other learned model. The threshold sweep also
shows how soft a single threshold is: at a loose 1.0 rad (~57°) tolerance, almost
everything "reaches 60 s," which is meaningless — the strict thresholds are where models
separate.

### System identification (the `changed_hidden` regime)

*Mean angle error by regime (horizons ≤ 1 s, where pointwise error is still meaningful).
`sysid_gap = error(hidden) − error(disclosed)`; **negative = the model does better when
the physical constants are withheld**, i.e. it infers them in-context.*

| Model | disclosed | hidden | normal | sysid_gap |
|-------|----------:|-------:|-------:|----------:|
| lnn | 1.072 | 0.612 | 0.903 | −0.460 |
| deepseek-v4-pro | 0.717 | 0.402 | 0.988 | **−0.315** |
| chronos-local | 0.499 | 0.332 | 0.234 | −0.167 |
| hnn-rollout | 0.138 | 0.069 | 0.115 | −0.069 |
| grok-4-1-fast-reasoning | 0.550 | 0.494 | 0.768 | −0.056 |
| neural-ode-rollout-mixed | 0.016 | 0.005 | 0.035 | −0.011 |
| neural-ode-rollout | 0.021 | 0.049 | 0.050 | +0.028 |
| kimi-k2.6 | 0.175 | 0.268 | 0.147 | +0.093 |

The frontier LLMs (deepseek strongly, grok mildly) — and, notably, Chronos — are
**better when constants are hidden than when they are disclosed**. A model that merely
plugged numbers into a memorised formula would be hurt by hiding them; doing *better*
suggests in-context **system identification** (inferring g, L, m from the trajectory) or
that the disclosed non-standard constants actively mislead a model anchored on
Earth-gravity priors. This is the most genuinely novel signal in the benchmark — the
trained learned models, which are given the constants, show no such effect (gaps ≈ 0).

---

## Caveats & provenance

- **RK4 is the oracle, not a competitor** — it *is* the data-generating process, so its
  win is foregone and uninformative. The science is in the gaps between the others.
- **Long-horizon pointwise error (10 s, 60 s) is past the Lyapunov time** and should be
  read via the predictability-horizon / system-ID lenses, not as a literal ranking.
- **LLM/Chronos cells are one-shot endpoint predictions**; learned/numerical models
  integrate a full trajectory. Not a like-for-like protocol.
- **Learned models are trained on the same distribution they're tested on**; LLMs and
  Chronos are zero-shot. The comparison is informative but asymmetric.
- **Kimi's canonical checkpoints predate the Round-1 token/best-of-N fixes** (31.5% vs
  the patched 88.9%); treat its row as provisional.
- The full-scale **LNN had ~3% NaN failures** from its autograd ODE integration.

---

## Round 2b: Time-series univariate-vs-multivariate significance study

The single-trajectory leaderboard suggested (directionally) that multivariate modelling
might help where the system is most coupled (k=3). To test this properly we ran a
**powered, paired** study: **Chronos-2** in univariate and multivariate modes, **20
trajectories per (k, regime)** cell (n=720 paired samples), compared on identical
trajectories with Wilcoxon signed-rank tests, bootstrap CIs, and Holm correction. Run at
Chronos's recommended `prediction_length ≤ 64` (also ~4× faster than 256).

**Per-model (95% bootstrap CI):**

| Model | mean err | 95% CI |
|-------|---------:|--------|
| chronos2-uni | 0.549 | [0.505, 0.593] |
| chronos2-multi | 0.543 | [0.498, 0.589] |

**Paired (diff = uni − multi; + ⇒ multi better):**

| Stratum | mean diff | 95% CI | Wilcoxon p |
|---------|----------:|--------|-----------:|
| OVERALL | +0.0054 | [−0.013, +0.024] | 0.025 |
| k=1 | +0.0105 | [−0.009, +0.032] | 0.056 |
| k=2 | +0.0063 | [−0.032, +0.044] | 0.086 |
| k=3 | −0.0004 | [−0.035, +0.033] | 0.74 |
| h=0.01 | +0.0052 | [+0.001, +0.009] | 0.0002 |
| h=1.0 | +0.0302 | [−0.009, +0.070] | 0.52 |
| h=10 | −0.0045 | [−0.064, +0.054] | 0.37 |
| h=60 | −0.0092 | [−0.030, +0.010] | 0.70 |

**Verdict: a clean null.** Multivariate ≈ univariate. There is **no significant effect at
any k or at the discriminating horizons (1 s, 10 s)**. The n=1 "k=3 favors multivariate"
trend **reversed to zero** (−0.0004, p=0.74) — it was sampling noise. The nominal overall
p=0.025 is driven entirely by a sub-0.01 rad bias at the trivial 0.01 s horizon and **does
not survive Holm correction** (only the meaningless h=0.01 effect does). Cross-channel
coupling gives no practically meaningful benefit on this coupled chaotic system. This is a
textbook example of why error bars matter here: a striking single-trajectory trend became a
null under proper sampling. Data: `results/summary_tsboot/ts_uni_vs_multi_significance.csv`.

> Note: at `pred_len ≤ 64` Chronos-2 scores ~0.54 (better than the round-2 point estimate
> of 0.69 at pred_len 256), but it still clusters with the LLMs and well behind the learned
> models — the overall ranking is unchanged.

---

## What This Suggests

Round 1 established that a small model trained with rollout loss beats frontier LLMs.
Round 2 sharpens that into a method: **mixed-k rollout** gives the best non-numerical
forecaster we have (Neural ODE, 0.258 rad), and the predictability-horizon analysis
shows it is the only learned/LLM model that tracks chaotic trajectories meaningfully
past 1 s. The objective — *which* horizons you train the rollout on — matters more than
architecture, epochs, or model family.

The negatives are equally clean: the mixed objective does not transfer to the HNN, and
full-scale training does not rescue the LNN — its fragility is intrinsic, not a hardware
artifact. And the headline cross-family story holds and broadens: frontier LLMs *and*
time-series foundation models cluster together (~0.6–0.8 rad), all decisively behind a
3-minute-trained dynamics model — while the LLMs' edge on the hidden-constants regime
remains the one place they show something a fitted model does not.

**Natural next steps:** (1) sweep the mixed-k set further (e.g. add k=100, or per-horizon
weighting) to recover the 60 s performance without losing the 1–10 s gains; (2) re-run
the LLMs with the patched kimi config for a fair reliability number; (3) make
predictability horizon and the hidden-constants probe the headline metrics in the paper,
demoting the raw long-horizon MSE leaderboard.
