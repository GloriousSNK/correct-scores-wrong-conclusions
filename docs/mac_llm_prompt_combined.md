# Combined Mac-agent prompt: rigorous LLM evaluation for the k-pendulum benchmark
# (Paste everything below the line into the Mac/API agent. Self-contained — does not
#  require the existing benchmark harness; can be implemented directly against the
#  shared JSON file.)

---

# Task: Statistically-rigorous LLM evaluation of physics forecasting (two parts)

You run the **LLM/API side** of a k-pendulum forecasting benchmark: given a pendulum's
state at t=0, predict its state at a future horizon T. The local machine handles all
non-LLM models (numerical integrators, learned dynamics, time-series foundation models).
Your job: produce **publication-grade, statistically-rigorous** LLM results, in two parts:
(A) a powered main comparison on a shared held-out test set, and (B) a controlled
in-context system-identification experiment. Both a positive and a null are publishable —
**do not steer toward any result.**

If the existing harness errors, ignore it: this prompt fully specifies the task and can be
implemented as a standalone script (read the JSON, call the API, score, run stats).

## Models
`kimi-k2.6`, `grok-4-1-fast-reasoning`, `deepseek-v4-pro`, via your OpenAI-compatible API.
Use reliability-fixed settings so failures are real, not artifacts (e.g. kimi
`max_tokens=65536`, `request_timeout=300`). **Single sample per (cell, trajectory) — no
best-of-N** (for error bars we want trajectory breadth, not depth).

## Global requirements (apply to BOTH parts)

**⚠️ Additive only — protect existing data.** Write to NEW dirs only
(`results/llm_main/`, `results/llm_sysid/`, `results/summary_llm/`). Treat all existing
`results/*` as read-only. Use resumable per-cell checkpoints so a restart never re-pays.

**Metric.** Per cell: mean over links of `|wrap(theta_pred - theta_true)|`, wrapped to
[-pi, pi]. **Reliability-adjusted:** a failed / unparseable / NaN cell is scored at
**pi/2** (the expected error of a random guess), NOT dropped. Also report raw success rate.

**Statistics.** Paired **Wilcoxon signed-rank** on per-trajectory differences (pair on
`movement_id` + horizon); **bootstrap 95% CIs** on means and mean-differences; **Holm**
correction across each family of tests; report effect sizes. State conclusions at the
significance level, including honest nulls.

**Cost.** Print a cell-count and cost estimate before running; **report total API cost at
the end. No hard cap.**

---

## PART A — Main powered comparison (shared held-out set)

Evaluate on the file **`heldout_llm_eval_set.json`** (transferred from the other machine).
These trajectories are **out-of-sample for every model** (seed 20260620, disjoint from the
learned models' training data) and **shared across machines**, so results are directly
comparable and can be paired with the local learned/numerical results by `movement_id`.

**File structure:**
- top level: `horizons` ([0.01, 1.0, 10.0, 60.0]), `regime_constants`, `metric`,
  `n_trajectories` (180), `trajectories`.
- each trajectory: `movement_id`, `k` (1–3), `regime` (normal | changed_disclosed |
  changed_hidden), `constants {g, L[], m[], damping}`, `initial_state {theta[], omega[]}`,
  `targets { "<horizon>": {theta[], omega[]} }`.

**Per (trajectory × horizon × prompting):** prompt the LLM with the `initial_state`; in the
**disclosed**/**normal** regimes include `constants`, in **changed_hidden** omit them; ask
for the state at that horizon as JSON `{"theta":[...],"omega":[...]}`; score against
`targets["<horizon>"]`.

**Grid:** all 3 LLMs × all 180 trajectories × horizons **{1.0, 10.0}** (the discriminating
range; you may add 0.01 and 60.0 for full comparability, but they carry little information —
every model is ~0 at 0.01 s and noise at 60 s) × prompting **{no_cot, cot}**.

**Analyses to output:**
- Per (model, prompting): mean error + 95% CI + success rate.
- Paired tests: **CoT − no_cot** (per model); each **model − model** pair; and the
  **system-ID gap** `err(changed_hidden) − err(changed_disclosed)` per model.
- NOTE on the gap: in this set `changed_hidden` and `changed_disclosed` use *different*
  constants, so this gap is **confounded** (hidden physics may simply be easier). Report it,
  but the clean test of system identification is Part B.

---

## PART B — Controlled in-context system identification (confound-free + mechanistic)

Part A's hidden-vs-disclosed gap conflates *disclosure* with *different constants*. Part B
removes that confound and tests the mechanism directly. **Generate your own data here**
(LLMs never train, so it is inherently out-of-sample); write to `results/dataset_llm_sysid/`
with a new seed you report.

**B1 — matched-constant disclosed-vs-hidden.**
- Pick **K = 4–5 non-standard constant sets**, sampled away from Earth-standard to limit
  memorised priors (e.g. g ∈ [1.6, 12], Lᵢ ∈ [0.5, 1.5], mᵢ ∈ [0.5, 2.0], damping ∈
  [0, 0.1]); fix and report the seed and the exact sets.
- For each set, generate trajectories with a ground-truth integrator (DOP853 or RK4 at
  dt=1e-3), recording the state at t=0 and at horizons {1, 10} s. ~10 trajectories/set.
- Evaluate each LLM under **two conditions on the SAME trajectories**: **DISCLOSED**
  (constants in prompt) vs **HIDDEN** (constants omitted), everything else identical, for
  k ∈ {1,2,3}. This is the clean paired contrast — same physics, only disclosure toggled.

**B2 — direct inference probe (the mechanism).**
- In the **HIDDEN** condition only, additionally require the model to output its **inferred
  constants** (g, each Lᵢ, each mᵢ, damping) in a structured field alongside its prediction.
- Compute, per model and per constant: **MAE and correlation (Spearman + Pearson) between
  inferred and true constants**, with bootstrap CIs.
- Also test whether **prediction accuracy tracks inference accuracy** across trajectories
  (does getting the constants right correlate with lower forecast error?). That link is the
  strongest evidence that any accuracy gap is *caused by* in-context system ID.

**B3 — difficulty control.** For each constant set, compute a numerical-integrator (RK4)
baseline error per cell so model error can be normalised by the intrinsic predictability of
that set (guards against difficulty imbalance across sets).

**Analyses to output:**
- Matched gap: paired Wilcoxon (hidden − disclosed) per k / horizon / overall; bootstrap
  CI; Holm; effect sizes.
- Inference probe: per-constant MAE + correlation (inferred vs true) per model, with CIs;
  plus the accuracy-vs-inference correlation.

---

## Outputs (write all to `results/summary_llm/`)

1. `llm_main_results_long.csv` — Part A, one row per (model, k, regime, horizon, prompting,
   movement_id): success, angle_error_mean, prompt_tokens, completion_tokens, latency_s.
2. `llm_main_significance.csv` — Part A paired tests (comparison, stratum, mean_diff,
   ci_lo, ci_hi, p_raw, p_holm, n, effect_size).
3. `sysid_results_long.csv` — Part B, one row per (model, constant_set, k, horizon,
   disclosure, trajectory_id): success, angle_error_mean, inferred constants (hidden cells),
   tokens, latency.
4. `sysid_significance.csv` — Part B matched hidden−disclosed paired tests.
5. `sysid_inference.csv` — per model × constant: MAE, Spearman, Pearson, CIs; plus the
   accuracy-vs-inference correlation.
6. `llm_summary.md` — per claim: result + CI + adjusted p; total API cost.

## How to read it (no steering)
- **Part A** establishes, with error bars, where the LLMs sit relative to each other and
  (once merged by `movement_id` with the local results) relative to learned/numerical/TS
  models — all out-of-sample on identical trajectories.
- **Part B** is the headline test: if hidden ≈ disclosed on *matched* constants AND inferred
  constants don't correlate with truth → the "better when hidden" effect was a difficulty
  confound (system-ID claim weakens). If hidden ≥ disclosed AND inferred constants track
  truth AND accuracy tracks inference → strong mechanistic evidence of in-context system
  identification. Report whichever way it lands.

Begin by printing the planned cell counts and a cost estimate for both parts, then run with
checkpointing into the new directories.
