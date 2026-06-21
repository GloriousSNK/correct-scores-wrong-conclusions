# PROJECT HANDOVER — read me first

This file is the single entry point for continuing this research project. If you are an
agent picking this up: **read this whole file, then follow the "Remaining work" checklist.**
It captures the full context, the current state of every run, the critical findings (and
mistakes already made and fixed), and exactly what is left to reach a rigorous, publishable
result.

---

## 0. TL;DR — the one thing you must not miss

The headline of the earlier rounds — *"a mixed-k Neural ODE is the best non-numerical
forecaster"* — was an artifact of **in-sample evaluation** (the learned models were tested
on their own training trajectories) **and single-trajectory sampling**. The first proper
**out-of-sample** evaluation (the "boot" run, 20 fresh trajectories/cell) overturns it:

- Learned models are ~2.4× worse out-of-sample (mixed-k NODE 0.258 → **0.625 rad**).
- The mixed-k advantage **does not replicate** (vs single-k: p≈0.09, n.s.).
- A **zero-shot time-series model (Chronos-2 ≈ 0.54) ties/beats the trained learned
  models** out-of-sample.
- What survives: numerical integrators dominate (RK4 0.097, symplectic 0.178), and
  **rollout-loss beats derivative-matching** (0.62 vs 0.83, significant).

The corrected, honest, *more interesting* story is about **evaluation rigor**. The paper
must be rewritten around the out-of-sample results. Do not reinstate the in-sample numbers.

---

## 1. The research project

**Question:** given a chaotic k-link pendulum's state at time t, how well can different
model families predict its state at t+T — and at what reliability/cost?

**Substrate:** k-pendulum, k∈{1,2,3} (chaotic for k≥2). Ground truth via high-accuracy
numerical integration. Metric: **mean absolute angle error (rad)**, wrapped to [−π,π],
averaged over links.

**Model families compared on identical inputs/metrics:**
- LLMs (kimi-k2.6, grok-4-1-fast-reasoning, deepseek-v4-pro) — run on a **separate Mac**
  via API (the colleague's side).
- Time-series foundation models — Chronos-t5 (univariate) and Chronos-2 (uni + multi),
  run locally.
- Learned dynamics — Neural ODE, HNN, LNN (derivative-matching, rollout, and mixed-k
  rollout variants), trained + run locally.
- Numerical integrators — Euler, RK4 (the data-generating oracle), Störmer–Verlet
  (symplectic).

**Contributions that hold up:** (1) a unified cross-family benchmark with a
**reliability-adjusted** metric (failures scored at π/2, not dropped) and
**chaos-appropriate** metrics (predictability horizon); (2) rollout-loss > derivative-MSE;
(3) the demonstration that in-sample + single-trajectory evaluation massively inflates
learned-model results — and that, evaluated properly, zero-shot foundation models are
competitive with trained dynamics models. The most novel *open* thread is in-context
**system identification** (see §6).

---

## 2. Critical findings & caveats (don't repeat the mistakes)

1. **In-sample leakage (most important).** The learned models were TRAINED on
   `results/dataset` (seed 42, 9 trajectories) and the "round 2" eval tested them on those
   SAME trajectories → inflated. Evidence it's leakage not noise: on the fresh boot set the
   **numerical models got better** (easier trajectories) while the **learned models got
   much worse** — they only diverge like that under overfitting. **Always evaluate
   out-of-sample** (the boot dataset, seed 20260620, is disjoint from training).
2. **Single-trajectory variance.** Round-2 used 1 trajectory/cell — wildly unreliable for
   chaotic systems. All real claims need ≥10–20 trajectories/cell with CIs. (Boot uses 20.)
3. **Reliability ≠ accuracy.** Averaging error only over a model's *answered* cells rewards
   abstention (it made kimi look like the best LLM when it answered 31.5% of cells). Always
   use the reliability-adjusted metric (failures → π/2) and report success rate separately.
4. **Chaos invalidates long-horizon pointwise error.** Past ~1 Lyapunov time (10s, 60s)
   error saturates to noise for everyone. Lead with the **predictability horizon** metric;
   the discriminating range is ~1–10 s.
5. **System-ID gap is confounded as currently measured.** The `changed_hidden` and
   `changed_disclosed` regimes use *different* constants, so "better when hidden" may just
   mean the hidden physics is easier. The clean test (matched constants + inference probe)
   is specified for the Mac run (§6).
6. **Multiple comparisons:** apply Holm/BH across families of tests (some "significant"
   results don't survive — e.g. the Chronos uni-vs-multi overall p=0.025 does not).
7. **Environment / data safety:** the project was originally under OneDrive, which
   **silently deleted the entire folder** mid-run (it can't handle the churn). It now lives
   in `C:\Users\srima\dp-work` — **never move it back under OneDrive.** All new runs write
   to NEW directories and treat existing results as read-only.

---

## 3. Environment & how to run (Windows machine)

- **Project root:** `C:\Users\srima\dp-work\Double-Pendulum copy`
- **Venv:** `.venv-win` (Python 3.13, CUDA torch 2.6.0+cu124 for the GTX 1070 Ti, 8GB).
  Run everything with `$env:PYTHONUTF8='1'` (Windows console can't print the scripts'
  Unicode otherwise) and, for GPU jobs, `$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'`.
- **GPU usage:** Chronos inference uses the GPU. Learned-model inference and dataset
  generation are CPU-bound (tiny nets + SciPy ODE solver / numpy physics); the GPU can't
  help those and that's expected/optimal. Sleep is disabled on AC for long runs.
- **Run a config-driven eval:** `python scripts/run_eval.py --config <cfg> [--models ...]
  [--checkpoint-dir ...]`. Aggregate: `python scripts/aggregate.py --config <cfg>
  --checkpoint-dir ... --summary-dir ...`.

---

## 4. Current state of runs (as of 2026-06-20 ~19:45)

- **Round-1/2 canonical (IN-SAMPLE, do not use as headline):** `results/checkpoints`
  (936 cells), summaries in `results/summary_round2`. LLM cells here predate fixes.
- **TS uni-vs-multi significance study: COMPLETE.** Chronos-2 uni vs multi, n=20 paired.
  Verdict: **null** (multivariate ≈ univariate). Output:
  `results/summary_tsboot/ts_uni_vs_multi_significance.csv`. Already written into the paper
  and `docs/results/round2_findings.md` (§"Round 2b").
- **Out-of-sample "boot" eval (learned + numerical): COMPLETE.** Config `config.boot.yaml`,
  dataset `results/dataset_boot` (seed 20260620, 20 traj/cell), checkpoints
  `results/checkpoints_boot`, summaries `results/summary_boot/` (`boot_model_cis.csv`,
  `boot_paired_contrasts.csv`, `leaderboard.csv`). All models complete at n=720 EXCEPT
  **lnn was stopped early at n=327** (its 60s-horizon autograd cells run minutes each →
  ~24h to finish; n=327 already gives a tight CI and it's the worst model, so finishing it
  would not change any conclusion). The Chronos-2 boot cells (1440) are included.
  **This is the definitive local out-of-sample leaderboard.**

  Final out-of-sample means (rad, 95% CI): rk4 0.097 [0.071,0.125] (oracle); symplectic
  0.178 [0.149,0.209]; euler 0.398 [0.351,0.445]; chronos2-multi 0.543 [0.498,0.589];
  chronos2-uni 0.549 [0.505,0.593]; neural-ode-rollout 0.622 [0.570,0.674];
  neural-ode-rollout-mixed 0.625 [0.573,0.679]; hnn-rollout 0.749 [0.694,0.807];
  hnn-rollout-mixed 0.804 [0.744,0.864]; neural-ode 0.833 [0.773,0.894]; hnn 0.838
  [0.782,0.898]; lnn 0.950 [0.851,1.051] (n=327). Holm-corrected contrasts: rollout >
  derivative-MSE SIG (p≈1.5e-7); mixed-k n.s. for both NODE (p=0.089) and HNN (p=0.088,
  directionally worse); symplectic ≫ mixed-k NODE (p≈1e-86); chronos2-uni vs mixed-k NODE
  n.s. (tied).
- **Mac LLM run: NOT STARTED** (colleague's side). Prompt ready (§5).

**Headline out-of-sample numbers already in hand (n=720 each, 95% CI):** rk4 0.097
[0.071,0.125]; symplectic 0.178 [0.149,0.209]; euler 0.398 [0.351,0.445]; chronos2-multi
0.543; chronos2-uni 0.549; neural-ode-rollout 0.622 [0.570,0.674]; neural-ode-rollout-mixed
0.625 [0.573,0.679]; neural-ode (deriv) 0.833 [0.773,0.894]. Paired (Holm): mixed-k vs
single-k NODE n.s. (p≈0.09); rollout vs deriv SIG (p≈1e-7); symplectic ≫ mixed-k NODE
(p≈1e-86).

---

## 5. The Mac (LLM) run — what the colleague's agent does

Full prompt: **`docs/mac_llm_prompt_combined.md`** (self-contained; works even if the old
harness errors). It has two parts:
- **Part A** — evaluate the 3 LLMs on the **shared held-out set**
  `results/heldout_llm_eval_set.json` (131 KB; also in the user's Downloads). This is the
  SAME out-of-sample trajectories the local models use, so results merge by `movement_id`.
- **Part B** — controlled in-context **system-identification** experiment (matched
  constants disclosed-vs-hidden + a probe asking the model to report its inferred
  constants). Generates its own data.

Outputs go to `results/summary_llm/` (schemas in the prompt). **Transfer
`heldout_llm_eval_set.json` to the Mac before running Part A.**

---

## 6. Remaining work — checklist to reach "completely rigorous"

Do these in order. Items marked [LOCAL] run on this machine; [MAC] on the colleague's.

1. **[LOCAL] Boot run — DONE.** `results/summary_boot/` holds the final out-of-sample CIs
   and paired contrasts (see §4 for numbers). LNN was stopped early at n=327 (sufficient;
   worst model). To re-derive: `python scripts/analyze_boot_cis.py --checkpoint-dir
   results/checkpoints_boot --summary-dir results/summary_boot`. (Optional: let LNN finish
   to n=720 someday, but it changes nothing.)
2. **[MAC] Run the combined LLM prompt** (`docs/mac_llm_prompt_combined.md`) on the shared
   held-out set + the system-ID experiment. Produces `results/summary_llm/*`.
3. **[MERGE] Build the unified out-of-sample leaderboard.** Pool `results/checkpoints_boot`
   (local: numerical, learned, Chronos) with the Mac's `llm_main_results_long.csv`, joined
   by `movement_id` + horizon. Every model is then out-of-sample on identical trajectories
   → a fair, paired, CI'd cross-family leaderboard. (Write a small merge+CI script mirroring
   `analyze_boot_cis.py`.)
4. **[ANALYSIS] Refresh the chaos + system-ID analyses** on the merged out-of-sample data:
   predictability horizon (`analyze_divergence.py`), and the *clean* system-ID result from
   Part B (matched constants + inference probe).
5. **[PAPER] Rewrite the paper** (`paper/neurips_workshop/main.tex`) around out-of-sample
   results:
   - Replace the in-sample leaderboard (Table `tab:overall`, `tab:horizon`) and the
     "mixed-k NODE is best non-numerical" claim with the out-of-sample numbers + CIs.
   - New headline framing: rigor — in-sample/single-trajectory eval inflated learned
     models; out-of-sample, zero-shot foundation models are competitive; rollout>deriv
     survives; mixed-k does not. Keep the reliability-adjusted + predictability-horizon
     methodology (those are strengths).
   - Fold in the Part B system-ID verdict (confound-free) as the novel result.
   - Add CIs/p-values (with Holm) to every comparative claim. State honest nulls.
   - Verify all citations in `references.bib` (entries marked `% VERIFY`); the paper is
     uncompiled (no local LaTeX) — build on Overleaf with the workshop's `.sty`.
6. **[OPTIONAL, raises the tier]** A second chaotic system (Lorenz / driven pendulum) for
   generality; broader LLM roster; develop the inference probe further. See
   `paper/neurips_workshop/README.md`.

---

## 7. File & script inventory

**Configs**
- `config.yaml` — main (round-2) config; dataset_dir=results/dataset (IN-SAMPLE; the canonical 9-traj set).
- `config.boot.yaml` — out-of-sample eval of learned+numerical on results/dataset_boot (seed 20260620, 20 traj).
- `config.tsboot.yaml` — Chronos-2 uni/multi significance study (pred_len≤64).

**Scripts (all in `scripts/`)**
- `generate_dataset.py` — make ground-truth trajectories (config-driven; trajectories_per_cell, seed).
- `run_eval.py` — evaluate selected models over the grid; resumable checkpoints. Supports kinds: numerical, llm (Azure), nim_llm, timeseries (Azure), **ts_local** (local Chronos/Chronos-2/Moirai), learned.
- `train_learned.py` — train Neural ODE / HNN / LNN; supports derivative-MSE, single-k rollout (`--rollout-steps`), and **mixed-k rollout** (`--rollout-ks`). LNN needs CUDA (second-order autograd; MPS can't).
- `aggregate.py` — checkpoints → leaderboard CSV/JSON.
- `analyze_divergence.py` — predictability-horizon (chaos-appropriate) + threshold sweep.
- `analyze_system_id.py` — system-ID gap (NOTE: confounded; see §2.5 / use Mac Part B for clean version).
- `analyze_ts_significance.py` — paired uni-vs-multi (or any two models) significance: Wilcoxon + bootstrap CI.
- `analyze_boot_cis.py` — per-model CIs + headline paired contrasts (Holm-corrected) for the boot run.
- `export_heldout.py` — produced `results/heldout_llm_eval_set.json` (the shared out-of-sample set for the Mac).
- `run_round2.ps1` — the round-2 orchestration (historical; produced the in-sample results).

**Models (`bench/models/`)**: `numerical.py`, `azure_llm.py`, `nim_llm.py`, `timeseries.py`
(Azure REST), `ts_local.py` (local Chronos/Chronos-2/Moirai; Chronos-2 supports
multivariate; Moirai blocked — uni2ts won't install on Python 3.13), `learned.py`
(NODE/HNN/LNN + the legacy stub).

**Docs / outputs**
- `docs/results/round2_findings.md` — narrative results log (incl. the TS significance null in §"Round 2b").
- `docs/mac_llm_prompt_combined.md` — the LLM-side prompt for the colleague's agent.
- `paper/neurips_workshop/{main.tex, references.bib, README.md}` — the paper (NEEDS the §6.5 rewrite).
- `RESEARCH_IDEA_EVALUATION.md`, `SPEC.md`, `README.md` — original design/critique.
- `results/summary_*` — leaderboards/analyses per run; `results/checkpoints_*` — per-cell results.

---

## 8. Data layout — which is which (avoid mixing in-sample and out-of-sample)

| Path | What | Sample status |
|---|---|---|
| `results/dataset` (seed 42, 9 traj) | learned models' TRAINING data + round-2 test | learned = IN-SAMPLE |
| `results/checkpoints` + `summary_round2` | round-2 leaderboard | learned numbers INFLATED |
| `results/dataset_boot` (seed 20260620, 180 traj) | held-out test set | OUT-OF-SAMPLE |
| `results/checkpoints_boot` + `summary_boot` | out-of-sample learned/numerical/Chronos | **use this** |
| `results/dataset_boot` → `heldout_llm_eval_set.json` | compact eval spec for the Mac | OUT-OF-SAMPLE |
| `results/checkpoints_tsboot` + `summary_tsboot` | Chronos-2 uni/multi significance | OUT-OF-SAMPLE |
| `results/summary_llm` (to be produced) | Mac LLM results | OUT-OF-SAMPLE |

**Merge key:** `movement_id` (e.g. `k2_changed_hidden_0007`) + `horizon`. Same movement_id
across machines = same trajectory (because both use the seed-20260620 held-out set).

---

## 9. Honest assessment / target venue

Aiming for a **NeurIPS-style benchmarking workshop**. Realistic outcome: **accept (likely)**,
poster most probable, oral plausible if pitched as an evaluation-rigor paper (the in-sample
inflation finding is the hook), not best paper as-is. Levers to raise it: a second chaotic
system, and developing the system-ID result (Part B) into a controlled, mechanistic,
powered claim. Keep every comparative claim backed by CIs + Holm-corrected p-values and
report nulls honestly — that rigor is the paper's main strength.
