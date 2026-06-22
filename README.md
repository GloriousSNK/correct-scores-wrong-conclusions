# Forecasting Chaos Across Model Families

A controlled benchmark that asks: **given the state of a chaotic $k$-pendulum at time
$t$, which family of modern models best predicts its state at $t+T$ — and at what
reliability and cost?** We place four model families on identical inputs and metrics —
**LLMs**, **time-series foundation models** (Chronos / Chronos-2), **learned dynamics
models** (Neural ODE, HNN, LNN), and **classical integrators** (Euler, RK4, symplectic)
— and, crucially, score every family on the **same out-of-sample held-out trajectories**.

📄 **Paper:** [`paper/neurips_workshop/DoublePendulum.pdf`](paper/neurips_workshop/DoublePendulum.pdf)
(NeurIPS workshop manuscript; LaTeX source in the same folder).

## Headline result (out-of-sample, reliability-adjusted)

Mean absolute angle error (rad) on the shared held-out set (180 held-out trajectories ×
horizons {1, 10}s; failed/missing cells scored at the random-guess baseline π/2; 95%
bootstrap CIs).

| # | Model | Err | Family |
|---|-------|----:|--------|
| – | rk4 *(oracle)* | 0.059 | numerical |
| 1 | symplectic | 0.205 | numerical |
| 2 | euler | 0.405 | numerical |
| **3** | **kimi-k2.6** | **0.538** | **LLM** |
| 4 | chronos-2 (multi) | 0.684 | time-series |
| 5 | chronos-2 (uni) | 0.697 | time-series |
| 6 | neural-ode-rollout-mixed | 0.844 | learned |
| 7 | neural-ode-rollout | 0.870 | learned |
| 8 | grok-4-1-fast-reasoning | 0.881 | LLM |
| … | … | … | … |
| 14 | lnn | 1.421 | learned |

## Key findings

- **In-sample evaluation of learned dynamics models is badly optimistic.** Scored on
  their own training trajectories the learned models look strong (~0.26 rad); on a shared
  *held-out* set their error roughly **triples** (0.84–1.06) and they fall **below** a
  zero-shot time-series model and a zero-shot LLM.
- **On a level out-of-sample field, a frontier LLM (kimi) is the best non-numerical
  forecaster** — ahead of the time-series models and every learned model. Classical
  integrators remain unbeaten.
- **The training objective generalises; the recipe does not.** Rollout loss beats
  derivative-matching (paired, $p_{\text{Holm}}=1.6\times10^{-7}$), but the in-sample
  "mixed-$k$ wins" result is a **statistical null** out of sample.
- **No in-context system identification.** A confound-free, matched-constant experiment
  plus a direct inference probe show LLMs do **not** recover hidden physical constants —
  they fall back on Earth-standard priors.
- **Reliability is a first-class axis.** Scoring abstentions (instead of averaging only
  over answered cells) reorders the board and exposes brittle models.

Full numbers, CIs, and significance tests: [`results/summary_llm/`](results/summary_llm)
(`unified_leaderboard.md`, `llm_summary.md`) and [`results/summary_boot/`](results/summary_boot).

## Repository layout

```
bench/                 # library
  simulator.py         # k-pendulum dynamics + Euler/RK4/leapfrog integrators
  metrics.py           # angle/coord/energy + predictability-horizon helpers
  prompts.py, rendering.py, schema.py, export.py, runner.py
  models/
    numerical.py       # Euler / RK4 / symplectic
    azure_llm.py       # OpenAI-compatible LLM client
    ts_local.py        # local Chronos / Chronos-2 (uni + multivariate)
    timeseries.py      # Azure-hosted time-series client
    learned.py         # Neural ODE / HNN / LNN (trained)
scripts/
  generate_dataset.py  run_eval.py  aggregate.py  train_learned.py
  analyze_divergence.py        # predictability horizon
  analyze_system_id.py         # system-identification gap
  analyze_ts_significance.py   # Chronos uni-vs-multi paired test
  analyze_boot_cis.py          # per-model CIs + paired contrasts
  export_heldout.py            # compact shared held-out eval spec
config.yaml            # main grid; config.boot.yaml / config.tsboot.yaml for sub-studies
paper/neurips_workshop/   # main.tex, references.bib, DoublePendulum.pdf
results/                  # summaries + trained weights (raw checkpoints via Release)
```

## Reproduce

```bash
# 1. Environment
python -m venv .venv && . .venv/Scripts/activate    # (or .venv/bin/activate on Unix)
pip install -r requirements.txt

# 2. Ground-truth dataset (deterministic from the config seed)
python scripts/generate_dataset.py --config config.yaml

# 3. Train the learned dynamics models (CUDA GPU recommended for the LNN)
python scripts/train_learned.py --models neural_ode hnn lnn
python scripts/train_learned.py --models neural_ode --rollout-ks 10 50   # mixed-k rollout

# 4. Evaluate (numerical / learned / local time-series are free; LLMs need API keys)
cp .env.example .env          # then add your OpenAI-compatible key for the LLM runs
python scripts/run_eval.py --config config.yaml

# 5. Aggregate + analyses (CIs, predictability horizon, system-ID, significance)
python scripts/aggregate.py --config config.yaml
python scripts/analyze_boot_cis.py
```

**Out-of-sample protocol.** Learned models are trained on one seed and evaluated on a
disjoint held-out seed; `scripts/export_heldout.py` produces the compact
`results/heldout_llm_eval_set.json` so the LLM side (run on a separate machine) scores the
*identical* trajectories, keyed by `movement_id`.

**Data release.** The committed `results/` holds the summaries, trained weights
(`learned_models/*.pt`), and the held-out spec. The full per-cell checkpoints and raw
datasets (~50k files) are attached as a GitHub **Release** asset to keep the repo lean.

A no-API-key smoke test:
```bash
python scripts/generate_dataset.py --smoke
python scripts/run_eval.py --smoke --models rk4 euler symplectic
python scripts/aggregate.py
```

## Authors

**Sriman Narayan Kandi** (lead, corresponding — kandisriman@gmail.com),
Trishant Srinivasan, Shrithik Shahapure.

## Citation

```bibtex
@inproceedings{kandi2026forecastingchaos,
  title  = {Forecasting Chaos Across Model Families: A Controlled Benchmark of LLMs,
            Time-Series Foundation Models, Learned Dynamics, and Numerical Integrators
            on k-Pendulum Systems},
  author = {Kandi, Sriman Narayan and Srinivasan, Trishant and Shahapure, Shrithik},
  year   = {2026}
}
```
