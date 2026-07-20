# Forecasting Chaos Across Model Families

This repository contains a controlled benchmark for comparing forecasting methods on
the $k$-link pendulum. Given the state of the system at time $t$, the task is to
predict the state at $t+T$ under shared out-of-sample trajectories and
reliability-adjusted scoring.

The benchmark evaluates four model families:

- large language models: Kimi, Grok, DeepSeek
- time-series foundation models: Chronos / Chronos-2
- learned dynamics models: Neural ODE, HNN, LNN
- classical integrators: Euler, RK4, symplectic

Paper: [`paper/manuscripts/ForecastingChaosAcrossModelFamilies.pdf`](paper/manuscripts/ForecastingChaosAcrossModelFamilies.pdf)

## Current Result

Mean absolute angle error in radians on the shared held-out set
($180$ trajectories, horizons $\{1,10\}$ s). Failed or missing cells are scored at
the random-guess baseline $\pi/2$.

| Rank | Model | Error | Family |
|---:|---|---:|---|
| reference | rk4 | 0.059 | numerical |
| 1 | symplectic | 0.205 | numerical |
| 2 | euler | 0.405 | numerical |
| 3 | kimi-k2.6 | 0.538 | LLM |
| 4 | chronos-2 multi | 0.684 | time-series |
| 5 | chronos-2 uni | 0.697 | time-series |
| 6 | nearest-neighbor | 0.725 | baseline |
| 7 | linearized | 0.855 | baseline |
| 8 | grok-4-1-fast-reasoning | 0.881 | LLM |
| 9 | hnn-rollout | 0.972 | learned |
| 10 | persistence | 1.004 | baseline |
| 11 | neural-ode-rollout-mixed | 1.036 | learned |
| 12 | hnn-rollout-mixed | 1.066 | learned |
| 13 | neural-ode | 1.110 | learned |
| 14 | neural-ode-rollout | 1.121 | learned |
| 15 | hnn | 1.138 | learned |
| 16 | constant-velocity | 1.319 | baseline |
| 17 | lnn | 1.349 | learned |

The numerical rows use the known equations and are reference methods. The comparison
of interest is among black-box methods. In this benchmark, Kimi has the lowest
reliability-adjusted error among the evaluated black-box methods, while Chronos-2 is
the lowest-error local zero-shot forecaster.

## Key Findings

- Held-out evaluation substantially changes the learned-dynamics conclusion. Learned
  models that appear strong in sample degrade out of sample and do not exceed the
  linearized or nearest-neighbor baselines under the current training budget.
- Reliability-adjusted scoring matters. Missing, refused, or invalid cells are scored
  instead of dropped.
- Chronos is evaluated in its standard history-conditioned form with a 10 s context
  window. LLMs and learned dynamics models are state-conditioned from $t=0$.
- Hidden-constant results require matched controls. An unadjusted regime contrast
  suggests lower error when constants are hidden, but the matched disclosure test and
  inference probe give little evidence that the evaluated LLMs infer hidden physical
  constants.
- The LLM evaluation is the only per-query billed run. Local numerical, learned, and
  Chronos evaluations use local compute.

## Repository Layout

```text
bench/
  simulator.py         # k-pendulum dynamics and numerical integrators
  metrics.py           # angle, coordinate, energy, reliability metrics
  prompts.py           # LLM prompt templates
  runner.py            # evaluation harness
  models/
    numerical.py       # Euler, RK4, symplectic
    azure_llm.py       # OpenAI-compatible LLM client
    ts_local.py        # local Chronos / Chronos-2
    learned.py         # Neural ODE / HNN / LNN
scripts/
  generate_dataset.py
  run_eval.py
  aggregate.py
  train_learned.py
  analyze_boot_cis.py
  analyze_divergence.py
  analyze_system_id.py
  analyze_ts_significance.py
paper/manuscripts/
  main.tex
  references.bib
  ForecastingChaosAcrossModelFamilies.pdf
results/
  summary files, trained weights, and held-out specifications
Historical log/
  superseded notes and earlier draft findings
```

## Reproduction

```bash
python -m venv .venv
. .venv/Scripts/activate      # Windows PowerShell/Git Bash users may need the matching activation script
pip install -r requirements.txt

python scripts/generate_dataset.py --config config.yaml
python scripts/train_learned.py --models neural_ode hnn lnn
python scripts/run_eval.py --config config.yaml
python scripts/aggregate.py --config config.yaml
python scripts/analyze_boot_cis.py
```

LLM evaluations require API credentials in `.env`. The local numerical, learned, and
Chronos evaluations do not require API calls.

## Data and Artifacts

The repository is configured to keep large raw result folders out of ordinary commits.
Summary tables, trained weights, held-out specifications, and manuscript artifacts should
be included or attached as release assets when publishing a submission snapshot. Do not
commit `.env` or other credential files.

## Authors

Sriman Narayan Kandi, Trishant Srinivasan.

Shrithik Shahapure provided compute access for the Azure AI Foundry large language model
evaluations and helped run those API evaluations.

## Citation

```bibtex
@inproceedings{kandi2026forecastingchaos,
  title  = {Forecasting Chaos Across Model Families: An Out-of-Sample,
            Reliability-Adjusted Benchmark},
  author = {Kandi, Sriman Narayan and Srinivasan, Trishant},
  year   = {2026}
}
```
