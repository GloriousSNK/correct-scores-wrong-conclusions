# Correct Scores, Wrong Scientific Conclusions

A claim audit for AI forecasts of chaotic dynamics. Given the state of a system at time
$t$, the task is to predict the state at $t+T$; the question this repository is built to
answer is not who scores best, but **which scientific claims a score actually supports**.

Five conditions connect a leaderboard number to a claim. Each one changes a conclusion
the raw comparison invites:

| Condition | Claim it tests | What it changes |
|---|---|---|
| Failure accounting | sample → population | Scoring refusals instead of dropping them reorders the board |
| Held-out generalization | in-sample → out-of-sample | Error rises $1.7$–$11\times$; simple baselines win |
| Matched disclosure | correlation → attribution | Unpaired $-0.390$ rad reverses to a paired $+0.069$ rad penalty |
| Structural identifiability | accuracy → recovery | Two exact symmetries make absolute constants unrecoverable |
| Cross-system transfer | one system → family claim | The pendulum ordering does not survive Lorenz-63 |

Systems: the $k$-link pendulum ($k\in\{1,2,3\}$, chaotic for $k\ge2$), Lorenz-63, and a
video-tracked physical double pendulum. Methods span large language models (Kimi, Grok,
DeepSeek), a time-series foundation model (Chronos-2), learned dynamics (Neural ODE, HNN,
LNN), sparse identification (SINDy, NVAR), numerical integrators (Euler, RK4, symplectic),
and four simple baselines.

## Manuscripts

Two workshop submissions share this evidence base:

- [`paper/manuscripts/ai4science.pdf`](paper/manuscripts/ai4science.pdf) — 8-page body,
  *Verification in the Age of AI Scientists* (NeurIPS 2026)
- [`paper/manuscripts/sim2science.pdf`](paper/manuscripts/sim2science.pdf) — 5-page body,
  *Sim2Science: ML with Imperfect Scientific Models* (NeurIPS 2026)

Both are anonymized for double-blind review; author information stays in LaTeX comments
until camera-ready. **Keep this repository private while the submissions are under
review** — it is otherwise a deanonymization path from the paper text.
`ForecastingChaosAcrossModelFamilies.pdf` and `main.tex` are the superseded single-paper
version, retained for history.

## Leaderboard

Reliability-adjusted mean absolute angle error (radians) on the shared held-out set
under the equalized hidden-regime run; horizons $\{1,10\}$ s, $360$ cells per model,
unanswered cells scored at the random-guess value $\pi/2$. RK4 generates the data and is
a reference, not a competitor.

| # | Model | Err | 95% CI | Answered | Family |
|---:|---|---:|---|---:|---|
| — | rk4 (reference) | 0.059 | [0.032, 0.089] | 100% | numerical |
| 1 | symplectic | 0.205 | [0.163, 0.251] | 100% | numerical |
| 2 | euler | 0.405 | [0.344, 0.470] | 100% | numerical |
| 3 | kimi-k2.6 | 0.538 | [0.475, 0.602] | 99% | LLM |
| 4 | chronos2-multi | 0.684 | [0.617, 0.752] | 100% | time-series |
| 5 | chronos2-uni | 0.697 | [0.631, 0.766] | 100% | time-series |
| 6 | nearest-neighbor | 0.725 | [0.657, 0.797] | 100% | baseline |
| 7 | linearized | 0.855 | [0.774, 0.939] | 100% | baseline |
| 8 | grok-4-1-fast-reasoning | 0.881 | [0.805, 0.957] | 82% | LLM |
| 9 | deepseek-v4-pro | 0.952 | [0.881, 1.027] | 100% | LLM |
| 10 | hnn-rollout | 0.972 | [0.898, 1.047] | 100% | learned |
| 11 | persistence | 1.004 | [0.934, 1.078] | 100% | baseline |
| 12 | neural-ode-rollout-mixed | 1.036 | [0.967, 1.104] | 100% | learned |
| 13 | hnn-rollout-mixed | 1.066 | [0.990, 1.143] | 100% | learned |
| 14 | neural-ode | 1.110 | [1.039, 1.181] | 100% | learned |
| 15 | neural-ode-rollout | 1.121 | [1.054, 1.190] | 100% | learned |
| 16 | hnn | 1.138 | [1.059, 1.218] | 100% | learned |
| 17 | constant-velocity | 1.319 | [1.241, 1.396] | 100% | baseline |
| 18 | lnn | 1.356 | [1.282, 1.430] | 94% | learned |

At this training budget every learned dynamics model sits below nearest-neighbor lookup
and the linearized pendulum, and all but one sit below persistence.

## Key Findings

- **Generalization.** The six stable learned models score $0.09$–$0.63$ rad on their own
  training trajectories against $0.97$–$1.14$ held out — a $1.7$–$11\times$ gap, widest
  for the model that looks best in sample (mixed-window Neural ODE).
- **Failure accounting.** Missing, refused, and invalid cells are scored, not dropped.
  Under a strict $\pi$ bound only one model moves materially, so the ordering is not an
  artifact of the imputation constant.
- **Matched disclosure.** An unpaired regime contrast says hiding physical constants
  *helps* ($-0.390$ rad); toggling disclosure on identical trajectories (900 paired
  cells, Wilcoxon + Holm) finds a $+0.069$ rad penalty. The naive result was a difficulty
  confound.
- **Identifiability.** Two exact scale symmetries, verified numerically below $10^{-13}$,
  make the absolute constants unrecoverable from angle trajectories — so "the model
  failed to recover $g$" is a category error, not a performance result. On Lorenz-63 the
  check inverts and SINDy recovers $(10, 28, 8/3)$ to $2.2\times10^{-13}$.
- **Transfer.** The pendulum ordering fails on Lorenz-63 across five held-out seeds: a
  system-trained Neural ODE beats zero-shot Chronos-2, reversing the pendulum result.
- **Real hardware.** Against a video-tracked physical pendulum the RK4 model with
  published constants becomes the *imperfect* model: excellent at $0.1$ s ($0.084$ rad,
  ${\sim}14\times$ tracking noise) but indistinguishable from persistence by $1.0$ s
  (paired difference $+0.018$, CI $[-0.121, +0.155]$). A fitted damping term gives a
  significant $1.0$ s gain (CI $[-0.202, -0.003]$; moving-block bootstrap
  $[-0.179, -0.019]$, stable across block lengths of 2–12 s).
- **Prompting, modality, repeatability.** Chain-of-thought never changes the model
  ordering and significantly *hurts* the best LLM ($+0.121$ rad), because reasoning
  consumes the token budget and unanswered cells rise from 6 to 50 of 360. Rendered
  images never beat coordinates. A second temperature-0 run agrees bit-identically on
  only $172/216$ cells and shifts the aggregate $+0.073$ rad — hosted inference is not
  exactly repeatable, which is why hosted results are reproduced from saved records
  rather than by re-calling the API.

## Repository Layout

```text
bench/
  simulator.py         # k-pendulum dynamics and numerical integrators
  lorenz.py            # Lorenz-63 system
  identifiability.py   # exact scale-symmetry checks
  metrics.py           # angle, coordinate, energy, reliability metrics
  prompts.py           # LLM prompt templates
  rendering.py         # image-modality rendering
  runner.py            # evaluation harness
  models/              # numerical, azure_llm, ts_local, learned predictors
scripts/
  generate_dataset.py                    # trajectories
  train_learned.py                       # Neural ODE / HNN / LNN
  run_eval.py, aggregate.py              # main grid
  eval_baselines.py                      # persistence, linearized, nearest-neighbor
  eval_insample.py                       # in-sample vs held-out table
  eval_realdata_myers.py                 # hardware check
  analyze_realdata_bootstrap.py          # hardware CIs (iid + moving-block)
  analyze_identifiability.py             # symmetry verification
  analyze_prompting_modality_repeat.py   # CoT, modality, repeatability
  run_lorenz_seed_sweep.py               # cross-system transfer
  run_training_budget_study.py           # nested budget sweep
  build_anonymized_artifact.py           # reviewer artifact builder
paper/manuscripts/
  ai4science.tex, sim2science.tex, references.bib, compiled PDFs
results/
  per-cell records, summaries, trained weights, held-out specifications
```

## Reproduction

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```

These run from committed records — no API calls, no GPU:

```bash
python scripts/analyze_identifiability.py
python scripts/analyze_realdata_bootstrap.py
python scripts/analyze_prompting_modality_repeat.py
python scripts/eval_realdata_myers.py
python -m unittest discover -s tests
```

Full pipeline (dataset generation, training, main grid):

```bash
python scripts/generate_dataset.py --config config.yaml
python scripts/train_learned.py --models neural_ode hnn lnn
python scripts/run_eval.py --config config.yaml
python scripts/aggregate.py --config config.yaml
python scripts/analyze_boot_cis.py
```

LLM evaluations need API credentials in `.env` (never commit it). Reported hosted-model
numbers are recomputed from the saved per-cell records; re-calling an API is a new
evaluation, not a reproduction.

## Reviewer Artifact

```bash
python scripts/build_anonymized_artifact.py
```

Builds two archives outside the repository: the main artifact (code, prompts, trained
weights, dataset and held-out specifications, and per-cell records for the main grid,
disclosure test, modality grid, Lorenz sweep, and hardware check) plus a sweeps archive.
Every staged text file is scanned against a banned-pattern list and the build aborts on
any hit, so the output is safe to attach to a double-blind submission.

## Authors

Sriman Narayan Kandi, Trishant Srinivasan.

Shrithik Shahapure provided compute access for the Azure AI Foundry large language model
evaluations and helped run those API evaluations.

## Citation

```bibtex
@inproceedings{kandi2026correctscores,
  title  = {Correct Scores, Wrong Scientific Conclusions:
            Auditing AI Forecasts of Chaotic Dynamics},
  author = {Kandi, Sriman Narayan and Srinivasan, Trishant},
  year   = {2026}
}
```
