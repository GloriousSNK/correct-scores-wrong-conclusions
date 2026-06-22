# NeurIPS workshop submission — build & checklist

Files:
- `main.tex` — the paper (integrates Round 1 + Round 2 / CUDA results).
- `references.bib` — bibliography (see citation note below).

## Build

This machine has no LaTeX toolchain, so the paper was not compiled here. Two options:

**Overleaf (easiest):** create a project, upload `main.tex` + `references.bib`, drop in
the workshop's official style file (e.g. `neurips_2025.sty`), set the compiler to
pdfLaTeX, and compile. `main.tex` auto-detects `neurips_2025.sty` / `neurips_2024.sty`
and falls back to a plain `article` layout if neither is present, so it builds even
without the style file (just not in final NeurIPS formatting).

**Local:** install MiKTeX or TeX Live, then:
```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Citations — VERIFY before submitting

`references.bib` contains well-established works (Neural ODE, HNN, LNN, Chronos,
TimesFM, Moirai, LLMTime, CoT, Minerva, dysts) plus entries marked `% VERIFY`
(the zero-shot-chaos paper and UGPhysics). Confirm every `% VERIFY` entry's authors,
venue, and arXiv id against the real source. A few very recent physics-LLM benchmarks
the project notes referenced could not be verified and were **deliberately omitted
rather than fabricated** — add the ones you trust to Related Work.

## To strengthen before submission (raises it from solid to safe-accept)

1. **Tighten to the page limit.** Workshop limits are typically 4–6 pages + refs;
   the current draft is fuller (closer to main-track length). Trim Discussion/Methods
   first; keep Tables 1–6.
2. **Error bars.** The canonical set uses ~1 trajectory/cell in places. Re-run with
   more trajectories/seeds and add ± std to the headline tables — reviewers will ask.
3. **Clean Kimi re-run.** Current Kimi checkpoints predate the token-budget fix
   (31.5% raw success). Re-run with the patched config for a fair reliability number,
   then update Tables 1–2.
4. **A second chaotic system** (Lorenz or driven pendulum) would convert "pendulum
   study" into "chaotic forecasting" and materially help generality claims.
5. **Deepen the system-ID result** (the most novel finding): regress each model's
   *implied* constants (g, L, m) against truth to show it is really inferring them.
6. **A figure or two** (error-vs-horizon curves; predictability-horizon bar chart)
   would help — currently all-tables.

Source data for every number is in `results/summary_round2/` and
`docs/results/round2_findings.md`.
