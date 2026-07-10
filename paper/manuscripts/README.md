# Manuscript Build Notes

Current manuscript:

- `main.tex`
- `references.bib`
- `neurips_2026.sty`
- `ForecastingChaosAcrossModelFamilies.pdf`

Build command:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The current checked PDF was compiled locally with MiKTeX and the NeurIPS 2026 style file.
It is currently set in preprint mode so author names are visible. For an anonymous or
workshop submission, switch the `\usepackage[preprint]{neurips_2026}` line in `main.tex`
to the target workshop option and set `\workshoptitle{...}` if required by that workshop.

The manuscript includes:

- main leaderboard
- input-information table
- known-physics vs black-box framing
- reliability-adjusted scoring
- hidden-constant matched control
- Chronos history-conditioning clarification
- HNN/LNN energy-conservation analysis
- NeurIPS paper checklist
- appendix prompt templates and supplemental result tables
