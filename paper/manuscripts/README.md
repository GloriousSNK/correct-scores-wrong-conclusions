# Manuscript Build Notes

Active submission: `tae.tex`, for the NeurIPS 2026 TAE (Trust-AI-Eval) workshop.

Files:

- `tae.tex` — the manuscript
- `references.bib`
- `neurips_2026.sty`

Build:

```bash
pdflatex -interaction=nonstopmode tae.tex
bibtex tae
pdflatex -interaction=nonstopmode tae.tex
pdflatex -interaction=nonstopmode tae.tex
```

Compiled locally with MiKTeX. The source uses `\usepackage[dblblindworkshop]{neurips_2026}`
and sets `\workshoptitle{TAE (Trust-AI-Eval): Can We Trust AI Evaluation?}` for double-blind
submission; the author block stays commented until camera-ready, when the option becomes
`[dblblindworkshop, final]`. Body is eight pages, references and appendices unlimited.
