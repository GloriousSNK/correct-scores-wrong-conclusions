"""Generate LaTeX table fragments from completed strengthening-study summaries."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def f3(value) -> str:
    return "--" if pd.isna(value) else f"{float(value):.3f}"


def balanced_table(path: Path) -> str:
    frame = pd.read_csv(path)
    rows = ["Model & Seeds & Error & Seed SD & 95\\% CI & Answered \\\\", "\\midrule"]
    for row in frame.itertuples(index=False):
        interval = f"[{f3(row.seed_bootstrap_low)}, {f3(row.seed_bootstrap_high)}]"
        rows.append(
            f"{row.model} & {int(row.seeds)} & {f3(row.error_mean)} & {f3(row.error_sd)} & "
            f"{interval} & {100 * float(row.answered_mean):.0f}\\% \\\\"
        )
    return "\n".join(rows)


def two_seed_table(path: Path) -> str:
    frame = pd.read_csv(path)
    rows = ["Model & Seed 20260620 & Seed 20260621 & Equal-seed mean & Answered \\\\",
            "\\midrule"]
    for _, row in frame.iterrows():
        rows.append(
            f"{row['model']} & {f3(row['20260620'])} & {f3(row['20260621'])} & "
            f"{f3(row['error_mean'])} & {100 * float(row['answered_mean']):.0f}\\% \\\\"
        )
    return "\n".join(rows)


def budget_table(path: Path) -> str:
    frame = pd.read_csv(path)
    rows = ["Model & Scope & Trajectories & Error & 95\\% CI \\\\", "\\midrule"]
    for row in frame.itertuples(index=False):
        interval = f"[{f3(row.hierarchical_bootstrap_low)}, {f3(row.hierarchical_bootstrap_high)}]"
        rows.append(
            f"{row.model} & {row.training_scope} & {int(row.total_trajectories)} & "
            f"{f3(row.error_mean)} & {interval} \\\\"
        )
    return "\n".join(rows)


def lorenz_table(path: Path) -> str:
    frame = pd.read_csv(path)
    pivot = frame.pivot(index="model", columns="horizon", values="nmae_mean")
    rows = ["Model & 0.1 s & 0.5 s & 1.0 s \\\\", "\\midrule"]
    for model, values in pivot.iterrows():
        rows.append(f"{model} & {f3(values.get(0.1))} & {f3(values.get(0.5))} & "
                    f"{f3(values.get(1.0))} \\\\ ")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="paper/manuscripts/generated/strengthening_tables.tex")
    args = parser.parse_args()
    sections = []
    inputs = [
        ("pendulum_two_seed", Path("results/two_seed_summary/pooled.csv"), two_seed_table),
        ("budget", Path("results/training_budget_study/summary/pooled.csv"), budget_table),
        ("lorenz", Path("results/lorenz_seed_sweep/summary_two_seed/pooled.csv"), lorenz_table),
    ]
    for name, path, renderer in inputs:
        if path.exists():
            sections.extend([f"% {name}", renderer(path), ""])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
