"""Build a checksummed, machine-readable reproducibility inventory."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


RESULT_ROOTS = [
    "results/identifiability",
    "results/balanced_seed_study/summary",
    "results/balanced_seed_study/scientific_baselines_final/summary",
    "results/training_budget_study/summary",
    "results/training_budget_study/run_config.json",
    "results/lorenz_seed_sweep/summary",
    "results/lorenz_seed_sweep/run_config.json",
    "results/hosted_repeat_plan/plan.json",
]
SOURCE_ROOTS = ["bench", "scripts", "tests", "config.yaml", "config.boot.yaml", "requirements.txt"]
PACKAGES = ["numpy", "scipy", "pandas", "torch", "pysindy", "scikit-learn", "chronos-forecasting"]


def files_under(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if root.is_dir():
        return [path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    return []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/reproducibility_artifact")
    parser.add_argument("--include-checkpoints", action="store_true")
    args = parser.parse_args()

    repository = Path.cwd().resolve()
    roots = [repository / value for value in RESULT_ROOTS + SOURCE_ROOTS]
    if args.include_checkpoints:
        roots.extend([
            repository / "results/balanced_seed_study/checkpoints",
            repository / "results/training_budget_study/checkpoints",
        ])
    files = sorted({path.resolve() for root in roots for path in files_under(root)})
    output = repository / args.output
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in files:
        rows.append({
            "relative_path": str(path.relative_to(repository)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        import torch
        cuda = {
            "available": torch.cuda.is_available(),
            "runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        cuda = {"available": False, "runtime": None, "device": None}
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "cuda": cuda,
        "environment_variables_recorded": [
            name for name in ("DP_LEARNED_DEVICE", "CUDA_VISIBLE_DEVICES") if os.getenv(name) is not None
        ],
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    commands = [
        ".venv/Scripts/python.exe scripts/analyze_identifiability.py",
        ".venv/Scripts/python.exe scripts/prepare_balanced_seed_datasets.py",
        ".venv/Scripts/python.exe scripts/run_training_budget_study.py",
        ".venv/Scripts/python.exe scripts/eval_scientific_baselines.py --output-root results/balanced_seed_study/scientific_baselines_final",
        ".venv/Scripts/python.exe scripts/aggregate_balanced_seed_study.py",
        ".venv/Scripts/python.exe scripts/eval_training_budget_study.py --aggregate-only",
        ".venv/Scripts/python.exe -m unittest discover -s tests -v",
    ]
    (output / "commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(rows), "bytes": sum(row["bytes"] for row in rows),
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
