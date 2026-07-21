"""Prepare five held-out pendulum seeds with five trajectories per cell.

The first two seeds reuse existing trajectory files through small manifest-only
views.  The remaining seeds are generated in the study directory.  No source
dataset is modified or copied.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def trajectory_index(movement_id: str) -> int:
    match = re.search(r"_(\d+)$", movement_id)
    if match is None:
        raise ValueError(f"Movement id has no trailing index: {movement_id}")
    return int(match.group(1))


def manifest_view(source: Path, destination: Path, *, seed: int, per_cell: int) -> None:
    entries = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    selected = [entry for entry in entries if trajectory_index(entry["movement_id"]) < per_cell]
    expected = 3 * 3 * per_cell
    if len(selected) != expected:
        raise RuntimeError(f"Expected {expected} entries for seed {seed}; found {len(selected)}")
    for entry in selected:
        path = Path(entry["file"])
        if not path.is_absolute():
            path = Path.cwd() / path
        entry["file"] = str(path.resolve())
        entry["seed"] = seed
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--root", default="results/balanced_seed_study/datasets")
    parser.add_argument("--trajectories-per-cell", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root)
    per_cell = args.trajectories_per_cell
    manifest_view(Path("results/dataset_boot"), root / "seed_20260620",
                  seed=20260620, per_cell=per_cell)
    manifest_view(Path("results/seed_sweep_two_seed_focused/datasets/seed_20260621"),
                  root / "seed_20260621", seed=20260621, per_cell=per_cell)

    for seed in (20260622, 20260623, 20260624):
        destination = root / f"seed_{seed}"
        manifest = destination / "manifest.json"
        if manifest.exists():
            entries = json.loads(manifest.read_text(encoding="utf-8"))
            if len(entries) == 3 * 3 * per_cell:
                print(f"Reusing {manifest}")
                continue
            raise RuntimeError(f"Incomplete existing manifest: {manifest}")
        command = [
            sys.executable, "scripts/generate_dataset.py", "--config", args.config,
            "--seed", str(seed), "--output-dir", str(destination),
            "--id-prefix", f"seed_{seed}__", "--trajectories-per-cell", str(per_cell),
            "--resume",
        ]
        print("+", " ".join(command))
        subprocess.run(command, check=True)

    summary = {}
    for manifest in sorted(root.glob("seed_*/manifest.json")):
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        summary[manifest.parent.name] = len(entries)
        normal = [entry for entry in entries if entry["regime"] == "normal"]
        normal_dir = root.parent / "datasets_normal" / manifest.parent.name
        normal_dir.mkdir(parents=True, exist_ok=True)
        (normal_dir / "manifest.json").write_text(json.dumps(normal, indent=2), encoding="utf-8")
    (root.parent / "dataset_inventory.json").write_text(
        json.dumps({"trajectories_per_cell": per_cell, "seeds": summary}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
