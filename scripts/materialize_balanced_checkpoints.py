"""Copy the first two seeds into the balanced five-seed checkpoint layout."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


MODELS = {
    "rk4", "symplectic", "euler", "chronos2-uni", "chronos2-multi",
    "neural-ode-rollout-mixed", "hnn-rollout", "hnn-rollout-mixed", "lnn",
}
LEARNED = {"neural-ode-rollout-mixed", "hnn-rollout", "hnn-rollout-mixed", "lnn"}


def trajectory_index(movement_id: str) -> int:
    match = re.search(r"_(\d+)$", movement_id)
    if match is None:
        raise ValueError(movement_id)
    return int(match.group(1))


def copy_selected(source: Path, destination: Path, *, allowed_models: set[str],
                  regime_filter=None, per_cell: int = 5) -> int:
    copied = 0
    destination.mkdir(parents=True, exist_ok=True)
    prefixes = tuple(f"{model}__" for model in allowed_models)
    for path in source.glob("*.json"):
        if not path.name.startswith(prefixes):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            cell = record["cell"]
        except Exception:
            continue
        if cell["model_name"] not in allowed_models or float(cell["horizon"]) not in (1.0, 10.0):
            continue
        if trajectory_index(cell["movement_id"]) >= per_cell:
            continue
        if regime_filter is not None and not regime_filter(cell):
            continue
        shutil.copy2(path, destination / path.name)
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="results/balanced_seed_study/checkpoints")
    parser.add_argument("--trajectories-per-cell", type=int, default=5)
    args = parser.parse_args()
    root = Path(args.output_root)

    seed20 = root / "seed_20260620"
    base_models = MODELS - {"chronos2-uni", "chronos2-multi"}
    copy_selected(
        Path("results/checkpoints_boot"), seed20, allowed_models=base_models,
        regime_filter=lambda cell: not (cell["model_name"] in LEARNED
                                        and cell["regime"] == "changed_hidden"),
        per_cell=args.trajectories_per_cell,
    )
    copy_selected(
        Path("results/checkpoints_boot_hidden_prior"), seed20, allowed_models=LEARNED,
        regime_filter=lambda cell: cell["regime"] == "changed_hidden",
        per_cell=args.trajectories_per_cell,
    )
    copy_selected(
        Path("results/checkpoints_tsboot"), seed20,
        allowed_models={"chronos2-uni", "chronos2-multi"},
        per_cell=args.trajectories_per_cell,
    )

    seed21 = root / "seed_20260621"
    copy_selected(
        Path("results/seed_sweep_two_seed_focused/checkpoints/seed_20260621"),
        seed21, allowed_models=MODELS, per_cell=args.trajectories_per_cell,
    )

    counts = {}
    for directory in (seed20, seed21):
        by_model = {model: 0 for model in sorted(MODELS)}
        for path in directory.glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            model = record["cell"]["model_name"]
            if model in by_model:
                by_model[model] += 1
        counts[directory.name] = by_model
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
