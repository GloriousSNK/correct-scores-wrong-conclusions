"""Resume pending LNN cells with independent CPU worker processes."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.metrics import all_pointwise_metrics
from bench.models.base import PredictionRequest
from bench.models.learned import LNNPredictor
from bench.runner import cell_checkpoint_path, load_checkpoint, save_checkpoint
from bench.schema import EvalCell, Prediction
from bench.simulator import PendulumParams
from scripts.run_eval import load_trajectory


_PREDICTOR = None
_CONFIG = None


def _k_values(values, k: int, default: float) -> list[float]:
    result = list(values or [default])
    while len(result) < k:
        result.append(result[-1])
    return result[:k]


def _normal_prior(config: dict, k: int) -> PendulumParams:
    normal = config["regimes"]["normal"]
    return PendulumParams.make(
        k=k, L=_k_values(normal.get("L"), k, 1.0),
        m=_k_values(normal.get("m"), k, 1.0),
        g=float(normal.get("g", 9.81)),
        damping=float(normal.get("damping", 0.0)),
    )


def _init_worker(config_path: str) -> None:
    global _PREDICTOR, _CONFIG
    os.environ["DP_LEARNED_DEVICE"] = "cpu"
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with open(config_path, "r", encoding="utf-8") as handle:
        _CONFIG = yaml.safe_load(handle)
    model = next(item for item in _CONFIG["models"] if item["name"] == "lnn")
    _PREDICTOR = LNNPredictor(
        name="lnn", checkpoint=model["checkpoint"],
        hidden=int(model.get("hidden", 256)), layers=int(model.get("layers", 3)),
    )


def _state_at(trajectory, horizon: float) -> np.ndarray:
    index = int(np.argmin(np.abs(trajectory.times - horizon)))
    return trajectory.states[index].copy()


def _run_task(task: dict) -> dict:
    trajectory = load_trajectory(task["file"])
    cell = EvalCell(**task["cell"])
    constants = trajectory.constants
    true_params = PendulumParams.make(
        k=cell.k, L=constants["L"], m=constants["m"],
        g=constants["g"], damping=constants["damping"],
    )
    prediction_params = (_normal_prior(_CONFIG, cell.k)
                         if cell.regime == "changed_hidden" else true_params)
    request = PredictionRequest(
        cell=cell, params=prediction_params,
        disclosed_params=None if cell.regime == "changed_hidden" else true_params,
        state0=_state_at(trajectory, 0.0), horizon=cell.horizon,
    )
    started = time.perf_counter()
    result = _PREDICTOR.predict(request)
    elapsed = time.perf_counter() - started
    metrics = {}
    if result.success:
        predicted = np.array(result.pred_theta + result.pred_omega, dtype=float)
        metrics = all_pointwise_metrics(predicted, _state_at(trajectory, cell.horizon), true_params)
    prediction = Prediction(
        cell=cell, pred_theta=result.pred_theta, pred_omega=result.pred_omega,
        latency_s=result.latency_s, success=result.success,
        error=result.error if not result.success else None, metrics=metrics,
    )
    save_checkpoint(task["checkpoint"], prediction)
    return {"slug": cell.slug(), "success": result.success, "elapsed": elapsed}


def _trajectory_index(movement_id: str) -> int:
    match = re.search(r"_(\d+)$", movement_id)
    return int(match.group(1)) if match else 0


def _build_tasks(dataset_dir: str, checkpoint_dir: str, horizons: list[float],
                 retry_failures: bool = False,
                 regimes: list[str] | None = None) -> list[dict]:
    with open(os.path.join(dataset_dir, "manifest.json"), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    regimes = regimes or ["normal", "changed_disclosed", "changed_hidden"]
    entries = {(int(item["k"]), item["regime"], _trajectory_index(item["movement_id"])): item
               for item in manifest}
    tasks = []
    indices = sorted({_trajectory_index(item["movement_id"]) for item in manifest})
    systems = sorted({int(item["k"]) for item in manifest})
    for index in indices:
        for regime in regimes:
            for k in systems:
                entry = entries.get((k, regime, index))
                if entry is None:
                    continue
                path = entry["file"]
                if not os.path.exists(path):
                    path = os.path.join(dataset_dir, f"{entry['movement_id']}.json")
                for horizon in horizons:
                    cell = EvalCell(
                        model_name="lnn", k=k, regime=regime, modality="coords",
                        horizon=horizon, prompting="no_cot", movement_id=entry["movement_id"],
                    )
                    checkpoint = cell_checkpoint_path(cell, checkpoint_dir)
                    existing = load_checkpoint(checkpoint)
                    if (existing is None
                            or (retry_failures and not bool(existing.get("success", True)))):
                        tasks.append({"cell": cell.__dict__, "file": path, "checkpoint": checkpoint})
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume LNN evaluation using a CPU process pool")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--horizons", type=float, nargs="+", default=[1.0, 10.0])
    parser.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument("--benchmark", action="store_true",
                        help="run at most one pending cell per horizon")
    parser.add_argument("--retry-failures", action="store_true",
                        help="rerun existing checkpoints whose success field is false")
    parser.add_argument("--regimes", nargs="+",
                        choices=["normal", "changed_disclosed", "changed_hidden"],
                        default=None, help="restrict evaluation to selected regimes")
    args = parser.parse_args()

    tasks = _build_tasks(args.dataset_dir, args.checkpoint_dir, args.horizons,
                         retry_failures=args.retry_failures, regimes=args.regimes)
    if args.benchmark:
        selected = []
        for horizon in args.horizons:
            selected.extend(task for task in tasks
                            if float(task["cell"]["horizon"]) == horizon and not any(
                                float(item["cell"]["horizon"]) == horizon for item in selected))
        tasks = selected
    print(f"Pending LNN cells: {len(tasks)}; CPU workers: {args.workers}")
    if not tasks:
        return

    started = time.perf_counter()
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.config,)) as pool:
        futures = [pool.submit(_run_task, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="lnn-cpu"):
            try:
                if not future.result()["success"]:
                    failures += 1
            except Exception as error:
                failures += 1
                tqdm.write(f"worker error: {error!r}")
    print(f"Completed {len(tasks)} cells in {time.perf_counter() - started:.1f}s; failures={failures}")


if __name__ == "__main__":
    main()
