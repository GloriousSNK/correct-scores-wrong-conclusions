from __future__ import annotations

import asyncio
import inspect
import json
import os
from typing import Iterable, Callable

import numpy as np
from tqdm import tqdm

from .schema import EvalCell, Prediction, Trajectory
from .simulator import PendulumParams
from .metrics import all_pointwise_metrics
from .models.base import PredictionRequest, PredictionResult

NA_MODALITY = "coords"
NA_PROMPTING = "no_cot"


def cell_checkpoint_path(cell: EvalCell, ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, f"{cell.slug()}.json")


def load_checkpoint(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_checkpoint(path: str, prediction: Prediction) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(prediction.to_dict(), f)
    os.replace(tmp, path)


def build_eval_cells(*, predictors: dict, systems: Iterable[int],
                     regimes: Iterable[str], modalities: Iterable[str],
                     horizons: Iterable[float], prompting: Iterable[str],
                     movement_ids: dict[int, list[str]]) -> list[EvalCell]:
    cells = []
    for mn, pred in predictors.items():
        mods = list(modalities) if getattr(pred, "uses_modality", False) else [NA_MODALITY]
        proms = list(prompting) if getattr(pred, "uses_prompting", False) else [NA_PROMPTING]
        for k in systems:
            for mv in movement_ids.get(k, []):
                for regime in regimes:
                    if not mv.startswith(f"k{k}_{regime}_"):
                        continue
                    for modality in mods:
                        for horizon in horizons:
                            for prompt in proms:
                                cells.append(EvalCell(
                                    model_name=mn, k=k, regime=regime,
                                    modality=modality, horizon=horizon,
                                    prompting=prompt, movement_id=mv,
                                ))
    return cells


def _state_at(traj: Trajectory, t: float) -> np.ndarray:
    idx = int(np.argmin(np.abs(traj.times - t)))
    return traj.states[idx].copy()


def _history_slice(traj: Trajectory) -> tuple[np.ndarray, np.ndarray]:
    mask = traj.times <= 0.0
    return traj.times[mask], traj.states[mask]


async def _run_one(predictor, req: PredictionRequest,
                   true_state_at_horizon: np.ndarray,
                   params: PendulumParams,
                   ckpt_path: str) -> Prediction:
    if predictor.is_async:
        res: PredictionResult = await predictor.apredict(req)
    else:
        res = await asyncio.to_thread(predictor.predict, req)

    if res.success:
        pred_state = np.array(res.pred_theta + res.pred_omega, dtype=float)
        try:
            metrics = all_pointwise_metrics(pred_state, true_state_at_horizon, params)
        except Exception as e:
            metrics = {"metrics_error": repr(e)}
    else:
        metrics = {}

    pred = Prediction(
        cell=req.cell,
        pred_theta=res.pred_theta,
        pred_omega=res.pred_omega,
        raw_response=res.raw_response,
        cot_text=res.cot_text,
        latency_s=res.latency_s,
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        success=res.success,
        error=res.error if not res.success else None,
        metrics=metrics,
    )
    save_checkpoint(ckpt_path, pred)
    return pred


async def run_all(*, predictors: dict[str, object],
                  cells: list[EvalCell],
                  trajectories: dict[str, Trajectory],
                  params_by_cell: Callable,
                  image_for: Callable,
                  ckpt_dir: str,
                  pbar_desc: str = "eval") -> list[Prediction]:
    os.makedirs(ckpt_dir, exist_ok=True)

    try:
        params_accepts_predictor = len(inspect.signature(params_by_cell).parameters) >= 2
    except (TypeError, ValueError):
        params_accepts_predictor = False

    results: list[Prediction] = []
    pending: list[tuple[EvalCell, str]] = []
    for cell in cells:
        ckpt = cell_checkpoint_path(cell, ckpt_dir)
        existing = load_checkpoint(ckpt)
        if existing is not None:
            ec = existing.get("cell", {})
            try:
                cell_obj = EvalCell(**ec)
            except TypeError:
                cell_obj = cell
            results.append(Prediction(
                cell=cell_obj,
                pred_theta=existing.get("pred_theta", []),
                pred_omega=existing.get("pred_omega", []),
                raw_response=existing.get("raw_response", ""),
                cot_text=existing.get("cot_text", ""),
                latency_s=existing.get("latency_s", 0.0),
                prompt_tokens=existing.get("prompt_tokens", 0),
                completion_tokens=existing.get("completion_tokens", 0),
                success=existing.get("success", True),
                error=existing.get("error"),
                metrics=existing.get("metrics", {}),
            ))
            continue
        pending.append((cell, ckpt))

    if not pending:
        return results

    async def _task(cell: EvalCell, ckpt: str):
        traj = trajectories[cell.movement_id]
        predictor = predictors[cell.model_name]
        params_pack = (params_by_cell(cell, predictor)
                       if params_accepts_predictor else params_by_cell(cell))
        if len(params_pack) == 3:
            params, disclosed, scoring_params = params_pack
        else:
            params, disclosed = params_pack
            scoring_params = params
        true_state_at_horizon = _state_at(traj, cell.horizon)
        state0 = _state_at(traj, 0.0)
        image_b64 = (image_for(cell, traj)
                     if cell.modality in ("images", "images_coords")
                     and getattr(predictor, "uses_modality", False)
                     else None)
        if getattr(predictor, "uses_history", False):
            hist_t, hist_s = _history_slice(traj)
        else:
            hist_t = hist_s = None
        req = PredictionRequest(
            cell=cell, params=params, disclosed_params=disclosed,
            state0=state0, horizon=cell.horizon, image_b64=image_b64,
            history_times=hist_t, history_states=hist_s,
        )
        return await _run_one(predictor, req, true_state_at_horizon, scoring_params, ckpt)

    tasks = [asyncio.create_task(_task(c, ck)) for c, ck in pending]
    pbar = tqdm(total=len(tasks), desc=pbar_desc)
    for fut in asyncio.as_completed(tasks):
        try:
            pred = await fut
            results.append(pred)
        except Exception as e:
            pbar.write(f"task error: {e!r}")
        pbar.update(1)
    pbar.close()
    return results

