from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from bench.schema import Trajectory, EvalCell
from bench.simulator import PendulumParams
from bench.rendering import render_state_b64
from bench.runner import build_eval_cells, run_all
from bench.models.numerical import NumericalPredictor
from bench.models.azure_llm import AzureFoundryPredictor
from bench.models.nim_llm import NimPredictor
from bench.models.timeseries import TimeSeriesPredictor
from bench.models.ts_local import LocalTimeSeriesPredictor
from bench.models.learned import NeuralODEPredictor, HNNPredictor, LNNPredictor, LearnedPredictor

def load_trajectory(path: str) -> Trajectory:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    times = np.array(d["time"])
    k = int(d["number_of_pendulums"])
    theta = np.array(d["theta"])
    omega = np.array(d["omega"])
    states = np.concatenate([theta, omega], axis=1)
    xy = np.stack([np.array(d["x"]), np.array(d["y"])], axis=-1)
    ke = np.array(d.get("kinetic_energy", []))
    pe = np.array(d["potential_energy"])
    e_total = np.array(d["total_energy"])
    if ke.size == 0:
        ke = e_total - pe
    return Trajectory(
        movement_id=d["movement_id"], k=k, regime=d["regime"],
        times=times, states=states, xy=xy,
        ke=ke, pe=pe, e_total=e_total,
        constants=d["constants"], initial_conditions=d["initial_conditions"],
    )

def load_all_trajectories(dataset_dir: str) -> dict[str, Trajectory]:
    manifest_path = os.path.join(dataset_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run `python scripts/generate_dataset.py` first."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    out = {}
    for entry in manifest:
        out[entry["movement_id"]] = load_trajectory(entry["file"])
    return out

def build_predictors(cfg: dict, only: set[str] | None = None) -> dict:
    azure_cfg = cfg.get("azure", {})
    preds = {}
    for mdef in cfg["models"]:
        name = mdef["name"]
        if only is not None and name not in only:
            continue
        kind = mdef["kind"]
        if kind == "numerical":
            preds[name] = NumericalPredictor(
                name=name, method=mdef["method"], step=float(mdef.get("step", 0.01))
            )
        elif kind == "llm":
            preds[name] = AzureFoundryPredictor(
                name=name, deployment=mdef["deployment"],
                vision=bool(mdef.get("vision", False)),
                concurrency=int(mdef.get("concurrency", 4)),
                request_timeout=float(mdef.get("request_timeout", azure_cfg.get("request_timeout", 120))),
                max_retries=int(azure_cfg.get("max_retries", 4)),
                api_version=str(azure_cfg.get("api_version", "2024-08-01-preview")),
                max_tokens=int(mdef.get("max_tokens", 16384)),
            )
        elif kind == "nim_llm":
            nim_cfg = cfg.get("nim", {})
            preds[name] = NimPredictor(
                name=name, deployment=mdef["deployment"],
                concurrency=int(mdef.get("concurrency", 4)),
                request_timeout=float(nim_cfg.get("request_timeout", 120)),
                max_retries=int(nim_cfg.get("max_retries", 4)),
            )
        elif kind == "timeseries":
            preds[name] = TimeSeriesPredictor(
                name=name, variant=mdef.get("variant", ""),
                max_prediction_length=int(mdef.get("max_prediction_length", 256)),
                max_context_length=int(mdef.get("max_context_length", 512)),
                concurrency=int(mdef.get("concurrency", 4)),
                request_timeout=float(azure_cfg.get("request_timeout", 120)),
                max_retries=int(azure_cfg.get("max_retries", 4)),
            )
        elif kind == "ts_local":
            preds[name] = LocalTimeSeriesPredictor(
                name=name, variant=mdef.get("variant", ""),
                model_id=mdef.get("model_id"),
                device=str(mdef.get("device", "auto")),
                dtype=str(mdef.get("dtype", "float32")),
                max_prediction_length=int(mdef.get("max_prediction_length", 256)),
                max_context_length=int(mdef.get("max_context_length", 512)),
                num_samples=int(mdef.get("num_samples", 20)),
                multivariate=bool(mdef.get("multivariate", False)),
            )
        elif kind == "learned":
            variant    = mdef.get("variant", "")
            checkpoint = mdef.get("checkpoint", "")
            hidden     = int(mdef.get("hidden", 256))
            layers     = int(mdef.get("layers", 3))
            if variant == "neural_ode":
                preds[name] = NeuralODEPredictor(name=name, checkpoint=checkpoint,
                                                  hidden=hidden, layers=layers)
            elif variant == "hnn":
                preds[name] = HNNPredictor(name=name, checkpoint=checkpoint,
                                            hidden=hidden, layers=layers)
            elif variant == "lnn":
                preds[name] = LNNPredictor(name=name, checkpoint=checkpoint,
                                            hidden=hidden, layers=layers)
            else:
                preds[name] = LearnedPredictor(name=name, variant=variant,
                                               checkpoint=checkpoint)
        else:
            raise ValueError(f"unknown model kind: {kind}")
    return preds

LEARNED_PREDICTOR_TYPES = (NeuralODEPredictor, HNNPredictor, LNNPredictor, LearnedPredictor)


def _k_values(values, k: int, default: float) -> list[float]:
    vals = list(values or [])
    if not vals:
        vals = [default]
    while len(vals) < k:
        vals.append(vals[-1])
    return vals[:k]


def _normal_prior_params(cfg: dict, k: int) -> PendulumParams:
    normal = cfg.get("regimes", {}).get("normal", {})
    return PendulumParams.make(
        k=k,
        L=_k_values(normal.get("L"), k, 1.0),
        m=_k_values(normal.get("m"), k, 1.0),
        g=float(normal.get("g", 9.81)),
        damping=float(normal.get("damping", 0.0)),
    )


def make_params_for_cell(
    cfg: dict,
    trajectories: dict[str, Trajectory],
    *,
    withhold_hidden_constants_for_learned: bool = False,
):
    def fn(cell: EvalCell, predictor=None):
        traj = trajectories[cell.movement_id]
        c = traj.constants
        true_p = PendulumParams.make(k=cell.k, L=c["L"], m=c["m"],
                                     g=c["g"], damping=c["damping"])
        if cell.regime == "changed_hidden":
            disclosed = None
        else:
            disclosed = true_p

        prediction_p = true_p
        if (withhold_hidden_constants_for_learned
                and cell.regime == "changed_hidden"
                and isinstance(predictor, LEARNED_PREDICTOR_TYPES)):
            prediction_p = _normal_prior_params(cfg, cell.k)
        return prediction_p, disclosed, true_p
    return fn

def make_image_renderer(cfg: dict, cache: dict, trajectories: dict[str, Trajectory]):
    rcfg = cfg.get("rendering", {})
    size = tuple(rcfg.get("image_size", [512, 512]))
    background = rcfg.get("background", "#0a0a0a")
    grid = bool(rcfg.get("grid", True))

    def fn(cell: EvalCell, traj: Trajectory) -> str:
        key = (cell.movement_id, "t0")
        if key in cache:
            return cache[key]
        c = traj.constants
        p = PendulumParams.make(k=cell.k, L=c["L"], m=c["m"],
                                g=c["g"], damping=c["damping"])
        idx0 = int(np.argmin(np.abs(traj.times - 0.0)))
        theta0 = traj.states[idx0, :cell.k]
        b64 = render_state_b64(theta0, p, size=size,
                               background=background, grid=grid)
        cache[key] = b64
        return b64
    return fn

def trajectories_by_k(trajectories: dict[str, Trajectory]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for mv_id, traj in trajectories.items():
        out.setdefault(traj.k, []).append(mv_id)
    return out

async def _amain(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_dir = args.dataset_dir or cfg["paths"]["dataset_dir"]
    trajectories = load_all_trajectories(dataset_dir)
    mv_by_k = trajectories_by_k(trajectories)

    only = set(args.models) if args.models else None
    predictors = build_predictors(cfg, only=only)
    if not predictors:
        print("No predictors selected. Use --models name1 name2 or remove the filter.")
        return

    systems = cfg["dataset"]["systems"]
    regimes = list(cfg["regimes"].keys())
    modalities = cfg["modalities"]
    horizons = args.horizons if args.horizons is not None else cfg["horizons_seconds"]
    prompting = cfg["prompting"]

    if args.smoke:
        regimes = regimes[:1]
        modalities = ["coords"]
        horizons = [horizons[0], horizons[1]]
        prompting = ["no_cot"]
        trimmed = {}
        seen = set()
        for mv, traj in trajectories.items():
            key = (traj.k, traj.regime)
            if key in seen:
                continue
            if traj.regime not in regimes:
                continue
            seen.add(key)
            trimmed[mv] = traj
        trajectories = trimmed
        mv_by_k = trajectories_by_k(trajectories)

    cells = build_eval_cells(
        predictors=predictors, systems=systems, regimes=regimes,
        modalities=modalities, horizons=horizons, prompting=prompting,
        movement_ids=mv_by_k,
    )
    print(f"Planned {len(cells)} eval cells across {len(predictors)} models.")

    params_fn = make_params_for_cell(
        cfg, trajectories,
        withhold_hidden_constants_for_learned=args.withhold_hidden_constants_for_learned,
    )
    image_fn = make_image_renderer(cfg, cache={}, trajectories=trajectories)

    ckpt_dir = args.checkpoint_dir or cfg["paths"]["checkpoints_dir"]
    if args.withhold_hidden_constants_for_learned and args.checkpoint_dir is None:
        ckpt_dir = ckpt_dir.rstrip("/\\") + "_hidden_prior"
        print(f"Withholding hidden-regime constants for learned predictors; writing to {ckpt_dir}/")
    results = await run_all(
        predictors=predictors, cells=cells, trajectories=trajectories,
        params_by_cell=params_fn, image_for=image_fn, ckpt_dir=ckpt_dir,
        pbar_desc="eval",
    )

    for p in predictors.values():
        close = getattr(p, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception:
                pass

    print(f"Done. {len(results)} results in {ckpt_dir}/")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="*",
                    help="restrict to these model names (default: all in config)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run for smoke testing")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="override checkpoints_dir from config (useful for best-of-N sampling)")
    ap.add_argument("--dataset-dir", default=None,
                    help="override dataset_dir from config (useful for held-out seed sweeps)")
    ap.add_argument("--horizons", type=float, nargs="*", default=None,
                    help="override config horizons_seconds")
    ap.add_argument("--withhold-hidden-constants-for-learned", action="store_true",
                    help="for changed_hidden learned-model cells, pass normal-regime constants as a neutral prior while scoring against true constants")
    args = ap.parse_args()
    asyncio.run(_amain(args))

if __name__ == "__main__":
    main()


