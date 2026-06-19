"""
Aggregate N independent evaluation runs into a best-of-N checkpoint directory.

For each cell slug, collects pred_theta/pred_omega from all N runs that
succeeded, takes the element-wise circular median, recomputes metrics against
the true trajectory state, and writes merged checkpoints to --output-dir.

Usage:
    python scripts/best_of_n.py \\
        --runs results/checkpoints results/checkpoints_bon/run2 \\
                results/checkpoints_bon/run3 results/checkpoints_bon/run4 \\
                results/checkpoints_bon/run5 \\
        --output-dir results/checkpoints_bon/merged \\
        --dataset-dir results/dataset \\
        --config config.yaml \\
        --models kimi-k2.6
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.metrics import all_pointwise_metrics
from bench.simulator import PendulumParams


def load_json(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_trajectories(dataset_dir: str) -> dict[str, dict]:
    manifest_path = os.path.join(dataset_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    trajs = {}
    for entry in manifest:
        with open(entry["file"]) as f:
            t = json.load(f)
        trajs[t["movement_id"]] = t
    return trajs


def true_state_at(traj: dict, horizon: float) -> np.ndarray:
    times = np.array(traj["time"])
    idx = int(np.argmin(np.abs(times - horizon)))
    theta = np.array(traj["theta"])[idx]
    omega = np.array(traj["omega"])[idx]
    return np.concatenate([theta, omega])


def circular_median(angles: np.ndarray) -> np.ndarray:
    """Element-wise median of angles in radians via wrap-and-median."""
    wrapped = (angles + np.pi) % (2 * np.pi) - np.pi
    return np.median(wrapped, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Checkpoint directories for each run (in order)")
    ap.add_argument("--output-dir", required=True,
                    help="Output directory for merged checkpoints")
    ap.add_argument("--dataset-dir", default="results/dataset")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Only process cells from these models (default: all)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Loading trajectories from {args.dataset_dir} …")
    trajs = load_trajectories(args.dataset_dir)
    print(f"  {len(trajs)} trajectories loaded.")

    # Collect all cell slugs across all runs
    all_slugs: set[str] = set()
    for run_dir in args.runs:
        for path in glob.glob(os.path.join(run_dir, "*.json")):
            slug = os.path.splitext(os.path.basename(path))[0]
            all_slugs.add(slug)

    print(f"Found {len(all_slugs)} unique cell slugs across {len(args.runs)} runs.")

    os.makedirs(args.output_dir, exist_ok=True)

    merged = 0
    skipped_no_success = 0
    skipped_wrong_model = 0

    for slug in sorted(all_slugs):
        # Load all N checkpoints for this slug
        checkpoints = []
        for run_dir in args.runs:
            d = load_json(os.path.join(run_dir, f"{slug}.json"))
            if d is not None:
                checkpoints.append(d)

        if not checkpoints:
            continue

        # Model filter
        cell = checkpoints[0].get("cell", {})
        model_name = cell.get("model_name", "")
        if args.models and model_name not in args.models:
            skipped_wrong_model += 1
            continue

        # Collect successful predictions
        successful = [d for d in checkpoints if d.get("success") and
                      d.get("pred_theta") and d.get("pred_omega")]

        if not successful:
            # Write through the most recent attempt (failure record)
            out = os.path.join(args.output_dir, f"{slug}.json")
            with open(out, "w") as f:
                json.dump(checkpoints[-1], f)
            skipped_no_success += 1
            continue

        # Compute element-wise circular median of theta and omega
        thetas = np.array([d["pred_theta"] for d in successful])   # (n_succ, k)
        omegas = np.array([d["pred_omega"] for d in successful])    # (n_succ, k)

        med_theta = circular_median(thetas).tolist()
        med_omega = np.median(omegas, axis=0).tolist()

        # Recompute metrics for the median prediction
        k = int(cell.get("k", len(med_theta)))
        mv_id = cell.get("movement_id", "")
        horizon = float(cell.get("horizon", 1.0))

        traj = trajs.get(mv_id)
        if traj is None:
            metrics = {}
        else:
            c = traj["constants"]
            params = PendulumParams.make(k=k, L=c["L"], m=c["m"],
                                         g=c["g"], damping=c["damping"])
            pred_state = np.array(med_theta + med_omega, dtype=float)
            true_state = true_state_at(traj, horizon)
            try:
                metrics = all_pointwise_metrics(pred_state, true_state, params)
            except Exception as e:
                metrics = {"metrics_error": repr(e)}

        # Build merged checkpoint (use first checkpoint as template)
        ref = successful[0]
        merged_ckpt = {
            "cell": ref["cell"],
            "pred_theta": med_theta,
            "pred_omega": med_omega,
            "raw_response": f"[best-of-{len(successful)}-median from {len(checkpoints)} runs]",
            "cot_text": "",
            "latency_s": float(np.mean([d.get("latency_s", 0) for d in successful])),
            "prompt_tokens": int(np.mean([d.get("prompt_tokens", 0) for d in successful])),
            "completion_tokens": int(np.mean([d.get("completion_tokens", 0) for d in successful])),
            "success": True,
            "error": None,
            "metrics": metrics,
            "_bon_n_runs": len(checkpoints),
            "_bon_n_success": len(successful),
        }

        out = os.path.join(args.output_dir, f"{slug}.json")
        with open(out, "w") as f:
            json.dump(merged_ckpt, f)
        merged += 1

    print(f"\nDone.")
    print(f"  Merged (median taken): {merged}")
    print(f"  All runs failed:       {skipped_no_success}")
    print(f"  Skipped (wrong model): {skipped_wrong_model}")
    print(f"  Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
