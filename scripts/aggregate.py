from __future__ import annotations

import argparse
import json
import os
import sys
import glob
from collections import defaultdict

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def flatten_metrics(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        if isinstance(v, list):
            for i, val in enumerate(v):
                out[f"{k}_{i}"] = val
        else:
            out[k] = v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="override checkpoints_dir from config")
    ap.add_argument("--summary-dir", default=None,
                    help="override summary_dir from config")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt_dir = args.checkpoint_dir or cfg["paths"]["checkpoints_dir"]
    summary_dir = args.summary_dir or cfg["paths"]["summary_dir"]
    os.makedirs(summary_dir, exist_ok=True)

    rows = []
    for path in glob.glob(os.path.join(ckpt_dir, "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        cell = d.get("cell", {})
        base = {
            "model": cell.get("model_name"),
            "k": cell.get("k"),
            "regime": cell.get("regime"),
            "modality": cell.get("modality"),
            "horizon": cell.get("horizon"),
            "prompting": cell.get("prompting"),
            "movement_id": cell.get("movement_id"),
            "success": d.get("success"),
            "error": d.get("error"),
            "latency_s": d.get("latency_s"),
            "prompt_tokens": d.get("prompt_tokens"),
            "completion_tokens": d.get("completion_tokens"),
        }
        base.update(flatten_metrics(d.get("metrics", {}) or {}))
        rows.append(base)

    if not rows:
        print(f"No checkpoints found in {ckpt_dir}/. Run eval first.")
        return

    df = pd.DataFrame(rows)
    long_path = os.path.join(summary_dir, "results_long.csv")
    df.to_csv(long_path, index=False)
    print(f"Wrote {long_path} ({len(df)} rows)")

    group_cols = ["model", "k", "modality", "horizon", "prompting", "regime"]
    metric_cols = [c for c in df.columns if c.startswith(("coord_error",
                                                          "angle_error",
                                                          "delta_",
                                                          "omega_mag",
                                                          "sign_match"))]
    metric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(df[c])]
    agg = df.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()
    success_rate = (df.groupby(group_cols, dropna=False)["success"]
                      .mean().reset_index().rename(columns={"success": "success_rate"}))
    agg = agg.merge(success_rate, on=group_cols)
    lead_path = os.path.join(summary_dir, "leaderboard.csv")
    agg.to_csv(lead_path, index=False)
    print(f"Wrote {lead_path} ({len(agg)} rows)")

    agg.to_json(os.path.join(summary_dir, "leaderboard.json"),
                orient="records", indent=2)

if __name__ == "__main__":
    main()
