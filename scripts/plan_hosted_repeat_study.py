"""Create a deterministic hosted-model repeat sample and nominal cost estimate."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PRICING = {
    "kimi-k2.6": {"input": 0.60, "output": 2.50},
    "deepseek-v4-pro": {"input": 0.27, "output": 1.10},
    "grok-4-1-fast-reasoning": {"input": 0.20, "output": 0.50},
}


def load_records(directory: Path) -> list[dict]:
    records = []
    for path in directory.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        record["_source"] = str(path)
        records.append(record)
    return records


def nominal_cost(model: str, prompt_tokens: float, completion_tokens: float, calls: int) -> float:
    rate = PRICING[model]
    return calls * (prompt_tokens * rate["input"] + completion_tokens * rate["output"]) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-checkpoints", default="results/llm_main")
    parser.add_argument("--sysid-checkpoints", default="results/llm_sysid")
    parser.add_argument("--output", default="results/hosted_repeat_plan/plan.json")
    parser.add_argument("--trajectories-per-stratum", type=int, default=2)
    parser.add_argument("--additional-repeats", type=int, default=2,
                        help="new calls per selected cell; existing result is repeat zero")
    args = parser.parse_args()

    main_records = [record for record in load_records(Path(args.main_checkpoints))
                    if record.get("prompting") == "no_cot" and record.get("model") in PRICING]
    grouped = defaultdict(list)
    for record in main_records:
        key = (record["model"], int(record["k"]), record["regime"], float(record["horizon"]))
        grouped[key].append(record)
    selected = []
    for key in sorted(grouped):
        candidates = sorted(grouped[key], key=lambda record: record["movement_id"])
        selected.extend(candidates[:args.trajectories_per_stratum])

    historical_main = defaultdict(list)
    for record in main_records:
        historical_main[record["model"]].append(record)
    historical_sysid = defaultdict(list)
    for record in load_records(Path(args.sysid_checkpoints)):
        if record.get("model") in PRICING:
            historical_sysid[record["model"]].append(record)

    repeat_cost = 0.0
    context_cost = 0.0
    estimates = {}
    for model in PRICING:
        main = historical_main[model]
        sysid = historical_sysid[model]
        main_prompt = float(np.mean([record.get("prompt_tokens", 0) for record in main]))
        main_completion = float(np.mean([record.get("completion_tokens", 0) for record in main]))
        sysid_prompt = float(np.mean([record.get("prompt_tokens", 0) for record in sysid]))
        sysid_completion = float(np.mean([record.get("completion_tokens", 0) for record in sysid]))
        selected_count = sum(record["model"] == model for record in selected)
        model_repeat_cost = nominal_cost(
            model, main_prompt, main_completion, selected_count * args.additional_repeats
        )
        # The context prompt adds eleven observed states. A 3x input-token multiplier is
        # deliberately conservative; completion length is estimated from the old probe.
        model_context_cost = nominal_cost(model, 3.0 * sysid_prompt, sysid_completion, 300)
        repeat_cost += model_repeat_cost
        context_cost += model_context_cost
        estimates[model] = {
            "historical_main_mean_prompt_tokens": main_prompt,
            "historical_main_mean_completion_tokens": main_completion,
            "historical_sysid_mean_prompt_tokens": sysid_prompt,
            "historical_sysid_mean_completion_tokens": sysid_completion,
            "repeat_cells": selected_count,
            "repeat_new_calls": selected_count * args.additional_repeats,
            "repeat_nominal_cost_usd": model_repeat_cost,
            "context_new_calls": 300,
            "context_nominal_cost_usd": model_context_cost,
        }

    plan = {
        "selection_seed": "lexicographic first movement ids within each stratum",
        "strata": ["model", "k", "regime", "horizon"],
        "prompting": "no_cot",
        "trajectories_per_stratum": args.trajectories_per_stratum,
        "selected_existing_cells": len(selected),
        "additional_repeats_per_cell": args.additional_repeats,
        "repeat_new_calls": len(selected) * args.additional_repeats,
        "redesigned_context_calls": 900,
        "repeat_nominal_cost_usd": repeat_cost,
        "context_nominal_cost_usd": context_cost,
        "combined_nominal_cost_usd": repeat_cost + context_cost,
        "pricing_usd_per_million_tokens": PRICING,
        "estimates_by_model": estimates,
        "selected_cells": [{key: value for key, value in record.items() if key != "_source"}
                           for record in selected],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps({key: plan[key] for key in (
        "selected_existing_cells", "repeat_new_calls", "redesigned_context_calls",
        "repeat_nominal_cost_usd", "context_nominal_cost_usd", "combined_nominal_cost_usd",
    )}, indent=2))


if __name__ == "__main__":
    main()
