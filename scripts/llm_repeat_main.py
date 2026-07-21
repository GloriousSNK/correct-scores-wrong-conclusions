"""Execute the precomputed stratified hosted-model repeat plan.

This script makes paid API calls.  It is intentionally separate from the
offline planner and is resumable by replicate checkpoint.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.llm_eval_main import MODELS, base_url, run_cell


def checkpoint_path(root: Path, cell: dict, replicate: int) -> Path:
    name = (f"{cell['model']}__{cell['movement_id']}__h{float(cell['horizon']):g}__"
            f"{cell['prompting']}__rep{replicate}.json")
    return root / name


async def amain(args) -> None:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    heldout = json.loads(Path(args.heldout).read_text(encoding="utf-8"))
    target_map = {}
    for trajectory in heldout["trajectories"]:
        for horizon, target in trajectory["targets"].items():
            target_map[(trajectory["movement_id"], float(horizon))] = target["theta"]

    root = Path(args.checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    pending = []
    for selected in plan["selected_cells"]:
        cell = {
            key: selected[key] for key in (
                "model", "k", "regime", "horizon", "prompting", "movement_id",
                "theta0", "omega0", "constants",
            )
        }
        true_theta = target_map[(cell["movement_id"], float(cell["horizon"]))]
        for replicate in range(1, int(plan["additional_repeats_per_cell"]) + 1):
            path = checkpoint_path(root, cell, replicate)
            if not path.exists():
                pending.append((cell, true_theta, path, replicate))
    if args.limit:
        pending = pending[:args.limit]
    print(f"Paid repeat study: {len(pending)} pending calls")
    if args.dry_run or not pending:
        return

    endpoint = base_url()
    key = os.getenv("AZURE_AI_FOUNDRY_KEY")
    clients = {}
    semaphores = {}
    for model, (deployment, _, timeout, concurrency) in MODELS.items():
        clients[model] = AsyncOpenAI(base_url=endpoint, api_key=key, timeout=timeout, max_retries=0)
        semaphores[model] = asyncio.Semaphore(concurrency)

    tasks = []
    for cell, true_theta, path, replicate in pending:
        deployment, max_tokens, timeout, _ = MODELS[cell["model"]]
        tagged = {**cell, "replicate": replicate}
        tasks.append(run_cell(
            clients[cell["model"]], cell["model"], deployment, max_tokens, timeout, 4,
            semaphores[cell["model"]], tagged, true_theta, str(path),
        ))
    completed = 0
    for future in asyncio.as_completed(tasks):
        await future
        completed += 1
        if completed % 25 == 0 or completed == len(tasks):
            print(f"  {completed}/{len(tasks)}", flush=True)
    for client in clients.values():
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="results/hosted_repeat_plan/plan.json")
    parser.add_argument("--heldout", default="results/heldout_llm_eval_set.json")
    parser.add_argument("--checkpoint-dir", default="results/llm_repeat/main")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    asyncio.run(amain(parser.parse_args()))


if __name__ == "__main__":
    main()
