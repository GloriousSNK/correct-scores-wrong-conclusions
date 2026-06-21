"""Part A: powered LLM evaluation on the shared out-of-sample held-out set.

Self-contained (does not need the benchmark harness): reads
results/heldout_llm_eval_set.json, calls the OpenAI-compatible Azure Foundry
API, scores reliability-adjusted angle error, writes resumable per-cell
checkpoints into results/llm_main/ (ADDITIVE — never touches existing results).

Grid: 3 LLMs x 180 trajectories x horizons {1.0, 10.0} x prompting {no_cot, cot}.
changed_hidden -> constants omitted; normal/changed_disclosed -> constants given.
Single sample per cell (no best-of-N).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import AsyncOpenAI

PI_2 = math.pi / 2.0
HORIZONS = [1.0, 10.0]
PROMPTING = ["no_cot", "cot"]

MODELS = {
    # name: (deployment, max_tokens, timeout, concurrency)
    "kimi-k2.6":               ("kimi-k2.6", 65536, 300.0, 6),
    "deepseek-v4-pro":         ("deepseek-v4-pro", 65536, 300.0, 6),
    "grok-4-1-fast-reasoning": ("grok-4-1-fast-reasoning", 65536, 300.0, 6),
}

SYSTEM_NOCOT = (
    "You are a careful physicist predicting the future state of a "
    "k-pendulum (a chain of k point masses on massless rods, hinged in a "
    "planar chain, swinging under gravity with viscous damping). "
    "Respond with ONLY a single JSON object on one line, no prose, no "
    "code fences. Schema: {\"theta\": [..k floats..], \"omega\": [..k floats..]} "
    "where theta_i is in radians (link i measured from straight-down, "
    "positive counter-clockwise) and omega_i is angular velocity in rad/s."
)
SYSTEM_COT = (
    "You are a careful physicist predicting the future state of a "
    "k-pendulum (a chain of k point masses on massless rods, hinged in a "
    "planar chain, swinging under gravity with viscous damping). "
    "Reason step-by-step about the dynamics, then output your final answer. "
    "Wrap the final answer in <answer>...</answer> tags containing ONLY a "
    "single JSON object: {\"theta\": [..k floats..], \"omega\": [..k floats..]}."
)


def constants_block(constants: dict | None) -> str:
    if constants is None:
        return ("Physical constants: HIDDEN. The numerical values of gravity, "
                "rod lengths, point masses, and damping are NOT disclosed. "
                "You must infer dynamics from the initial conditions alone.")
    return (f"Physical constants:\n"
            f"  g (gravity, m/s^2): {constants['g']}\n"
            f"  L (rod lengths, m): {list(constants['L'])}\n"
            f"  m (point masses, kg): {list(constants['m'])}\n"
            f"  damping (viscous coefficient on omega): {constants['damping']}\n")


def build_user_prompt(k, theta0, omega0, constants, horizon) -> str:
    return (f"Number of links k = {k}\n"
            f"{constants_block(constants)}"
            f"Initial state at t=0:\n"
            f"  theta_0 (rad): {list(theta0)}\n"
            f"  omega_0 (rad/s): {list(omega0)}\n"
            f"Predict the state at t = {horizon} seconds.\n"
            f"Return theta and omega as length-{k} JSON arrays.")


def parse_response(text: str, k: int, used_cot: bool):
    payload = text.strip()
    if used_cot:
        i = payload.find("<answer>")
        j = payload.rfind("</answer>")
        if i != -1 and j != -1 and j > i:
            payload = payload[i + len("<answer>"):j].strip()
    if payload.startswith("```"):
        payload = payload.strip("`")
        nl = payload.find("\n")
        if nl != -1:
            payload = payload[nl + 1:]
        if payload.endswith("```"):
            payload = payload[:-3]
        payload = payload.strip()
    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object found: {text[:160]!r}")
    obj = json.loads(payload[start:end + 1])
    theta, omega = obj.get("theta"), obj.get("omega")
    if not isinstance(theta, list) or not isinstance(omega, list):
        raise ValueError(f"theta/omega missing: {obj}")
    if len(theta) != k or len(omega) != k:
        raise ValueError(f"expected len {k}, got {len(theta)}/{len(omega)}")
    return [float(x) for x in theta], [float(x) for x in omega]


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def angle_error_mean(pred_theta, true_theta) -> float:
    return float(np.abs(wrap(np.asarray(pred_theta) - np.asarray(true_theta))).mean())


def base_url() -> str:
    ep = os.getenv("AZURE_AI_FOUNDRY_ENDPOINT")
    if not ep:
        sys.exit("Missing AZURE_AI_FOUNDRY_ENDPOINT in .env")
    ep = ep.rstrip("/")
    for suffix in ("/models", "/openai/v1", "/openai"):
        if ep.endswith(suffix):
            ep = ep[:-len(suffix)]
            break
    return ep.rstrip("/") + "/openai/v1"


def ckpt_path(ckpt_dir, model, mv, horizon, prompting):
    safe = f"{model}__{mv}__h{horizon:g}__{prompting}.json"
    return os.path.join(ckpt_dir, safe)


async def run_cell(client, model, dep, max_tokens, timeout, retries, sem,
                   cell, true_theta, path):
    used_cot = cell["prompting"] == "cot"
    sysmsg = SYSTEM_COT if used_cot else SYSTEM_NOCOT
    user = build_user_prompt(cell["k"], cell["theta0"], cell["omega0"],
                             cell["constants"], cell["horizon"])
    messages = [{"role": "system", "content": sysmsg},
                {"role": "user", "content": user}]
    last_err = None
    async with sem:
        for attempt in range(retries + 1):
            t0 = time.perf_counter()
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        messages=messages, model=dep,
                        temperature=0.0, max_tokens=max_tokens),
                    timeout=timeout)
                latency = time.perf_counter() - t0
                msg = resp.choices[0].message
                text = msg.content or ""
                reasoning = getattr(msg, "reasoning_content", None)
                if reasoning is None and getattr(msg, "model_extra", None):
                    reasoning = msg.model_extra.get("reasoning_content")
                usage = getattr(resp, "usage", None)
                p_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
                c_tok = int(getattr(usage, "completion_tokens", 0) or 0)
                theta = None
                perr = None
                for cand in (text, reasoning):
                    if not cand:
                        continue
                    try:
                        theta, omega = parse_response(cand, cell["k"], used_cot)
                        perr = None
                        break
                    except Exception as e:
                        perr = e
                if theta is None:
                    rec = {**cell, "success": False, "error": f"parse: {perr}",
                           "angle_error_mean": PI_2, "prompt_tokens": p_tok,
                           "completion_tokens": c_tok, "latency_s": latency}
                else:
                    rec = {**cell, "success": True, "error": None,
                           "pred_theta": theta, "pred_omega": omega,
                           "angle_error_mean": angle_error_mean(theta, true_theta),
                           "prompt_tokens": p_tok, "completion_tokens": c_tok,
                           "latency_s": latency}
                _save(path, rec)
                return rec
            except Exception as e:
                last_err = e
                if attempt < retries:
                    await asyncio.sleep(min(2 ** attempt + 0.5, 30.0))
    rec = {**cell, "success": False, "error": f"api: {last_err!r}",
           "angle_error_mean": PI_2, "prompt_tokens": 0,
           "completion_tokens": 0, "latency_s": 0.0}
    _save(path, rec)
    return rec


def _save(path, rec):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, path)


async def amain(args):
    data = json.load(open(args.heldout))
    trajs = data["trajectories"]
    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    models = args.models or list(MODELS.keys())
    # build pending cell list
    cells = []
    for model in models:
        for t in trajs:
            k = t["k"]
            theta0 = t["initial_state"]["theta"]
            omega0 = t["initial_state"]["omega"]
            for h in HORIZONS:
                tgt = t["targets"][str(h)]
                for pr in PROMPTING:
                    disclosed = None if t["regime"] == "changed_hidden" else t["constants"]
                    cells.append({
                        "model": model, "k": k, "regime": t["regime"],
                        "horizon": h, "prompting": pr,
                        "movement_id": t["movement_id"],
                        "theta0": theta0, "omega0": omega0,
                        "constants": disclosed,
                        "_true_theta": tgt["theta"],
                    })

    if args.limit:
        cells = cells[:args.limit]
    print(f"Part A: {len(cells)} cells across {len(models)} models.")
    done = 0
    pending = []
    for c in cells:
        path = ckpt_path(ckpt_dir, c["model"], c["movement_id"],
                         c["horizon"], c["prompting"])
        if os.path.exists(path):
            done += 1
        else:
            pending.append((c, path))
    print(f"  {done} already done, {len(pending)} to run.")
    if not pending:
        return

    bu = base_url()
    key = os.getenv("AZURE_AI_FOUNDRY_KEY")
    clients = {}
    sems = {}
    for model in models:
        dep, mt, to, conc = MODELS[model]
        clients[model] = AsyncOpenAI(base_url=bu, api_key=key, timeout=to, max_retries=0)
        sems[model] = asyncio.Semaphore(conc)

    tasks = []
    for c, path in pending:
        dep, mt, to, conc = MODELS[c["model"]]
        cell = {kk: vv for kk, vv in c.items() if kk != "_true_theta"}
        tasks.append(run_cell(clients[c["model"]], c["model"], dep, mt, to, 4,
                              sems[c["model"]], cell, c["_true_theta"], path))

    completed = 0
    for fut in asyncio.as_completed(tasks):
        await fut
        completed += 1
        if completed % 50 == 0 or completed == len(tasks):
            print(f"  {completed}/{len(tasks)} done", flush=True)

    for cl in clients.values():
        await cl.close()
    print("Part A complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", default="results/heldout_llm_eval_set.json")
    ap.add_argument("--checkpoint-dir", default="results/llm_main")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--limit", type=int, default=0, help="cap cells (smoke test)")
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
