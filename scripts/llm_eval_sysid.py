"""Part B: controlled in-context system-identification evaluation.

Reads results/dataset_llm_sysid/sysid_eval_set.json (matched constant sets,
k in {1,2,3}). Evaluates each LLM on the SAME trajectories under two conditions:
  DISCLOSED  -> physical constants given in the prompt
  HIDDEN     -> constants omitted; model estimates dynamics from the supplied
                observations and additionally reports a parameter estimate.

Everything else is held identical (no_cot, temperature 0) so the only toggle is
disclosure -> a confound-free paired contrast. Reliability-adjusted scoring
(failure = pi/2 rad). Resumable per-cell checkpoints into results/llm_sysid/
(ADDITIVE -- never touches existing results).

Grid: 3 LLMs x 150 trajectories x horizons {1.0,10.0} x disclosure {disclosed,hidden}.
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

from bench.identifiability import equivalence_log_residual

PI_2 = math.pi / 2.0
HORIZONS = [1.0, 10.0]
DISCLOSURES = ["disclosed", "hidden"]

MODELS = {
    "kimi-k2.6":               ("kimi-k2.6", 65536, 300.0, 6),
    "deepseek-v4-pro":         ("deepseek-v4-pro", 65536, 300.0, 6),
    "grok-4-1-fast-reasoning": ("grok-4-1-fast-reasoning", 65536, 300.0, 6),
}

SYSTEM_DISCLOSED = (
    "You are a careful physicist predicting the future state of a "
    "k-pendulum (a chain of k point masses on massless rods, hinged in a "
    "planar chain, swinging under gravity with viscous damping). "
    "Respond with ONLY a single JSON object on one line, no prose, no "
    "code fences. Schema: {\"theta\": [..k floats..], \"omega\": [..k floats..]} "
    "where theta_i is in radians (link i measured from straight-down, "
    "positive counter-clockwise) and omega_i is angular velocity in rad/s."
)
SYSTEM_HIDDEN = (
    "You are a careful physicist predicting the future state of a "
    "k-pendulum (a chain of k point masses on massless rods, hinged in a "
    "planar chain, swinging under gravity with viscous damping). "
    "The physical constants are NOT disclosed; estimate the dynamics from the "
    "observed states. Respond with ONLY a single JSON object on one "
    "line, no prose, no code fences. Schema: "
    "{\"theta\": [..k floats..], \"omega\": [..k floats..], "
    "\"inferred_constants\": {\"g\": float, \"L\": [..k floats..], "
    "\"m\": [..k floats..], \"damping\": float}} "
    "where theta_i is in radians (link i measured from straight-down, positive "
    "counter-clockwise), omega_i is angular velocity in rad/s, g is gravity in "
    "m/s^2, L are rod lengths in m, m are point masses in kg, and damping is the "
    "viscous coefficient on omega. Give your best numerical estimate for every "
    "constant even if uncertain."
)
SYSTEM_CONTEXT = (
    "You are a careful physicist predicting the future state of a k-pendulum "
    "from an observed trajectory. Respond with ONLY one JSON object on one line, "
    "with no prose or code fences. Schema: {\"theta\": [..k floats..], "
    "\"omega\": [..k floats..], \"inferred_constants\": {\"g\": float, "
    "\"L\": [..k floats..], \"m\": [..k floats..], \"damping\": float}}. "
    "Give one numerical estimate for every field."
)


def constants_block_disclosed(constants: dict) -> str:
    return (f"Physical constants:\n"
            f"  g (gravity, m/s^2): {constants['g']}\n"
            f"  L (rod lengths, m): {list(constants['L'])}\n"
            f"  m (point masses, kg): {list(constants['m'])}\n"
            f"  damping (viscous coefficient on omega): {constants['damping']}\n")


CONSTANTS_BLOCK_HIDDEN = (
    "Physical constants: HIDDEN. The numerical values of gravity, rod lengths, "
    "point masses, and damping are NOT disclosed. Infer them from the dynamics.\n")


def build_user_prompt(k, theta0, omega0, disclosure, constants, horizon,
                      history=None) -> str:
    cblock = (constants_block_disclosed(constants) if disclosure == "disclosed"
              else CONSTANTS_BLOCK_HIDDEN)
    tail = (f"Return theta and omega as length-{k} JSON arrays, plus inferred_constants "
            f"(g, L[{k}], m[{k}], damping)."
            if history
            else f"Return theta and omega as length-{k} JSON arrays."
            if disclosure == "disclosed"
            else (f"Return theta and omega as length-{k} JSON arrays, plus your "
                  f"inferred_constants (g, L[{k}], m[{k}], damping)."))
    history_block = ""
    if history:
        rows = [f"  t={row['t']}: theta={row['theta']}, omega={row['omega']}"
                for row in history]
        history_block = "Observed trajectory ending at t=0:\n" + "\n".join(rows) + "\n"
    return (f"Number of links k = {k}\n"
            f"{cblock}"
            f"{history_block}"
            f"Initial state at t=0:\n"
            f"  theta_0 (rad): {list(theta0)}\n"
            f"  omega_0 (rad/s): {list(omega0)}\n"
            f"Predict the state at t = {horizon} seconds.\n"
            f"{tail}")


def _extract_json(text: str) -> dict:
    payload = text.strip()
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
    return json.loads(payload[start:end + 1])


def parse_response(text: str, k: int, disclosure: str, require_constants: bool = False):
    obj = _extract_json(text)
    theta, omega = obj.get("theta"), obj.get("omega")
    if not isinstance(theta, list) or not isinstance(omega, list):
        raise ValueError(f"theta/omega missing: {obj}")
    if len(theta) != k or len(omega) != k:
        raise ValueError(f"expected len {k}, got {len(theta)}/{len(omega)}")
    theta = [float(x) for x in theta]
    omega = [float(x) for x in omega]
    inferred = None
    if disclosure == "hidden" or require_constants:
        ic = obj.get("inferred_constants")
        if isinstance(ic, dict):
            inferred = _coerce_constants(ic, k)
    return theta, omega, inferred


def _coerce_constants(ic: dict, k: int):
    out = {}
    try:
        out["g"] = float(ic.get("g")) if ic.get("g") is not None else None
    except (TypeError, ValueError):
        out["g"] = None
    try:
        out["damping"] = (float(ic.get("damping"))
                          if ic.get("damping") is not None else None)
    except (TypeError, ValueError):
        out["damping"] = None
    for key in ("L", "m"):
        v = ic.get(key)
        if isinstance(v, list) and len(v) >= k:
            try:
                out[key] = [float(x) for x in v[:k]]
            except (TypeError, ValueError):
                out[key] = None
        else:
            out[key] = None
    return out


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


def ckpt_path(ckpt_dir, model, mv, horizon, disclosure):
    safe = f"{model}__{mv}__h{horizon:g}__{disclosure}.json"
    return os.path.join(ckpt_dir, safe)


def _save(path, rec):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, path)


async def run_cell(client, dep, max_tokens, timeout, retries, sem,
                   cell, true_theta, path):
    disclosure = cell["disclosure"]
    context_study = bool(cell.get("history"))
    sysmsg = (SYSTEM_CONTEXT if context_study else
              SYSTEM_DISCLOSED if disclosure == "disclosed" else SYSTEM_HIDDEN)
    user = build_user_prompt(cell["k"], cell["theta0"], cell["omega0"],
                             disclosure, cell.get("constants"), cell["horizon"],
                             history=cell.get("history"))
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
                theta = inferred = None
                perr = None
                for cand in (text, reasoning):
                    if not cand:
                        continue
                    try:
                        theta, omega, inferred = parse_response(
                            cand, cell["k"], disclosure, require_constants=context_study)
                        perr = None
                        break
                    except Exception as e:
                        perr = e
                if theta is None:
                    rec = {**cell, "success": False, "error": f"parse: {perr}",
                           "angle_error_mean": PI_2, "pred_theta": None,
                           "pred_omega": None, "inferred_constants": None,
                           "prompt_tokens": p_tok, "completion_tokens": c_tok,
                           "latency_s": latency}
                else:
                    identifiable = None
                    if inferred is not None:
                        try:
                            identifiable = equivalence_log_residual(
                                inferred, cell["true_constants"], cell["k"])
                        except (KeyError, TypeError, ValueError):
                            identifiable = None
                    rec = {**cell, "success": True, "error": None,
                           "pred_theta": theta, "pred_omega": omega,
                           "inferred_constants": inferred,
                           "identifiable_parameter_error": identifiable,
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
           "angle_error_mean": PI_2, "pred_theta": None, "pred_omega": None,
           "inferred_constants": None, "prompt_tokens": 0,
           "completion_tokens": 0, "latency_s": 0.0}
    _save(path, rec)
    return rec


async def amain(args):
    data = json.load(open(args.dataset))
    trajs = data["trajectories"]
    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    models = args.models or list(MODELS.keys())
    cells = []
    for model in models:
        for t in trajs:
            k = t["k"]
            theta0 = t["initial_state"]["theta"]
            omega0 = t["initial_state"]["omega"]
            for h in HORIZONS:
                tgt = t["targets"][str(h)]
                for disc in DISCLOSURES:
                    cells.append({
                        "model": model, "k": k, "set_id": t["set_id"],
                        "horizon": h, "disclosure": disc,
                        "movement_id": t["movement_id"],
                        "theta0": theta0, "omega0": omega0,
                        "constants": t["constants"], "history": t.get("history"),
                        "true_constants": t["constants"],
                        "difficulty": t.get("rk4_coarse_difficulty", {}).get(str(h)),
                        "_true_theta": tgt["theta"],
                    })

    if args.limit:
        cells = cells[:args.limit]
    print(f"Part B: {len(cells)} cells across {len(models)} models.")
    done = 0
    pending = []
    for c in cells:
        path = ckpt_path(ckpt_dir, c["model"], c["movement_id"],
                         c["horizon"], c["disclosure"])
        if os.path.exists(path):
            done += 1
        else:
            pending.append((c, path))
    print(f"  {done} already done, {len(pending)} to run.")
    if args.dry_run or not pending:
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
        tasks.append(run_cell(clients[c["model"]], dep, mt, to, 4,
                              sems[c["model"]], cell, c["_true_theta"], path))

    completed = 0
    for fut in asyncio.as_completed(tasks):
        await fut
        completed += 1
        if completed % 50 == 0 or completed == len(tasks):
            print(f"  {completed}/{len(tasks)} done", flush=True)

    for cl in clients.values():
        await cl.close()
    print("Part B complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="results/dataset_llm_sysid/sysid_eval_set.json")
    ap.add_argument("--checkpoint-dir", default="results/llm_sysid")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--limit", type=int, default=0, help="cap cells (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="plan cells without API calls")
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
