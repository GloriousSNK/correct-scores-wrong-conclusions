from __future__ import annotations

import asyncio
import os
import time
from typing import Optional

from ..prompts import (SYSTEM_COT, SYSTEM_NOCOT, build_user_prompt,
                       parse_response)
from .base import PredictionRequest, PredictionResult

try:
    from openai import AsyncOpenAI
    _HAVE_AZURE = True
except ImportError:
    _HAVE_AZURE = False

def _resolve_endpoint_and_key(model_name: str) -> tuple[str, str]:
    upper = model_name.upper().replace("-", "_").replace(".", "_")
    ep = os.getenv(f"AZURE_{upper}_ENDPOINT") or os.getenv("AZURE_AI_FOUNDRY_ENDPOINT")
    key = os.getenv(f"AZURE_{upper}_KEY") or os.getenv("AZURE_AI_FOUNDRY_KEY")
    if not ep or not key:
        raise RuntimeError(
            f"Missing endpoint/key for model {model_name!r}. Set "
            f"AZURE_AI_FOUNDRY_ENDPOINT and AZURE_AI_FOUNDRY_KEY in .env, or "
            f"the per-model AZURE_{upper}_ENDPOINT / AZURE_{upper}_KEY."
        )
    return ep, key

def _to_openai_base_url(endpoint: str) -> str:
    """Foundry AIServices resources serve an OpenAI-compatible /openai/v1 route.
    The legacy /models inference path 404s on these resources, so normalize any
    configured endpoint (with or without a trailing /models) to /openai/v1."""
    ep = endpoint.rstrip("/")
    for suffix in ("/models", "/openai/v1", "/openai"):
        if ep.endswith(suffix):
            ep = ep[: -len(suffix)]
            break
    return ep.rstrip("/") + "/openai/v1"

class AzureFoundryPredictor:
    is_async = True
    uses_modality = True
    uses_prompting = True
    uses_history = False

    def __init__(self, name: str, deployment: str, *, vision: bool = False,
                 concurrency: int = 6, request_timeout: float = 120.0,
                 max_retries: int = 4, api_version: str = "2024-08-01-preview"):
        if not _HAVE_AZURE:
            raise RuntimeError(
                "openai SDK is not installed. Run "
                "`pip install -r requirements.txt`."
            )
        self.name = name
        self.deployment = deployment
        self.vision = vision
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.api_version = api_version
        self._sem = asyncio.Semaphore(concurrency)
        self._client: Optional[AsyncOpenAI] = None

    async def _ensure_client(self):
        if self._client is None:
            ep, key = _resolve_endpoint_and_key(self.name)
            self._client = AsyncOpenAI(
                base_url=_to_openai_base_url(ep),
                api_key=key,
                timeout=self.request_timeout,
                max_retries=0,  # retry/backoff handled in apredict
            )

    async def aclose(self):
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def apredict(self, req: PredictionRequest) -> PredictionResult:
        await self._ensure_client()
        cell = req.cell
        used_cot = (cell.prompting == "cot")
        system_msg = SYSTEM_COT if used_cot else SYSTEM_NOCOT
        user_text = build_user_prompt(req, k=cell.k, modality=cell.modality)

        modality = cell.modality
        if modality in ("images", "images_coords"):
            if not self.vision:
                return PredictionResult(
                    pred_theta=[float("nan")] * cell.k,
                    pred_omega=[float("nan")] * cell.k,
                    success=False,
                    error=f"model {self.name} not configured for vision modality",
                )
            if req.image_b64 is None:
                return PredictionResult(
                    pred_theta=[float("nan")] * cell.k,
                    pred_omega=[float("nan")] * cell.k,
                    success=False, error="no image supplied for vision modality",
                )
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{req.image_b64}"}},
                ],
            }
        else:
            user_msg = {"role": "user", "content": user_text}

        messages = [{"role": "system", "content": system_msg}, user_msg]

        last_err: Optional[BaseException] = None
        async with self._sem:
            for attempt in range(self.max_retries + 1):
                t0 = time.perf_counter()
                try:
                    resp = await asyncio.wait_for(
                        self._client.chat.completions.create(
                            messages=messages,
                            model=self.deployment,
                            temperature=0.0,
                            # Thinking models (kimi-k2.6) burn ~7-8k tokens reasoning
                            # on this task before emitting the answer; a tight cap
                            # leaves content empty. 16k gives headroom to finish.
                            max_tokens=16384,
                        ),
                        timeout=self.request_timeout,
                    )
                    latency = time.perf_counter() - t0
                    msg = resp.choices[0].message
                    text = msg.content or ""
                    # Reasoning models (grok-*-reasoning, kimi, gpt-5) put their
                    # chain-of-thought in a separate reasoning_content field and
                    # can leave `content` empty when the budget is tight.
                    reasoning = getattr(msg, "reasoning_content", None)
                    if reasoning is None and getattr(msg, "model_extra", None):
                        reasoning = msg.model_extra.get("reasoning_content")
                    usage = getattr(resp, "usage", None)
                    p_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
                    c_tok = int(getattr(usage, "completion_tokens", 0) or 0)
                    # Parse content first; fall back to the JSON answer reasoning
                    # models often leave inside reasoning_content.
                    theta = omega = cot = None
                    perr: Optional[Exception] = None
                    for candidate in (text, reasoning):
                        if not candidate:
                            continue
                        try:
                            theta, omega, cot = parse_response(candidate, cell.k, used_cot)
                            perr = None
                            break
                        except Exception as pe:
                            perr = pe
                    if theta is None:
                        return PredictionResult(
                            pred_theta=[float("nan")] * cell.k,
                            pred_omega=[float("nan")] * cell.k,
                            raw_response=(text or reasoning or ""), latency_s=latency,
                            prompt_tokens=p_tok, completion_tokens=c_tok,
                            success=False, error=f"parse: {perr}",
                        )
                    return PredictionResult(
                        pred_theta=theta, pred_omega=omega,
                        raw_response=text, cot_text=cot,
                        latency_s=latency,
                        prompt_tokens=p_tok, completion_tokens=c_tok,
                        success=True,
                    )
                except Exception as e:
                    last_err = e
                    if attempt < self.max_retries:
                        backoff = min(2 ** attempt + 0.5, 30.0)
                        await asyncio.sleep(backoff)
                    else:
                        break

        return PredictionResult(
            pred_theta=[float("nan")] * cell.k,
            pred_omega=[float("nan")] * cell.k,
            success=False, error=f"api: {last_err!r}",
        )
