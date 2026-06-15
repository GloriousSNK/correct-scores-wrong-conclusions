from __future__ import annotations

import asyncio
import os
import time
from typing import Optional

try:
    from openai import AsyncOpenAI
    _HAVE_OPENAI = True
except ImportError:
    _HAVE_OPENAI = False

from ..prompts import SYSTEM_COT, SYSTEM_NOCOT, build_user_prompt, parse_response
from .base import PredictionRequest, PredictionResult

_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NimPredictor:
    """NVIDIA NIM inference predictor (OpenAI-compatible API, text-only)."""

    is_async = True
    uses_modality = False   # text-only; image modalities fall back to coords
    uses_prompting = True
    uses_history = False

    def __init__(self, name: str, deployment: str, *,
                 concurrency: int = 4,
                 request_timeout: float = 120.0,
                 max_retries: int = 4):
        if not _HAVE_OPENAI:
            raise RuntimeError(
                "openai package is not installed. Run `pip install -r requirements.txt`."
            )
        api_key = os.getenv("NVIDIA_NIM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "NVIDIA_NIM_API_KEY is not set. Add it to your .env file."
            )
        self.name = name
        self.deployment = deployment
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self._sem = asyncio.Semaphore(concurrency)
        self._client = AsyncOpenAI(base_url=_NIM_BASE_URL, api_key=api_key)

    async def aclose(self):
        await self._client.close()

    async def apredict(self, req: PredictionRequest) -> PredictionResult:
        cell = req.cell
        used_cot = (cell.prompting == "cot")
        system_msg = SYSTEM_COT if used_cot else SYSTEM_NOCOT
        # Always use coords modality — NIM models are text-only
        user_text = build_user_prompt(req, k=cell.k, modality="coords")

        last_err: Optional[BaseException] = None
        async with self._sem:
            for attempt in range(self.max_retries + 1):
                t0 = time.perf_counter()
                try:
                    resp = await asyncio.wait_for(
                        self._client.chat.completions.create(
                            model=self.deployment,
                            messages=[
                                {"role": "system", "content": system_msg},
                                {"role": "user",   "content": user_text},
                            ],
                            temperature=0.0,
                            max_tokens=2048 if used_cot else 256,
                        ),
                        timeout=self.request_timeout,
                    )
                    latency = time.perf_counter() - t0
                    text = resp.choices[0].message.content or ""
                    usage = resp.usage
                    p_tok = int(getattr(usage, "prompt_tokens",     0) or 0)
                    c_tok = int(getattr(usage, "completion_tokens", 0) or 0)
                    try:
                        theta, omega, cot = parse_response(text, cell.k, used_cot)
                    except Exception as pe:
                        return PredictionResult(
                            pred_theta=[float("nan")] * cell.k,
                            pred_omega=[float("nan")] * cell.k,
                            raw_response=text, latency_s=latency,
                            prompt_tokens=p_tok, completion_tokens=c_tok,
                            success=False, error=f"parse: {pe}",
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
                        await asyncio.sleep(min(2 ** attempt + 0.5, 30.0))
                    else:
                        break

        return PredictionResult(
            pred_theta=[float("nan")] * cell.k,
            pred_omega=[float("nan")] * cell.k,
            success=False, error=f"api: {last_err!r}",
        )
