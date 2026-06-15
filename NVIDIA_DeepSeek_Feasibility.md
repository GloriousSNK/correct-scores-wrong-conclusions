# DeepSeek V4 Pro on NVIDIA NIM — Benchmark Feasibility Assessment

**Project:** Can LLMs Predict Physics? (Double Pendulum Benchmark)
**Model:** DeepSeek V4 Pro via [build.nvidia.com](https://build.nvidia.com/deepseek-ai/deepseek-v4-pro)
**Date:** June 2026

---

## NVIDIA NIM Free Tier (2026)

| Parameter | Value |
|---|---|
| Default rate limit | 40 requests/min (RPM) |
| Upgradeable rate limit | 200 RPM (on request) |
| Credit allowance | ~1,000–5,000 on signup |
| Credit system status | Likely phased out — current system appears to be **pure RPM throttling** with no hard call cap |
| Per-token billing | None on free tier |
| Credit card required | No |

> **Note:** There is conflicting public information about whether the credit system is still active. The safest approach is to run a small test batch first to confirm no credit wall is hit.

---

## Model Constraints

DeepSeek V4 Pro is a **text-only** model — it does not support image inputs. This has direct implications for the benchmark:

| Modality | Compatible? | Outcome |
|---|---|---|
| `coords` | Yes | Full predictions |
| `images` | No | Returns NaN (wasted call) |
| `images_coords` | No | Returns NaN (wasted call) |

Only the `coords` modality produces useful results.

---

## Benchmark Call Volume

Based on `config.yaml`:

| Factor | Value |
|---|---|
| Pendulum systems (k) | 1, 2, 3 |
| Trajectories per cell | 20 |
| Prediction horizons | 0.01s, 1.0s, 10.0s, 60.0s |
| Modalities | coords, images, images_coords |
| Prompting strategies | no_cot, cot |
| Regimes | normal, changed_disclosed, changed_hidden |

**Total calls (full run, all modalities):** `3 × 20 × 4 × 3 × 2 × 3 = 4,320`

**Useful calls (coords only):** `3 × 20 × 4 × 1 × 2 × 3 = 1,440`

---

## Prompt Size Per Call

Prompts are small — just initial state numerics and physical constants:

| Component | Approximate Size |
|---|---|
| Input tokens (coords modality) | 150–300 tokens |
| Output tokens (no_cot) | ≤ 256 tokens |
| Output tokens (cot) | ≤ 2,048 tokens |
| Images | N/A — model is text-only |

Context window size is **not a limiting factor**.

---

## Free Tier Coverage

### If credits are phased out (pure RPM model) — most likely

| Scenario | Calls | Time at 40 RPM |
|---|---|---|
| Full run (all modalities) | 4,320 | ~108 min |
| Coords-only (recommended) | 1,440 | ~36 min |

**Result: Free tier covers the full benchmark run.** No hard call cap, just throughput throttling.

### If the old credit system is still active

| Credit allocation | Covers coords-only run (1,440 calls)? |
|---|---|
| 1,000 credits | No — ~30% short |
| 5,000 credits | Yes — with headroom |

---

## Recommendations

1. **Restrict modality to `coords` only** for the DeepSeek V4 Pro entry in `config.yaml` — image modalities will always return NaN and waste API quota.

2. **Lower concurrency from 6 to 3–4** to avoid hitting the 40 RPM wall and triggering 429 errors during the run.

3. **Run a 5-call smoke test first** to confirm whether the credit system is active before committing to the full 1,440-call run.

4. **Request a 200 RPM increase** from NVIDIA if you want to cut the 36-minute minimum runtime significantly.

5. **Integration work required:** The project currently uses `azure.ai.inference` SDK. NVIDIA NIM exposes an OpenAI-compatible REST API (`https://integrate.api.nvidia.com/v1`), so a new predictor class using `openai.AsyncOpenAI` is needed before testing can begin.

---

## Config Change Needed

In `config.yaml`, add DeepSeek V4 Pro with `coords`-only modality override:

```yaml
models:
  - name: deepseek-v4-pro
    kind: llm
    deployment: deepseek-ai/deepseek-v4-pro
    vision: false
    concurrency: 4   # stay under 40 RPM
```

And run with a filtered modality set to avoid wasted calls on image inputs.

---

*Sources: [NVIDIA NIM Free Tier (decodethefuture.org)](https://decodethefuture.org/en/nvidia-nim-api-explained/) · [NIM Pricing Limits (decodethefuture.org)](https://decodethefuture.org/en/nvidia-nim-api-pricing-limits-guide/) · [NVIDIA Developer Forums — Rate Limit Increase](https://forums.developer.nvidia.com/t/api-rate-limit-increase-for-nvidia-nim/366043)*
