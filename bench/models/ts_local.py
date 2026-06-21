"""
Local time-series foundation-model predictors (run on a local GPU, no Azure).

This is the local counterpart to `timeseries.py` (which calls Azure ML REST
endpoints). It loads the model weights from HuggingFace and runs inference on the
machine's GPU, so the time-series comparison can be run without deploying any
cloud endpoint.

Each pendulum state is treated as 2k independent univariate channels
(theta_1..k, omega_1..k). The history window (t <= 0) is resampled onto a uniform
grid, the model forecasts `prediction_length` steps, and the value at the step
closest to the requested horizon is returned as the prediction.

Supported variants (install the matching package):
    chronos  ->  pip install chronos-forecasting   (amazon/chronos-t5-*)
    timesfm  ->  pip install timesfm                (google/timesfm-*)
    moirai   ->  pip install uni2ts                 (Salesforce/moirai-*)

Chronos is the lightest dependency and the default reference implementation.
"""
from __future__ import annotations

import threading
import time
import numpy as np

from .base import PredictionRequest, PredictionResult

VARIANTS = {"chronos", "chronos2", "timesfm", "moirai"}

DEFAULT_MODEL_IDS = {
    "chronos":  "amazon/chronos-t5-large",   # univariate only
    "chronos2": "amazon/chronos-2",          # supports univariate and multivariate
    "timesfm":  "google/timesfm-1.0-200m",
    "moirai":   "Salesforce/moirai-1.0-R-large",  # any-variate (uni or multi)
}

# Variants that can forecast all channels jointly (multivariate=True).
MULTIVARIATE_CAPABLE = {"chronos2", "moirai"}


def _pick_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class LocalTimeSeriesPredictor:
    """Forecast pendulum state with a locally-run time-series foundation model."""

    is_async = False
    uses_modality = False
    uses_prompting = False
    uses_history = True

    def __init__(self, name: str, variant: str, *, model_id: str | None = None,
                 device: str = "auto", dtype: str = "float32",
                 max_prediction_length: int = 256, max_context_length: int = 512,
                 num_samples: int = 20, multivariate: bool = False):
        if variant not in VARIANTS:
            raise ValueError(f"unknown time-series variant {variant!r}; "
                             f"expected one of {sorted(VARIANTS)}")
        if multivariate and variant not in MULTIVARIATE_CAPABLE:
            raise ValueError(f"variant {variant!r} cannot run multivariate; "
                             f"multivariate-capable: {sorted(MULTIVARIATE_CAPABLE)}")
        self.name = name
        self.variant = variant
        self.model_id = model_id or DEFAULT_MODEL_IDS[variant]
        self.device = _pick_device(device)
        self.dtype = dtype
        self.max_prediction_length = max_prediction_length
        self.max_context_length = max_context_length
        self.num_samples = num_samples
        self.multivariate = multivariate
        self._pipe = None  # lazily loaded
        # The async runner dispatches sync predictors across many worker threads.
        # On a single small GPU we must (a) load the model exactly once and
        # (b) serialize inference so concurrent cells don't each load a copy or
        # blow up VRAM with parallel activations.
        self._gpu_lock = threading.Lock()

    # ── model loading (lazy, per variant) ─────────────────────────────────────
    def _ensure_loaded(self):
        if self._pipe is not None:
            return
        loader = getattr(self, f"_load_{self.variant}")
        self._pipe = loader()

    def _torch_dtype(self):
        import torch
        return {"float32": torch.float32, "bfloat16": torch.bfloat16,
                "float16": torch.float16}.get(self.dtype, torch.float32)

    def _free_cuda(self):
        # Release cached/intermediate GPU memory between cells. Without this,
        # per-cell forecast allocations accumulate and OOM on a small GPU.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_chronos(self):
        from chronos import ChronosPipeline  # type: ignore
        return ChronosPipeline.from_pretrained(
            self.model_id, device_map=self.device, torch_dtype=self._torch_dtype())

    def _load_chronos2(self):
        from chronos import Chronos2Pipeline  # type: ignore
        return Chronos2Pipeline.from_pretrained(self.model_id, device_map=self.device)

    def _load_timesfm(self):
        import timesfm  # type: ignore
        backend = "gpu" if self.device.startswith("cuda") else "cpu"
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                context_len=self.max_context_length,
                horizon_len=self.max_prediction_length,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=self.model_id),
        )
        return tfm

    def _load_moirai(self):
        import torch
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule  # type: ignore
        module = MoiraiModule.from_pretrained(self.model_id)
        # Forecast object is rebuilt per-call (it needs prediction_length); keep module.
        return {"module": module, "MoiraiForecast": MoiraiForecast}

    # ── context resampling (ported from timeseries.py) ────────────────────────
    def _resample_context(self, req: PredictionRequest) -> tuple[np.ndarray, float, int]:
        k = req.cell.k
        n_channels = 2 * k
        horizon = req.horizon

        ht = req.history_times
        hs = req.history_states
        if ht is None or hs is None or len(ht) < 2:
            raise ValueError("time-series predictor requires history_times/states")
        ctx_seconds = float(ht[-1] - ht[0])
        if ctx_seconds <= 0:
            raise ValueError("history window has non-positive duration")

        dt = max(horizon / self.max_prediction_length,
                 ctx_seconds / self.max_context_length)
        prediction_length = max(1, int(round(horizon / dt)))
        ctx_len = max(8, min(self.max_context_length,
                             int(round(ctx_seconds / dt))))
        t_end = float(ht[-1])
        t_start = max(float(ht[0]), t_end - ctx_len * dt)

        channels = np.empty((n_channels, ctx_len), dtype=np.float32)
        grid = np.linspace(t_start, t_end, ctx_len, endpoint=True)
        for ch_idx in range(n_channels):
            channels[ch_idx] = np.interp(grid, ht, hs[:, ch_idx])
        return channels, dt, prediction_length

    # ── per-variant forecasting → (n_channels, prediction_length) ─────────────
    def _forecast_chronos(self, context: np.ndarray, prediction_length: int) -> np.ndarray:
        import torch
        # Chronos expands each series to `num_samples` sequences, so the effective
        # batch through the T5 is (n_channels * num_samples). On an 8GB GPU that
        # OOMs for k=3 (6 channels). Forecast one channel at a time to bound the
        # batch to num_samples, freeing the cache between channels.
        outs = []
        for ch in context:
            ctx = [torch.tensor(ch, dtype=torch.float32)]
            fc = self._pipe.predict(ctx, prediction_length,
                                    num_samples=self.num_samples)  # (1, num_samples, pred_len)
            outs.append(np.median(fc.detach().cpu().numpy(), axis=1)[0])  # (pred_len,)
            del fc
            self._free_cuda()
        return np.stack(outs, axis=0)  # (n_channels, pred_len)

    def _forecast_chronos2(self, context: np.ndarray, prediction_length: int) -> np.ndarray:
        """Chronos-2: supports univariate (per-channel) and multivariate (joint).

        predict() returns one tensor per input element of shape
        (n_variates, n_quantiles, pred_len); we take the 0.5-quantile (median).
        """
        qs = list(getattr(self._pipe, "quantiles", [0.5]))
        qidx = qs.index(0.5) if 0.5 in qs else len(qs) // 2

        def _np(x):
            return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

        ctx = context.astype(np.float32)
        if self.multivariate:
            # one 2-d element (n_variates, history) -> joint forecast
            out = self._pipe.predict([ctx], prediction_length=prediction_length)
            arr = _np(out[0])                       # (n_var, n_q, pred_len)
            med = arr[:, qidx, :]                   # (n_var, pred_len)
        else:
            # list of 1-d series -> independent per-channel forecast
            out = self._pipe.predict([ch for ch in ctx],
                                     prediction_length=prediction_length)
            med = np.stack([_np(o)[0, qidx, :] for o in out], axis=0)  # (n_ch, pred_len)
        return med

    def _forecast_timesfm(self, context: np.ndarray, prediction_length: int) -> np.ndarray:
        inputs = [ch for ch in context]
        freq = [0] * len(inputs)
        point_forecast, _ = self._pipe.forecast(inputs, freq=freq)
        pf = np.asarray(point_forecast)
        return pf[:, :prediction_length]

    def _forecast_moirai(self, context: np.ndarray, prediction_length: int) -> np.ndarray:
        import torch
        ctx_len = context.shape[1]
        MoiraiForecast = self._pipe["MoiraiForecast"]
        forecast = MoiraiForecast(
            module=self._pipe["module"],
            prediction_length=prediction_length,
            context_length=ctx_len,
            patch_size="auto",
            num_samples=self.num_samples,
            target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        )
        preds = []
        for ch in context:
            past_target = torch.tensor(ch, dtype=torch.float32).reshape(1, ctx_len, 1)
            past_observed = torch.ones_like(past_target, dtype=torch.bool)
            past_is_pad = torch.zeros(1, ctx_len, dtype=torch.bool)
            out = forecast(past_target=past_target,
                           past_observed_target=past_observed,
                           past_is_pad=past_is_pad)
            samples = out[0].detach().cpu().numpy()  # (num_samples, pred_len)
            preds.append(np.median(samples, axis=0))
        return np.stack(preds, axis=0)

    def predict(self, req: PredictionRequest) -> PredictionResult:
        k = req.cell.k
        try:
            context, dt, prediction_length = self._resample_context(req)
        except Exception as e:
            return PredictionResult(pred_theta=[float("nan")] * k,
                                    pred_omega=[float("nan")] * k,
                                    success=False, error=f"context: {e!r}")
        # Serialize load + inference: one model load, one GPU computation at a
        # time (the runner would otherwise call this from many threads at once).
        forecaster = getattr(self, f"_forecast_{self.variant}")
        t0 = time.perf_counter()
        with self._gpu_lock:
            try:
                self._ensure_loaded()
            except Exception as e:
                return PredictionResult(pred_theta=[float("nan")] * k,
                                        pred_omega=[float("nan")] * k,
                                        success=False, error=f"load: {e!r}")
            try:
                forecast = forecaster(context, prediction_length)  # (n_channels, pred_len)
                final = np.asarray(forecast)[:, -1]
                theta = final[:k].tolist()
                omega = final[k:2 * k].tolist()
                return PredictionResult(pred_theta=theta, pred_omega=omega,
                                        latency_s=time.perf_counter() - t0, success=True)
            except Exception as e:
                return PredictionResult(pred_theta=[float("nan")] * k,
                                        pred_omega=[float("nan")] * k,
                                        success=False, error=f"forecast: {e!r}")
            finally:
                self._free_cuda()

    async def apredict(self, req: PredictionRequest) -> PredictionResult:
        return self.predict(req)
