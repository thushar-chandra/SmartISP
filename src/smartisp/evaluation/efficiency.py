"""Runtime and memory measurement."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EfficiencyConfig:
    """Measurement protocol settings."""

    device: str = "auto"
    warmup_runs: int = 5
    timed_runs: int = 20
    synchronize: bool = True
    include_host_memory: bool = True
    include_gpu_memory: bool = True

    def __post_init__(self) -> None:
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")
        if self.timed_runs <= 0:
            raise ValueError("timed_runs must be positive")


@dataclass(frozen=True)
class EfficiencyReport:
    """Latency and memory statistics for one model/pipeline configuration."""

    device: str
    latency_ms_p50: float
    latency_ms_p90: float
    latency_ms_p95: float
    throughput_fps: float
    peak_gpu_memory_mb: float | None = None
    peak_host_memory_mb: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        for value in (
            self.latency_ms_p50,
            self.latency_ms_p90,
            self.latency_ms_p95,
            self.throughput_fps,
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("latency and throughput values must be finite and non-negative")


def percentile(samples: Sequence[float], quantile: float) -> float:
    """Linear-interpolated percentile with ``quantile`` in ``[0, 1]``."""
    if not samples:
        raise ValueError("samples must not be empty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return float(ordered[0])
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summarize_latencies(samples_ms: Sequence[float]) -> dict[str, float]:
    """Summarize latency samples using median, p90, p95, and mean."""
    if not samples_ms:
        raise ValueError("samples_ms must not be empty")
    if any(value < 0 or not math.isfinite(value) for value in samples_ms):
        raise ValueError("latency samples must be finite and non-negative")
    return {
        "p50": percentile(samples_ms, 0.50),
        "p90": percentile(samples_ms, 0.90),
        "p95": percentile(samples_ms, 0.95),
        "mean": float(sum(samples_ms) / len(samples_ms)),
    }


def _torch_module() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _cuda_available(torch: Any) -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def _synchronize(torch: Any, enabled: bool) -> None:
    if enabled and _cuda_available(torch):
        torch.cuda.synchronize()


def _peak_host_memory_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)


def measure_inference(
    callable_: Any,
    inputs: Any,
    config: EfficiencyConfig | None = None,
) -> EfficiencyReport:
    """Time repeated inference calls and record peak GPU/host memory."""
    settings = config or EfficiencyConfig()
    torch = _torch_module()
    if settings.device == "auto":
        device = "cuda" if _cuda_available(torch) else "cpu"
    else:
        device = settings.device

    def _invoke() -> Any:
        if isinstance(inputs, tuple):
            return callable_(*inputs)
        if isinstance(inputs, list):
            return callable_(*inputs)
        return callable_(inputs)

    try:
        import psutil

        process = psutil.Process()
        process.cpu_percent(None)
    except ImportError:
        process = None

    if torch is not None and _cuda_available(torch) and settings.include_gpu_memory:
        torch.cuda.reset_peak_memory_stats()

    samples_ms: list[float] = []
    peak_host = 0.0
    for _ in range(settings.warmup_runs):
        if torch is not None:
            with torch.no_grad():
                _invoke()
        else:
            _invoke()
        _synchronize(torch, settings.synchronize)

    for _ in range(settings.timed_runs):
        start = time.perf_counter()
        if torch is not None:
            with torch.no_grad():
                _invoke()
        else:
            _invoke()
        _synchronize(torch, settings.synchronize)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
        if settings.include_host_memory:
            peak_host = max(peak_host, _peak_host_memory_mb() or 0.0)

    summary = summarize_latencies(samples_ms)
    peak_gpu: float | None = None
    if torch is not None and _cuda_available(torch) and settings.include_gpu_memory:
        peak_gpu = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)

    notes = "single-process wall-clock measurement"
    if process is not None:
        notes += "; host memory is process RSS, not isolated model allocation"

    return EfficiencyReport(
        device=device,
        latency_ms_p50=summary["p50"],
        latency_ms_p90=summary["p90"],
        latency_ms_p95=summary["p95"],
        throughput_fps=1000.0 / max(summary["p50"], 1e-9),
        peak_gpu_memory_mb=peak_gpu,
        peak_host_memory_mb=peak_host if settings.include_host_memory else None,
        notes=notes,
    )
