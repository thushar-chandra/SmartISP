"""Runtime and memory measurement interfaces."""

from __future__ import annotations

import math
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


def measure_inference(
    callable_: Any,
    inputs: Any,
    config: EfficiencyConfig | None = None,
) -> EfficiencyReport:
    raise NotImplementedError(
        "measure_inference is an interface placeholder; planned: warmup, device "
        "synchronization, torch.cuda.max_memory_allocated, psutil RSS, and "
        "p50/p90/p95 reporting"
    )
