"""Image-quality metric interfaces.

The metric implementations are intentionally not faked. When added, each
function must document data range, color space, channel handling, and
preprocessing so that reported numbers are reproducible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class MetricName(StrEnum):
    """Supported evaluation metrics."""

    PSNR = "psnr"
    SSIM = "ssim"
    LPIPS = "lpips"
    DELTA_E_2000 = "delta_e_2000"


METRIC_HIGHER_IS_BETTER: dict[MetricName, bool] = {
    MetricName.PSNR: True,
    MetricName.SSIM: True,
    MetricName.LPIPS: False,
    MetricName.DELTA_E_2000: False,
}


@dataclass(frozen=True)
class MetricConfig:
    """Shared metric settings."""

    data_range: float = 1.0
    ssim_window_size: int = 11
    ssim_sigma: float = 1.5
    lpips_net: str = "alex"
    delta_e_white_point: str = "D65"

    def __post_init__(self) -> None:
        if self.data_range <= 0:
            raise ValueError("data_range must be positive")
        if self.ssim_window_size <= 0 or self.ssim_window_size % 2 == 0:
            raise ValueError("ssim_window_size must be a positive odd number")
        if self.ssim_sigma <= 0:
            raise ValueError("ssim_sigma must be positive")
        if self.lpips_net not in {"alex", "vgg", "squeeze"}:
            raise ValueError("lpips_net must be alex, vgg, or squeeze")


@dataclass(frozen=True)
class MetricValue:
    """A metric value with its optimization direction."""

    name: MetricName
    value: float

    @property
    def higher_is_better(self) -> bool:
        return METRIC_HIGHER_IS_BETTER[self.name]


def psnr(prediction: Any, target: Any, config: MetricConfig | None = None) -> float:
    raise NotImplementedError(
        "psnr is an interface placeholder; planned: data-range-aware PSNR with "
        "explicit per-image and dataset aggregation"
    )


def ssim(prediction: Any, target: Any, config: MetricConfig | None = None) -> float:
    raise NotImplementedError(
        "ssim is an interface placeholder; planned: Gaussian-window SSIM with "
        "documented channel reduction"
    )


def lpips_distance(
    prediction: Any,
    target: Any,
    config: MetricConfig | None = None,
) -> float:
    raise NotImplementedError(
        "lpips_distance is an interface placeholder; planned: pretrained LPIPS "
        "with a pinned backbone, reported as a distance (lower is better)"
    )


def delta_e_2000(
    prediction: Any,
    target: Any,
    config: MetricConfig | None = None,
) -> float:
    raise NotImplementedError(
        "delta_e_2000 is an interface placeholder; planned: CIEDE2000 in Lab "
        "after an explicit sRGB-to-linear-to-Lab conversion"
    )


def evaluate_pair(
    prediction: Any,
    target: Any,
    metrics: Sequence[MetricName] = (
        MetricName.PSNR,
        MetricName.SSIM,
        MetricName.LPIPS,
        MetricName.DELTA_E_2000,
    ),
    config: MetricConfig | None = None,
) -> dict[str, float]:
    """Evaluate one prediction/target pair, preserving metric names as keys."""
    functions = {
        MetricName.PSNR: psnr,
        MetricName.SSIM: ssim,
        MetricName.LPIPS: lpips_distance,
        MetricName.DELTA_E_2000: delta_e_2000,
    }
    return {name.value: functions[name](prediction, target, config) for name in metrics}
