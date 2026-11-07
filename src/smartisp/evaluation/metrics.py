"""Image-quality metrics with explicit data ranges."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

_LPIPS_CACHE: dict[tuple[str, str], Any] = {}


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


def _to_numpy(image: Any) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError("batch metrics require one image at a time")
        array = array[0]
    if array.ndim == 3 and array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3}:
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float32) / float(np.iinfo(array.dtype).max)
    else:
        array = array.astype(np.float32)
    return array


def _prepare_pair(prediction: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    prediction_array = _to_numpy(prediction)
    target_array = _to_numpy(target)
    if prediction_array.shape != target_array.shape:
        raise ValueError(
            f"prediction and target shapes differ: {prediction_array.shape} vs {target_array.shape}"
        )
    return prediction_array, target_array


def psnr(prediction: Any, target: Any, config: MetricConfig | None = None) -> float:
    """Peak signal-to-noise ratio in dB for arrays scaled to ``data_range``."""
    settings = config or MetricConfig()
    prediction_array, target_array = _prepare_pair(prediction, target)
    mse = float(np.mean((prediction_array - target_array) ** 2))
    if mse <= 0:
        return math.inf
    return float(10.0 * math.log10(settings.data_range**2 / mse))


def _global_ssim(
    prediction_array: np.ndarray, target_array: np.ndarray, data_range: float
) -> float:
    mu_pred = float(np.mean(prediction_array))
    mu_target = float(np.mean(target_array))
    var_pred = float(np.var(prediction_array))
    var_target = float(np.var(target_array))
    covariance = float(np.mean((prediction_array - mu_pred) * (target_array - mu_target)))
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    return ((2 * mu_pred * mu_target + c1) * (2 * covariance + c2)) / (
        (mu_pred**2 + mu_target**2 + c1) * (var_pred + var_target + c2)
    )


def ssim(prediction: Any, target: Any, config: MetricConfig | None = None) -> float:
    """Mean structural similarity with a Gaussian window.

    Multichannel images are reduced by averaging the per-channel SSIM, which is
    the behavior of ``skimage.metrics.structural_similarity`` with
    ``channel_axis=-1``. Small images fall back to the global SSIM formula.
    """
    settings = config or MetricConfig()
    prediction_array, target_array = _prepare_pair(prediction, target)
    height, width = prediction_array.shape[:2]
    window = min(settings.ssim_window_size, height, width)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return float(_global_ssim(prediction_array, target_array, settings.data_range))

    from skimage.metrics import structural_similarity

    channel_axis = -1 if prediction_array.ndim == 3 else None
    value = structural_similarity(
        prediction_array,
        target_array,
        data_range=settings.data_range,
        channel_axis=channel_axis,
        gaussian_weights=True,
        sigma=settings.ssim_sigma,
        win_size=window,
        use_sample_covariance=False,
    )
    return float(value)


def delta_e_2000(prediction: Any, target: Any, config: MetricConfig | None = None) -> float:
    """Mean CIEDE2000 color difference after sRGB -> Lab conversion."""
    settings = config or MetricConfig()
    prediction_array, target_array = _prepare_pair(prediction, target)
    from skimage.color import deltaE_ciede2000, rgb2lab

    def _lab(array: np.ndarray) -> np.ndarray:
        scaled = np.clip(array / settings.data_range, 0.0, 1.0)
        if scaled.ndim == 2:
            scaled = np.repeat(scaled[..., None], 3, axis=-1)
        return rgb2lab(scaled)

    difference = deltaE_ciede2000(_lab(prediction_array), _lab(target_array))
    return float(np.mean(difference))


def _lpips_model(net: str) -> Any:
    import torch

    if net == "squeeze":
        raise ValueError("the lpips package does not provide a squeeze backbone")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (net, device)
    if key not in _LPIPS_CACHE:
        import lpips

        model = lpips.LPIPS(net=net).to(device)
        model.eval()
        _LPIPS_CACHE[key] = model
    return _LPIPS_CACHE[key]


def lpips_distance(
    prediction: Any,
    target: Any,
    config: MetricConfig | None = None,
) -> float:
    """Learned perceptual image patch similarity as a distance (lower is better)."""
    settings = config or MetricConfig()
    prediction_array, target_array = _prepare_pair(prediction, target)
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "LPIPS requires PyTorch and the 'lpips' package; install the torch extra"
        ) from exc

    model = _lpips_model(settings.lpips_net)
    device = next(model.parameters()).device
    height, width = prediction_array.shape[:2]
    factor = 1
    while min(height, width) * factor < 64:
        factor += 1

    def _tensor(array: np.ndarray) -> Any:
        scaled = np.clip(array / settings.data_range, 0.0, 1.0)
        if scaled.ndim == 2:
            scaled = np.repeat(scaled[..., None], 3, axis=-1)
        tensor = torch.from_numpy(scaled).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device) * 2.0 - 1.0
        if factor > 1:
            tensor = torch.nn.functional.interpolate(
                tensor,
                scale_factor=factor,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp(-1.0, 1.0)
        return tensor

    with torch.no_grad():
        value = model(_tensor(prediction_array), _tensor(target_array))
    return float(value.mean().item())


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
