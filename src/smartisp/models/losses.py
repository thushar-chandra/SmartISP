"""Differentiable restoration losses."""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without the torch extra
    raise ImportError(
        "smartisp.models.losses requires PyTorch; install the 'torch' extra "
        "(pip install -e '.[torch]') or use the Colab runtime"
    ) from exc

from smartisp.models.config import LossConfig, LossKind

__all__ = ["LossConfig", "LossKind", "RestorationLoss", "build_loss"]


def _gaussian_window(
    window_size: int, sigma: float, channels: int, device: Any, dtype: Any
) -> torch.Tensor:
    coordinates = torch.arange(window_size, device=device, dtype=dtype)
    coordinates -= window_size // 2
    gaussian = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel = gaussian[:, None] @ gaussian[None, :]
    kernel = kernel / kernel.sum()
    return kernel.expand(channels, 1, window_size, window_size).contiguous()


def ssim_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Return ``1 - SSIM`` averaged over batch and channels."""
    channels = prediction.shape[1]
    size = min(window_size, prediction.shape[-1], prediction.shape[-2])
    if size % 2 == 0:
        size -= 1
    size = max(size, 3)
    window = _gaussian_window(size, sigma, channels, prediction.device, prediction.dtype)
    padding = size // 2
    mu_pred = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu_target = F.conv2d(target, window, padding=padding, groups=channels)
    mu_pred_sq = mu_pred**2
    mu_target_sq = mu_target**2
    mu_product = mu_pred * mu_target
    sigma_pred_sq = (
        F.conv2d(prediction * prediction, window, padding=padding, groups=channels) - mu_pred_sq
    )
    sigma_target_sq = (
        F.conv2d(target * target, window, padding=padding, groups=channels) - mu_target_sq
    )
    sigma_product = (
        F.conv2d(prediction * target, window, padding=padding, groups=channels) - mu_product
    )
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    similarity = ((2 * mu_product + c1) * (2 * sigma_product + c2)) / (
        (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    )
    return 1.0 - similarity.mean()


def charbonnier_loss(
    prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3
) -> torch.Tensor:
    return torch.sqrt((prediction - target) ** 2 + epsilon**2).mean()


class RestorationLoss(nn.Module):
    """Composite reconstruction objective.

    Supported terms are Charbonnier/L1 reconstruction, differentiable SSIM, and
    an optional LPIPS perceptual term. LPIPS requires the ``lpips`` package and
    downloads pretrained weights on first use.
    """

    def __init__(self, config: LossConfig | None = None) -> None:
        super().__init__()
        self.config = config or LossConfig()
        self._perceptual: nn.Module | None = None

    def _perceptual_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self._perceptual is None:
            try:
                import lpips
            except ImportError as exc:
                raise ImportError(
                    "perceptual loss requires the 'lpips' package (install the project torch extra)"
                ) from exc
            self._perceptual = lpips.LPIPS(net=self.config.perceptual_net).to(prediction.device)
            self._perceptual.eval()
        return self._perceptual(prediction * 2 - 1, target * 2 - 1).mean()

    def forward(self, prediction: Any, target: Any) -> Any:
        if not torch.is_tensor(prediction) or not torch.is_tensor(target):
            raise TypeError("RestorationLoss expects torch tensors")
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have the same shape")

        config = self.config
        kind = config.kind
        if kind is LossKind.L1:
            return config.weight * F.l1_loss(prediction, target)
        if kind is LossKind.CHARBONNIER:
            return config.weight * charbonnier_loss(prediction, target, config.epsilon)
        if kind is LossKind.SSIM:
            return config.weight * ssim_loss(prediction, target)
        if kind is LossKind.PERCEPTUAL:
            return config.weight * self._perceptual_loss(prediction, target)

        total = config.charbonnier_weight * charbonnier_loss(prediction, target, config.epsilon)
        total = total + config.ssim_weight * ssim_loss(prediction, target)
        if config.perceptual_weight > 0:
            total = total + config.perceptual_weight * self._perceptual_loss(prediction, target)
        return config.weight * total


def build_loss(config: LossConfig | None = None) -> RestorationLoss:
    """Construct the configured loss."""
    return RestorationLoss(config)
