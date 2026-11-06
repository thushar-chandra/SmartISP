"""Loss-function interfaces for restoration training."""

from __future__ import annotations

from typing import Any

try:
    from torch import nn
except ImportError as exc:
    raise ImportError(
        "smartisp.models.losses requires PyTorch; install the 'torch' extra "
        "(pip install -e '.[torch]') or use the Colab runtime"
    ) from exc

from smartisp.models.config import LossConfig, LossKind

__all__ = ["LossConfig", "LossKind", "RestorationLoss", "build_loss"]


class RestorationLoss(nn.Module):
    """Composite loss that will be built from configured terms."""

    def __init__(self, config: LossConfig | None = None) -> None:
        super().__init__()
        self.config = config or LossConfig()

    def forward(self, prediction: Any, target: Any) -> Any:
        raise NotImplementedError(
            "RestorationLoss.forward is an interface placeholder; planned: "
            "Charbonnier + SSIM + optional LPIPS perceptual loss"
        )


def build_loss(config: LossConfig | None = None) -> RestorationLoss:
    """Construct the configured loss; term implementations are added later."""
    return RestorationLoss(config)
