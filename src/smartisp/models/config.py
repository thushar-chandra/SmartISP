"""Model configuration objects that do not require PyTorch to import."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class UNetConfig:
    """Architecture settings for the planned restoration U-Net."""

    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 32
    max_channels: int = 512
    depth: int = 4
    groups: int = 8
    norm: str = "group"
    activation: str = "silu"
    dropout: float = 0.0
    residual: bool = True
    attention: bool = False
    upsample: str = "bilinear"

    def __post_init__(self) -> None:
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("input and output channels must be positive")
        if self.base_channels <= 0 or self.max_channels < self.base_channels:
            raise ValueError("channels must satisfy 0 < base_channels <= max_channels")
        if not 1 <= self.depth <= 8:
            raise ValueError("depth must be between 1 and 8")
        if self.groups <= 0:
            raise ValueError("groups must be positive")
        if self.norm not in {"group", "batch", "instance", "none"}:
            raise ValueError("norm must be group, batch, instance, or none")
        if self.activation not in {"relu", "leaky_relu", "gelu", "silu"}:
            raise ValueError("unsupported activation")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.upsample not in {"bilinear", "transpose"}:
            raise ValueError("upsample must be bilinear or transpose")


class LossKind(StrEnum):
    """Planned loss terms."""

    L1 = "l1"
    CHARBONNIER = "charbonnier"
    SSIM = "ssim"
    PERCEPTUAL = "perceptual"
    COMPOSITE = "composite"


@dataclass(frozen=True)
class LossConfig:
    """Weights and parameters for a restoration objective."""

    kind: LossKind = LossKind.COMPOSITE
    weight: float = 1.0
    charbonnier_weight: float = 1.0
    ssim_weight: float = 0.1
    perceptual_weight: float = 0.0
    epsilon: float = 1e-3
    perceptual_net: str = "alex"

    def __post_init__(self) -> None:
        weights = (
            self.weight,
            self.charbonnier_weight,
            self.ssim_weight,
            self.perceptual_weight,
        )
        if any(value < 0 for value in weights):
            raise ValueError("loss weights must be non-negative")
        if sum(weights) <= 0:
            raise ValueError("at least one loss weight must be positive")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.perceptual_net not in {"alex", "vgg"}:
            raise ValueError("perceptual_net must be 'alex' or 'vgg'")
