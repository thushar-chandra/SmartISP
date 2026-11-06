"""Sharpening interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from smartisp.classical.base import ISPContext, interface_not_implemented
from smartisp.types import RGBImage


class SharpenMethod(StrEnum):
    """Planned sharpening methods."""

    UNSHARP_MASK = "unsharp_mask"
    LAPLACIAN = "laplacian"
    GUIDED = "guided"


@dataclass(frozen=True)
class SharpenConfig:
    """Configuration for :class:`Sharpener`."""

    method: SharpenMethod = SharpenMethod.UNSHARP_MASK
    radius: float = 1.0
    amount: float = 1.0
    threshold: float = 0.0
    clamp: bool = True

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if self.amount < 0:
            raise ValueError("amount must be non-negative")
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")


class Sharpener:
    """Restore local contrast after tone mapping."""

    name = "sharpen"

    def __init__(self, config: SharpenConfig | None = None) -> None:
        self.config = config or SharpenConfig()

    def process(self, image: RGBImage, context: ISPContext | None = None) -> RGBImage:
        raise interface_not_implemented(
            self.name,
            "implement unsharp masking and edge-aware sharpening with halo control",
        )
