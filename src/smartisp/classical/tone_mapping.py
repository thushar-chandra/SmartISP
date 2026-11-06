"""Tone mapping interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from smartisp.classical.base import ISPContext, interface_not_implemented
from smartisp.types import ColorSpace, RGBImage


class ToneMappingMethod(StrEnum):
    """Planned tone mapping operators."""

    REINHARD = "reinhard"
    FILMIC = "filmic"
    ACES = "aces"
    GAMMA = "gamma"


@dataclass(frozen=True)
class ToneMappingConfig:
    """Configuration for :class:`ToneMapper`."""

    method: ToneMappingMethod = ToneMappingMethod.FILMIC
    exposure: float = 0.0
    gamma: float = 2.2
    white_point: float = 1.0
    output_color_space: ColorSpace = ColorSpace.SRGB

    def __post_init__(self) -> None:
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.white_point <= 0:
            raise ValueError("white_point must be positive")


class ToneMapper:
    """Compress linear scene radiance into a display-referred image."""

    name = "tone_mapping"

    def __init__(self, config: ToneMappingConfig | None = None) -> None:
        self.config = config or ToneMappingConfig()

    def process(self, image: RGBImage, context: ISPContext | None = None) -> RGBImage:
        raise interface_not_implemented(
            self.name,
            "apply exposure, a selectable tone curve, and the sRGB transfer function",
        )
