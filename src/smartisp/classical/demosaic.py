"""Bayer demosaicing interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from smartisp.classical.base import ISPContext, interface_not_implemented
from smartisp.types import ColorSpace, RawFrame, RGBImage


class DemosaicMethod(StrEnum):
    """Planned demosaicing algorithm families."""

    BILINEAR = "bilinear"
    MALVAR = "malvar"
    EDGE_AWARE = "edge_aware"


@dataclass(frozen=True)
class DemosaicConfig:
    """Configuration for :class:`Demosaicer`."""

    method: DemosaicMethod = DemosaicMethod.EDGE_AWARE
    output_color_space: ColorSpace = ColorSpace.LINEAR_SRGB
    align_to_even: bool = True

    def __post_init__(self) -> None:
        if self.output_color_space is ColorSpace.SRGB:
            raise ValueError("demosaicing must output linear RGB, not display sRGB")


class Demosaicer:
    """Reconstruct a linear RGB image from a single-channel Bayer frame."""

    name = "demosaic"

    def __init__(self, config: DemosaicConfig | None = None) -> None:
        self.config = config or DemosaicConfig()

    def process(self, image: RawFrame, context: ISPContext | None = None) -> RGBImage:
        raise interface_not_implemented(
            self.name,
            "implement and benchmark bilinear, Malvar, and edge-aware demosaicing "
            "with CFA-aware border handling",
        )
