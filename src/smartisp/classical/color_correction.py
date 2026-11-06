"""Camera color correction matrix interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from smartisp.classical.base import ISPContext, interface_not_implemented
from smartisp.types import RGBImage


class ColorCorrectionMethod(StrEnum):
    """Planned color correction sources."""

    MATRIX = "matrix"
    CAMERA_METADATA = "camera_metadata"
    LEARNED = "learned"


@dataclass(frozen=True)
class ColorCorrectionConfig:
    """Configuration for :class:`ColorCorrector`."""

    method: ColorCorrectionMethod = ColorCorrectionMethod.CAMERA_METADATA
    matrix: tuple[tuple[float, float, float], ...] | None = None
    clamp: bool = True

    def __post_init__(self) -> None:
        if self.matrix is not None and (
            len(self.matrix) != 3 or any(len(row) != 3 for row in self.matrix)
        ):
            raise ValueError("matrix must be 3x3")
        if self.method is ColorCorrectionMethod.MATRIX and self.matrix is None:
            raise ValueError("matrix is required when method='matrix'")


class ColorCorrector:
    """Map camera primaries into a working RGB space."""

    name = "color_correction"

    def __init__(self, config: ColorCorrectionConfig | None = None) -> None:
        self.config = config or ColorCorrectionConfig()

    def process(self, image: RGBImage, context: ISPContext | None = None) -> RGBImage:
        raise interface_not_implemented(
            self.name,
            "apply a camera or configured 3x3 CCM in linear RGB, then clip if requested",
        )
