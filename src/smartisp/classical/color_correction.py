"""Camera color correction matrix application."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from smartisp.classical.base import ISPContext
from smartisp.types import RGBImage


class ColorCorrectionMethod(StrEnum):
    """Supported color correction sources."""

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
        if not isinstance(image, RGBImage):
            raise TypeError("ColorCorrector requires an RGBImage input")
        if self.config.method is ColorCorrectionMethod.LEARNED:
            raise NotImplementedError("learned color correction is not part of this release")

        matrix = self.config.matrix
        if matrix is None and context is not None:
            matrix = context.color_matrix
        data = np.asarray(image.data, dtype=np.float32)
        if matrix is None:
            corrected = data.copy()
            applied = False
        else:
            corrected = data @ np.asarray(matrix, dtype=np.float32).T
            applied = True
        if self.config.clamp:
            np.clip(corrected, 0.0, 1.0, out=corrected)

        color_space = image.color_space
        if context is not None and context.working_space is not None and applied:
            color_space = context.working_space
        metadata = dict(image.metadata)
        metadata["color_correction"] = {"applied": applied, "matrix": matrix}
        return RGBImage(
            data=corrected,
            color_space=color_space,
            bit_depth=image.bit_depth,
            metadata=metadata,
        )
