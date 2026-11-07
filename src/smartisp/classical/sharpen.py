"""Sharpening."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from smartisp.classical.base import ISPContext
from smartisp.types import RGBImage


class SharpenMethod(StrEnum):
    """Supported sharpening methods."""

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
        if not isinstance(image, RGBImage):
            raise TypeError("Sharpener requires an RGBImage input")

        data = np.asarray(image.data, dtype=np.float32)
        config = self.config
        if config.amount == 0.0:
            sharpened = data.copy()
        elif config.method is SharpenMethod.LAPLACIAN:
            detail = cv2.Laplacian(data, cv2.CV_32F, ksize=3)
            sharpened = data + config.amount * detail
        else:
            sigma = max(0.1, float(config.radius))
            blurred = cv2.GaussianBlur(data, (0, 0), sigmaX=sigma, sigmaY=sigma)
            detail = data - blurred
            if config.threshold > 0:
                detail = np.where(np.abs(detail) >= config.threshold, detail, 0.0)
            sharpened = data + config.amount * detail
        if config.clamp:
            np.clip(sharpened, 0.0, 1.0, out=sharpened)
        metadata = dict(image.metadata)
        metadata["sharpen"] = {"method": config.method.value, "amount": config.amount}
        return RGBImage(
            data=sharpened.astype(np.float32),
            color_space=image.color_space,
            bit_depth=image.bit_depth,
            metadata=metadata,
        )
