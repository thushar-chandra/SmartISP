"""Tone mapping."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from smartisp.classical.base import ISPContext
from smartisp.types import ColorSpace, RGBImage


class ToneMappingMethod(StrEnum):
    """Supported tone mapping operators."""

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


def _srgb_encode(linear: np.ndarray) -> np.ndarray:
    threshold = 0.0031308
    return np.where(
        linear <= threshold,
        12.92 * linear,
        (1.055 * np.power(np.maximum(linear, threshold), 1.0 / 2.4)) - 0.055,
    )


def _hable_filmic(value: np.ndarray, white_point: float) -> np.ndarray:
    a, b, c, d, e, f = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    curved = (value * (a * value + c * b) + d * e) / (value * (a * value + b) + d * f) - e / f
    white = (white_point * (a * white_point + c * b) + d * e) / (
        white_point * (a * white_point + b) + d * f
    ) - e / f
    return curved / max(white, 1e-8)


def _tone_curve(linear: np.ndarray, config: ToneMappingConfig) -> np.ndarray:
    method = config.method
    if method is ToneMappingMethod.REINHARD:
        return linear / (1.0 + linear)
    if method is ToneMappingMethod.ACES:
        mapped = (linear * (2.51 * linear + 0.03)) / (linear * (2.43 * linear + 0.59) + 0.14)
        return np.clip(mapped, 0.0, 1.0)
    if method is ToneMappingMethod.FILMIC:
        return _hable_filmic(linear, config.white_point)
    if method is ToneMappingMethod.GAMMA:
        return np.power(np.clip(linear, 0.0, 1.0), 1.0 / config.gamma)
    raise ValueError(f"unsupported tone mapping method: {method!r}")


class ToneMapper:
    """Compress linear scene radiance into a display-referred image."""

    name = "tone_mapping"

    def __init__(self, config: ToneMappingConfig | None = None) -> None:
        self.config = config or ToneMappingConfig()

    def process(self, image: RGBImage, context: ISPContext | None = None) -> RGBImage:
        if not isinstance(image, RGBImage):
            raise TypeError("ToneMapper requires an RGBImage input")

        linear = np.asarray(image.data, dtype=np.float32) * (2.0**self.config.exposure)
        linear = np.maximum(linear, 0.0)
        mapped = _tone_curve(linear, self.config)
        if (
            self.config.output_color_space is ColorSpace.LINEAR_SRGB
            or self.config.method is ToneMappingMethod.GAMMA
        ):
            display = np.clip(mapped, 0.0, 1.0)
        else:
            display = _srgb_encode(np.clip(mapped, 0.0, 1.0))
        display = np.clip(display, 0.0, 1.0).astype(np.float32)

        metadata = dict(image.metadata)
        metadata["tone_mapping"] = {
            "method": self.config.method.value,
            "exposure": self.config.exposure,
            "gamma": self.config.gamma,
        }
        return RGBImage(
            data=display,
            color_space=self.config.output_color_space,
            bit_depth=8,
            metadata=metadata,
        )
