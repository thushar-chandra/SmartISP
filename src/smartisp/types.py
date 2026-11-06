"""Shared image and metadata types used by every SmartISP subsystem."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Channel(StrEnum):
    """Physical color channel represented by a CFA site."""

    RED = "R"
    GREEN = "G"
    BLUE = "B"


class CFAPattern(StrEnum):
    """Two-by-two color filter array layouts, written in row-major order."""

    RGGB = "RGGB"
    BGGR = "BGGR"
    GRBG = "GRBG"
    GBRG = "GBRG"

    def color_at(self, row: int, col: int) -> Channel:
        """Return the CFA channel at an absolute sensor coordinate."""
        if row < 0 or col < 0:
            raise ValueError("row and col must be non-negative")
        index = (row % 2) * 2 + (col % 2)
        return Channel(self.value[index])


class ColorSpace(StrEnum):
    """Color space labels attached to image buffers."""

    LINEAR_SRGB = "linear_srgb"
    SRGB = "srgb"
    DISPLAY_P3 = "display_p3"
    UNKNOWN = "unknown"


def _require_ndim(data: Any, expected: int, name: str) -> None:
    ndim = getattr(data, "ndim", None)
    if ndim is None:
        shape = getattr(data, "shape", None)
        ndim = len(shape) if shape is not None else None
    if ndim != expected:
        raise TypeError(f"{name} must be a {expected}-D array-like object, got ndim={ndim}")


@dataclass(frozen=True)
class RawFrame:
    """A single-channel Bayer mosaic with the metadata needed to process it.

    ``data`` is expected to be a NumPy array or a framework tensor of shape
    ``(height, width)``. Values may be unnormalized sensor codes or floats in
    the range ``[0, 1]``; ``bit_depth`` and the black/white levels describe the
    encoding. No subsystem may reinterpret an sRGB image as ``RawFrame``.
    """

    data: Any
    cfa: CFAPattern
    bit_depth: int = 16
    black_level: float = 0.0
    white_level: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_ndim(self.data, 2, "RawFrame.data")
        if not 1 <= self.bit_depth <= 32:
            raise ValueError("bit_depth must be between 1 and 32")
        if self.black_level < 0:
            raise ValueError("black_level must be non-negative")
        if self.white_level is not None and self.white_level <= self.black_level:
            raise ValueError("white_level must be greater than black_level")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)


@dataclass(frozen=True)
class RGBImage:
    """A three-channel image with an explicit color space label."""

    data: Any
    color_space: ColorSpace = ColorSpace.LINEAR_SRGB
    bit_depth: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_ndim(self.data, 3, "RGBImage.data")
        if self.data.shape[-1] != 3:
            raise ValueError("RGBImage.data must have three channels in the last axis")
        if self.bit_depth is not None and not 1 <= self.bit_depth <= 32:
            raise ValueError("bit_depth must be between 1 and 32 when provided")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)


@dataclass(frozen=True)
class WhiteBalanceGains:
    """Per-channel multiplicative gains used by white balance."""

    red: float
    green: float
    blue: float

    def __post_init__(self) -> None:
        for name, value in (("red", self.red), ("green", self.green), ("blue", self.blue)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} gain must be a finite positive number")

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.red, self.green, self.blue)


ImageBuffer = RawFrame | RGBImage
