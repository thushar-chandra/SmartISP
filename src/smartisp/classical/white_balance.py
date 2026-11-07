"""White balance estimation and application."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from smartisp.classical.base import ISPContext
from smartisp.types import (
    CFAPattern,
    Channel,
    ColorSpace,
    ImageBuffer,
    RawFrame,
    RGBImage,
    WhiteBalanceGains,
)

_EPSILON = 1e-8


class WhiteBalanceMethod(StrEnum):
    """Supported white balance estimators."""

    GRAY_WORLD = "gray_world"
    WHITE_PATCH = "white_patch"
    CAMERA_METADATA = "camera_metadata"
    LEARNED = "learned"


@dataclass(frozen=True)
class WhiteBalanceConfig:
    """Configuration for :class:`WhiteBalancer`."""

    method: WhiteBalanceMethod = WhiteBalanceMethod.CAMERA_METADATA
    min_gain: float = 0.25
    max_gain: float = 4.0
    normalize_green: bool = True

    def __post_init__(self) -> None:
        if self.min_gain <= 0 or self.max_gain <= self.min_gain:
            raise ValueError("gain limits must satisfy 0 < min_gain < max_gain")


def _channel_mask(cfa: CFAPattern, channel: Channel, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for row in range(2):
        for col in range(2):
            if cfa.color_at(row, col) is channel:
                mask[row::2, col::2] = True
    return mask


def _bayer_channel_values(data: np.ndarray, cfa: CFAPattern) -> list[np.ndarray]:
    return [
        data[_channel_mask(cfa, channel, data.shape)]
        for channel in (Channel.RED, Channel.GREEN, Channel.BLUE)
    ]


def _clip_range(data: np.ndarray) -> float:
    if np.issubdtype(data.dtype, np.integer):
        return float(np.iinfo(data.dtype).max)
    return max(1.0, float(np.nanmax(data)) if data.size else 1.0)


class WhiteBalancer:
    """Estimate and apply per-channel gains.

    The interface accepts either a Bayer frame or an RGB image so that the
    pipeline can place white balance before or after demosaicing. Placement is
    an explicit configuration decision, not an implicit assumption.
    """

    name = "white_balance"

    def __init__(self, config: WhiteBalanceConfig | None = None) -> None:
        self.config = config or WhiteBalanceConfig()

    def _raw_means(self, image: RawFrame) -> list[float]:
        data = np.asarray(image.data, dtype=np.float32)
        values = _bayer_channel_values(data, image.cfa)
        return [float(np.mean(v)) if v.size else 1.0 for v in values]

    def _rgb_means(self, image: RGBImage) -> list[float]:
        data = np.asarray(image.data, dtype=np.float32)
        reshaped = data.reshape(-1, 3)
        return [float(np.mean(reshaped[:, i])) for i in range(3)]

    def _white_patch(self, image: ImageBuffer) -> list[float]:
        if isinstance(image, RawFrame):
            data = np.asarray(image.data, dtype=np.float32)
            values = _bayer_channel_values(data, image.cfa)
        else:
            data = np.asarray(image.data, dtype=np.float32).reshape(-1, 3)
            values = [data[:, i] for i in range(3)]
        return [float(np.percentile(v, 99.5)) if v.size else 1.0 for v in values]

    def estimate(
        self,
        image: ImageBuffer,
        context: ISPContext | None = None,
    ) -> WhiteBalanceGains:
        method = self.config.method
        if method is WhiteBalanceMethod.LEARNED:
            raise NotImplementedError("learned illuminant estimation is not part of this release")

        metadata_gains = context.camera_white_balance if context is not None else None
        if method is WhiteBalanceMethod.CAMERA_METADATA and metadata_gains is not None:
            gains = [max(float(value), _EPSILON) for value in metadata_gains]
        else:
            if method is WhiteBalanceMethod.WHITE_PATCH:
                means = self._white_patch(image)
            else:
                means = (
                    self._raw_means(image)
                    if isinstance(image, RawFrame)
                    else self._rgb_means(image)
                )
            green_reference = means[1] if means[1] > _EPSILON else 1.0
            gains = [green_reference / max(value, _EPSILON) for value in means]
        if self.config.normalize_green and gains[1] > _EPSILON:
            gains = [value / gains[1] for value in gains]
        gains = [min(max(value, self.config.min_gain), self.config.max_gain) for value in gains]
        return WhiteBalanceGains(red=gains[0], green=gains[1], blue=gains[2])

    def apply(
        self,
        image: ImageBuffer,
        gains: WhiteBalanceGains,
        context: ISPContext | None = None,
    ) -> ImageBuffer:
        gain_tuple = gains.as_tuple()
        if isinstance(image, RawFrame):
            data = np.asarray(image.data).astype(np.float32, copy=True)
            gain_map = np.ones_like(data)
            for index, channel in enumerate((Channel.RED, Channel.GREEN, Channel.BLUE)):
                mask = _channel_mask(image.cfa, channel, data.shape)
                gain_map[mask] = gain_tuple[index]
            data *= gain_map
            np.clip(data, 0.0, _clip_range(np.asarray(image.data)), out=data)
            metadata = dict(image.metadata)
            metadata["white_balance_gains"] = gain_tuple
            return replace(image, data=data, metadata=metadata)

        data = np.asarray(image.data).astype(np.float32, copy=True)
        data *= np.asarray(gain_tuple, dtype=np.float32)
        np.clip(data, 0.0, _clip_range(np.asarray(image.data)), out=data)
        metadata = dict(image.metadata)
        metadata["white_balance_gains"] = gain_tuple
        return RGBImage(
            data=data,
            color_space=image.color_space
            if image.color_space is not ColorSpace.UNKNOWN
            else ColorSpace.LINEAR_SRGB,
            bit_depth=image.bit_depth,
            metadata=metadata,
        )

    def process(
        self,
        image: ImageBuffer,
        context: ISPContext | None = None,
    ) -> ImageBuffer:
        gains = self.estimate(image, context)
        return self.apply(image, gains, context)
