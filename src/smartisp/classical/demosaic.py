"""Bayer demosaicing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from smartisp.classical.base import ISPContext
from smartisp.types import CFAPattern, Channel, ColorSpace, RawFrame, RGBImage

_EPSILON = 1e-8

_CV2_BAYER_CODES: dict[CFAPattern, int] = {
    CFAPattern.RGGB: cv2.COLOR_BayerBG2BGR,
    CFAPattern.BGGR: cv2.COLOR_BayerRG2BGR,
    CFAPattern.GRBG: cv2.COLOR_BayerGB2BGR,
    CFAPattern.GBRG: cv2.COLOR_BayerGR2BGR,
}


class DemosaicMethod(StrEnum):
    """Supported demosaicing algorithm families."""

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


def _channel_mask(cfa: CFAPattern, channel: Channel, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for row in range(2):
        for col in range(2):
            if cfa.color_at(row, col) is channel:
                mask[row::2, col::2] = True
    return mask


def _normalized_convolution(
    raw: np.ndarray,
    cfa: CFAPattern,
    kernel: np.ndarray,
) -> np.ndarray:
    height, width = raw.shape
    channels: list[np.ndarray] = []
    for channel in (Channel.RED, Channel.GREEN, Channel.BLUE):
        mask = _channel_mask(cfa, channel, (height, width))
        sparse = np.where(mask, raw, 0.0).astype(np.float32)
        weight = mask.astype(np.float32)
        numerator = cv2.filter2D(sparse, -1, kernel, borderType=cv2.BORDER_REFLECT)
        denominator = cv2.filter2D(weight, -1, kernel, borderType=cv2.BORDER_REFLECT)
        interpolated = numerator / np.maximum(denominator, _EPSILON)
        interpolated[mask] = raw[mask]
        channels.append(interpolated)
    return np.stack(channels, axis=-1)


def _bilinear(raw: np.ndarray, cfa: CFAPattern) -> np.ndarray:
    kernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=np.float32)
    kernel /= kernel.sum()
    return _normalized_convolution(raw, cfa, kernel)


def _malvar(raw: np.ndarray, cfa: CFAPattern) -> np.ndarray:
    kernel = np.array(
        [
            [1.0, 4.0, 6.0, 4.0, 1.0],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [6.0, 24.0, 36.0, 24.0, 6.0],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [1.0, 4.0, 6.0, 4.0, 1.0],
        ],
        dtype=np.float32,
    )
    kernel /= kernel.sum()
    return _normalized_convolution(raw, cfa, kernel)


def _opencv_edge_aware(raw: np.ndarray, cfa: CFAPattern) -> np.ndarray:
    code = _CV2_BAYER_CODES[cfa]
    maximum = float(np.nanmax(raw)) if raw.size else 1.0
    if maximum <= 1.5:
        maximum = 1.0
    encoded = np.clip(raw / maximum * 65535.0, 0.0, 65535.0).astype(np.uint16)
    bgr = cv2.cvtColor(encoded, code)
    rgb = bgr[..., ::-1].astype(np.float32) / 65535.0 * maximum
    return np.clip(rgb, 0.0, maximum)


class Demosaicer:
    """Reconstruct a linear RGB image from a single-channel Bayer frame."""

    name = "demosaic"

    def __init__(self, config: DemosaicConfig | None = None) -> None:
        self.config = config or DemosaicConfig()

    def process(self, image: RawFrame, context: ISPContext | None = None) -> RGBImage:
        if not isinstance(image, RawFrame):
            raise TypeError("Demosaicer requires a RawFrame input")

        raw = np.asarray(image.data, dtype=np.float32)
        cfa = image.cfa
        if cfa is None and context is not None:
            cfa = context.cfa
        if cfa is None:
            raise ValueError("demosaicing requires an explicit CFA pattern")

        method = self.config.method
        if method is DemosaicMethod.BILINEAR:
            rgb = _bilinear(raw, cfa)
        elif method is DemosaicMethod.MALVAR:
            rgb = _malvar(raw, cfa)
        else:
            rgb = _opencv_edge_aware(raw, cfa)

        if self.config.align_to_even:
            height = rgb.shape[0] - (rgb.shape[0] % 2)
            width = rgb.shape[1] - (rgb.shape[1] % 2)
            rgb = rgb[:height, :width]

        np.clip(rgb, 0.0, 1.0, out=rgb)
        metadata = dict(image.metadata)
        metadata["demosaic"] = {"method": method.value, "cfa": cfa.value}
        return RGBImage(
            data=rgb,
            color_space=self.config.output_color_space,
            metadata=metadata,
        )
