"""Bayer-domain preprocessing: black level, normalization, and clipping."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from smartisp.classical.base import ISPContext
from smartisp.types import CFAPattern, Channel, RawFrame

_EPSILON = 1e-8


class BlackLevelMethod(StrEnum):
    """Supported ways to estimate or apply sensor black level."""

    FIXED = "fixed"
    PER_CHANNEL = "per_channel"


@dataclass(frozen=True)
class BayerPreprocessConfig:
    """Configuration for :class:`BayerPreprocessor`."""

    apply_black_level: bool = True
    normalize: bool = True
    clip: bool = True
    black_level_method: BlackLevelMethod = BlackLevelMethod.FIXED
    white_level: float | None = None
    output_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.white_level is not None and self.white_level <= 0:
            raise ValueError("white_level must be positive when provided")
        if self.output_dtype not in {"float32", "float16"}:
            raise ValueError("output_dtype must be float32 or float16")


def _channel_mask(cfa: CFAPattern, channel: Channel, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for row in range(2):
        for col in range(2):
            if cfa.color_at(row, col) is channel:
                mask[row::2, col::2] = True
    return mask


def _is_normalized(data: np.ndarray) -> bool:
    if not np.issubdtype(data.dtype, np.floating) or data.size == 0:
        return False
    return bool(np.nanmax(data) <= 1.5)


class BayerPreprocessor:
    """Convert sensor codes into a normalized Bayer mosaic.

    Integer sensor codes are normalized with ``(code - black) / (white - black)``.
    Floating-point frames already in ``[0, 1]`` are preserved so that repeated
    pipeline runs do not divide by a 16-bit white level. The CFA pattern and
    provenance metadata are propagated unchanged.
    """

    name = "bayer_preprocess"

    def __init__(self, config: BayerPreprocessConfig | None = None) -> None:
        self.config = config or BayerPreprocessConfig()

    def process(self, image: RawFrame, context: ISPContext | None = None) -> RawFrame:
        if not isinstance(image, RawFrame):
            raise TypeError("BayerPreprocessor requires a RawFrame input")

        data = np.asarray(image.data)
        work = data.astype(np.float32, copy=True)
        original_dtype = str(data.dtype)
        already_normalized = _is_normalized(data)

        black = image.black_level
        if context is not None and context.black_level:
            black = context.black_level
        white = self.config.white_level
        if white is None:
            white = image.white_level
        if white is None and context is not None:
            white = context.white_level
        if white is None:
            if np.issubdtype(data.dtype, np.integer):
                white = float(np.iinfo(data.dtype).max)
            elif already_normalized:
                white = 1.0
            else:
                white = float(np.nanmax(data)) if data.size else 1.0

        if self.config.apply_black_level:
            if self.config.black_level_method is BlackLevelMethod.PER_CHANNEL:
                cfa = image.cfa
                for channel in (Channel.RED, Channel.GREEN, Channel.BLUE):
                    mask = _channel_mask(cfa, channel, work.shape)
                    if mask.any():
                        estimate = float(np.percentile(work[mask], 0.1))
                        work[mask] -= estimate
            else:
                work -= black

        if self.config.normalize:
            denominator = max(float(white) - float(black), _EPSILON)
            if not (already_normalized and denominator <= 1.5):
                work /= denominator

        if self.config.clip:
            np.clip(work, 0.0, 1.0, out=work)

        dtype = np.float16 if self.config.output_dtype == "float16" else np.float32
        output = work.astype(dtype)
        metadata = dict(image.metadata)
        metadata.update(
            {
                "bayer_preprocess": {
                    "source_dtype": original_dtype,
                    "black_level": float(black),
                    "white_level": float(white),
                    "normalized": bool(self.config.normalize),
                }
            }
        )
        return replace(
            image,
            data=output,
            black_level=0.0,
            white_level=1.0,
            metadata=metadata,
        )
