"""Bayer-domain preprocessing: black level, normalization, and clipping."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from smartisp.classical.base import ISPContext, interface_not_implemented
from smartisp.types import RawFrame


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


class BayerPreprocessor:
    """Convert sensor codes into a normalized Bayer mosaic.

    The planned implementation estimates per-channel black levels when
    requested, subtracts them, divides by the white level, and optionally
    clips the result to ``[0, 1]``. It preserves the CFA pattern and metadata.
    """

    name = "bayer_preprocess"

    def __init__(self, config: BayerPreprocessConfig | None = None) -> None:
        self.config = config or BayerPreprocessConfig()

    def process(self, image: RawFrame, context: ISPContext | None = None) -> RawFrame:
        raise interface_not_implemented(
            self.name,
            "estimate black level, normalize to [0, 1], clip, and propagate provenance",
        )
