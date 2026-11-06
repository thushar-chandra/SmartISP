"""Dataset preprocessing interfaces.

All transforms are domain-aware. Bayer RAW normalization is separate from sRGB
display encoding, and paired transforms must apply identical geometry to input
and target images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smartisp.data.domains import Domain
from smartisp.types import RawFrame


@dataclass(frozen=True)
class NormalizationConfig:
    """How a buffer is scaled before entering a model."""

    domain: Domain
    output_range: tuple[float, float] = (0.0, 1.0)
    subtract_mean: bool = False
    mean: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        low, high = self.output_range
        if high <= low:
            raise ValueError("output_range must be increasing")
        if self.subtract_mean and self.mean is None:
            raise ValueError("mean is required when subtract_mean=True")


@dataclass(frozen=True)
class PatchConfig:
    """Deterministic or random patch extraction settings."""

    size: int = 256
    scale: int = 1
    random: bool = True
    pad_if_smaller: bool = True

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be positive")
        if self.scale <= 0:
            raise ValueError("scale must be positive")


def normalize_raw(raw: RawFrame, config: NormalizationConfig) -> RawFrame:
    raise NotImplementedError(
        "normalize_raw is an interface placeholder; planned: domain-aware black-level "
        "correction and scaling without treating sRGB input as RAW"
    )


def random_crop(image: Any, config: PatchConfig, seed: int | None = None) -> Any:
    raise NotImplementedError(
        "random_crop is an interface placeholder; planned: synchronized crops for "
        "paired input/target buffers and deterministic seeding"
    )


def synchronized_augment(
    image: Any,
    target: Any,
    *,
    horizontal_flip: bool = True,
    vertical_flip: bool = True,
    rotations: tuple[int, ...] = (0, 90, 180, 270),
    seed: int | None = None,
) -> tuple[Any, Any]:
    raise NotImplementedError(
        "synchronized_augment is an interface placeholder; planned: geometry-only "
        "augmentation that keeps input and target aligned"
    )
