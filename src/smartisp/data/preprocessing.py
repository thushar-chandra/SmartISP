"""Domain-aware preprocessing helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from smartisp.data.domains import Domain
from smartisp.types import RawFrame

_EPSILON = 1e-8


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
    """Black-level correct and rescale a Bayer frame into ``output_range``."""
    if config.domain is not Domain.BAYER_RAW:
        raise ValueError("normalize_raw only accepts a BAYER_RAW normalization domain")
    data = np.asarray(raw.data)
    work = data.astype(np.float32, copy=True)
    if np.issubdtype(data.dtype, np.integer):
        white = float(np.iinfo(data.dtype).max)
    else:
        white = float(raw.white_level if raw.white_level is not None else 1.0)
    black = float(raw.black_level)
    normalized = (work - black) / max(white - black, _EPSILON)
    if config.subtract_mean and config.mean is not None:
        normalized -= float(config.mean[0])
    low, high = config.output_range
    scaled = low + normalized * (high - low)
    return replace(
        raw,
        data=np.clip(scaled, low, high).astype(np.float32),
        black_level=0.0,
        white_level=high,
    )


def _pad_to(array: np.ndarray, size: int) -> np.ndarray:
    height, width = array.shape[:2]
    pad_height = max(0, size - height)
    pad_width = max(0, size - width)
    if not pad_height and not pad_width:
        return array
    padding: tuple[tuple[int, int], ...]
    if array.ndim == 3:
        padding = ((0, pad_height), (0, pad_width), (0, 0))
    else:
        padding = ((0, pad_height), (0, pad_width))
    return np.pad(array, padding, mode="reflect")


def random_crop(image: Any, config: PatchConfig, seed: int | None = None) -> np.ndarray:
    """Extract one square patch, padding with reflection when necessary."""
    array = np.asarray(image)
    size = config.size
    if config.pad_if_smaller:
        array = _pad_to(array, size)
    height, width = array.shape[:2]
    if height < size or width < size:
        raise ValueError("image is smaller than the requested patch")
    if not config.random:
        top = (height - size) // 2
        left = (width - size) // 2
    else:
        rng = random.Random(seed)
        top = rng.randint(0, height - size)
        left = rng.randint(0, width - size)
    return np.ascontiguousarray(array[top : top + size, left : left + size])


def random_crop_pair(
    image: np.ndarray,
    target: np.ndarray,
    config: PatchConfig,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one synchronized crop to an input/target pair."""
    image_array = np.asarray(image)
    target_array = np.asarray(target)
    if image_array.shape[:2] != target_array.shape[:2]:
        raise ValueError("input and target must have matching spatial shapes")
    size = config.size
    if config.pad_if_smaller:
        image_array = _pad_to(image_array, size)
        target_array = _pad_to(target_array, size)
    height, width = image_array.shape[:2]
    if height < size or width < size:
        raise ValueError("pair is smaller than the requested patch")
    if not config.random:
        top = (height - size) // 2
        left = (width - size) // 2
    else:
        rng = random.Random(seed)
        top = rng.randint(0, height - size)
        left = rng.randint(0, width - size)
    return (
        np.ascontiguousarray(image_array[top : top + size, left : left + size]),
        np.ascontiguousarray(target_array[top : top + size, left : left + size]),
    )


def synchronized_augment(
    image: Any,
    target: Any,
    *,
    horizontal_flip: bool = True,
    vertical_flip: bool = True,
    rotations: tuple[int, ...] = (0, 90, 180, 270),
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply geometry-only augmentation identically to input and target."""
    image_array = np.asarray(image)
    target_array = np.asarray(target)
    if image_array.shape[:2] != target_array.shape[:2]:
        raise ValueError("input and target must have matching spatial shapes")
    rng = random.Random(seed)
    if horizontal_flip and rng.random() < 0.5:
        image_array = image_array[:, ::-1]
        target_array = target_array[:, ::-1]
    if vertical_flip and rng.random() < 0.5:
        image_array = image_array[::-1, :]
        target_array = target_array[::-1, :]
    if rotations:
        turns = rng.choice(rotations)
        if turns:
            image_array = np.rot90(image_array, turns // 90)
            target_array = np.rot90(target_array, turns // 90)
    return np.ascontiguousarray(image_array), np.ascontiguousarray(target_array)
