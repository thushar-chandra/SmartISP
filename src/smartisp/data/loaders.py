"""Torch data loader construction and pairing collation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DataLoaderConfig:
    """Data loader settings shared by training and validation."""

    batch_size: int = 8
    shuffle: bool = True
    num_workers: int = 2
    pin_memory: bool = True
    drop_last: bool = False
    persistent_workers: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")


def _stack(values: list[Any]) -> Any:
    first = values[0]
    if hasattr(first, "detach"):
        import torch

        return torch.stack(values, dim=0)
    if isinstance(first, np.ndarray):
        return np.stack(values, axis=0)
    return values


def collate_paired(samples: Any) -> Any:
    """Stack array/tensor fields and keep provenance metadata as lists."""
    if not samples:
        return {}
    keys = set(samples[0])
    batch: dict[str, Any] = {}
    for key in keys:
        values = [sample[key] for sample in samples]
        first = values[0]
        if isinstance(first, np.ndarray) or hasattr(first, "detach"):
            batch[key] = _stack(values)
        else:
            batch[key] = values
    return batch


def build_dataloader(dataset: Any, config: DataLoaderConfig) -> Any:
    """Build a torch ``DataLoader`` without importing torch at module import time."""
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "build_dataloader requires PyTorch; install the 'torch' extra or use Colab"
        ) from exc
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        persistent_workers=config.persistent_workers,
        collate_fn=collate_paired,
    )
