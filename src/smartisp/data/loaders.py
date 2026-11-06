"""Torch data loader interfaces kept separate from dataset discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


def collate_paired(samples: Any) -> Any:
    raise NotImplementedError(
        "collate_paired is an interface placeholder; planned: stack input/target "
        "tensors and collate domain metadata without dropping provenance"
    )


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
