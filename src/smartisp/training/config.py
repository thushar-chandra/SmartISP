"""Training configuration parsed from YAML without importing PyTorch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from smartisp.data.domains import Domain


@dataclass(frozen=True)
class TrainConfig:
    """Flat, validated training settings."""

    experiment: str = "smartisp"
    seed: int = 0
    device: str = "auto"
    epochs: int = 100
    batch_size: int = 8
    num_workers: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    amp: bool = True
    grad_clip: float = 1.0
    patch_size: int = 256
    val_interval: int = 1
    data_root: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")
    output_dir: Path = Path("outputs")
    resume: Path | None = None
    train_split: str = "train"
    val_split: str = "val"
    input_domain: Domain = Domain.SRGB
    target_domain: Domain = Domain.SRGB

    def __post_init__(self) -> None:
        if not self.experiment:
            raise ValueError("experiment must not be empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.grad_clip < 0:
            raise ValueError("grad_clip must be non-negative")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.val_interval <= 0:
            raise ValueError("val_interval must be positive")
        if not self.train_split or not self.val_split:
            raise ValueError("split names must not be empty")
        if self.train_split == self.val_split:
            raise ValueError("train and validation splits must differ")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> TrainConfig:
        """Build a config from the ``training``/``paths`` YAML sections."""
        known = {field.name for field in cls.__dataclass_fields__.values()}
        values: dict[str, Any] = {}
        for key, value in mapping.items():
            if key not in known:
                continue
            if key in {"data_root", "checkpoint_dir", "output_dir"}:
                values[key] = Path(value)
            elif key == "resume":
                values[key] = Path(value) if value is not None else None
            elif key in {"input_domain", "target_domain"}:
                values[key] = value if isinstance(value, Domain) else Domain(value)
            else:
                values[key] = value
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("data_root", "checkpoint_dir", "output_dir"):
            payload[key] = str(payload[key])
        payload["resume"] = str(self.resume) if self.resume is not None else None
        payload["input_domain"] = self.input_domain.value
        payload["target_domain"] = self.target_domain.value
        return payload
