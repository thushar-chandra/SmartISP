"""Training loop contracts."""

from __future__ import annotations

from typing import Any

from smartisp.training.config import TrainConfig


class Trainer:
    """Own the training loop, checkpointing, and validation cadence.

    The implementation is deferred so that data provenance, model architecture,
    and loss definitions can be finalized first.
    """

    def __init__(self, config: TrainConfig) -> None:
        self.config = config

    def fit(self) -> Any:
        raise NotImplementedError(
            "Trainer.fit is an interface placeholder; planned: seeded data loaders, "
            "mixed precision, checkpointing, validation hooks, and metric logging"
        )

    def validate(self) -> dict[str, float]:
        raise NotImplementedError(
            "Trainer.validate is an interface placeholder; planned: deterministic "
            "validation pass with PSNR, SSIM, LPIPS, and Delta E aggregation"
        )
