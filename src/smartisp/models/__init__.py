"""Learned restoration models and losses.

Torch-dependent classes are imported lazily so that data, evaluation, and
configuration modules remain usable in a lightweight CPU environment.
"""

from __future__ import annotations

from typing import Any

from smartisp.models.config import LossConfig, LossKind, UNetConfig

__all__ = [
    "LossConfig",
    "LossKind",
    "RestorationLoss",
    "UNetConfig",
    "UNetRestorer",
    "build_loss",
    "build_unet",
]


def __getattr__(name: str) -> Any:
    if name == "UNetRestorer":
        from smartisp.models.unet import UNetRestorer

        return UNetRestorer
    if name == "build_unet":
        from smartisp.models.unet import build_unet

        return build_unet
    if name in {"RestorationLoss", "build_loss"}:
        from smartisp.models.losses import RestorationLoss, build_loss

        return {
            "RestorationLoss": RestorationLoss,
            "build_loss": build_loss,
        }[name]
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(__all__)
