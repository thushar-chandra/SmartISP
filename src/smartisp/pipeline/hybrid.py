"""Hybrid composition of classical ISP stages and learned restoration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from smartisp.classical.base import ISPContext
from smartisp.classical.pipeline import ClassicalISPPipeline


class HybridMode(StrEnum):
    """Where learned restoration is inserted relative to the classical ISP."""

    CLASSICAL_ONLY = "classical_only"
    LEARNED_ONLY = "learned_only"
    PRE_ISP = "pre_isp"
    POST_ISP = "post_isp"


@runtime_checkable
class RestorationModel(Protocol):
    """Minimal contract for a learned restoration module."""

    def restore(self, image: Any) -> Any:
        """Restore an image buffer and return the same kind of buffer."""


@dataclass(frozen=True)
class HybridPipelineConfig:
    """Configuration for :class:`HybridISPPipeline`."""

    mode: HybridMode = HybridMode.POST_ISP

    def __post_init__(self) -> None:
        if not isinstance(self.mode, HybridMode):
            raise TypeError("mode must be a HybridMode")


class HybridISPPipeline:
    """Run classical and learned stages in an explicit, inspectable order.

    ``PRE_ISP`` applies restoration before the classical pipeline (typically on
    RAW or linear data), while ``POST_ISP`` applies it to the display-referred
    RGB output. The two placements require different checkpoints and input
    domains; this class never silently converts between them.
    """

    def __init__(
        self,
        *,
        classical: ClassicalISPPipeline | None = None,
        restorer: RestorationModel | None = None,
        config: HybridPipelineConfig | None = None,
    ) -> None:
        self.config = config or HybridPipelineConfig()
        self.classical = classical
        self.restorer = restorer
        needs_classical = self.config.mode in {
            HybridMode.CLASSICAL_ONLY,
            HybridMode.PRE_ISP,
            HybridMode.POST_ISP,
        }
        needs_restoration = self.config.mode in {
            HybridMode.LEARNED_ONLY,
            HybridMode.PRE_ISP,
            HybridMode.POST_ISP,
        }
        if needs_classical and self.classical is None:
            raise ValueError(f"mode={self.config.mode.value!r} requires a classical pipeline")
        if needs_restoration and self.restorer is None:
            raise ValueError(f"mode={self.config.mode.value!r} requires a restoration model")

    def run_classical(
        self,
        image: Any,
        context: ISPContext | None = None,
    ) -> Any:
        if self.classical is None:
            raise ValueError("no classical pipeline was configured")
        return self.classical.run(image, context)

    def run_restoration(self, image: Any) -> Any:
        if self.restorer is None:
            raise ValueError("no restoration model was configured")
        return self.restorer.restore(image)

    def run(self, image: Any, context: ISPContext | None = None) -> Any:
        mode = self.config.mode
        if mode is HybridMode.CLASSICAL_ONLY:
            return self.run_classical(image, context)
        if mode is HybridMode.LEARNED_ONLY:
            return self.run_restoration(image)
        if mode is HybridMode.PRE_ISP:
            restored = self.run_restoration(image)
            return self.run_classical(restored, context)
        if mode is HybridMode.POST_ISP:
            classical_output = self.run_classical(image, context)
            return self.run_restoration(classical_output)
        raise ValueError(f"unsupported hybrid mode: {mode!r}")
