"""Composition of classical ISP stages."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from smartisp.classical.base import ISPContext, ISPStage
from smartisp.classical.bayer import BayerPreprocessor
from smartisp.classical.color_correction import ColorCorrector
from smartisp.classical.demosaic import Demosaicer
from smartisp.classical.denoise import Denoiser
from smartisp.classical.sharpen import Sharpener
from smartisp.classical.tone_mapping import ToneMapper
from smartisp.classical.white_balance import WhiteBalancer
from smartisp.types import ImageBuffer


@dataclass(frozen=True)
class PipelineTrace:
    """Names of the stages that processed one image."""

    stage_names: tuple[str, ...]


class ClassicalISPPipeline:
    """Execute an ordered sequence of ISP stages.

    The pipeline is intentionally explicit: callers decide stage order and
    whether white balance runs in the Bayer or RGB domain. The default order is
    Bayer preprocessing, white balance, demosaic, color correction, denoise,
    tone mapping, and sharpening.
    """

    def __init__(self, stages: Sequence[ISPStage]) -> None:
        if not stages:
            raise ValueError("a pipeline requires at least one stage")
        names = tuple(stage.name for stage in stages)
        if len(set(names)) != len(names):
            raise ValueError(f"stage names must be unique, got {names}")
        self.stages = tuple(stages)

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages)

    def run(
        self,
        image: ImageBuffer,
        context: ISPContext | None = None,
    ) -> ImageBuffer:
        """Run every stage in order.

        If the caller supplies a context without a CFA pattern and the input
        frame carries one, the frame's explicit pattern is used. The pipeline
        never guesses a pattern from the array shape.
        """
        image_cfa = getattr(image, "cfa", None)
        if context is None:
            active_context = ISPContext(cfa=image_cfa)
        elif context.cfa is None and image_cfa is not None:
            active_context = replace(context, cfa=image_cfa)
        else:
            active_context = context
        current: ImageBuffer = image
        for stage in self.stages:
            current = stage.process(current, active_context)
        return current

    def run_with_trace(
        self,
        image: ImageBuffer,
        context: ISPContext | None = None,
    ) -> tuple[ImageBuffer, PipelineTrace]:
        result = self.run(image, context)
        return result, PipelineTrace(stage_names=self.stage_names)


def default_stages() -> tuple[ISPStage, ...]:
    """Return the planned classical baseline in a documented order."""
    stages: tuple[ISPStage, ...] = (
        BayerPreprocessor(),
        WhiteBalancer(),
        Demosaicer(),
        ColorCorrector(),
        Denoiser(),
        ToneMapper(),
        Sharpener(),
    )
    return stages


def build_default_pipeline() -> ClassicalISPPipeline:
    """Construct the classical baseline pipeline."""
    return ClassicalISPPipeline(default_stages())
