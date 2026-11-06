from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from smartisp.classical import (
    ClassicalISPPipeline,
    ISPContext,
    build_default_pipeline,
    default_stages,
)
from smartisp.types import CFAPattern, RawFrame


class DummyStage:
    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log

    def process(self, image: Any, context: ISPContext | None = None) -> Any:
        self.log.append(self.name)
        return f"{image}->{self.name}"


def test_pipeline_runs_stages_in_order() -> None:
    log: list[str] = []
    pipeline = ClassicalISPPipeline(
        [DummyStage("one", log), DummyStage("two", log), DummyStage("three", log)]
    )
    result = pipeline.run("start", ISPContext())
    assert result == "start->one->two->three"
    assert log == ["one", "two", "three"]
    assert pipeline.stage_names == ("one", "two", "three")


def test_pipeline_copies_explicit_frame_cfa_into_context() -> None:
    seen: list[ISPContext | None] = []

    class ContextStage:
        name = "context"

        def process(self, image: Any, context: ISPContext | None = None) -> Any:
            seen.append(context)
            return image

    frame = RawFrame(data=np.zeros((2, 2), dtype=np.uint16), cfa=CFAPattern.RGGB)
    pipeline = ClassicalISPPipeline([ContextStage()])
    pipeline.run(frame, ISPContext(black_level=1.0))
    assert seen[0] is not None
    assert seen[0].cfa is CFAPattern.RGGB
    assert seen[0].black_level == 1.0


def test_pipeline_trace() -> None:
    log: list[str] = []
    pipeline = ClassicalISPPipeline([DummyStage("one", log), DummyStage("two", log)])
    _, trace = pipeline.run_with_trace("start")
    assert trace.stage_names == ("one", "two")


def test_pipeline_rejects_duplicate_or_empty_stages() -> None:
    log: list[str] = []
    with pytest.raises(ValueError):
        ClassicalISPPipeline([])
    with pytest.raises(ValueError):
        ClassicalISPPipeline([DummyStage("one", log), DummyStage("one", log)])


def test_default_pipeline_order() -> None:
    pipeline = build_default_pipeline()
    assert pipeline.stage_names == (
        "bayer_preprocess",
        "white_balance",
        "demosaic",
        "color_correction",
        "denoise",
        "tone_mapping",
        "sharpen",
    )
    assert len(default_stages()) == 7
