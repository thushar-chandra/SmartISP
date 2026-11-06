from __future__ import annotations

from typing import Any

import pytest

from smartisp.classical import ClassicalISPPipeline, ISPContext
from smartisp.pipeline import HybridISPPipeline, HybridMode, HybridPipelineConfig


class DummyStage:
    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log

    def process(self, image: Any, context: ISPContext | None = None) -> Any:
        self.log.append(self.name)
        return f"{image}->{self.name}"


class DummyRestorer:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def restore(self, image: Any) -> Any:
        self.log.append("restore")
        return f"{image}->restore"


def _classical(log: list[str]) -> ClassicalISPPipeline:
    return ClassicalISPPipeline([DummyStage("isp", log)])


def test_classical_only_does_not_require_restorer() -> None:
    log: list[str] = []
    pipeline = HybridISPPipeline(
        classical=_classical(log),
        config=HybridPipelineConfig(mode=HybridMode.CLASSICAL_ONLY),
    )
    assert pipeline.run("input") == "input->isp"
    assert log == ["isp"]


def test_post_isp_order() -> None:
    log: list[str] = []
    pipeline = HybridISPPipeline(
        classical=_classical(log),
        restorer=DummyRestorer(log),
        config=HybridPipelineConfig(mode=HybridMode.POST_ISP),
    )
    assert pipeline.run("input") == "input->isp->restore"
    assert log == ["isp", "restore"]


def test_pre_isp_order() -> None:
    log: list[str] = []
    pipeline = HybridISPPipeline(
        classical=_classical(log),
        restorer=DummyRestorer(log),
        config=HybridPipelineConfig(mode=HybridMode.PRE_ISP),
    )
    assert pipeline.run("input") == "input->restore->isp"
    assert log == ["restore", "isp"]


def test_learned_only_order() -> None:
    log: list[str] = []
    pipeline = HybridISPPipeline(
        restorer=DummyRestorer(log),
        config=HybridPipelineConfig(mode=HybridMode.LEARNED_ONLY),
    )
    assert pipeline.run("input") == "input->restore"
    assert log == ["restore"]


def test_modes_validate_required_components() -> None:
    log: list[str] = []
    with pytest.raises(ValueError):
        HybridISPPipeline(
            classical=_classical(log),
            config=HybridPipelineConfig(mode=HybridMode.POST_ISP),
        )
    with pytest.raises(ValueError):
        HybridISPPipeline(config=HybridPipelineConfig(mode=HybridMode.CLASSICAL_ONLY))
