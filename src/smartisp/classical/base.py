"""Classical ISP stage contracts and shared context."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from smartisp.types import CFAPattern, ColorSpace


@dataclass(frozen=True)
class ISPContext:
    """Sensor and camera metadata shared by classical and learned stages."""

    cfa: CFAPattern | None = None
    black_level: float = 0.0
    white_level: float | None = None
    camera_white_balance: tuple[float, float, float] | None = None
    color_matrix: tuple[tuple[float, float, float], ...] | None = None
    working_space: ColorSpace = ColorSpace.LINEAR_SRGB
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.black_level < 0:
            raise ValueError("black_level must be non-negative")
        if self.white_level is not None and self.white_level <= self.black_level:
            raise ValueError("white_level must be greater than black_level")
        if self.color_matrix is not None and len(self.color_matrix) != 3:
            raise ValueError("color_matrix must have three rows")
        if self.color_matrix is not None and any(len(row) != 3 for row in self.color_matrix):
            raise ValueError("color_matrix must be 3x3")


@runtime_checkable
class ISPStage(Protocol):
    """A processing stage that consumes and returns an image-like buffer."""

    name: str

    def process(self, image: Any, context: ISPContext | None = None) -> Any:
        """Transform ``image`` and return the next buffer in the pipeline."""


def interface_not_implemented(stage: str, planned: str) -> NotImplementedError:
    """Create a consistent error for stages that are still interface-only."""
    return NotImplementedError(
        f"{stage} is an interface placeholder. Planned implementation: {planned}"
    )
