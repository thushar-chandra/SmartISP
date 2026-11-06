"""Classical denoising interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from smartisp.classical.base import ISPContext, interface_not_implemented
from smartisp.types import RGBImage


class DenoiseMethod(StrEnum):
    """Planned classical denoisers."""

    BILATERAL = "bilateral"
    NON_LOCAL_MEANS = "non_local_means"
    GUIDED = "guided"


@dataclass(frozen=True)
class DenoiseConfig:
    """Configuration for :class:`Denoiser`."""

    method: DenoiseMethod = DenoiseMethod.BILATERAL
    strength: float = 1.0
    sigma_color: float = 0.05
    sigma_space: float = 2.0
    patch_size: int = 5
    search_window: int = 21
    preserve_edges: bool = True

    def __post_init__(self) -> None:
        if self.strength < 0:
            raise ValueError("strength must be non-negative")
        if self.sigma_color <= 0 or self.sigma_space <= 0:
            raise ValueError("sigma values must be positive")
        if self.patch_size < 1 or self.search_window < self.patch_size:
            raise ValueError("search_window must be at least patch_size")


class Denoiser:
    """Reduce sensor and demosaicing noise before tone mapping."""

    name = "denoise"

    def __init__(self, config: DenoiseConfig | None = None) -> None:
        self.config = config or DenoiseConfig()

    def process(self, image: RGBImage, context: ISPContext | None = None) -> RGBImage:
        raise interface_not_implemented(
            self.name,
            "implement bilateral and non-local-means baselines with edge preservation",
        )
