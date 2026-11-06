"""White balance estimation and application interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from smartisp.classical.base import ISPContext, interface_not_implemented
from smartisp.types import ImageBuffer, WhiteBalanceGains


class WhiteBalanceMethod(StrEnum):
    """Planned white balance estimators."""

    GRAY_WORLD = "gray_world"
    WHITE_PATCH = "white_patch"
    CAMERA_METADATA = "camera_metadata"
    LEARNED = "learned"


@dataclass(frozen=True)
class WhiteBalanceConfig:
    """Configuration for :class:`WhiteBalancer`."""

    method: WhiteBalanceMethod = WhiteBalanceMethod.CAMERA_METADATA
    min_gain: float = 0.25
    max_gain: float = 4.0
    normalize_green: bool = True

    def __post_init__(self) -> None:
        if self.min_gain <= 0 or self.max_gain <= self.min_gain:
            raise ValueError("gain limits must satisfy 0 < min_gain < max_gain")


class WhiteBalancer:
    """Estimate and apply per-channel gains.

    The interface accepts either a Bayer frame or an RGB image so that the
    pipeline can place white balance before or after demosaicing. Placement is
    an explicit configuration decision, not an implicit assumption.
    """

    name = "white_balance"

    def __init__(self, config: WhiteBalanceConfig | None = None) -> None:
        self.config = config or WhiteBalanceConfig()

    def estimate(
        self,
        image: ImageBuffer,
        context: ISPContext | None = None,
    ) -> WhiteBalanceGains:
        raise interface_not_implemented(
            self.name,
            "estimate robust illuminant gains from camera metadata or gray-world statistics",
        )

    def apply(
        self,
        image: ImageBuffer,
        gains: WhiteBalanceGains,
        context: ISPContext | None = None,
    ) -> ImageBuffer:
        raise interface_not_implemented(
            self.name,
            "apply gains to the appropriate Bayer or RGB channels with clipping",
        )

    def process(
        self,
        image: ImageBuffer,
        context: ISPContext | None = None,
    ) -> ImageBuffer:
        gains = self.estimate(image, context)
        return self.apply(image, gains, context)
