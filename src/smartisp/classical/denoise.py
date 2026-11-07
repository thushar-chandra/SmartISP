"""Classical denoising."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from smartisp.classical.base import ISPContext
from smartisp.types import RGBImage


class DenoiseMethod(StrEnum):
    """Supported classical denoisers."""

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


def _non_local_means(data: np.ndarray, config: DenoiseConfig) -> np.ndarray:
    from skimage.restoration import denoise_nl_means

    patch_size = max(3, config.patch_size)
    patch_distance = max(patch_size + 1, config.search_window // 2)
    channels = [
        denoise_nl_means(
            data[..., index],
            h=config.strength * config.sigma_color,
            sigma=config.sigma_color,
            patch_size=patch_size,
            patch_distance=patch_distance,
            fast_mode=True,
            preserve_range=True,
        )
        for index in range(3)
    ]
    return np.stack(channels, axis=-1).astype(np.float32)


class Denoiser:
    """Reduce sensor and demosaicing noise before tone mapping."""

    name = "denoise"

    def __init__(self, config: DenoiseConfig | None = None) -> None:
        self.config = config or DenoiseConfig()

    def process(self, image: RGBImage, context: ISPContext | None = None) -> RGBImage:
        if not isinstance(image, RGBImage):
            raise TypeError("Denoiser requires an RGBImage input")

        data = np.asarray(image.data, dtype=np.float32)
        config = self.config
        if config.strength == 0.0:
            filtered = data.copy()
        elif config.method is DenoiseMethod.NON_LOCAL_MEANS:
            filtered = _non_local_means(data, config)
        elif config.method is DenoiseMethod.GUIDED:
            try:
                filtered = cv2.ximgproc.guidedFilter(  # type: ignore[attr-defined]
                    guide=data,
                    src=data,
                    radius=max(1, round(config.sigma_space)),
                    eps=config.sigma_color**2,
                )
            except (AttributeError, cv2.error):
                filtered = cv2.bilateralFilter(
                    data,
                    d=max(1, round(config.sigma_space) * 2 + 1),
                    sigmaColor=config.sigma_color * config.strength,
                    sigmaSpace=config.sigma_space,
                )
        else:
            filtered = cv2.bilateralFilter(
                data,
                d=max(1, round(config.sigma_space) * 2 + 1),
                sigmaColor=config.sigma_color * config.strength,
                sigmaSpace=config.sigma_space,
            )
        maximum = max(1.0, float(np.nanmax(data)) if data.size else 1.0)
        np.clip(filtered, 0.0, maximum, out=filtered)
        metadata = dict(image.metadata)
        metadata["denoise"] = {"method": config.method.value, "strength": config.strength}
        return RGBImage(
            data=filtered,
            color_space=image.color_space,
            bit_depth=image.bit_depth,
            metadata=metadata,
        )
