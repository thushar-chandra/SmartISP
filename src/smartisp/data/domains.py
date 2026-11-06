"""Image-domain provenance rules.

SmartISP never infers that an sRGB image is Bayer RAW. The domain is carried
explicitly through configuration and validated against file provenance before
any preprocessing occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from smartisp.types import CFAPattern


class Domain(StrEnum):
    """Physical encoding of an image buffer or dataset file."""

    BAYER_RAW = "bayer_raw"
    LINEAR_RGB = "linear_rgb"
    SRGB = "srgb"


BAYER_RAW_EXTENSIONS: frozenset[str] = frozenset(
    {".npy", ".raw", ".tif", ".tiff", ".dng", ".arw", ".cr2", ".nef"}
)
LINEAR_RGB_EXTENSIONS: frozenset[str] = frozenset({".npy", ".tif", ".tiff", ".exr"})
SRGB_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
)
CAMERA_RAW_EXTENSIONS: frozenset[str] = frozenset({".raw", ".dng", ".arw", ".cr2", ".nef"})


class DomainMismatchError(ValueError):
    """Raised when a declared image domain contradicts file provenance."""


def normalize_suffix(suffix: str) -> str:
    """Return a lowercase suffix with a leading dot."""
    lowered = suffix.lower()
    return lowered if lowered.startswith(".") else f".{lowered}"


def validate_domain_for_suffix(domain: Domain, suffix: str, role: str) -> None:
    """Reject accidental reinterpretation of display images as sensor data."""
    normalized = normalize_suffix(suffix)
    if domain is Domain.BAYER_RAW and normalized not in BAYER_RAW_EXTENSIONS:
        raise DomainMismatchError(
            f"{role} declares {domain.value} but has suffix {normalized!r}. "
            "Display-referred files such as SIDD sRGB PNGs are not Bayer RAW."
        )
    if domain is Domain.LINEAR_RGB and normalized not in LINEAR_RGB_EXTENSIONS:
        raise DomainMismatchError(
            f"{role} declares {domain.value} but has suffix {normalized!r}; "
            "use .npy, .tif, .tiff, or .exr for linear RGB arrays."
        )
    if domain is Domain.SRGB and normalized in CAMERA_RAW_EXTENSIONS:
        raise DomainMismatchError(
            f"{role} declares {domain.value} but has camera RAW suffix {normalized!r}."
        )


@dataclass(frozen=True)
class DatasetProvenance:
    """Human- and machine-readable origin of a dataset split."""

    name: str
    domain: Domain
    cfa: CFAPattern | None = None
    is_synthetic: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.domain is Domain.BAYER_RAW and self.cfa is None:
            raise ValueError("Bayer RAW provenance must specify the CFA pattern")

    def validate_file(self, path: str | Path) -> None:
        validate_domain_for_suffix(self.domain, Path(path).suffix, role=self.name)
