"""Paired dataset discovery with strict domain/provenance checks."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smartisp.data.domains import Domain, validate_domain_for_suffix


@dataclass(frozen=True)
class PairedDatasetConfig:
    """Configuration for a paired restoration dataset.

    The defaults describe SIDD-style sRGB PNG pairs. They are deliberately not
    Bayer RAW. A RAW dataset must set ``input_domain``/``target_domain``,
    provide ``.npy``/``.raw``/``.tiff`` files, and carry CFA metadata.
    """

    root: str | Path
    input_dir: str = "input"
    target_dir: str = "target"
    input_domain: Domain = Domain.SRGB
    target_domain: Domain = Domain.SRGB
    input_extensions: tuple[str, ...] = (".png",)
    target_extensions: tuple[str, ...] | None = None
    crop_size: int | None = None
    scale: int = 1
    seed: int = 0
    recursive: bool = True
    allow_domain_override: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        input_extensions = tuple(_normalize_extension(ext) for ext in self.input_extensions)
        target_extensions = tuple(
            _normalize_extension(ext)
            for ext in (
                self.target_extensions
                if self.target_extensions is not None
                else self.input_extensions
            )
        )
        if not input_extensions or not target_extensions:
            raise ValueError("extensions must not be empty")
        object.__setattr__(self, "input_extensions", input_extensions)
        object.__setattr__(self, "target_extensions", target_extensions)
        for field_name in ("input_dir", "target_dir"):
            value = getattr(self, field_name)
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{field_name} must be a relative path inside root")
        if self.crop_size is not None and self.crop_size <= 0:
            raise ValueError("crop_size must be positive when provided")
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.allow_domain_override:
            for extension in input_extensions:
                validate_domain_for_suffix(self.input_domain, extension, role="input")
            for extension in target_extensions:
                validate_domain_for_suffix(self.target_domain, extension, role="target")


def _normalize_extension(extension: str) -> str:
    lowered = extension.lower()
    if not lowered.startswith("."):
        lowered = f".{lowered}"
    if len(lowered) < 2:
        raise ValueError("extensions must contain a suffix such as '.png'")
    return lowered


@dataclass(frozen=True)
class PairedEntry:
    """Paths for one input/target pair."""

    relative_key: str
    input_path: Path
    target_path: Path


def _index_files(directory: Path, extensions: tuple[str, ...], recursive: bool) -> dict[str, Path]:
    iterator: Iterator[Path]
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    indexed: dict[str, Path] = {}
    for path in sorted(iterator):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        key = path.relative_to(directory).with_suffix("").as_posix()
        if key in indexed:
            raise ValueError(f"duplicate stem {key!r} under {directory}")
        indexed[key] = path
    return indexed


def discover_pairs(
    root: str | Path,
    *,
    input_dir: str = "input",
    target_dir: str = "target",
    input_extensions: tuple[str, ...] = (".png",),
    target_extensions: tuple[str, ...] | None = None,
    recursive: bool = True,
) -> tuple[PairedEntry, ...]:
    """Discover same-stem input/target pairs and fail on unmatched entries."""
    root_path = Path(root)
    input_root = root_path / input_dir
    target_root = root_path / target_dir
    if not input_root.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_root}")
    if not target_root.is_dir():
        raise FileNotFoundError(f"target directory does not exist: {target_root}")

    normalized_inputs = tuple(_normalize_extension(ext) for ext in input_extensions)
    normalized_targets = tuple(
        _normalize_extension(ext)
        for ext in (target_extensions if target_extensions is not None else input_extensions)
    )
    inputs = _index_files(input_root, normalized_inputs, recursive)
    targets = _index_files(target_root, normalized_targets, recursive)
    missing_targets = sorted(set(inputs) - set(targets))
    missing_inputs = sorted(set(targets) - set(inputs))
    if missing_targets or missing_inputs:
        raise ValueError(
            "unmatched pairs; "
            f"missing targets={missing_targets[:5]}, missing inputs={missing_inputs[:5]}"
        )
    return tuple(
        PairedEntry(
            relative_key=key,
            input_path=inputs[key],
            target_path=targets[key],
        )
        for key in sorted(inputs)
    )


class PairedImageDataset:
    """Index a paired dataset and expose tensor-ready samples.

    File discovery and provenance validation are implemented now. Decoding and
    tensor conversion are intentionally deferred to the training stage.
    """

    def __init__(self, config: PairedDatasetConfig) -> None:
        self.config = config
        root = Path(config.root)
        if not root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {root}")
        self.entries = discover_pairs(
            root,
            input_dir=config.input_dir,
            target_dir=config.target_dir,
            input_extensions=config.input_extensions,
            target_extensions=config.target_extensions,
            recursive=config.recursive,
        )
        if not config.allow_domain_override:
            for entry in self.entries:
                validate_domain_for_suffix(
                    config.input_domain, entry.input_path.suffix, role="input"
                )
                validate_domain_for_suffix(
                    config.target_domain, entry.target_path.suffix, role="target"
                )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError(
            "PairedImageDataset.__getitem__ is an interface placeholder; planned: "
            "domain-aware decoding, synchronized crops, normalization, and metadata"
        )

    def __repr__(self) -> str:
        return (
            f"PairedImageDataset(root={self.config.root!r}, pairs={len(self)}, "
            f"input_domain={self.config.input_domain.value!r}, "
            f"target_domain={self.config.target_domain.value!r})"
        )

    @property
    def relative_keys(self) -> tuple[str, ...]:
        return tuple(entry.relative_key for entry in self.entries)
