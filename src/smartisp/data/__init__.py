"""Dataset discovery, provenance, and preprocessing contracts."""

from smartisp.data.dataset import (
    PairedDatasetConfig,
    PairedEntry,
    PairedImageDataset,
    discover_pairs,
)
from smartisp.data.domains import (
    BAYER_RAW_EXTENSIONS,
    CAMERA_RAW_EXTENSIONS,
    LINEAR_RGB_EXTENSIONS,
    SRGB_EXTENSIONS,
    DatasetProvenance,
    Domain,
    DomainMismatchError,
    validate_domain_for_suffix,
)
from smartisp.data.loaders import (
    DataLoaderConfig,
    build_dataloader,
    collate_paired,
)
from smartisp.data.preprocessing import (
    NormalizationConfig,
    PatchConfig,
    normalize_raw,
    random_crop,
    synchronized_augment,
)

__all__ = [
    "BAYER_RAW_EXTENSIONS",
    "CAMERA_RAW_EXTENSIONS",
    "LINEAR_RGB_EXTENSIONS",
    "SRGB_EXTENSIONS",
    "DataLoaderConfig",
    "DatasetProvenance",
    "Domain",
    "DomainMismatchError",
    "NormalizationConfig",
    "PairedDatasetConfig",
    "PairedEntry",
    "PairedImageDataset",
    "PatchConfig",
    "build_dataloader",
    "collate_paired",
    "discover_pairs",
    "normalize_raw",
    "random_crop",
    "synchronized_augment",
    "validate_domain_for_suffix",
]
