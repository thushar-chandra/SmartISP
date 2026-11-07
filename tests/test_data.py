from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from smartisp.data import (
    DatasetProvenance,
    Domain,
    DomainMismatchError,
    PairedDatasetConfig,
    PairedImageDataset,
    discover_pairs,
)
from smartisp.types import CFAPattern


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_srgb_png_pairs_are_discovered(tmp_path: Path) -> None:
    _touch(tmp_path / "input" / "a.png")
    _touch(tmp_path / "target" / "a.png")
    _touch(tmp_path / "input" / "nested" / "b.png")
    _touch(tmp_path / "target" / "nested" / "b.png")
    config = PairedDatasetConfig(
        root=tmp_path,
        input_domain=Domain.SRGB,
        target_domain=Domain.SRGB,
        input_extensions=(".png",),
    )
    dataset = PairedImageDataset(config)
    assert len(dataset) == 2
    assert dataset.relative_keys == ("a", "nested/b")


def test_srgb_png_cannot_be_declared_as_bayer_raw(tmp_path: Path) -> None:
    _touch(tmp_path / "input" / "a.png")
    _touch(tmp_path / "target" / "a.png")
    with pytest.raises(DomainMismatchError):
        PairedDatasetConfig(
            root=tmp_path,
            input_domain=Domain.BAYER_RAW,
            target_domain=Domain.SRGB,
            input_extensions=(".png",),
        )


def test_raw_to_srgb_pairs_support_different_extensions(tmp_path: Path) -> None:
    _touch(tmp_path / "input" / "scene.npy")
    _touch(tmp_path / "target" / "scene.png")
    config = PairedDatasetConfig(
        root=tmp_path,
        input_domain=Domain.BAYER_RAW,
        target_domain=Domain.SRGB,
        input_extensions=(".npy",),
        target_extensions=(".png",),
    )
    dataset = PairedImageDataset(config)
    assert len(dataset) == 1
    assert dataset.relative_keys == ("scene",)


def test_unmatched_pairs_fail(tmp_path: Path) -> None:
    _touch(tmp_path / "input" / "a.png")
    _touch(tmp_path / "target" / "b.png")
    with pytest.raises(ValueError):
        discover_pairs(tmp_path, input_extensions=(".png",), target_extensions=(".png",))


def test_dataset_provenance_requires_cfa_for_raw() -> None:
    with pytest.raises(ValueError):
        DatasetProvenance(name="raw", domain=Domain.BAYER_RAW)
    provenance = DatasetProvenance(name="raw", domain=Domain.BAYER_RAW, cfa=CFAPattern.RGGB)
    provenance.validate_file("frame.npy")
    with pytest.raises(DomainMismatchError):
        provenance.validate_file("frame.png")


def test_config_rejects_escaping_directories(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        PairedDatasetConfig(root=tmp_path, input_dir="../outside")


def _write_png(path: Path, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.random.default_rng(seed).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    Image.fromarray(array).save(path)


def test_dataset_getitem_decodes_srgb_pair(tmp_path: Path) -> None:
    _write_png(tmp_path / "input" / "a.png", seed=1)
    _write_png(tmp_path / "target" / "a.png", seed=2)
    dataset = PairedImageDataset(PairedDatasetConfig(root=tmp_path, crop_size=8))
    sample = dataset[0]
    assert sample["input"].shape == (8, 8, 3)
    assert sample["target"].shape == (8, 8, 3)
    assert sample["input"].dtype == np.float32
    assert 0.0 <= float(sample["input"].min()) <= float(sample["input"].max()) <= 1.0
    assert sample["key"] == "a"


def test_dataset_getitem_requires_cfa_for_raw(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "target").mkdir()
    np.save(tmp_path / "input" / "a.npy", np.zeros((8, 8), dtype=np.uint16))
    _write_png(tmp_path / "target" / "a.png")
    config = PairedDatasetConfig(
        root=tmp_path,
        input_domain=Domain.BAYER_RAW,
        target_domain=Domain.SRGB,
        input_extensions=(".npy",),
        target_extensions=(".png",),
    )
    dataset = PairedImageDataset(config)
    with pytest.raises(ValueError, match="cfa"):
        dataset[0]
    raw_config = PairedDatasetConfig(
        root=tmp_path,
        input_domain=Domain.BAYER_RAW,
        target_domain=Domain.SRGB,
        input_extensions=(".npy",),
        target_extensions=(".png",),
        cfa=CFAPattern.RGGB,
    )
    sample = PairedImageDataset(raw_config)[0]
    assert sample["input"].shape == (8, 8)
