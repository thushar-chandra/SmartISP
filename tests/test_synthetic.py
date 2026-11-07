"""Tests for the synthetic paired dataset generator."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from smartisp.data.dataset import discover_pairs
from smartisp.data.synthetic import generate_synthetic_pairs


def test_generated_pairs_are_discoverable_and_degraded(tmp_path: Path) -> None:
    listing = generate_synthetic_pairs(tmp_path, train=2, val=1, size=64, seed=7)
    assert len(listing["train"]) == 2
    assert len(listing["val"]) == 1
    pairs = discover_pairs(tmp_path / "val")
    assert len(pairs) == 1
    for pair in discover_pairs(tmp_path / "train"):
        target = cv2.imread(str(pair.target_path))
        degraded = cv2.imread(str(pair.input_path))
        assert target is not None and degraded is not None
        assert target.shape == degraded.shape
        assert not np.array_equal(target, degraded)


def test_generator_validates_arguments(tmp_path: pytest.TempdirFactory) -> None:
    with pytest.raises(ValueError):
        generate_synthetic_pairs(tmp_path, train=0, val=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        generate_synthetic_pairs(tmp_path, train=1, val=1, size=16)  # type: ignore[arg-type]
