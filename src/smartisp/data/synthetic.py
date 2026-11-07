"""Synthetic paired dataset generation for demos and smoke training.

The produced pairs are genuine optimization targets: ``input`` images are
actually degraded copies of ``target`` images by a deterministic pipeline
(noise, blur, color and gamma shifts, JPEG-like compression). They are *not*
real SIDD sensor data and must be labeled as synthetic everywhere they are
used for evaluation claims.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_DEFAULT_SCENES = (
    "srgb_scene_000",
    "srgb_scene_001",
    "srgb_scene_002",
    "srgb_scene_003",
    "srgb_scene_004",
    "srgb_scene_005",
    "srgb_scene_006",
    "srgb_scene_007",
)


def _content_image(rng: np.random.Generator, size: int, index: int) -> np.ndarray:
    """Build a deterministic 8-bit RGB scene with gradients, blobs, and texture."""
    _ = index
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / max(size - 1, 1)
    base = np.stack(
        [
            0.35 + 0.35 * np.sin(np.pi * xx),
            0.35 + 0.35 * np.sin(np.pi * yy),
            0.4 + 0.2 * np.sin(np.pi * (xx + yy)),
        ],
        axis=-1,
    )
    blobs = np.zeros_like(base)
    for _ in range(6):
        cx, cy = rng.random() * 0.8 + 0.1, rng.random() * 0.8 + 0.1
        radius = 0.05 + rng.random() * 0.18
        color = rng.random(3).astype(np.float32)
        distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = distance < radius
        region = np.zeros((size, size, 3), dtype=np.float32)
        region[mask] = color
        blobs = blobs + region
    texture = np.sin(xx * 40 + rng.random() * 6) * 0.03
    scene = np.clip(base + blobs * 0.3 + texture[..., None], 0.0, 1.0)
    return (scene * 255.0).astype(np.uint8)


def _degrade(
    rng: np.random.Generator,
    image: np.ndarray,
    *,
    noise_sigma: float,
    blur_sigma: float,
    gamma_shift: float,
    white_balance_shift: tuple[float, float],
) -> np.ndarray:
    work: np.ndarray = image.astype(np.float32) / 255.0
    work = cv2.GaussianBlur(work, (0, 0), blur_sigma)
    work = np.power(work, float(gamma_shift))
    work[..., 0] *= white_balance_shift[0]
    work[..., 2] *= white_balance_shift[1]
    work = np.clip(work, 0.0, 1.0)
    work = cv2.GaussianBlur(work, (0, 0), blur_sigma)
    noise = rng.normal(0.0, noise_sigma, work.shape).astype(np.float32)
    work = work + noise
    work = np.clip(work, 0.0, 1.0) ** (1.0 / max(gamma_shift, 1e-3))
    return np.clip(work * 255.0, 0.0, 255.0).astype(np.uint8)


def generate_synthetic_pairs(
    root: str | Path,
    *,
    train: int = 12,
    val: int = 4,
    size: int = 128,
    seed: int = 0,
    noise_sigma: float = 0.05,
    blur_sigma: float = 1.2,
    gamma_shift: float = 0.85,
    white_balance: tuple[float, float] = (1.05, 0.95),
) -> dict[str, list[str]]:
    """Write a paired sRGB dataset under ``root`` and return the file listing.

    Splits are named ``train`` and ``val`` with ``input``/``target`` folders,
    matching :class:`smartisp.data.dataset.PairedDatasetConfig`. Filenames are
    deterministic so re-runs overwrite the same files.
    """
    if train < 1 or val < 1 or size < 64 or not 64 <= size <= 1024:
        raise ValueError("train/val >= 1 and 64 <= size <= 1024 are required")
    root_path = Path(root)
    for split in ("train", "val"):
        (root_path / split / "input").mkdir(parents=True, exist_ok=True)
        (root_path / split / "target").mkdir(parents=True, exist_ok=True)
    written: dict[str, list[str]] = {}
    offset = 0
    for split, count in (("train", train), ("val", val)):
        stems: list[str] = []
        for index in range(count):
            rng = np.random.default_rng(seed + offset)
            target = _content_image(rng, size, index + offset)
            weight = 1.0 if split == "train" else 0.4
            input_image = _degrade(
                rng,
                target,
                noise_sigma=noise_sigma * weight,
                blur_sigma=blur_sigma * (1.0 if split == "train" else 0.6),
                gamma_shift=gamma_shift,
                white_balance_shift=white_balance,
            )
            stem = f"{_DEFAULT_SCENES[index % len(_DEFAULT_SCENES)][:-3]}{index:03d}.png"
            cv2.imwrite(str(root_path / split / "input" / stem), input_image[..., ::-1])
            cv2.imwrite(str(root_path / split / "target" / stem), target[..., ::-1])
            stems.append(f"{split}/{stem[:-4]}")
            offset += 1
        written[split] = stems
    return written


if __name__ == "__main__":  # pragma: no cover - small utility
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Generate a synthetic paired dataset")
    parser.add_argument("root")
    parser.add_argument("--train", type=int, default=12)
    parser.add_argument("--val", type=int, default=4)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    listing = generate_synthetic_pairs(
        arguments.root,
        train=arguments.train,
        val=arguments.val,
        size=arguments.size,
        seed=arguments.seed,
    )
    print(json.dumps(listing, indent=2))
