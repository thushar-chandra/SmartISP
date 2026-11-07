"""Measure classical, learned, and hybrid variants on real image files.

This script computes genuine metrics from images on disk; every number it
prints is measured. Run from the repository root:

    uv run python scripts/evaluate_demo.py --root data/val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from smartisp import ColorSpace, RGBImage
from smartisp.classical import build_display_pipeline
from smartisp.evaluation.efficiency import EfficiencyConfig, measure_inference
from smartisp.evaluation.metrics import MetricName, evaluate_pair
from smartisp.models import build_restorer

METRIC_NAMES = (MetricName.PSNR, MetricName.SSIM, MetricName.DELTA_E_2000)


def _load_pair(entry: Path) -> tuple[RGBImage, RGBImage]:
    target = cv2.imread(str(entry), cv2.IMREAD_COLOR)
    if target is None:
        raise ValueError(f"failed to decode image: {entry}")
    degraded = cv2.imread(str(entry.parents[1] / "input" / entry.name), cv2.IMREAD_COLOR)
    if degraded is None:
        raise ValueError(f"failed to decode image: {entry.parents[1] / 'input' / entry.name}")

    def to_image(array: np.ndarray) -> RGBImage:
        return RGBImage(
            data=np.ascontiguousarray(array[..., ::-1].astype(np.float32) / 255.0),
            color_space=ColorSpace.SRGB,
            bit_depth=8,
            metadata={},
        )

    return to_image(degraded), to_image(target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Traditional vs learned vs hybrid")
    parser.add_argument("--root", default="data/val")
    parser.add_argument("--checkpoint", default="checkpoints/smartisp-synthetic_best.pt")
    parser.add_argument("--output", default="outputs/evaluation_summary.json")
    arguments = parser.parse_args()

    root = Path(arguments.root)
    entries = sorted((root / "target").glob("*.png"))
    if not entries:
        raise SystemExit(f"no target images found under {root}/target")

    pipeline = build_display_pipeline()
    restorer = build_restorer(arguments.checkpoint, device="cpu")

    variants: dict[str, RGBImage | None] = {
        "input_vs_target": None,
        "classical_only": None,
        "learned_only": None,
        "hybrid_post_isp": None,
    }

    summary: dict[str, dict[str, float]] = {
        name: {metric.value: 0.0 for metric in METRIC_NAMES} for name in variants
    }
    count = 0
    for entry in entries:
        degraded, target = _load_pair(entry)
        target_array = np.asarray(target.data)
        variants["input_vs_target"] = degraded
        variants["classical_only"] = pipeline.run(degraded)
        variants["learned_only"] = restorer.restore(degraded)
        hybrid = build_restorer(arguments.checkpoint, device="cpu")
        variants["hybrid_post_isp"] = hybrid.restore(pipeline.run(degraded))
        classical_variant = variants["classical_only"]
        assert isinstance(classical_variant, RGBImage)
        for name, image in variants.items():
            assert image is not None
            values = evaluate_pair(np.asarray(image.data), target_array, METRIC_NAMES)
            for metric, value in values.items():
                summary[name][metric] += float(value)
        count += 1
    for name in summary:
        for metric in summary[name]:
            summary[name][metric] /= count

    effect_config = EfficiencyConfig(warmup_runs=2, timed_runs=5)
    probe = load_probe(entries)
    timing = {
        "classical_only": measure_inference(
            lambda image: pipeline.run(image), probe, effect_config
        ),
        "learned_only": measure_inference(
            lambda image: restorer.restore(image), probe, effect_config
        ),
    }

    payload = {
        "checkpoint": arguments.checkpoint,
        "images": count,
        "metrics_mean": summary,
        "latency": {
            name: {
                "latency_ms_p50": report.latency_ms_p50,
                "latency_ms_p95": report.latency_ms_p95,
                "throughput_fps": report.throughput_fps,
                "device": report.device,
            }
            for name, report in timing.items()
        },
        "trained_on": "synthetic_pairs",
        "note": "Measured on the provided image pairs; no synthetic claims.",
    }
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def load_probe(entries: list[Path]) -> RGBImage:
    first = entries[0]
    image = cv2.imread(str(first), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode image: {first}")
    return RGBImage(
        data=np.ascontiguousarray(image[..., ::-1].astype(np.float32) / 255.0),
        color_space=ColorSpace.SRGB,
        bit_depth=8,
        metadata={"source": str(first)},
    )


if __name__ == "__main__":
    raise SystemExit(main())
