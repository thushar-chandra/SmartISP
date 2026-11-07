"""Ablation and failure-case analysis contracts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from smartisp.pipeline.hybrid import HybridMode


@dataclass(frozen=True)
class AblationVariant:
    """One controlled comparison in the evaluation matrix."""

    name: str
    description: str
    mode: HybridMode
    classical_stages: tuple[str, ...] = ()
    checkpoint: str | None = None
    hypothesis: str = ""


def default_ablation_plan() -> tuple[AblationVariant, ...]:
    """Return the planned ablation matrix, not measured results."""
    return (
        AblationVariant(
            name="classical_baseline",
            description="Full classical ISP without learned restoration.",
            mode=HybridMode.CLASSICAL_ONLY,
            classical_stages=(
                "bayer_preprocess",
                "white_balance",
                "demosaic",
                "color_correction",
                "denoise",
                "tone_mapping",
                "sharpen",
            ),
            hypothesis="Classical baseline establishes a non-learned reference.",
        ),
        AblationVariant(
            name="learned_only",
            description="U-Net restoration without classical ISP stages.",
            mode=HybridMode.LEARNED_ONLY,
            checkpoint="<learned-only checkpoint>",
            hypothesis="Measures learned capacity without hand-crafted priors.",
        ),
        AblationVariant(
            name="hybrid_pre_isp",
            description="Learned restoration before classical ISP.",
            mode=HybridMode.PRE_ISP,
            checkpoint="<pre-ISP checkpoint>",
            hypothesis="Tests whether RAW-domain restoration helps demosaicing and denoising.",
        ),
        AblationVariant(
            name="hybrid_post_isp",
            description="Classical ISP followed by learned restoration.",
            mode=HybridMode.POST_ISP,
            checkpoint="<post-ISP checkpoint>",
            hypothesis="Tests display-domain artifact removal and texture recovery.",
        ),
        AblationVariant(
            name="no_denoise",
            description="Hybrid pipeline with classical denoising disabled.",
            mode=HybridMode.POST_ISP,
            classical_stages=(
                "bayer_preprocess",
                "white_balance",
                "demosaic",
                "color_correction",
                "tone_mapping",
                "sharpen",
            ),
            hypothesis="Isolates the contribution of classical denoising.",
        ),
        AblationVariant(
            name="no_tone_mapping",
            description="Hybrid pipeline with tone mapping disabled.",
            mode=HybridMode.POST_ISP,
            classical_stages=(
                "bayer_preprocess",
                "white_balance",
                "demosaic",
                "color_correction",
                "denoise",
                "sharpen",
            ),
            hypothesis="Isolates tone-curve effects on perceptual metrics.",
        ),
        AblationVariant(
            name="no_sharpen",
            description="Hybrid pipeline with sharpening disabled.",
            mode=HybridMode.POST_ISP,
            classical_stages=(
                "bayer_preprocess",
                "white_balance",
                "demosaic",
                "color_correction",
                "denoise",
                "tone_mapping",
            ),
            hypothesis="Checks whether sharpening amplifies noise or halos.",
        ),
    )


class FailureCategory(StrEnum):
    """Planned failure taxonomy."""

    HIGHLIGHT_CLIPPING = "highlight_clipping"
    LOW_LIGHT_NOISE = "low_light_noise"
    COLOR_CAST = "color_cast"
    DEMOSAIC_ARTIFACT = "demosaic_artifact"
    MOIRE = "moire"
    MOTION_BLUR = "motion_blur"
    DOMAIN_MISMATCH = "domain_mismatch"
    LEARNED_ARTIFACT = "learned_artifact"
    OTHER = "other"


@dataclass(frozen=True)
class FailureCase:
    """A reproducible failure observation with artifacts and metrics."""

    case_id: str
    category: FailureCategory
    description: str
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()


def summarize_failure_cases(cases: list[FailureCase]) -> dict[str, int]:
    """Count observed failure cases by category."""
    counts = Counter(case.category.value for case in cases)
    return dict(sorted(counts.items()))


def run_ablation(variant: AblationVariant, dataset: Any, config: Any) -> Any:
    """Evaluate one ablation variant over a paired dataset.

    ``config`` must be a mapping with a ``"process"`` callable that accepts one
    dataset sample and returns an image array. Metrics default to PSNR, SSIM,
    and CIEDE2000. The returned payload contains measured values only.
    """
    from collections.abc import Mapping

    from smartisp.evaluation.metrics import MetricName, evaluate_pair

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping with a 'process' callable")
    process = config.get("process")
    if not callable(process):
        raise ValueError("config['process'] must be callable")
    metric_names = [
        MetricName(name) for name in config.get("metrics", ["psnr", "ssim", "delta_e_2000"])
    ]
    limit = config.get("limit")
    count = len(dataset) if limit is None else min(len(dataset), int(limit))
    totals: dict[str, float] = {}
    for index in range(count):
        sample = dataset[index]
        prediction = process(sample)
        target = sample["target"] if isinstance(sample, Mapping) else sample
        values = evaluate_pair(prediction, target, metric_names, config.get("metric_config"))
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value
    metrics = {name: value / count for name, value in totals.items()} if count else {}
    return {
        "variant": variant.name,
        "mode": variant.mode.value,
        "images": count,
        "metrics": metrics,
    }


def classify_failure(image: Any, reference: Any, metrics: dict[str, float]) -> FailureCase:
    """Assign a conservative failure category from measured metrics.

    The classifier only reports categories that the supplied metrics support.
    It never claims a visual diagnosis without the caller attaching artifacts or
    notes to the returned case.
    """
    psnr_value = metrics.get("psnr")
    ssim_value = metrics.get("ssim")
    delta_e = metrics.get("delta_e_2000")
    if psnr_value is not None and psnr_value < 20.0:
        category = FailureCategory.LEARNED_ARTIFACT
        description = "low PSNR indicates a large reconstruction error"
    elif ssim_value is not None and ssim_value < 0.6:
        category = FailureCategory.LEARNED_ARTIFACT
        description = "low SSIM indicates structural distortion"
    elif delta_e is not None and delta_e > 12.0:
        category = FailureCategory.COLOR_CAST
        description = "high CIEDE2000 indicates a color shift"
    elif (
        psnr_value is not None
        and psnr_value > 40.0
        and ssim_value is not None
        and ssim_value > 0.98
    ):
        category = FailureCategory.OTHER
        description = "metrics indicate a close match; inspect residual artifacts"
    else:
        category = FailureCategory.OTHER
        description = "no metric threshold was crossed"
    case_id = metrics.get("case_id", "case")
    return FailureCase(
        case_id=str(case_id),
        category=category,
        description=description,
        metrics=dict(metrics),
    )
