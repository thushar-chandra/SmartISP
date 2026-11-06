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
    raise NotImplementedError(
        "run_ablation is an interface placeholder; planned: execute each variant "
        "with fixed seeds, collect metrics, and emit machine-readable reports"
    )


def classify_failure(image: Any, reference: Any, metrics: dict[str, float]) -> FailureCase:
    raise NotImplementedError(
        "classify_failure is an interface placeholder; planned: combine metric "
        "thresholds with visual inspection notes instead of automatic claims"
    )
