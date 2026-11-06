"""Metrics, efficiency measurements, and analysis contracts."""

from smartisp.evaluation.analysis import (
    AblationVariant,
    FailureCase,
    FailureCategory,
    classify_failure,
    default_ablation_plan,
    run_ablation,
    summarize_failure_cases,
)
from smartisp.evaluation.efficiency import (
    EfficiencyConfig,
    EfficiencyReport,
    measure_inference,
    percentile,
    summarize_latencies,
)
from smartisp.evaluation.metrics import (
    METRIC_HIGHER_IS_BETTER,
    MetricConfig,
    MetricName,
    MetricValue,
    delta_e_2000,
    evaluate_pair,
    lpips_distance,
    psnr,
    ssim,
)

__all__ = [
    "METRIC_HIGHER_IS_BETTER",
    "AblationVariant",
    "EfficiencyConfig",
    "EfficiencyReport",
    "FailureCase",
    "FailureCategory",
    "MetricConfig",
    "MetricName",
    "MetricValue",
    "classify_failure",
    "default_ablation_plan",
    "delta_e_2000",
    "evaluate_pair",
    "lpips_distance",
    "measure_inference",
    "percentile",
    "psnr",
    "run_ablation",
    "ssim",
    "summarize_failure_cases",
    "summarize_latencies",
]
