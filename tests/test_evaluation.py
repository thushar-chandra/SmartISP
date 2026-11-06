from __future__ import annotations

import pytest

from smartisp.evaluation import (
    FailureCase,
    FailureCategory,
    MetricConfig,
    MetricName,
    MetricValue,
    default_ablation_plan,
    delta_e_2000,
    lpips_distance,
    percentile,
    psnr,
    ssim,
    summarize_failure_cases,
    summarize_latencies,
)
from smartisp.evaluation.metrics import METRIC_HIGHER_IS_BETTER


def test_percentile_and_latency_summary() -> None:
    samples = [10.0, 20.0, 30.0, 40.0]
    assert percentile(samples, 0.5) == 25.0
    assert percentile(samples, 0.9) == pytest.approx(37.0)
    assert percentile(samples, 0.95) == pytest.approx(38.5)
    summary = summarize_latencies(samples)
    assert summary["p50"] == 25.0
    assert summary["mean"] == 25.0


def test_percentile_validation() -> None:
    with pytest.raises(ValueError):
        percentile([], 0.5)
    with pytest.raises(ValueError):
        percentile([1.0], 1.5)
    with pytest.raises(ValueError):
        summarize_latencies([-1.0])


def test_metric_directions() -> None:
    assert METRIC_HIGHER_IS_BETTER[MetricName.PSNR]
    assert METRIC_HIGHER_IS_BETTER[MetricName.SSIM]
    assert not METRIC_HIGHER_IS_BETTER[MetricName.LPIPS]
    assert not METRIC_HIGHER_IS_BETTER[MetricName.DELTA_E_2000]
    assert MetricValue(MetricName.LPIPS, 0.1).higher_is_better is False


def test_metric_implementations_are_placeholders() -> None:
    with pytest.raises(NotImplementedError):
        psnr(None, None)
    with pytest.raises(NotImplementedError):
        ssim(None, None)
    with pytest.raises(NotImplementedError):
        lpips_distance(None, None)
    with pytest.raises(NotImplementedError):
        delta_e_2000(None, None)


def test_metric_config_validation() -> None:
    with pytest.raises(ValueError):
        MetricConfig(data_range=0)
    with pytest.raises(ValueError):
        MetricConfig(ssim_window_size=10)
    with pytest.raises(ValueError):
        MetricConfig(lpips_net="unknown")


def test_default_ablation_plan_is_planned_not_measured() -> None:
    variants = default_ablation_plan()
    names = {variant.name for variant in variants}
    assert {
        "classical_baseline",
        "learned_only",
        "hybrid_pre_isp",
        "hybrid_post_isp",
        "no_denoise",
        "no_tone_mapping",
        "no_sharpen",
    } == names


def test_failure_case_summary() -> None:
    cases = [
        FailureCase("a", FailureCategory.LOW_LIGHT_NOISE, "noise"),
        FailureCase("b", FailureCategory.LOW_LIGHT_NOISE, "noise"),
        FailureCase("c", FailureCategory.HIGHLIGHT_CLIPPING, "clipping"),
    ]
    assert summarize_failure_cases(cases) == {
        "highlight_clipping": 1,
        "low_light_noise": 2,
    }
