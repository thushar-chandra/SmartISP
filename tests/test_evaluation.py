from __future__ import annotations

import importlib.util

import numpy as np
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


def test_metric_implementations() -> None:
    rng = np.random.default_rng(0)
    target = rng.random((32, 32, 3), dtype=np.float32)
    prediction = np.clip(target + 0.05 * rng.standard_normal(target.shape), 0.0, 1.0)
    assert np.isfinite(psnr(prediction, target))
    assert -1.0 <= ssim(prediction, target) <= 1.0
    assert delta_e_2000(prediction, target) >= 0.0
    assert psnr(target, target) == float("inf")
    assert ssim(target, target) == pytest.approx(1.0, abs=1e-4)
    assert delta_e_2000(target, target) == pytest.approx(0.0, abs=1e-6)


def test_psnr_known_value() -> None:
    prediction = np.zeros((8, 8, 3), dtype=np.float32)
    target = np.full((8, 8, 3), 0.5, dtype=np.float32)
    assert psnr(prediction, target) == pytest.approx(6.0206, abs=1e-3)


def test_lpips_reports_missing_dependency_or_runs() -> None:
    target = np.random.default_rng(1).random((16, 16, 3), dtype=np.float32)
    if importlib.util.find_spec("lpips") is None:
        with pytest.raises(ImportError):
            lpips_distance(target, target)
    else:
        assert lpips_distance(target, target) == pytest.approx(0.0, abs=1e-4)


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
