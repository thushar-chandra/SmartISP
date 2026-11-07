from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _app_module() -> Any:
    pytest.importorskip("streamlit")
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    path = Path(__file__).resolve().parents[1] / "apps" / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("smartisp_streamlit_app", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rgb_image(size: int = 48) -> Any:
    from smartisp import ColorSpace, RGBImage

    rng = np.random.default_rng(3)
    return RGBImage(
        data=rng.random((size, size, 3), dtype=np.float32),
        color_space=ColorSpace.SRGB,
        bit_depth=8,
        metadata={"source": "test"},
    )


def test_streamlit_app_imports() -> None:
    app = _app_module()
    assert callable(app.main)
    assert callable(app._run_pipeline)


def test_classical_only_mode_runs() -> None:
    app = _app_module()
    image = _rgb_image()
    result = app._run_pipeline(image, app.HybridMode.CLASSICAL_ONLY)
    assert result["trained"] is None
    out = np.asarray(result["image"].data)
    assert out.shape == np.asarray(image.data).shape


def test_learned_mode_with_explicit_untrained_flag() -> None:
    pytest.importorskip("torch")
    app = _app_module()
    result = app._run_pipeline(_rgb_image(), app.HybridMode.LEARNED_ONLY, allow_untrained=True)
    assert result["trained"] is False


def test_post_isp_with_trained_synthetic_checkpoint() -> None:
    pytest.importorskip("torch")
    app = _app_module()
    checkpoints = [Path(path) for path in app._known_checkpoints()]
    checkpoint = next((path for path in checkpoints if "synthetic" in path.stem), None)
    if checkpoint is None:
        pytest.skip("no synthetic checkpoint on disk")
    result = app._run_pipeline(_rgb_image(), app.HybridMode.POST_ISP, checkpoint)
    assert result["trained"] is True
    assert app._checkpoint_trained_on(checkpoint) == "synthetic_pairs"


def test_raw_input_rejects_rgb_only_restorer() -> None:
    pytest.importorskip("torch")
    from smartisp import CFAPattern, RawFrame

    app = _app_module()
    raw = RawFrame(
        data=np.zeros((64, 64), dtype=np.uint16),
        cfa=CFAPattern.BGGR,
        bit_depth=16,
        black_level=64.0,
        metadata={},
    )
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        app._run_pipeline(raw, app.HybridMode.LEARNED_ONLY, allow_untrained=True)


def test_png_download_bytes() -> None:
    app = _app_module()
    processed = app._run_pipeline(_rgb_image(), app.HybridMode.CLASSICAL_ONLY)
    assert app._png_bytes(processed["image"]).startswith(b"\x89PNG")


def test_reference_metrics() -> None:
    pytest.importorskip("cv2")
    app = _app_module()
    reference = _rgb_image()
    values = app._metrics_for(reference, reference, [app.MetricName.PSNR, app.MetricName.SSIM])
    assert values["psnr"] > 30
    assert abs(values["ssim"] - 1.0) < 0.05
