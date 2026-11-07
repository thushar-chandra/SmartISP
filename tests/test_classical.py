from __future__ import annotations

import numpy as np
import pytest

from smartisp.classical import (
    BayerPreprocessConfig,
    BayerPreprocessor,
    ColorCorrectionConfig,
    ColorCorrectionMethod,
    ColorCorrector,
    DemosaicConfig,
    Demosaicer,
    DemosaicMethod,
    DenoiseConfig,
    Denoiser,
    ISPContext,
    SharpenConfig,
    Sharpener,
    ToneMapper,
    ToneMappingConfig,
    WhiteBalanceConfig,
    WhiteBalancer,
    build_default_pipeline,
)
from smartisp.types import CFAPattern, ColorSpace, RawFrame, RGBImage


def _gradient_rgb(height: int = 32, width: int = 32) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    return np.stack(
        [xx / max(width - 1, 1), yy / max(height - 1, 1), np.full_like(xx, 0.5, dtype=float)],
        axis=-1,
    ).astype(np.float32)


def _raw_frame(cfa: CFAPattern = CFAPattern.RGGB) -> RawFrame:
    rgb = _gradient_rgb()
    mosaic = np.zeros(rgb.shape[:2], dtype=np.float32)
    for row in range(2):
        for col in range(2):
            channel = cfa.color_at(row, col).value
            mosaic[row::2, col::2] = rgb[row::2, col::2, "RGB".index(channel)]
    return RawFrame(data=mosaic, cfa=cfa, bit_depth=16, black_level=0.0, white_level=1.0)


def _rgb_image() -> RGBImage:
    return RGBImage(data=_gradient_rgb(), color_space=ColorSpace.LINEAR_SRGB)


def test_bayer_preprocess_normalizes_integer_codes() -> None:
    rgb = _gradient_rgb(16, 16)
    mosaic = np.zeros((16, 16), dtype=np.uint16)
    for row in range(2):
        for col in range(2):
            channel = CFAPattern.RGGB.color_at(row, col).value
            mosaic[row::2, col::2] = (rgb[row::2, col::2, "RGB".index(channel)] * 65535).astype(
                np.uint16
            )
    frame = RawFrame(data=mosaic, cfa=CFAPattern.RGGB, bit_depth=16)
    result = BayerPreprocessor().process(frame)
    assert result.data.dtype == np.float32
    assert result.data.min() >= 0.0 and result.data.max() <= 1.0
    assert result.white_level == 1.0


def test_white_balance_estimates_and_applies_gains() -> None:
    frame = _raw_frame()
    balancer = WhiteBalancer(WhiteBalanceConfig())
    gains = balancer.estimate(frame)
    assert gains.red > 0 and gains.green > 0 and gains.blue > 0
    balanced = balancer.apply(frame, gains)
    assert balanced.shape == frame.shape


def test_demosaic_variants_preserve_channels() -> None:
    frame = _raw_frame()
    for method in (DemosaicMethod.BILINEAR, DemosaicMethod.MALVAR, DemosaicMethod.EDGE_AWARE):
        result = Demosaicer(DemosaicConfig(method=method)).process(frame)
        assert isinstance(result, RGBImage)
        assert result.data.shape[-1] == 3
        assert not np.allclose(result.data[..., 0], result.data[..., 2])


def test_rgb_stages_process_images() -> None:
    image = _rgb_image()
    balanced = WhiteBalancer().process(image)
    assert isinstance(balanced, RGBImage)
    corrected = ColorCorrector(ColorCorrectionConfig()).process(balanced)
    denoised = Denoiser(DenoiseConfig(strength=0.5)).process(corrected)
    mapped = ToneMapper(ToneMappingConfig()).process(denoised)
    sharpened = Sharpener(SharpenConfig(amount=0.5)).process(mapped)
    assert mapped.color_space is ColorSpace.SRGB
    assert sharpened.shape == image.shape
    assert 0.0 <= float(sharpened.data.min()) <= float(sharpened.data.max()) <= 1.0


def test_default_pipeline_runs_on_raw_frame() -> None:
    result = build_default_pipeline().run(_raw_frame())
    assert isinstance(result, RGBImage)
    assert result.color_space is ColorSpace.SRGB
    assert result.data.shape[-1] == 3
    assert np.isfinite(result.data).all()


def test_demosaic_rejects_display_space_output() -> None:
    with pytest.raises(ValueError):
        DemosaicConfig(output_color_space=ColorSpace.SRGB)


def test_color_correction_requires_matrix_for_matrix_method() -> None:
    with pytest.raises(ValueError):
        ColorCorrectionConfig(method=ColorCorrectionMethod.MATRIX)
    with pytest.raises(ValueError):
        ColorCorrectionConfig(matrix=((1.0, 0.0), (0.0, 1.0)))


def test_context_validates_color_matrix() -> None:
    with pytest.raises(ValueError):
        ISPContext(color_matrix=((1.0, 0.0, 0.0),))
    valid = ISPContext(
        camera_white_balance=(2.0, 1.0, 1.5),
        color_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    assert valid.camera_white_balance == (2.0, 1.0, 1.5)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        BayerPreprocessConfig(output_dtype="uint8")
    with pytest.raises(ValueError):
        WhiteBalanceConfig(min_gain=2.0, max_gain=1.0)
    with pytest.raises(ValueError):
        DenoiseConfig(patch_size=8, search_window=4)
    with pytest.raises(ValueError):
        ToneMappingConfig(gamma=0.0)
    with pytest.raises(ValueError):
        SharpenConfig(radius=0.0)
