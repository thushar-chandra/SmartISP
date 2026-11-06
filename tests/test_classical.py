from __future__ import annotations

import pytest

from smartisp.classical import (
    BayerPreprocessConfig,
    BayerPreprocessor,
    ColorCorrectionConfig,
    ColorCorrectionMethod,
    ColorCorrector,
    DemosaicConfig,
    Demosaicer,
    DenoiseConfig,
    Denoiser,
    ISPContext,
    SharpenConfig,
    Sharpener,
    ToneMapper,
    ToneMappingConfig,
    WhiteBalanceConfig,
    WhiteBalancer,
)
from smartisp.types import ColorSpace


@pytest.mark.parametrize(
    "stage",
    [
        BayerPreprocessor(),
        Demosaicer(),
        WhiteBalancer(),
        ColorCorrector(),
        Denoiser(),
        ToneMapper(),
        Sharpener(),
    ],
)
def test_stages_are_interface_placeholders(stage: object) -> None:
    with pytest.raises(NotImplementedError):
        stage.process(None)  # type: ignore[attr-defined]


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
