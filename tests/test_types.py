from __future__ import annotations

import pytest

from smartisp import CFAPattern, Channel, RawFrame, RGBImage, WhiteBalanceGains


class Array:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.ndim = len(shape)


def test_cfa_color_lookup() -> None:
    assert CFAPattern.RGGB.color_at(0, 0) is Channel.RED
    assert CFAPattern.RGGB.color_at(0, 1) is Channel.GREEN
    assert CFAPattern.RGGB.color_at(1, 0) is Channel.GREEN
    assert CFAPattern.RGGB.color_at(1, 1) is Channel.BLUE
    assert CFAPattern.BGGR.color_at(0, 0) is Channel.BLUE
    assert CFAPattern.GBRG.color_at(0, 0) is Channel.GREEN


def test_cfa_rejects_negative_coordinates() -> None:
    with pytest.raises(ValueError):
        CFAPattern.RGGB.color_at(-1, 0)


def test_raw_frame_requires_single_channel_mosaic() -> None:
    frame = RawFrame(data=Array((4, 5)), cfa=CFAPattern.RGGB)
    assert frame.shape == (4, 5)
    with pytest.raises(TypeError):
        RawFrame(data=Array((4, 5, 3)), cfa=CFAPattern.RGGB)


def test_raw_frame_validates_levels_and_bit_depth() -> None:
    with pytest.raises(ValueError):
        RawFrame(data=Array((2, 2)), cfa=CFAPattern.RGGB, bit_depth=0)
    with pytest.raises(ValueError):
        RawFrame(data=Array((2, 2)), cfa=CFAPattern.RGGB, black_level=-1)
    with pytest.raises(ValueError):
        RawFrame(
            data=Array((2, 2)),
            cfa=CFAPattern.RGGB,
            black_level=100,
            white_level=100,
        )


def test_rgb_image_requires_three_channels() -> None:
    image = RGBImage(data=Array((4, 5, 3)))
    assert image.shape == (4, 5, 3)
    with pytest.raises(ValueError):
        RGBImage(data=Array((4, 5, 4)))
    with pytest.raises(TypeError):
        RGBImage(data=Array((4, 5)))


def test_white_balance_gains_must_be_positive() -> None:
    gains = WhiteBalanceGains(1.0, 1.0, 1.2)
    assert gains.as_tuple() == (1.0, 1.0, 1.2)
    with pytest.raises(ValueError):
        WhiteBalanceGains(0.0, 1.0, 1.0)
