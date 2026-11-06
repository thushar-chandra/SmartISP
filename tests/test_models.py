from __future__ import annotations

import pytest

from smartisp.models.config import LossConfig, LossKind, UNetConfig


def test_unet_config_validation() -> None:
    config = UNetConfig()
    assert config.in_channels == 3
    with pytest.raises(ValueError):
        UNetConfig(depth=0)
    with pytest.raises(ValueError):
        UNetConfig(norm="unknown")
    with pytest.raises(ValueError):
        UNetConfig(upsample="unknown")


def test_loss_config_validation() -> None:
    config = LossConfig(kind=LossKind.COMPOSITE)
    assert config.kind is LossKind.COMPOSITE
    with pytest.raises(ValueError):
        LossConfig(weight=0, charbonnier_weight=0, ssim_weight=0, perceptual_weight=0)
    with pytest.raises(ValueError):
        LossConfig(epsilon=0)
    with pytest.raises(ValueError):
        LossConfig(perceptual_net="unknown")


def test_unet_interface_placeholder() -> None:
    pytest.importorskip("torch")
    from smartisp.models.unet import UNetRestorer, build_unet

    model = build_unet(UNetConfig(base_channels=8, max_channels=64))
    assert isinstance(model, UNetRestorer)
    with pytest.raises(NotImplementedError):
        model.restore(None)


def test_loss_interface_placeholder() -> None:
    pytest.importorskip("torch")
    from smartisp.models.losses import RestorationLoss, build_loss

    loss = build_loss(LossConfig())
    assert isinstance(loss, RestorationLoss)
    with pytest.raises(NotImplementedError):
        loss(None, None)
