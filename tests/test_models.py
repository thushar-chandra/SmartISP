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


def test_unet_forward_handles_arbitrary_sizes() -> None:
    torch = pytest.importorskip("torch")
    from smartisp.models.unet import UNetRestorer, build_unet

    config = UNetConfig(base_channels=4, max_channels=16, depth=2, groups=2)
    model = build_unet(config)
    assert isinstance(model, UNetRestorer)
    inputs = torch.rand(2, 3, 30, 34)
    output = model.restore(inputs)
    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()


def test_unet_residual_is_additive_at_initialization() -> None:
    torch = pytest.importorskip("torch")
    from smartisp.models.unet import UNetRestorer

    model = UNetRestorer(UNetConfig(base_channels=4, max_channels=8, depth=1, residual=True))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    inputs = torch.full((1, 3, 8, 8), 0.25)
    output = model(inputs)
    assert torch.allclose(output, inputs)


def test_loss_forward_and_backward() -> None:
    torch = pytest.importorskip("torch")
    from smartisp.models.losses import build_loss

    loss = build_loss(LossConfig(kind=LossKind.COMPOSITE))
    prediction = torch.rand(1, 3, 16, 16, requires_grad=True)
    target = torch.rand(1, 3, 16, 16)
    value = loss(prediction, target)
    assert torch.isfinite(value)
    value.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_rgb_restorer_adapter() -> None:
    pytest.importorskip("torch")
    import numpy as np

    from smartisp.models.inference import build_restorer
    from smartisp.types import ColorSpace, RGBImage

    restorer = build_restorer(None, device="cpu")
    assert restorer.is_trained is False
    image = RGBImage(
        data=np.random.default_rng(0).random((16, 16, 3), dtype=np.float32),
        color_space=ColorSpace.SRGB,
    )
    output = restorer.restore(image)
    assert isinstance(output, RGBImage)
    assert output.shape == image.shape
    assert output.data.min() >= 0.0 and output.data.max() <= 1.0
