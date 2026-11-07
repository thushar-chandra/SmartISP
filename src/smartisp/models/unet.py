"""Residual U-Net restoration model."""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without the torch extra
    raise ImportError(
        "smartisp.models.unet requires PyTorch; install the 'torch' extra "
        "(pip install -e '.[torch]') or use the Colab runtime"
    ) from exc

from smartisp.models.config import UNetConfig


def _activation(name: str) -> nn.Module:
    return {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(0.01, inplace=True),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(inplace=True),
    }[name]


def _group_count(channels: int, requested: int) -> int:
    groups = min(requested, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(groups, 1)


def _normalization(channels: int, config: UNetConfig) -> nn.Module:
    if config.norm == "group":
        return nn.GroupNorm(_group_count(channels, config.groups), channels)
    if config.norm == "batch":
        return nn.BatchNorm2d(channels)
    if config.norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    return nn.Identity()


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with normalization and activation."""

    def __init__(self, in_channels: int, out_channels: int, config: UNetConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            _normalization(out_channels, config),
            _activation(config.activation),
        ]
        if config.dropout > 0:
            layers.append(nn.Dropout2d(config.dropout))
        layers.extend(
            [
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                _normalization(out_channels, config),
                _activation(config.activation),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SqueezeExcite(nn.Module):
    """Channel attention at the bottleneck."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, config: UNetConfig) -> None:
        super().__init__()
        self.block: nn.Module
        self.mode = config.upsample
        if config.upsample == "transpose":
            self.block = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetRestorer(nn.Module):
    """Residual U-Net for smartphone image restoration.

    The model accepts and returns tensors of shape ``(N, C, H, W)``. When
    ``residual`` is enabled and the channel counts match, the prediction is
    added to the input, which makes the network learn an additive correction.
    Spatial dimensions are padded internally to a multiple of ``2 ** depth`` and
    cropped back, so arbitrary image sizes are accepted.
    """

    def __init__(self, config: UNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or UNetConfig()
        self.input_channels = self.config.in_channels
        self.output_channels = self.config.out_channels

        features: list[int] = []
        in_channels = self.config.in_channels
        self.encoders = nn.ModuleList()
        for level in range(self.config.depth):
            out_channels = min(self.config.base_channels * (2**level), self.config.max_channels)
            self.encoders.append(ConvBlock(in_channels, out_channels, self.config))
            features.append(out_channels)
            in_channels = out_channels

        bottleneck_channels = min(
            self.config.base_channels * (2**self.config.depth), self.config.max_channels
        )
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(in_channels, bottleneck_channels, self.config)
        self.attention = (
            SqueezeExcite(bottleneck_channels) if self.config.attention else nn.Identity()
        )

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current_channels = bottleneck_channels
        for skip_channels in reversed(features):
            self.upsamples.append(UpsampleBlock(current_channels, skip_channels, self.config))
            self.decoders.append(ConvBlock(skip_channels * 2, skip_channels, self.config))
            current_channels = skip_channels

        self.head = nn.Conv2d(current_channels, self.config.out_channels, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.GroupNorm, nn.BatchNorm2d, nn.InstanceNorm2d)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Any) -> Any:
        if not torch.is_tensor(x):
            raise TypeError("UNetRestorer.forward expects a torch tensor (N, C, H, W)")
        if x.ndim != 4:
            raise ValueError("input must have shape (N, C, H, W)")

        height, width = x.shape[-2:]
        factor = 2**self.config.depth
        pad_h = (-height) % factor
        pad_w = (-width) % factor
        padded = x
        if pad_h or pad_w:
            padded = nn.functional.pad(padded, (0, pad_w, 0, pad_h), mode="replicate")

        skips: list[torch.Tensor] = []
        current = padded
        for encoder in self.encoders:
            current = encoder(current)
            skips.append(current)
            current = self.pool(current)

        current = self.bottleneck(current)
        current = self.attention(current)

        for upsample, decoder, skip in zip(
            self.upsamples, self.decoders, reversed(skips), strict=True
        ):
            current = upsample(current)
            current = decoder(torch.cat([skip, current], dim=1))

        output = self.head(current)
        if self.config.residual and self.input_channels == self.output_channels:
            output = output + padded
        return output[..., :height, :width]

    def restore(self, x: Any) -> Any:
        """Convenience alias used by the hybrid pipeline."""
        return self.forward(x)


def build_unet(config: UNetConfig | None = None) -> UNetRestorer:
    """Construct a restoration model."""
    return UNetRestorer(config)
