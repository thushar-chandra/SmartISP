"""U-Net restoration model interface.

The encoder/decoder implementation is intentionally deferred. This module fixes
the public constructor, configuration, and forward contract so that datasets,
training loops, evaluation code, and the Streamlit app can be developed against
a stable API.
"""

from __future__ import annotations

from typing import Any

try:
    from torch import nn
except ImportError as exc:
    raise ImportError(
        "smartisp.models.unet requires PyTorch; install the 'torch' extra "
        "(pip install -e '.[torch]') or use the Colab runtime"
    ) from exc

from smartisp.models.config import UNetConfig


class UNetRestorer(nn.Module):
    """Residual U-Net for smartphone image restoration.

    Planned architecture: a configurable encoder/decoder with skip connections,
    optional residual output (``out = input + residual``), normalization, and
    attention at the bottleneck. ``forward`` will accept and return tensors of
    shape ``(N, C, H, W)`` in the configured input/output domains.
    """

    def __init__(self, config: UNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or UNetConfig()
        self.input_channels = self.config.in_channels
        self.output_channels = self.config.out_channels

    def forward(self, x: Any) -> Any:
        raise NotImplementedError(
            "UNetRestorer.forward is an interface placeholder; planned: build the "
            "encoder/decoder blocks, skip connections, and residual prediction"
        )

    def restore(self, x: Any) -> Any:
        """Convenience alias used by the hybrid pipeline."""
        return self.forward(x)


def build_unet(config: UNetConfig | None = None) -> UNetRestorer:
    """Construct a restoration model; layer construction is added in stage 3."""
    return UNetRestorer(config)
