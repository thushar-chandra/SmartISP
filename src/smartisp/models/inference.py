"""Inference adapters and checkpoint helpers for learned restoration."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from smartisp.models.config import UNetConfig
from smartisp.types import ColorSpace, RawFrame, RGBImage


def resolve_device(device: str = "auto") -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this process")
    return device


def _as_float_image(data: Any) -> np.ndarray:
    array = np.asarray(data)
    if np.issubdtype(array.dtype, np.integer):
        scale = float(np.iinfo(array.dtype).max)
        array = array.astype(np.float32) / max(scale, 1.0)
    else:
        array = array.astype(np.float32)
    maximum = float(np.nanmax(array)) if array.size else 1.0
    if maximum > 1.5 and maximum <= 255.5:
        array = array / 255.0
    return np.ascontiguousarray(np.clip(array, 0.0, 1.0))


class RGBRestorer:
    """Adapt a PyTorch U-Net to the ``RestorationModel`` buffer contract."""

    def __init__(
        self,
        model: Any,
        *,
        device: str = "auto",
        clamp: bool = True,
        checkpoint: str | None = None,
        is_trained: bool = False,
    ) -> None:
        import torch

        self.model = model.to(resolve_device(device))
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.clamp = clamp
        self.checkpoint = checkpoint
        self.is_trained = is_trained
        self._torch = torch

    def restore(self, image: Any) -> Any:
        if isinstance(image, RawFrame):
            raise TypeError(
                "RGBRestorer accepts display or linear RGB, not a Bayer RawFrame; "
                "use a RAW-domain checkpoint and adapter for pre-ISP restoration"
            )
        if not isinstance(image, RGBImage):
            raise TypeError("RGBRestorer requires an RGBImage input")

        data = _as_float_image(image.data)
        tensor = self._torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        t0 = time.perf_counter()
        with self._torch.no_grad():
            output = self.model(tensor)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        restored = output.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        if self.clamp:
            restored = np.clip(restored, 0.0, 1.0)
        metadata = dict(image.metadata)
        metadata["restoration"] = {
            "checkpoint": self.checkpoint,
            "trained_weights": self.is_trained,
            "device": str(self.device),
            "latency_ms": latency_ms,
        }
        return RGBImage(
            data=restored.astype(np.float32),
            color_space=image.color_space
            if image.color_space is not ColorSpace.UNKNOWN
            else ColorSpace.SRGB,
            bit_depth=image.bit_depth,
            metadata=metadata,
        )


def save_checkpoint(
    path: str | Path,
    model: Any,
    model_config: UNetConfig,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "model_config": asdict(model_config),
        "format_version": 1,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, destination)
    return destination


def load_restorer(
    checkpoint: str | Path | None,
    *,
    device: str = "auto",
    model_config: UNetConfig | None = None,
) -> RGBRestorer:
    """Load a checkpoint, or build an explicitly untrained model when omitted."""
    import torch

    from smartisp.models.unet import UNetRestorer

    if checkpoint is None:
        config = model_config or UNetConfig()
        return RGBRestorer(
            UNetRestorer(config),
            device=device,
            checkpoint=None,
            is_trained=False,
        )

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    stored_config = payload.get("model_config") if isinstance(payload, dict) else None
    config = UNetConfig(**stored_config) if stored_config else (model_config or UNetConfig())
    model = UNetRestorer(config)
    state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state)
    return RGBRestorer(
        model,
        device=device,
        checkpoint=str(path),
        is_trained=True,
    )


def build_restorer(
    checkpoint: str | Path | None = None,
    *,
    device: str = "auto",
    model_config: UNetConfig | None = None,
) -> RGBRestorer:
    """Public alias used by the app and evaluation scripts."""
    return load_restorer(checkpoint, device=device, model_config=model_config)
