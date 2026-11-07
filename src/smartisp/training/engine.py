"""Training loop, validation, and checkpoint management."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from smartisp.data.dataset import PairedDatasetConfig, PairedImageDataset
from smartisp.data.loaders import DataLoaderConfig, build_dataloader
from smartisp.models.config import LossConfig, UNetConfig
from smartisp.models.inference import resolve_device, save_checkpoint
from smartisp.training.config import TrainConfig


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _dataset_for(config: TrainConfig, split: str) -> PairedImageDataset:
    root = Path(config.data_root) / split
    if not (root / "input").is_dir() or not (root / "target").is_dir():
        raise FileNotFoundError(
            f"paired split not found: {root}. Expected {root}/input and {root}/target"
        )
    return PairedImageDataset(
        PairedDatasetConfig(
            root=root,
            input_domain=config.input_domain,
            target_domain=config.target_domain,
            crop_size=config.patch_size if split == config.train_split else None,
            seed=config.seed,
            augment=split == config.train_split,
        )
    )


def _batch_to_device(batch: Mapping[str, Any], device: Any) -> tuple[Any, Any]:
    import torch

    def convert(array: Any) -> Any:
        if array.ndim == 4:
            tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(0, 3, 1, 2)
        elif array.ndim == 3:
            tensor = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(1)
        else:
            tensor = torch.from_numpy(np.ascontiguousarray(array))
        return tensor.float().to(device)

    return convert(batch["input"]), convert(batch["target"])


def _tensor_to_image(tensor: Any) -> np.ndarray:
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().float().cpu()
    array = np.asarray(tensor)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in {1, 3}:
        array = np.moveaxis(array, 0, -1)
    return array


class Trainer:
    """Own the training loop, checkpointing, and validation cadence."""

    def __init__(
        self,
        config: TrainConfig,
        *,
        model_config: UNetConfig | None = None,
        loss_config: LossConfig | None = None,
        limit_batches: int | None = None,
    ) -> None:
        self.config = config
        self.model_config = model_config or UNetConfig()
        self.loss_config = loss_config or LossConfig()
        self.limit_batches = limit_batches

    def _device(self) -> str:
        return resolve_device(self.config.device)

    def fit(self) -> dict[str, Any]:
        from smartisp.models.losses import build_loss
        from smartisp.models.unet import build_unet

        _seed_everything(self.config.seed)
        config = self.config
        train_dataset = _dataset_for(config, config.train_split)
        try:
            val_dataset: PairedImageDataset | None = _dataset_for(config, config.val_split)
        except FileNotFoundError:
            val_dataset = None

        loader_config = DataLoaderConfig(
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.device != "cpu",
        )
        train_loader = build_dataloader(train_dataset, loader_config)
        val_loader = (
            build_dataloader(
                val_dataset,
                DataLoaderConfig(
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=config.num_workers,
                    pin_memory=config.device != "cpu",
                ),
            )
            if val_dataset is not None
            else None
        )

        device = self._device()
        model = build_unet(self.model_config).to(device)
        loss_fn = build_loss(self.loss_config)
        import torch

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        use_amp = config.amp and str(device).startswith("cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        checkpoint_dir = Path(config.checkpoint_dir)
        output_dir = Path(config.output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        if config.resume is not None:
            self._resume(model, optimizer, config.resume)

        best_psnr = -float("inf")
        history: list[dict[str, Any]] = []
        for epoch in range(1, config.epochs + 1):
            model.train()
            running = 0.0
            batches = 0
            for batch_index, batch in enumerate(train_loader):
                if self.limit_batches is not None and batch_index >= self.limit_batches:
                    break
                inputs, targets = _batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    prediction = model(inputs)
                    loss = loss_fn(prediction, targets)
                scaler.scale(loss).backward()
                if config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                running += float(loss.detach())
                batches += 1

            entry: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": running / max(batches, 1),
                "batches": batches,
            }
            if val_loader is not None and epoch % config.val_interval == 0:
                metrics = self.validate(model, val_loader, device)
                entry.update(metrics)
                psnr_value = metrics.get("psnr", float("-inf"))
                if psnr_value > best_psnr:
                    best_psnr = psnr_value
                    save_checkpoint(
                        checkpoint_dir / f"{config.experiment}_best.pt",
                        model,
                        self.model_config,
                        extra={"epoch": epoch, **entry},
                    )
            history.append(entry)
            save_checkpoint(
                checkpoint_dir / f"{config.experiment}_last.pt",
                model,
                self.model_config,
                extra={"epoch": epoch, "history": history},
            )
            (output_dir / f"{config.experiment}_history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            print(json.dumps(entry, sort_keys=True))

        return {"history": history, "checkpoints": str(checkpoint_dir)}

    def _resume(self, model: Any, optimizer: Any, checkpoint: Path) -> None:
        import torch

        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model_state", payload)
        model.load_state_dict(state)
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])

    def validate(
        self,
        model: Any,
        loader: Any,
        device: str,
    ) -> dict[str, float]:
        from smartisp.evaluation.metrics import psnr as compute_psnr
        from smartisp.evaluation.metrics import ssim as compute_ssim

        model.eval()
        psnr_values: list[float] = []
        ssim_values: list[float] = []
        with __import__("torch").no_grad():
            for batch in loader:
                inputs, targets = _batch_to_device(batch, device)
                prediction = model(inputs)
                for index in range(prediction.shape[0]):
                    pred_image = _tensor_to_image(prediction[index])
                    target_image = _tensor_to_image(targets[index])
                    psnr_values.append(compute_psnr(pred_image, target_image))
                    ssim_values.append(compute_ssim(pred_image, target_image))
        return {
            "psnr": float(np.mean(psnr_values)) if psnr_values else float("nan"),
            "ssim": float(np.mean(ssim_values)) if ssim_values else float("nan"),
            "images": len(psnr_values),
        }


def validate_checkpoint(
    checkpoint: str | Path,
    *,
    data_root: str | Path,
    split: str = "val",
    batch_size: int = 4,
    num_workers: int = 0,
    device: str = "auto",
    metrics: tuple[str, ...] = ("psnr", "ssim", "delta_e_2000"),
) -> dict[str, Any]:
    """Evaluate a saved checkpoint on a paired split and return measured metrics."""
    import torch

    from smartisp.evaluation.metrics import MetricName, evaluate_pair
    from smartisp.models.unet import UNetRestorer

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    stored = payload.get("model_config") if isinstance(payload, dict) else None
    model_config = UNetConfig(**stored) if stored else UNetConfig()
    model = UNetRestorer(model_config)
    model.load_state_dict(payload.get("model_state", payload))
    resolved_device = resolve_device(device)
    model.to(resolved_device).eval()

    root = Path(data_root) / split
    dataset = PairedImageDataset(PairedDatasetConfig(root=root))
    loader = build_dataloader(
        dataset,
        DataLoaderConfig(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=resolved_device != "cpu",
        ),
    )
    metric_names = [MetricName(name) for name in metrics]
    metric_values: dict[str, list[float]] = {name.value: [] for name in metric_names}
    with torch.no_grad():
        for batch in loader:
            inputs, targets = _batch_to_device(batch, resolved_device)
            prediction = model(inputs)
            for index in range(prediction.shape[0]):
                pred_image = _tensor_to_image(prediction[index])
                target_image = _tensor_to_image(targets[index])
                values = evaluate_pair(pred_image, target_image, metric_names)
                for name, value in values.items():
                    metric_values[name].append(value)
    summary = {
        "checkpoint": str(path),
        "split": split,
        "images": len(dataset),
        "metrics": {
            name: float(np.mean(values)) if values else float("nan")
            for name, values in metric_values.items()
        },
        "trained_weights": True,
    }
    return summary
