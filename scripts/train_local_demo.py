"""Local (CPU-capable) demo training on synthetic degraded/clean pairs.

Produces a genuinely trained checkpoint, but trained on synthetic data, not on
real SIDD sensor images. Run from the repository root:

    uv run python scripts/train_local_demo.py
    SMARTISP_EPOCHS=10 uv run python scripts/train_local_demo.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from smartisp.data.synthetic import generate_synthetic_pairs
from smartisp.models.config import LossConfig, UNetConfig
from smartisp.training.config import TrainConfig
from smartisp.training.engine import Trainer


def main() -> int:
    data_root = Path(os.environ.get("SMARTISP_DATA_ROOT", "data"))
    train_count = int(os.environ.get("SMARTISP_SYNTHETIC_TRAIN", "12"))
    val_count = int(os.environ.get("SMARTISP_SYNTHETIC_VAL", "4"))
    size = int(os.environ.get("SMARTISP_SYNTHETIC_SIZE", "128"))
    generate_synthetic_pairs(data_root, train=train_count, val=val_count, size=size, seed=0)
    epochs = int(os.environ.get("SMARTISP_EPOCHS", "40"))
    base_channels = int(os.environ.get("SMARTISP_BASE_CHANNELS", "16"))
    depth = int(os.environ.get("SMARTISP_DEPTH", "3"))

    train_config = TrainConfig(
        experiment="smartisp-synthetic",
        epochs=epochs,
        batch_size=int(os.environ.get("SMARTISP_BATCH_SIZE", "4")),
        patch_size=int(os.environ.get("SMARTISP_PATCH_SIZE", "64")),
        learning_rate=float(os.environ.get("SMARTISP_LEARNING_RATE", "0.002")),
        data_root=data_root,
        device=os.environ.get("SMARTISP_DEVICE", "auto"),
    )
    model_config = UNetConfig(base_channels=base_channels, depth=depth)
    trainer = Trainer(train_config, model_config=model_config, loss_config=LossConfig())
    result = trainer.fit()

    history = result["history"]
    last = history[-1]
    summary = {
        "checkpoint_directory": result["checkpoints"],
        "experiment": train_config.experiment,
        "trained_on": "synthetic_pairs",
        "data_root": data_root,
        "epochs": epochs,
        "final_train_loss": last.get("train_loss"),
        "final_val_psnr": last.get("psnr"),
        "final_val_ssim": last.get("ssim"),
        "note": (
            "Trained on deterministically degraded synthetic pairs. Not trained "
            "on real SIDD sensor data; do not quote these metrics as dataset "
            "benchmarks."
        ),
    }
    path = Path(train_config.output_dir) / f"{train_config.experiment}_training_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
