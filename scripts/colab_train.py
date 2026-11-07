"""SmartISP training on the Colab T4 runtime.

Run with the Colab CLI (Linux hosts only; it is executed in the remote kernel):

    colab new -s smartisp-t4 --gpu T4
    colab install -s smartisp-t4 -r requirements/colab.txt
    colab exec -s smartisp-t4 \
        --env SMARTISP_REPO_URL=https://github.com/thushar-chandra/SmartISP.git \
        --env SMARTISP_SYNTHETIC=true --env SMARTISP_EPOCHS=5 \
        -f scripts/colab_train.py

Environment variables (all optional, defaults shown):

    SMARTISP_REPO_URL    https://github.com/thushar-chandra/SmartISP.git
    SMARTISP_REPO_DIR    /content/SmartISP
    SMARTISP_CONFIG      configs/colab_t4.yaml
    SMARTISP_SYNTHETIC   true    generate a deterministic degraded/clean pair set
    SMARTISP_DATA_ROOT   data    used when SMARTISP_SYNTHETIC is enabled
    SMARTISP_EPOCHS      from config
    SMARTISP_BATCH_SIZE  from config
    SMARTISP_PATCH_SIZE  from config
    SMARTISP_LEARNING_RATE  from config
    SMARTISP_BASE_CHANNELS  from config
    SMARTISP_LIMIT_BATCHES  none

Checkpoints land in ``<repo>/checkpoints`` and the history JSON in
``<repo>/outputs`` inside the ephemeral Colab VM. Copy them out (e.g. to Drive)
before the session ends; the script prints their absolute paths.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def _ensure_import_path() -> None:
    repo_url = os.environ.get(
        "SMARTISP_REPO_URL", "https://github.com/thushar-chandra/SmartISP.git"
    )
    repo_dir = Path(os.environ.get("SMARTISP_REPO_DIR", "/content/SmartISP"))
    try:
        import smartisp

        print("smartisp import: OK")
        return
    except ImportError:
        pass
    if not repo_dir.exists():
        print(f"cloning {repo_url} -> {repo_dir}")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], check=True)
    sys.path.insert(0, str(repo_dir / "src"))
    import smartisp  # noqa: F401

    print("smartisp import: OK (from clone)")


def _env_int(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def main() -> int:
    import torch

    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    _ensure_import_path()

    from smartisp.data.synthetic import generate_synthetic_pairs
    from smartisp.models.config import LossConfig, UNetConfig
    from smartisp.training.config import TrainConfig
    from smartisp.training.engine import Trainer

    config_path = os.environ.get("SMARTISP_CONFIG", "configs/colab_t4.yaml")
    candidates = [
        Path(os.environ.get("SMARTISP_REPO_DIR", "/content/SmartISP")),
        Path(os.environ.get("SMARTISP_REPO_DIR", ".")),
        Path.cwd(),
    ]
    root = next((candidate for candidate in candidates if (candidate / config_path).is_file()), Path.cwd())
    import yaml

    raw = yaml.safe_load((root / config_path).read_text(encoding="utf-8"))
    training = dict(raw.get("training", {}))
    model = dict(raw.get("model", {}))
    paths = raw.get("paths", {})

    data_root = os.environ.get("SMARTISP_DATA_ROOT", str(training.get("data_root", "data")))
    if os.environ.get("SMARTISP_SYNTHETIC", "true").lower() in {"1", "true", "yes"}:
        listing = generate_synthetic_pairs(
            Path(data_root),
            train=int(os.environ.get("SMARTISP_SYNTHETIC_TRAIN", "12")),
            val=int(os.environ.get("SMARTISP_SYNTHETIC_VAL", "4")),
            size=int(os.environ.get("SMARTISP_SYNTHETIC_SIZE", "128")),
            seed=int(os.environ.get("SMARTISP_SEED", raw.get("project", {}).get("seed", 0))),
        )
        print("synthetic pairs generated:", json.dumps(listing))

    patches = {
        "data_root": data_root,
        "checkpoint_dir": os.environ.get(
            "SMARTISP_CHECKPOINT_DIR", str(paths.get("checkpoint_dir", "checkpoints"))
        ),
        "output_dir": os.environ.get(
            "SMARTISP_OUTPUT_DIR", str(paths.get("output_dir", "outputs"))
        ),
        "patch_size": _env_int("SMARTISP_PATCH_SIZE", None),
        "epochs": _env_int("SMARTISP_EPOCHS", None),
        "batch_size": _env_int("SMARTISP_BATCH_SIZE", None),
        "num_workers": _env_int("SMARTISP_WORKERS", None),
        "learning_rate": (
            float(os.environ["SMARTISP_LEARNING_RATE"])
            if os.environ.get("SMARTISP_LEARNING_RATE") is not None
            else None
        ),
        "seed": _env_int("SMARTISP_SEED", None),
        "resume": os.environ.get("SMARTISP_RESUME") or None,
    }
    train_config = TrainConfig.from_mapping(
        {**{**training, **paths}, **{k: v for k, v in patches.items() if v is not None}}
    )
    model["base_channels"] = _env_int("SMARTISP_BASE_CHANNELS", model.get("base_channels"))
    model["depth"] = _env_int("SMARTISP_DEPTH", model.get("depth"))
    model_config = UNetConfig(**model)
    loss_config = LossConfig(**dict(raw.get("loss", {})))

    trainer = Trainer(
        train_config,
        model_config=model_config,
        loss_config=loss_config,
        limit_batches=_env_int("SMARTISP_LIMIT_BATCHES", None),
    )
    result = trainer.fit()
    checkpoint_dir = Path(train_config.checkpoint_dir).resolve()
    output_dir = Path(train_config.output_dir).resolve()

    best = checkpoint_dir / f"{train_config.experiment}_best.pt"
    last = checkpoint_dir / f"{train_config.experiment}_last.pt"
    history_path = output_dir / f"{train_config.experiment}_history.json"
    print("TRAINING COMPLETE")
    print(json.dumps(result["history"], indent=2))
    print("artifacts:")
    print(" best:", best)
    print(" last:", last)
    print(" history:", history_path)
    print("copy these out of the ephemeral Colab VM before the session ends.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
