"""SmartISP learned-restoration inference on the Colab T4 runtime.

Run with the Colab CLI (Linux hosts only; it is executed in the remote kernel):

    colab exec -s smartisp-t4 \
        --env SMARTISP_REPO_URL=https://github.com/thushar-chandra/SmartISP.git \
        --env SMARTISP_CHECKPOINT=/content/SmartISP/checkpoints/smartisp-post-isp_best.pt \
        --env SMARTISP_IMAGE=/content/input.png \
        -f scripts/colab_inference.py

Environment variables:

    SMARTISP_REPO_URL    https://github.com/thushar-chandra/SmartISP.git
    SMARTISP_REPO_DIR    /content/SmartISP
    SMARTISP_CONFIG      configs/colab_t4.yaml (source of model settings for
                         untrained fallback inference)
    SMARTISP_CHECKPOINT  required for a trained restoration; without it the
                         script runs an explicitly UNTRAINED model and says so
    SMARTISP_IMAGE       input image path or https URL
    SMARTISP_OUTPUT      defaults to <SMARTISP_OUTPUT_DIR>/restored.png
    SMARTISP_OUTPUT_DIR  defaults to <SMARTISP_REPO_DIR>/outputs
    SMARTISP_DEVICE      auto|cpu|cuda

The output PNG is written inside the ephemeral Colab VM; copy it out before the
session ends. The printed JSON records whether trained weights were used.
"""

from __future__ import annotations

import os
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


def main() -> int:
    import torch

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    image_source = os.environ.get("SMARTISP_IMAGE")
    if not image_source:
        raise SystemExit("set SMARTISP_IMAGE to an input image path or URL")

    _ensure_import_path()

    import cv2
    import numpy as np

    from smartisp import ColorSpace, RGBImage
    from smartisp.models import build_restorer

    checkpoint = os.environ.get("SMARTISP_CHECKPOINT")
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise SystemExit(f"checkpoint does not exist: {checkpoint_path}")
        restorer = build_restorer(checkpoint_path, device=os.environ.get("SMARTISP_DEVICE", "auto"))
        print("using trained checkpoint:", checkpoint_path)
    else:
        config_path = Path(os.environ.get("SMARTISP_CONFIG", "configs/colab_t4.yaml"))
        model_config = None
        try:
            import yaml

            model = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}).get("model", {})
            from smartisp.models.config import UNetConfig

            model_config = UNetConfig(**model)
        except (OSError, ValueError) as exc:
            print(f"note: could not read {config_path} ({exc}); default UNetConfig")
        restorer = build_restorer(
            None, device=os.environ.get("SMARTISP_DEVICE", "auto"), model_config=model_config
        )
        print("WARNING: no checkpoint configured; output uses UNTRAINED random weights")

    source = Path(image_source)
    if "://" in image_source:
        import shutil
        import urllib.request

        source = Path(os.environ.get("SMARTISP_REPO_DIR", "data")) / "inference_input.png"
        source.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(image_source) as response, source.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        print("downloaded image ->", source)
    if not source.is_file():
        raise SystemExit(f"image does not exist: {source}")

    decoded = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if decoded is None:
        raise SystemExit(f"failed to decode image: {source}")
    pixels = np.ascontiguousarray(decoded[..., ::-1].astype(np.float32) / 255.0)
    input_image = RGBImage(
        data=pixels, color_space=ColorSpace.SRGB, bit_depth=8, metadata={"source": str(source)}
    )
    restored = restorer.restore(input_image)
    is_trained = bool(restorer.is_trained)

    output_dir = Path(
        os.environ.get(
            "SMARTISP_OUTPUT_DIR",
            Path(os.environ.get("SMARTISP_REPO_DIR", ".")) / "outputs",
        )
    )
    output = Path(
        os.environ["SMARTISP_OUTPUT"]
        if os.environ.get("SMARTISP_OUTPUT")
        else output_dir / "restored.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(np.asarray(restored.data) * 255.0, 0.0, 255.0).astype(np.uint8)
    cv2.imwrite(str(output), data[..., ::-1])

    import json

    print(
        "INFERENCE COMPLETE",
        json.dumps(
            {
                "input": str(source),
                "output": str(output),
                "trained_weights": is_trained,
                "checkpoint": restorer.checkpoint,
                "device": str(restorer.device),
            },
            indent=2,
        ),
    )
    print("copy the output out of the ephemeral Colab VM before the session ends.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
