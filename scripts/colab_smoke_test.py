"""Colab T4 smoke test for the SmartISP scaffold.

Run from a Colab CLI session with:
    colab exec -s smartisp-t4 -f scripts/colab_smoke_test.py

The script verifies CUDA, the installed package interfaces, and a small GPU
computation. It does not train a model and does not produce evaluation results.
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-url",
        default=os.environ.get(
            "SMARTISP_REPO_URL", "https://github.com/thushar-chandra/SmartISP.git"
        ),
    )
    parser.add_argument("--repo-dir", default="/content/SmartISP")
    parser.add_argument("--skip-clone", action="store_true")
    return parser.parse_args()


def ensure_import_path(args: argparse.Namespace) -> None:
    try:
        importlib.import_module("smartisp")

        print("smartisp import: OK")
        return
    except ImportError:
        pass
    if args.skip_clone:
        raise SystemExit("smartisp is not importable and --skip-clone was requested")
    repo_dir = Path(args.repo_dir)
    if not repo_dir.exists():
        print(f"cloning {args.repo_url} -> {repo_dir}")
        subprocess.run(
            ["git", "clone", "--depth", "1", args.repo_url, str(repo_dir)],
            check=True,
        )
    sys.path.insert(0, str(repo_dir / "src"))
    importlib.import_module("smartisp")

    print("smartisp import: OK (from clone)")


def main() -> int:
    args = parse_args()
    import torch

    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("FAILED: remote CUDA is unavailable")
    gpu = torch.cuda.get_device_properties(0)
    print("GPU:", gpu.name)
    print("GPU memory GB:", round(gpu.total_memory / (1024**3), 2))

    ensure_import_path(args)
    from smartisp import __version__
    from smartisp.models.config import UNetConfig
    from smartisp.models.unet import UNetRestorer

    print("SmartISP version:", __version__)
    model = UNetRestorer(UNetConfig(depth=2, base_channels=8)).to("cuda")
    output = model(torch.zeros(1, 3, 64, 68, device="cuda"))
    if tuple(output.shape) != (1, 3, 64, 68):
        raise SystemExit("FAILED: U-Net produced an unexpected output shape")

    matrix = torch.randn(2000, 2000, device="cuda")
    result = matrix @ matrix
    torch.cuda.synchronize()
    print("GPU computation: OK")
    print("Output shape:", tuple(result.shape))
    print("SMOKE TEST PASSED: infrastructure and interfaces are ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
