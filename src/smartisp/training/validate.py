"""Validation CLI scaffold."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from smartisp.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smartisp-validate",
        description="Validate a SmartISP checkpoint (interface scaffold).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Base YAML configuration.",
    )
    parser.add_argument("--override", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint", type=Path, required=False)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["psnr", "ssim", "lpips", "delta_e_2000"],
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    raw = load_config(args.config, *args.override)
    payload = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "split": args.split,
        "metrics": args.metrics,
        "config_paths": [str(args.config), *(str(path) for path in args.override)],
        "config_keys": sorted(raw),
    }
    if args.checkpoint is not None and not args.checkpoint.exists():
        print(f"error: checkpoint does not exist: {args.checkpoint}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        print("dry-run: validation implementation is intentionally not available yet")
        return 0
    print(
        "SmartISP validation is not implemented in this foundation release. "
        "Use --dry-run to inspect the resolved evaluation plan.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
