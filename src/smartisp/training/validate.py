"""Validation CLI."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from smartisp.config import load_config
from smartisp.training.engine import validate_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smartisp-validate",
        description="Validate a SmartISP checkpoint on a paired split.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Base YAML configuration.",
    )
    parser.add_argument("--override", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint", type=Path, required=False)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["psnr", "ssim", "delta_e_2000"],
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    raw = load_config(args.config, *args.override)
    paths = raw.get("paths", {})
    data_root = args.data_root or Path(paths.get("data_root", "data"))
    payload = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "split": args.split,
        "data_root": str(data_root),
        "metrics": args.metrics,
        "config_paths": [str(args.config), *(str(path) for path in args.override)],
        "config_keys": sorted(raw),
    }
    if args.checkpoint is not None and not args.checkpoint.exists():
        print(f"error: checkpoint does not exist: {args.checkpoint}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        print("dry-run: evaluation plan resolved; no validation performed")
        return 0
    if args.checkpoint is None:
        print("error: --checkpoint is required for validation", file=sys.stderr)
        return 2
    try:
        result = validate_checkpoint(
            args.checkpoint,
            data_root=data_root,
            split=args.split,
            batch_size=args.batch_size,
            device=args.device,
            metrics=tuple(args.metrics),
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
