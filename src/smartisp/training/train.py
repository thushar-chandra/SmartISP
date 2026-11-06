"""Training CLI scaffold."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from smartisp.config import load_config
from smartisp.training.config import TrainConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smartisp-train",
        description="Train the SmartISP restoration model (interface scaffold).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Base YAML configuration.",
    )
    parser.add_argument(
        "--override",
        type=Path,
        action="append",
        default=[],
        help="Additional YAML overlay; may be repeated.",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print configuration without starting training.",
    )
    return parser


def resolve_section(args: argparse.Namespace) -> dict[str, Any]:
    raw = load_config(args.config, *args.override)
    section = dict(raw.get("training", {}))
    paths_section = raw.get("paths", {})
    section.setdefault("data_root", paths_section.get("data_root", "data"))
    section.setdefault("checkpoint_dir", paths_section.get("checkpoint_dir", "checkpoints"))
    section.setdefault("output_dir", paths_section.get("output_dir", "outputs"))
    if args.data_root is not None:
        section["data_root"] = args.data_root
    if args.checkpoint_dir is not None:
        section["checkpoint_dir"] = args.checkpoint_dir
    if args.output_dir is not None:
        section["output_dir"] = args.output_dir
    if args.device is not None:
        section["device"] = args.device
    if args.epochs is not None:
        section["epochs"] = args.epochs
    if args.resume is not None:
        section["resume"] = args.resume
    return section


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = TrainConfig.from_mapping(resolve_section(args))
    if args.dry_run:
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        print("dry-run: training implementation is intentionally not available yet")
        return 0

    if config.resume is not None and not config.resume.exists():
        print(f"error: checkpoint does not exist: {config.resume}", file=sys.stderr)
        return 2
    print(
        "SmartISP training is not implemented in this foundation release. "
        "Use --dry-run to validate configuration, or follow docs/methodology.md "
        "for the planned implementation stages.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
