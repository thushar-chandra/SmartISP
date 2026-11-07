"""Training CLI."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from smartisp.config import load_config
from smartisp.models.config import LossConfig, UNetConfig
from smartisp.training.config import TrainConfig
from smartisp.training.engine import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smartisp-train",
        description="Train the SmartISP restoration model.",
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
    parser.add_argument("--limit-batches", type=int, default=None)
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


def _model_config(raw: dict[str, Any]) -> UNetConfig:
    values = raw.get("model", {})
    known = {field.name for field in UNetConfig.__dataclass_fields__.values()}
    return UNetConfig(**{key: value for key, value in values.items() if key in known})


def _loss_config(raw: dict[str, Any]) -> LossConfig:
    values = raw.get("loss", {})
    known = {field.name for field in LossConfig.__dataclass_fields__.values()}
    return LossConfig(**{key: value for key, value in values.items() if key in known})


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    raw = load_config(args.config, *args.override)
    section = resolve_section(args)
    config = TrainConfig.from_mapping(section)
    if args.dry_run:
        payload = config.to_dict()
        payload["model"] = _model_config(raw).__dict__
        payload["loss"] = _loss_config(raw).__dict__
        print(json.dumps(payload, indent=2, sort_keys=True))
        print("dry-run: configuration resolved; no training performed")
        return 0

    if config.resume is not None and not config.resume.exists():
        print(f"error: checkpoint does not exist: {config.resume}", file=sys.stderr)
        return 2
    try:
        trainer = Trainer(
            config,
            model_config=_model_config(raw),
            loss_config=_loss_config(raw),
            limit_batches=args.limit_batches,
        )
        result = trainer.fit()
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "completed", **result}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
