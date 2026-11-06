from __future__ import annotations

import json
from pathlib import Path

import pytest

from smartisp.training import train, validate


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        "paths:\n"
        "  data_root: data\n"
        "  checkpoint_dir: checkpoints\n"
        "  output_dir: outputs\n"
        "training:\n"
        "  experiment: test-run\n"
        "  epochs: 2\n"
        "  batch_size: 1\n"
        "  train_split: train\n"
        "  val_split: val\n",
        encoding="utf-8",
    )
    return path


def test_train_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _config(tmp_path)
    assert train.main(["--config", str(config), "--dry-run"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output.split("\ndry-run:")[0])
    assert payload["experiment"] == "test-run"
    assert payload["epochs"] == 2
    assert payload["resume"] is None


def test_train_dry_run_records_resume(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _config(tmp_path)
    checkpoint = tmp_path / "resume.pth"
    assert train.main(["--config", str(config), "--resume", str(checkpoint), "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out.split("\ndry-run:")[0])
    assert payload["resume"] == str(checkpoint)


def test_train_without_implementation_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(tmp_path)
    assert train.main(["--config", str(config)]) == 2
    error = capsys.readouterr().err
    assert "not implemented" in error


def test_train_missing_resume_returns_error(tmp_path: Path) -> None:
    config = _config(tmp_path)
    missing = tmp_path / "missing.pth"
    assert train.main(["--config", str(config), "--resume", str(missing)]) == 2


def test_validate_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _config(tmp_path)
    assert validate.main(["--config", str(config), "--dry-run"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output.split("\ndry-run:")[0])
    assert payload["split"] == "val"
    assert "psnr" in payload["metrics"]


def test_validate_missing_checkpoint_returns_error(tmp_path: Path) -> None:
    config = _config(tmp_path)
    missing = tmp_path / "missing.pth"
    assert validate.main(["--config", str(config), "--checkpoint", str(missing)]) == 2
