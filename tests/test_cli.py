from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

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


def _write_pair(root: Path, split: str, name: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for subdir in ("input", "target"):
        directory = root / split / subdir
        directory.mkdir(parents=True, exist_ok=True)
        array = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
        Image.fromarray(array).save(directory / f"{name}.png")


@pytest.fixture()
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Config whose relative paths cannot be resolved against the repository."""
    monkeypatch.chdir(tmp_path)
    return _config(tmp_path)


def test_train_dry_run(isolated_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert train.main(["--config", str(isolated_config), "--dry-run"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output.split("\ndry-run:")[0])
    assert payload["experiment"] == "test-run"
    assert payload["epochs"] == 2
    assert payload["resume"] is None


def test_train_dry_run_records_resume(
    isolated_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = isolated_config.parent / "resume.pth"
    command = ["--config", str(isolated_config), "--resume", str(checkpoint), "--dry-run"]
    assert train.main(command) == 0
    payload = json.loads(capsys.readouterr().out.split("\ndry-run:")[0])
    assert payload["resume"] == str(checkpoint)


def test_train_missing_data_returns_error(
    isolated_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert train.main(["--config", str(isolated_config)]) == 2
    assert "error:" in capsys.readouterr().err


def test_train_missing_resume_returns_error(isolated_config: Path) -> None:
    missing = isolated_config.parent / "missing.pth"
    assert train.main(["--config", str(isolated_config), "--resume", str(missing)]) == 2


def test_train_tiny_synthetic_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("torch")
    data_root = tmp_path / "data"
    _write_pair(data_root, "train", "a", seed=1)
    _write_pair(data_root, "val", "a", seed=2)
    config = tmp_path / "tiny.yaml"
    config.write_text(
        "paths:\n"
        f"  data_root: {data_root}\n"
        f"  checkpoint_dir: {tmp_path / 'checkpoints'}\n"
        f"  output_dir: {tmp_path / 'outputs'}\n"
        "training:\n"
        "  experiment: tiny\n"
        "  epochs: 1\n"
        "  batch_size: 1\n"
        "  num_workers: 0\n"
        "  patch_size: 16\n"
        "  train_split: train\n"
        "  val_split: val\n"
        "model:\n"
        "  base_channels: 4\n"
        "  max_channels: 16\n"
        "  depth: 2\n"
        "  groups: 2\n"
        "loss:\n"
        "  kind: composite\n",
        encoding="utf-8",
    )
    assert train.main(["--config", str(config), "--limit-batches", "1"]) == 0
    assert (tmp_path / "checkpoints" / "tiny_last.pt").is_file()


def test_validate_dry_run(isolated_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert validate.main(["--config", str(isolated_config), "--dry-run"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output.split("\ndry-run:")[0])
    assert payload["split"] == "val"
    assert "psnr" in payload["metrics"]


def test_validate_missing_checkpoint_returns_error(isolated_config: Path) -> None:
    missing = isolated_config.parent / "missing.pth"
    assert validate.main(["--config", str(isolated_config), "--checkpoint", str(missing)]) == 2


def test_validate_requires_checkpoint(
    isolated_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert validate.main(["--config", str(isolated_config)]) == 2
    assert "required" in capsys.readouterr().err
