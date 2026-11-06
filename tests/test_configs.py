from __future__ import annotations

from pathlib import Path

from smartisp.config import load_config, load_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_default_and_colab_configs_merge() -> None:
    config = load_config(ROOT / "configs" / "default.yaml", ROOT / "configs" / "colab_t4.yaml")
    assert config["training"]["device"] == "cuda"
    assert config["paths"]["data_root"] == "/content/data"
    assert config["pipeline"]["mode"] == "post_isp"


def test_dataset_example_manifests_parse() -> None:
    paths = sorted((ROOT / "configs" / "datasets").glob("*.yaml.example"))
    assert paths
    for path in paths:
        payload = load_yaml(path)
        assert "dataset" in payload
