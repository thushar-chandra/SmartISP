from __future__ import annotations

from pathlib import Path

import pytest

from smartisp.config import deep_merge, expand_env, load_config, load_yaml


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"b": 1}, "c": 2}
    override = {"a": {"d": 3}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 1, "d": 3}, "c": 2}
    assert base == {"a": {"b": 1}, "c": 2}


def test_load_config_merges_and_expands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMARTISP_TEST_ROOT", "/tmp/smartisp")
    first = tmp_path / "base.yaml"
    first.write_text(
        "paths:\n  data_root: ${SMARTISP_TEST_ROOT}\n  fallback: ${MISSING:-value}\n",
        encoding="utf-8",
    )
    second = tmp_path / "override.yaml"
    second.write_text("paths:\n  output_dir: outputs\n", encoding="utf-8")
    config = load_config(first, second)
    assert config == {
        "paths": {
            "data_root": "/tmp/smartisp",
            "fallback": "value",
            "output_dir": "outputs",
        }
    }


def test_empty_environment_variable_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SMARTISP_EMPTY", "")
    assert expand_env("${SMARTISP_EMPTY:-fallback}") == "fallback"
    assert expand_env("${SMARTISP_EMPTY}") == ""


def test_missing_environment_variable_without_default(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("value: ${SMARTISP_MISSING_VARIABLE}\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_config(path)


def test_expand_env_recurses() -> None:
    assert expand_env(["a", {"b": "c"}]) == ["a", {"b": "c"}]


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(TypeError):
        load_yaml(path)
