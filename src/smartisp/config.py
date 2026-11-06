"""YAML configuration loading with environment expansion and deep merging."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        value = os.environ.get(name)
        if value:
            return value
        if default is not None:
            return default
        if value is not None:
            return value
        raise KeyError(f"environment variable {name!r} is not set and has no default")

    return _ENV_PATTERN.sub(replace, value)


def expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` and ``${VAR:-default}`` in configuration values."""
    if isinstance(value, str):
        return _expand_string(value)
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_env(item) for item in value)
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"{resolved} must contain a top-level mapping")
    return payload


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursive merge of two mappings without mutating either input."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def load_config(
    *paths: str | Path,
    overrides: Mapping[str, Any] | None = None,
    expand_environment: bool = True,
) -> dict[str, Any]:
    """Load and merge YAML files left-to-right, then apply optional overrides."""
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_merge(merged, load_yaml(path))
    if overrides is not None:
        merged = deep_merge(merged, overrides)
    if expand_environment:
        merged = expand_env(merged)
    return merged
