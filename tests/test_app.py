from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def test_streamlit_app_imports() -> None:
    pytest.importorskip("streamlit")
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    path = Path(__file__).resolve().parents[1] / "apps" / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("smartisp_streamlit_app", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
    assert callable(module._run_pipeline)
