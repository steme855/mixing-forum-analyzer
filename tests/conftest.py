"""Pytest configuration to ensure local packages are importable."""

from __future__ import annotations

import sys
from pathlib import Path
import importlib


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    src_path = root / "src"
    if src_path.exists():
        path_str = str(src_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_src_on_path()

# Ensure optional dependency stubs are registered for the tests
try:
    importlib.import_module("sitecustomize")
except ModuleNotFoundError:
    pass
