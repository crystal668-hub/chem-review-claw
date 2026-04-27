from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from typing import Any

import pytest


@pytest.fixture
def skill_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def add_scripts_to_path(skill_root: Path) -> None:
    scripts_dir = str(skill_root / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


@pytest.fixture
def write_request(tmp_path: Path) -> Any:
    def _write(payload: dict[str, Any], name: str = "request.json") -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def load_module() -> Any:
    def _load(name: str) -> Any:
        if name in sys.modules:
            return importlib.reload(sys.modules[name])
        return importlib.import_module(name)

    return _load
