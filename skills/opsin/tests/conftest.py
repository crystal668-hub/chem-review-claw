from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = SKILL_ROOT / "scripts"
RESULT_FILENAME = "result.json"


def load_script_module(script_name: str):
    module_path = SCRIPTS_ROOT / f"{script_name}.py"
    assert module_path.exists(), f"missing script: {module_path}"
    module_name = f"opsin_{script_name}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec and spec.loader, f"unable to load module spec: {module_path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def write_request(tmp_path: Path, payload: dict) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return request_path
