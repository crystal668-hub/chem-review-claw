#!/usr/bin/env python3
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any

from workflow_api import WorkflowPackage, WorkflowSpec


class WorkflowLoadError(RuntimeError):
    pass


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise WorkflowLoadError(f"Could not load workflow module from path: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_workflow_spec(payload: dict[str, Any]) -> WorkflowSpec:
    kind = str(payload.get("kind") or "").strip()
    class_name = str(payload.get("class") or payload.get("class_name") or "").strip()
    module = str(payload.get("module") or "").strip() or None
    path = str(payload.get("path") or "").strip() or None
    if not kind:
        raise WorkflowLoadError("Workflow package spec is missing `kind`.")
    if not class_name:
        raise WorkflowLoadError("Workflow package spec is missing `class` / `class_name`.")
    if kind == "python-module" and not module:
        raise WorkflowLoadError("python-module workflow spec requires `module`.")
    if kind == "python-path" and not path:
        raise WorkflowLoadError("python-path workflow spec requires `path`.")
    if kind not in {"python-module", "python-path"}:
        raise WorkflowLoadError(f"Unsupported workflow package kind: {kind}")
    return WorkflowSpec(kind=kind, class_name=class_name, module=module, path=path)


def load_workflow_package(payload: dict[str, Any]) -> WorkflowPackage:
    spec = parse_workflow_spec(payload)
    if spec.kind == "python-module":
        module = importlib.import_module(str(spec.module))
    else:
        module_path = Path(str(spec.path)).expanduser().resolve()
        if not module_path.is_file():
            raise WorkflowLoadError(f"Workflow package path does not exist: {module_path}")
        module = _load_module_from_path(f"workflow_package_{module_path.stem}", module_path)

    workflow_cls = getattr(module, spec.class_name, None)
    if workflow_cls is None:
        raise WorkflowLoadError(
            f"Workflow class `{spec.class_name}` not found in {spec.module or spec.path}."
        )
    workflow = workflow_cls()
    required_attrs = ("workflow_id", "version", "roles")
    for attr in required_attrs:
        if not hasattr(workflow, attr):
            raise WorkflowLoadError(f"Loaded workflow is missing required attribute `{attr}`.")
    required_methods = (
        "initialize_run",
        "compute_next_action",
        "submit_artifact",
        "advance",
        "build_status",
        "build_summary",
        "finalize",
    )
    for method in required_methods:
        if not callable(getattr(workflow, method, None)):
            raise WorkflowLoadError(f"Loaded workflow is missing required method `{method}`.")
    return workflow
