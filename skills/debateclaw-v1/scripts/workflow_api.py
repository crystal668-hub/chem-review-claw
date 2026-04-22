#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class WorkflowSpec:
    kind: str
    class_name: str
    module: str | None = None
    path: str | None = None


@runtime_checkable
class WorkflowPackage(Protocol):
    workflow_id: str
    version: str
    roles: list[str]

    def initialize_run(self, run_config: dict[str, Any]) -> dict[str, Any]: ...
    def compute_next_action(self, state: dict[str, Any], role: str) -> dict[str, Any]: ...
    def submit_artifact(
        self,
        state: dict[str, Any],
        role: str,
        artifact_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]: ...
    def advance(self, state: dict[str, Any]) -> dict[str, Any]: ...
    def build_status(self, state: dict[str, Any], role: str) -> dict[str, Any]: ...
    def build_summary(self, state: dict[str, Any]) -> dict[str, Any]: ...
    def finalize(self, state: dict[str, Any]) -> dict[str, Any]: ...
