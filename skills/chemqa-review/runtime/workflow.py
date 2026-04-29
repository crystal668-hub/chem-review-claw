#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    from .state_models import ALL_ROLES, CANDIDATE_OWNER, REVIEWER_LANES, initial_state
except ImportError:
    runtime_dir = Path(__file__).resolve().parent
    runtime_dir_str = str(runtime_dir)
    if runtime_dir_str not in sys.path:
        sys.path.insert(0, runtime_dir_str)
    from state_models import ALL_ROLES, CANDIDATE_OWNER, REVIEWER_LANES, initial_state


class ChemQAWorkflow:
    """Inactive native ChemQA workflow package scaffold.

    Live ChemQA execution is controlled by DebateClaw's SQLite-backed
    ``debate_state.py`` plus ``chemqa_review_openclaw_driver.py``. This scaffold
    only preserves the future workflow-package hook shape and must not be treated
    as the live control plane.
    """

    workflow_id = "chemqa-review@1"
    version = "1"
    status = "scaffold"
    active = False
    live_control_plane = "debate_state_driver"
    roles = list(ALL_ROLES)

    def initialize_run(self, run_config: dict[str, Any]) -> dict[str, Any]:
        return initial_state(run_config)

    def compute_next_action(self, state: dict[str, Any], role: str) -> dict[str, Any]:
        phase = str(state.get("phase") or "propose")
        if role == "debate-coordinator":
            return {"agent": role, "phase": phase, "action": "wait"}
        if phase == "propose":
            return {"agent": role, "phase": phase, "action": "propose" if role == CANDIDATE_OWNER else "wait"}
        if phase == "review":
            return {
                "agent": role,
                "phase": phase,
                "action": "review" if role in REVIEWER_LANES else "wait",
                "target_owner": CANDIDATE_OWNER if role in REVIEWER_LANES else None,
            }
        if phase == "rebuttal":
            return {"agent": role, "phase": phase, "action": "rebuttal" if role == CANDIDATE_OWNER else "wait"}
        if phase in {"done", "failed"}:
            return {"agent": role, "phase": phase, "action": "stop"}
        return {"agent": role, "phase": phase, "action": "wait"}

    def submit_artifact(
        self,
        state: dict[str, Any],
        role: str,
        artifact_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        updated = dict(state)
        updated.setdefault("artifacts", []).append(
            {
                "role": role,
                "artifact_type": artifact_type,
                "payload": payload,
            }
        )
        return updated

    def advance(self, state: dict[str, Any]) -> dict[str, Any]:
        return dict(state)

    def build_status(self, state: dict[str, Any], role: str) -> dict[str, Any]:
        action_payload = self.compute_next_action(state, role)
        return {
            "workflow": self.workflow_id,
            "phase": state.get("phase"),
            "status": state.get("status"),
            "candidate_owner": state.get("candidate_owner"),
            "review_round": state.get("review_round"),
            "rebuttal_round": state.get("rebuttal_round"),
            "required_reviewer_lanes": list(state.get("required_reviewer_lanes") or []),
            "agent_view": action_payload,
        }

    def build_summary(self, state: dict[str, Any]) -> dict[str, Any]:
        return dict(state)

    def finalize(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "workflow": self.workflow_id,
            "phase": state.get("phase"),
            "status": state.get("status"),
            "candidate_owner": state.get("candidate_owner"),
            "acceptance_status": state.get("acceptance_status"),
            "failure_reason": state.get("failure_reason"),
        }
