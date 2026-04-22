#!/usr/bin/env python3
from __future__ import annotations

from typing import Any


CANDIDATE_OWNER = "proposer-1"
REVIEWER_LANES = ["proposer-2", "proposer-3", "proposer-4", "proposer-5"]
ALL_ROLES = ["debate-coordinator", CANDIDATE_OWNER, *REVIEWER_LANES]
ALLOWED_PHASES = ["propose", "review", "rebuttal", "done", "failed"]


def initial_state(run_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "workflow_id": "chemqa-review@1",
        "phase": "propose",
        "status": "running",
        "goal": str(run_config.get("goal") or ""),
        "candidate_owner": CANDIDATE_OWNER,
        "candidate_revision": 0,
        "review_round": 0,
        "rebuttal_round": 0,
        "max_review_rounds": int(run_config.get("max_review_rounds") or 3),
        "max_rebuttal_rounds": int(run_config.get("max_rebuttal_rounds") or 2),
        "required_reviewer_lanes": list(REVIEWER_LANES),
        "candidate_submission": {},
        "reviews_by_round": {},
        "rebuttals": [],
        "acceptance_status": None,
        "failure_reason": None,
    }
