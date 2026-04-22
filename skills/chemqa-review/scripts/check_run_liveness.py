#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from bundle_common import resolve_python_interpreter, resolve_skill_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ChemQA run liveness and obvious blockers.")
    parser.add_argument("--skill-root", default=str(Path(__file__).resolve().parents[1]), help="chemqa-review skill root")
    parser.add_argument("--team", required=True, help="ChemQA team / run id")
    parser.add_argument("--agent", default="debate-coordinator", help="Agent to use for the compact snapshot")
    return parser.parse_args()


def run_json(command: list[str], env: dict[str, str]) -> dict[str, Any] | list[Any]:
    result = subprocess.run(command, env=env, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"Command failed ({result.returncode}): {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return json.loads(result.stdout)


def main() -> int:
    args = parse_args()
    skill_root = resolve_skill_root(args.skill_root)
    env = os.environ.copy()
    snapshot_script = skill_root / "scripts" / "chemqa_review_state_snapshot.py"

    snapshot = run_json(
        [
            resolve_python_interpreter(),
            str(snapshot_script),
            "--skill-root",
            str(skill_root),
            "--team",
            args.team,
            "--agent",
            args.agent,
            "--force",
        ],
        env,
    )
    task_list: list[dict[str, Any]] = run_json(
        ["clawteam", "--json", "task", "list", args.team],
        env,
    )

    coordinator_status = "unknown"
    for task in task_list:
        if str(task.get("owner")) == "debate-coordinator":
            coordinator_status = str(task.get("status") or "unknown")
            break

    payload = {
        "team": args.team,
        "healthy": not snapshot.get("missing_proposer_submissions") and not snapshot.get("missing_required_reviewer_lanes"),
        "phase": snapshot.get("phase"),
        "status": snapshot.get("status"),
        "progress": snapshot.get("phase_progress"),
        "missing_roles": snapshot.get("missing_proposer_submissions") or snapshot.get("missing_required_reviewer_lanes") or [],
        "coordinator_task_status": coordinator_status,
        "recommendation": (
            "advance-or-wait"
            if not (snapshot.get("missing_proposer_submissions") or snapshot.get("missing_required_reviewer_lanes"))
            else "restart-missing-agents-or-fail-run"
        ),
        "snapshot": snapshot,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
