#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from bundle_common import default_runtime_dir, dump_json, load_json, resolve_skill_root
from chemqa_review_artifacts import CANDIDATE_OWNER, REVIEWER_ROLES, liveness_summary, missing_proposer_submissions, missing_required_reviewer_lanes, qualifying_candidate_reviews
from control_store import FileControlStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Return a compact cached ChemQA review state snapshot.")
    parser.add_argument("--skill-root", default=str(Path(__file__).resolve().parents[1]), help="chemqa-review skill root")
    parser.add_argument("--team", required=True, help="Debate team / run id")
    parser.add_argument("--agent", required=True, help="Agent name, e.g. proposer-1")
    parser.add_argument("--runtime-dir", help="Path to deployed DebateClaw runtime helpers")
    parser.add_argument("--cooldown-seconds", type=int, default=180, help="Minimum seconds between real state fetches")
    parser.add_argument("--force", action="store_true", help="Bypass cooldown and fetch fresh state")
    return parser.parse_args()


def run_json(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"Command failed ({result.returncode}): {' '.join(command)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Command did not return JSON: {' '.join(command)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        ) from exc


def compact_phase_progress(payload: dict[str, Any] | None) -> dict[str, Any]:
    progress = dict(payload or {})
    return {
        "kind": progress.get("kind"),
        "actual": progress.get("actual"),
        "expected": progress.get("expected"),
        "complete": progress.get("complete"),
    }


def build_summary(*, status_payload: dict[str, Any], next_action_payload: dict[str, Any]) -> dict[str, Any]:
    phase = status_payload.get("phase") or next_action_payload.get("phase")
    qualifying_reviews = qualifying_candidate_reviews(status_payload)
    liveness = liveness_summary(status_payload)
    return {
        "team": status_payload.get("team_name") or next_action_payload.get("team"),
        "agent": next_action_payload.get("agent"),
        "status": status_payload.get("status"),
        "phase": phase,
        "action": next_action_payload.get("action"),
        "advance_ready": bool(next_action_payload.get("advance_ready")),
        "phase_progress": compact_phase_progress(status_payload.get("phase_progress") or next_action_payload.get("phase_progress")),
        "review_round": status_payload.get("review_round"),
        "rebuttal_round": status_payload.get("rebuttal_round"),
        "final_candidates_count": len(status_payload.get("final_candidates") or []),
        "proposals_count": len(status_payload.get("proposals") or []),
        "reviews_count": len(status_payload.get("reviews") or []),
        "rebuttals_count": len(status_payload.get("rebuttals") or []),
        "candidate_owner": CANDIDATE_OWNER,
        "required_reviewer_lanes": list(REVIEWER_ROLES),
        "missing_proposer_submissions": missing_proposer_submissions(status_payload) if phase == "propose" else [],
        "qualifying_candidate_reviews_count": len(qualifying_reviews),
        "missing_required_reviewer_lanes": missing_required_reviewer_lanes(status_payload) if phase in {"review", "done"} else [],
        "healthy": bool(liveness.get("healthy")),
        "phase_signature": liveness.get("phase_signature"),
    }


def state_key(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": summary.get("status"),
        "phase": summary.get("phase"),
        "action": summary.get("action"),
        "advance_ready": summary.get("advance_ready"),
        "phase_progress": summary.get("phase_progress"),
        "review_round": summary.get("review_round"),
        "rebuttal_round": summary.get("rebuttal_round"),
        "final_candidates_count": summary.get("final_candidates_count"),
        "proposals_count": summary.get("proposals_count"),
        "reviews_count": summary.get("reviews_count"),
        "rebuttals_count": summary.get("rebuttals_count"),
        "missing_proposer_submissions": summary.get("missing_proposer_submissions"),
        "qualifying_candidate_reviews_count": summary.get("qualifying_candidate_reviews_count"),
        "missing_required_reviewer_lanes": summary.get("missing_required_reviewer_lanes"),
        "phase_signature": summary.get("phase_signature"),
    }


def cache_path_for(skill_root: Path, team: str, agent: str) -> Path:
    safe_team = "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in team)
    safe_agent = "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in agent)
    return skill_root / "generated" / "state-snapshots" / f"{safe_team}--{safe_agent}.json"


def load_terminal_status(store: FileControlStore, run_id: str) -> dict[str, Any]:
    path = store.control / "run-status" / f"{run_id}.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    status = str(payload.get("status") or "")
    terminal_state = str(payload.get("terminal_state") or "")
    if status == "terminal_failure":
        return payload
    if status == "done" and terminal_state in {"failed", "cancelled"}:
        return payload
    return {}


def main() -> int:
    args = parse_args()
    skill_root = resolve_skill_root(args.skill_root)
    store = FileControlStore(skill_root)
    runtime_root = Path(args.runtime_dir).expanduser().resolve() if args.runtime_dir else default_runtime_dir()
    debate_state = runtime_root / "debate_state.py"
    if not debate_state.is_file():
        raise SystemExit(f"Missing runtime helper: {debate_state}")

    cache_path = cache_path_for(skill_root, args.team, args.agent)
    terminal_status = load_terminal_status(store, args.team)
    now = time.time()
    cache_payload: dict[str, Any] | None = None
    if cache_path.is_file():
        try:
            cache_payload = load_json(cache_path)
        except Exception:
            cache_payload = None

    if cache_payload and not args.force:
        checked_at = float(cache_payload.get("checked_at_epoch", 0))
        age_seconds = max(0.0, now - checked_at)
        if age_seconds < args.cooldown_seconds:
            payload = dict(cache_payload.get("snapshot") or {})
            payload.update(
                {
                    "cached_recent": True,
                    "age_seconds": round(age_seconds, 1),
                    "cooldown_seconds": args.cooldown_seconds,
                    "cooldown_remaining_seconds": max(0, int(args.cooldown_seconds - age_seconds)),
                    "guidance": "Recent cached snapshot is still valid. Keep working unless you changed state or received new coordination.",
                }
            )
            if terminal_status:
                payload["terminal_failure"] = terminal_status
                payload["guidance"] = "Run is in explicit terminal failure state. Do not keep polling or burning tokens."
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

    status_payload = run_json(
        [
            str(debate_state),
            "status",
            "--team",
            args.team,
            "--agent",
            args.agent,
            "--json",
        ]
    )
    next_action_payload = run_json(
        [
            str(debate_state),
            "next-action",
            "--team",
            args.team,
            "--agent",
            args.agent,
            "--json",
        ]
    )
    analysis_payload = status_payload
    phase = str(status_payload.get("phase") or next_action_payload.get("phase") or "")
    if phase in {"review", "done"} and (status_payload.get("reviews") or status_payload.get("status") == "done"):
        analysis_payload = run_json(
            [
                str(debate_state),
                "summary",
                "--team",
                args.team,
                "--json",
                "--include-bodies",
            ]
        )
    summary = build_summary(status_payload=analysis_payload, next_action_payload=next_action_payload)

    previous_snapshot = dict(cache_payload.get("snapshot") or {}) if cache_payload else {}
    unchanged = state_key(previous_snapshot) == state_key(summary) if previous_snapshot else False
    summary.update(
        {
            "cached_recent": False,
            "age_seconds": 0,
            "cooldown_seconds": args.cooldown_seconds,
            "cooldown_remaining_seconds": args.cooldown_seconds,
            "unchanged_since_last_real_check": unchanged,
            "guidance": (
                "State unchanged since the last real check. Continue working instead of polling again unless you cause or expect a state change."
                if unchanged
                else "Fresh compact snapshot fetched. Continue with the next concrete action."
            ),
        }
    )
    if terminal_status:
        summary["terminal_failure"] = terminal_status
        summary["guidance"] = "Run is in explicit terminal failure state. Do not keep polling or burning tokens."

    dump_json(
        cache_path,
        {
            "team": args.team,
            "agent": args.agent,
            "checked_at_epoch": now,
            "snapshot": summary,
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
