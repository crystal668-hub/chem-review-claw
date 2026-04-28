#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bundle_common import (
    REQUIRED_SKILLS,
    dependency_report,
    missing_skills_from_report,
    resolve_skill_root,
    role_slot_map,
    safe_session_id,
    slot_id_map,
)
from control_store import FileControlStore


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_run_id(preset_ref: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{preset_ref.split('@', 1)[0]}-{stamp}"


def apply_override(*, resolved: dict[str, Any], preset: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    rule = preset.get("overrides", {}).get(key)
    if not rule or rule.get("exposure") != "tunable":
        raise SystemExit(f"Preset `{preset['id']}@{preset['version']}` does not allow override for `{key}`.")
    if "min" in rule and value < rule["min"]:
        raise SystemExit(f"Override `{key}`={value} is below minimum {rule['min']}.")
    if "max" in rule and value > rule["max"]:
        raise SystemExit(f"Override `{key}`={value} is above maximum {rule['max']}.")
    resolved[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile a chemqa-review run plan.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="chemqa-review skill root")
    parser.add_argument("--preset", required=True, help="Preset ref, e.g. chemqa-review@1")
    parser.add_argument("--goal", required=True, help="Debate goal or task prompt")
    parser.add_argument("--run-id", help="Optional explicit run id")
    parser.add_argument("--additional-file-workspace", help="Optional extra file workspace locator")
    parser.add_argument("--answer-kind", help="Resolved answer kind for ChemQA Artifact Flow")
    parser.add_argument("--model-profile", help="Override model profile")
    parser.add_argument("--slot-set", choices=("default", "A", "B"), default="default", help="Debate slot set to bind this run to")
    parser.add_argument("--proposer-count", type=int)
    parser.add_argument("--review-rounds", type=int)
    parser.add_argument("--rebuttal-rounds", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--evidence-mode", help="Override evidence mode")
    parser.add_argument("--priority", default="normal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = resolve_skill_root(args.root)
    dependency_payload = dependency_report(root)
    missing_skills = missing_skills_from_report(dependency_payload)
    if missing_skills:
        raise SystemExit("Missing required sibling skills: " + ", ".join(missing_skills))

    store = FileControlStore(root)
    preset = store.get_preset(args.preset)
    workflow = store.get_workflow(preset["workflow_ref"])
    config_snapshot = store.get_config_snapshot("react-reviewed-default")

    resolved = dict(preset.get("defaults", {}))
    apply_override(resolved=resolved, preset=preset, key="model_profile", value=args.model_profile)
    apply_override(resolved=resolved, preset=preset, key="review_rounds", value=args.review_rounds)
    apply_override(resolved=resolved, preset=preset, key="rebuttal_rounds", value=args.rebuttal_rounds)
    apply_override(resolved=resolved, preset=preset, key="max_epochs", value=args.max_epochs)
    apply_override(resolved=resolved, preset=preset, key="evidence_mode", value=args.evidence_mode)

    if resolved["model_profile"] not in preset.get("allowed_model_profiles", []):
        raise SystemExit(
            f"Model profile `{resolved['model_profile']}` is not allowed by preset `{args.preset}`."
        )

    if args.proposer_count is not None and args.proposer_count != resolved["proposer_count"]:
        raise SystemExit("chemqa-review fixes proposer_count to the preset default and does not allow overrides.")

    model_profile = store.get_model_profile(resolved["model_profile"])
    logical_slot_models = dict(model_profile.get("slot_models") or {})
    required_slots = ("debate-coordinator", "debate-1", "debate-2", "debate-3", "debate-4", "debate-5")
    missing_slots = [slot_id for slot_id in required_slots if slot_id not in logical_slot_models]
    if missing_slots:
        raise SystemExit("Model profile is missing required slots: " + ", ".join(missing_slots))

    role_map = dict(workflow.get("role_map") or {})
    prompt_pack = dict(preset.get("prompt_pack") or {})
    role_contracts = dict(prompt_pack.get("role_contracts") or {})
    shared_modules = list(prompt_pack.get("shared_modules") or [])
    role_modules = {
        role_name: list(modules)
        for role_name, modules in dict(prompt_pack.get("role_modules") or {}).items()
    }

    slot_ids = slot_id_map(args.slot_set)
    role_slots = role_slot_map(args.slot_set)
    slot_models = {
        slot_ids[logical_slot_id]: payload
        for logical_slot_id, payload in logical_slot_models.items()
    }

    run_id = args.run_id or default_run_id(args.preset)
    session_assignments = {
        role_slots["debate-coordinator"]: safe_session_id("chemqa-review", run_id, "coordinator"),
        role_slots["proposer-1"]: safe_session_id("chemqa-review", run_id, "proposer-1"),
        role_slots["proposer-2"]: safe_session_id("chemqa-review", run_id, "proposer-2"),
        role_slots["proposer-3"]: safe_session_id("chemqa-review", run_id, "proposer-3"),
        role_slots["proposer-4"]: safe_session_id("chemqa-review", run_id, "proposer-4"),
        role_slots["proposer-5"]: safe_session_id("chemqa-review", run_id, "proposer-5"),
    }
    additional_file_workspace = (args.additional_file_workspace or "").strip() or None
    answer_kind = (args.answer_kind or "generic_semantic_answer").strip() or "generic_semantic_answer"
    prompt_assembly = {
        role_name: {
            "contracts": list(role_contracts.get(role_name, [])),
            "modules": list(role_modules.get(role_name, shared_modules)),
            "semantic_role": role_map[role_name],
        }
        for role_name in role_map
    }
    run_plan = {
        "run_id": run_id,
        "created_at": iso_now(),
        "request_snapshot": {
            "preset_ref": args.preset,
            "goal": args.goal,
            "inputs": {"additional_file_workspace": additional_file_workspace},
            "metadata": {"priority": args.priority, "answer_kind": answer_kind},
            "overrides": resolved,
        },
        "workflow_ref": f"{workflow['id']}@{workflow['version']}",
        "preset_ref": f"{preset['id']}@{preset['version']}",
        "engine_workflow_ref": str(workflow.get("engine_workflow_ref") or "chemqa-review@1"),
        "engine_preset_ref": str(preset.get("engine_preset_ref") or "chemqa-review@1"),
        "resolved_model_profile": model_profile,
        "slot_set": args.slot_set,
        "slot_assignments": slot_models,
        "session_assignments": session_assignments,
        "prompt_assembly": prompt_assembly,
        "launch_spec": {
            "backend": resolved.get("backend", "subprocess"),
            "final_decider": resolved.get("final_decider", "outer-entry-agent"),
            "proposer_slots": [
                role_slots["proposer-1"],
                role_slots["proposer-2"],
                role_slots["proposer-3"],
                role_slots["proposer-4"],
                role_slots["proposer-5"],
            ],
            "coordinator_slot": role_slots["debate-coordinator"],
            "role_slots": role_slots,
            "engine_workflow_name": str(workflow.get("engine_workflow_name") or workflow.get("id") or "chemqa-review"),
        },
        "protocol_defaults": {
            "proposer_count": resolved["proposer_count"],
            "review_rounds": resolved.get("review_rounds"),
            "rebuttal_rounds": resolved.get("rebuttal_rounds"),
            "max_epochs": resolved.get("max_epochs"),
            "evidence_mode": resolved.get("evidence_mode"),
        },
        "runtime_context": {
            "additional_file_workspace": additional_file_workspace,
            "answer_kind": answer_kind,
            "final_decider": resolved.get("final_decider"),
            "backend": resolved.get("backend", "subprocess"),
            "evidence_mode": resolved.get("evidence_mode"),
            "workflow_package": {
                "kind": "python-path",
                "path": str((root / "runtime" / "workflow.py").resolve()),
                "class": "ChemQAWorkflow"
            },
            "chemqa_review": {
                "slot_set": args.slot_set,
                "slot_ids": slot_ids,
                "role_slots": role_slots,
                "engine_skill_root": str(root.parent / "debateclaw-v1"),
                "required_skills": list(REQUIRED_SKILLS),
                "role_map": role_map,
                "artifact_contract_version": "react-reviewed-v2",
                "max_epochs": resolved.get("max_epochs"),
                "react_reviewed_config_snapshot": config_snapshot,
                "stop_loss": {
                    "stale_timeout_seconds": 300,
                    "respawn_cooldown_seconds": 120,
                    "max_model_attempts": 2,
                    "lane_retry_budget": 2,
                    "phase_repair_budget": 2,
                    "max_respawns_per_role_phase_signature": 2,
                    "stagnation_basis": "phase_signature",
                    "terminal_failure_artifact": "chemqa_review_failure.yaml"
                },
                "acceptance_policy": {
                    "candidate_owner": "proposer-1",
                    "required_reviewer_lanes": ["proposer-2", "proposer-3", "proposer-4", "proposer-5"],
                    "require_all_reviewers": True,
                    "review_failure_blocks_acceptance": True,
                    "stop_on_no_blocking_items": True,
                    "ignore_transport_placeholders": True,
                    "count_only_review_phase_artifacts": True,
                    "count_only_reviews_targeting_candidate_owner": True,
                    "synthetic_reviews_do_not_satisfy_required_reviews": True,
                    "missing_required_review_blocks_acceptance": True,
                },
            },
        },
        "artifacts_root": f"<debate-run-root>/{run_id}/artifacts",
        "protocol_state_path": f"<debate-run-root>/{run_id}/state.db",
        "status": "planned",
    }
    if not args.dry_run:
        store.save_run_plan(run_plan)
        store.update_run_status(run_id, {"run_id": run_id, "status": "planned", "updated_at": iso_now()})
    print(json.dumps(run_plan, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
