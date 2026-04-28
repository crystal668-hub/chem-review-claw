#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from bundle_common import dump_json, resolve_skill_root, write_text
from chemqa_artifact_flow import finalization_from_protocol, resolve_answer_kind

REQUIRED_REVIEWER_LANES = ("proposer-2", "proposer-3", "proposer-4", "proposer-5")
EXPECTED_CANDIDATE_OWNER = "proposer-1"
ALLOWED_VERDICTS = {"blocking", "non_blocking", "insufficient_evidence"}


def find_protocol_file(source_dir: Path) -> Path:
    candidates = (
        source_dir / "chemqa_review_protocol.yaml",
        source_dir / "chemqa_review_protocol.yml",
        source_dir / "chemqa_review_protocol.json",
        source_dir / "debate-coordinator" / "chemqa_review_protocol.yaml",
        source_dir / "debate-coordinator" / "chemqa_review_protocol.json",
        source_dir / "coordinator" / "chemqa_review_protocol.yaml",
        source_dir / "coordinator" / "chemqa_review_protocol.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(f"Could not find chemqa_review_protocol.yaml under {source_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild react_reviewed-style artifacts from chemqa-review outputs.")
    parser.add_argument("--skill-root", help="chemqa-review skill root; defaults to this script's parent bundle")
    parser.add_argument("--source-dir", required=True, help="Directory containing chemqa_review_protocol.yaml")
    parser.add_argument("--output-dir", required=True, help="Output directory for rebuilt artifacts")
    parser.add_argument("--protocol-file", help="Optional explicit protocol YAML path")
    parser.add_argument("--answer-kind", help="Resolved ChemQA answer kind for Artifact Flow validation")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def clone_jsonish(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def load_protocol(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Protocol file did not decode to a mapping: {path}")
    return payload


def stringify_final_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("final_markdown", "final_text", "answer", "direct_answer", "value"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
            if isinstance(candidate, (int, float)):
                return str(candidate)
        if "final_integer" in value:
            return str(value["final_integer"])
        return json.dumps(value, indent=2, ensure_ascii=False)
    if isinstance(value, list):
        return json.dumps(value, indent=2, ensure_ascii=False)
    return str(value)


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return list(value.values())
    return [value]


def candidate_owner(protocol: dict[str, Any]) -> str:
    submission = as_dict(protocol.get("candidate_submission"))
    owner = submission.get("owner")
    if isinstance(owner, str) and owner.strip():
        return owner
    final_answer = as_dict(protocol.get("final_answer"))
    owner = final_answer.get("accepted_owner")
    if isinstance(owner, str) and owner.strip():
        return owner
    return EXPECTED_CANDIDATE_OWNER


def review_completion_dict(protocol: dict[str, Any]) -> dict[str, Any]:
    value = protocol.get("review_completion_status")
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return {"status": value}
    return {}


def reviewer_trajectories_dict(protocol: dict[str, Any]) -> dict[str, Any]:
    value = protocol.get("reviewer_trajectories")
    return value if isinstance(value, dict) else {}


def review_statuses_dict(protocol: dict[str, Any]) -> dict[str, Any]:
    value = protocol.get("review_statuses")
    if isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    if isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, dict):
                key = str(item.get("target") or item.get("reviewer_role") or index)
                result[key] = item
    return result


def find_lane_candidate_review(*, protocol: dict[str, Any], lane: str, owner: str) -> dict[str, Any]:
    reviewer_trajectories = reviewer_trajectories_dict(protocol)
    lane_payload = as_dict(reviewer_trajectories.get(lane))
    candidate_review = as_dict(lane_payload.get("candidate_review"))
    if candidate_review:
        target = candidate_review.get("target_owner") or candidate_review.get("target")
        if target in (None, owner):
            merged = dict(candidate_review)
            if not as_dict(merged.get("artifact")):
                merged["artifact"] = clone_jsonish(lane_payload.get("artifact") or {})
            return merged

    owner_status = as_dict(review_statuses_dict(protocol).get(owner))
    for item in as_list(owner_status.get("reviewers")):
        if not isinstance(item, dict):
            continue
        if item.get("reviewer") == lane:
            fallback = dict(item)
            fallback.setdefault("target_owner", owner)
            fallback.setdefault("target_kind", "candidate_submission")
            return fallback
    return {}


def validate_protocol(protocol: dict[str, Any]) -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    terminal_state = str(protocol.get("terminal_state") or "completed")
    acceptance_status = str(protocol.get("acceptance_status") or "")
    if terminal_state == "failed":
        if not str(protocol.get("failure_reason") or protocol.get("terminal_failure_artifact") or "").strip():
            errors.append("failed protocol must include `failure_reason` or `terminal_failure_artifact`")
        return {"errors": errors, "warnings": warnings}

    if acceptance_status != "accepted":
        return {"errors": errors, "warnings": warnings}

    owner = candidate_owner(protocol)
    if owner != EXPECTED_CANDIDATE_OWNER:
        errors.append(
            f"accepted protocol must keep `{EXPECTED_CANDIDATE_OWNER}` as candidate owner, found `{owner}`"
        )

    review_completion = review_completion_dict(protocol)
    if review_completion.get("status") != "complete":
        errors.append("accepted protocol requires review_completion_status.status = `complete`")
    if review_completion.get("required_candidate_reviews_expected") != 4:
        errors.append("accepted protocol requires review_completion_status.required_candidate_reviews_expected = 4")
    if review_completion.get("required_candidate_reviews_submitted") != 4:
        errors.append("accepted protocol requires review_completion_status.required_candidate_reviews_submitted = 4")
    if review_completion.get("required_fixed_reviewer_lanes_complete") is not True:
        errors.append("accepted protocol requires required_fixed_reviewer_lanes_complete = true")
    if "transport_placeholders_ignored" not in review_completion:
        errors.append("accepted protocol must report review_completion_status.transport_placeholders_ignored")
    if "non_candidate_reviews_ignored" not in review_completion:
        errors.append("accepted protocol must report review_completion_status.non_candidate_reviews_ignored")
    if "synthetic_reviews_excluded_from_acceptance" not in review_completion:
        errors.append("accepted protocol must report review_completion_status.synthetic_reviews_excluded_from_acceptance")

    for lane in REQUIRED_REVIEWER_LANES:
        lane_review = find_lane_candidate_review(protocol=protocol, lane=lane, owner=owner)
        if not lane_review:
            errors.append(f"missing qualifying candidate review for {lane}")
            continue

        if lane_review.get("artifact_kind") != "formal_review":
            errors.append(f"{lane} review must declare artifact_kind = `formal_review`")
        if lane_review.get("phase") != "review":
            errors.append(f"{lane} review must declare phase = `review`")
        if lane_review.get("reviewer_lane") != lane:
            errors.append(f"{lane} review must declare reviewer_lane = `{lane}`")
        if lane_review.get("target_owner") != owner and lane_review.get("target") != owner:
            errors.append(f"{lane} review must target `{owner}`")
        if lane_review.get("target_kind") != "candidate_submission":
            errors.append(f"{lane} review must declare target_kind = `candidate_submission`")
        if lane_review.get("synthetic") is not False:
            errors.append(f"{lane} review is synthetic or missing `synthetic: false`; synthetic reviews cannot satisfy acceptance")
        if lane_review.get("counts_for_acceptance") is not True:
            errors.append(f"{lane} review must declare counts_for_acceptance = true")
        verdict = lane_review.get("verdict")
        if verdict not in ALLOWED_VERDICTS:
            errors.append(f"{lane} review must declare verdict in {sorted(ALLOWED_VERDICTS)}")
        review_items = lane_review.get("review_items")
        if not isinstance(review_items, list):
            errors.append(f"{lane} review must provide review_items as a list")
        if lane_review.get("blocking") is True and verdict != "blocking":
            errors.append(f"{lane} review has blocking=true but verdict is not `blocking`")
        if verdict == "blocking":
            errors.append(f"accepted protocol cannot include blocking verdict from {lane}")
        if not as_dict(lane_review.get("artifact")):
            warnings.append(f"{lane} review is missing artifact metadata")

    final_review_items = protocol.get("final_review_items")
    acceptance_items = []
    if isinstance(final_review_items, dict):
        acceptance_items = as_list(final_review_items.get("blocking_items")) + as_list(final_review_items.get("non_blocking_items"))
    else:
        acceptance_items = as_list(final_review_items)
    for item in acceptance_items:
        if not isinstance(item, dict):
            errors.append("final_review_items entries must be objects")
            continue
        reviewer = item.get("reviewer")
        if reviewer not in REQUIRED_REVIEWER_LANES:
            errors.append(f"final_review_items contains non-required reviewer `{reviewer}`")
        if item.get("synthetic") is not False:
            errors.append(
                f"final_review_items for reviewer `{reviewer}` is synthetic or missing `synthetic: false`; synthetic reviews cannot support acceptance"
            )

    return {"errors": errors, "warnings": warnings}


def main() -> int:
    args = parse_args()
    _skill_root = resolve_skill_root(args.skill_root, file_hint=__file__)
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = Path(args.protocol_file).expanduser().resolve() if args.protocol_file else find_protocol_file(source_dir)
    protocol = load_protocol(protocol_path)
    run_id = str(protocol.get("run_id") or output_dir.name).strip() or output_dir.name
    answer_kind = resolve_answer_kind(
        {
            "answer_kind": args.answer_kind or protocol.get("answer_kind"),
            "eval_kind": protocol.get("eval_kind"),
            "dataset": protocol.get("dataset"),
            "track": protocol.get("track"),
        }
    )

    validation = validate_protocol(protocol)
    if validation["errors"]:
        payload = {
            "status": "error",
            "reason": "protocol_validation_failed",
            "protocol_file": str(protocol_path),
            "validation": validation,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 2

    artifact_paths = {
        "candidate_submission": str(dump_json(output_dir / "candidate_submission.json", clone_jsonish(protocol.get("candidate_submission") or {}))),
        "acceptance_decision": str(dump_json(output_dir / "acceptance_decision.json", clone_jsonish(protocol.get("acceptance_decision") or {}))),
        "submission_trace": str(dump_json(output_dir / "submission_trace.json", clone_jsonish(protocol.get("submission_trace") or []))),
        "submission_cycles": str(dump_json(output_dir / "submission_cycles.json", clone_jsonish(protocol.get("submission_cycles") or []))),
        "proposer_trajectory": str(dump_json(output_dir / "proposer_trajectory.json", clone_jsonish(protocol.get("proposer_trajectory") or {}))),
        "reviewer_trajectories": str(dump_json(output_dir / "reviewer_trajectories.json", clone_jsonish(protocol.get("reviewer_trajectories") or {}))),
        "review_statuses": str(dump_json(output_dir / "review_statuses.json", clone_jsonish(protocol.get("review_statuses") or []))),
        "final_review_items": str(dump_json(output_dir / "final_review_items.json", clone_jsonish(protocol.get("final_review_items") or []))),
        "final_answer": str(write_text(output_dir / "final_answer.md", stringify_final_answer(protocol.get("final_answer")).strip() + "\n")),
    }
    if protocol.get("acceptance_status") == "accepted":
        artifact_paths["final_submission"] = str(
            dump_json(output_dir / "final_submission.json", clone_jsonish(protocol.get("candidate_submission") or {}))
        )

    finalization = finalization_from_protocol(
        protocol=protocol,
        output_dir=output_dir,
        run_id=run_id,
        answer_kind=answer_kind,
    )
    qa_result = dict(finalization.qa_result)
    qa_result.update(
        {
            "question": str(protocol.get("question") or qa_result.get("question") or ""),
            "language": "en",
            "workflow_mode": "react_reviewed",
            "sections": clone_jsonish(protocol.get("sections") or []),
            "citations": clone_jsonish(protocol.get("citations") or []),
            "claim_trace": clone_jsonish(protocol.get("claim_trace") or []),
            "submission_trace": clone_jsonish(protocol.get("submission_trace") or []),
            "review_completion_status": str(review_completion_dict(protocol).get("status") or protocol.get("review_completion_status") or "incomplete"),
            "overall_confidence": clone_jsonish(
                protocol.get("overall_confidence")
                or {"level": "low", "score": 0.0, "rationale": "Protocol did not provide confidence."}
            ),
            "section_confidence": clone_jsonish(protocol.get("section_confidence") or []),
            "insufficient_evidence": bool(protocol.get("insufficient_evidence", False)),
            "limitations_summary": str(protocol.get("limitations_summary") or protocol.get("failure_reason") or ""),
            "retrieval_diagnostics_summary": str(protocol.get("retrieval_diagnostics_summary") or ""),
            "execution_warnings": clone_jsonish(list(protocol.get("execution_warnings") or []) + validation["warnings"]),
            "time_elapsed": float(protocol.get("time_elapsed") or 0.0),
        }
    )
    qa_result_artifact_paths = dict(qa_result.get("artifact_paths") or {})
    qa_result_artifact_paths.update(artifact_paths)
    qa_result_artifact_paths.update(finalization.artifact_paths)
    qa_result["artifact_paths"] = qa_result_artifact_paths
    dump_json(output_dir / "qa_result.json", qa_result)
    artifact_paths.update(finalization.artifact_paths)

    payload = {
        "protocol_file": str(protocol_path),
        "output_dir": str(output_dir),
        "artifact_paths": artifact_paths,
        "answer_kind": answer_kind,
        "terminal_state": finalization.terminal_state,
        "status_overlay": finalization.status_overlay,
        "validation": validation,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
