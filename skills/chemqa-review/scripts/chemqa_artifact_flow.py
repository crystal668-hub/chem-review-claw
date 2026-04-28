#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ANSWER_KINDS = {
    "numeric_short_answer",
    "short_text_answer",
    "multi_part_research_answer",
    "multiple_choice",
    "structure_answer",
    "generic_semantic_answer",
}

REVIEWER_LANES = ("proposer-2", "proposer-3", "proposer-4", "proposer-5")
CANDIDATE_OWNER = "proposer-1"


@dataclass
class ArtifactValidation:
    artifact: dict[str, Any]
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class CandidateState:
    candidate_view: dict[str, Any]
    review_items: dict[str, dict[str, Any]]
    validation_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class FinalizationResult:
    terminal_state: str
    status_overlay: dict[str, Any]
    qa_result: dict[str, Any]
    artifact_paths: dict[str, str]


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clone_jsonish(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def resolve_answer_kind(metadata: dict[str, Any] | None) -> str:
    payload = metadata or {}
    explicit = clean_text(payload.get("answer_kind") or payload.get("chemqa_answer_kind")).lower()
    if explicit in ANSWER_KINDS:
        return explicit

    eval_kind = clean_text(payload.get("eval_kind") or payload.get("kind")).lower()
    dataset = clean_text(payload.get("dataset")).lower()
    track = clean_text(payload.get("track") or payload.get("subset")).lower()
    final_answer_kind = clean_text(payload.get("final_answer_kind")).lower()

    if eval_kind in {"chembench_open_ended", "frontierscience_olympiad"}:
        return "numeric_short_answer"
    if eval_kind == "frontierscience_research" or (dataset == "frontierscience" and track == "research"):
        return "multi_part_research_answer"
    if eval_kind == "superchem_multiple_choice_rpf" or "multiple_choice" in eval_kind:
        return "multiple_choice"
    if eval_kind == "conformabench_constructive" or final_answer_kind in {"single_smiles", "smiles", "structure"}:
        return "structure_answer"
    if dataset == "chembench":
        return "numeric_short_answer"
    if dataset == "superchem" and isinstance(payload.get("options"), dict):
        return "multiple_choice"
    if dataset == "conformabench":
        return "structure_answer"
    return "generic_semantic_answer"


def _artifact_id(*parts: Any) -> str:
    raw = "|".join(clean_text(part) for part in parts if clean_text(part))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return digest


def _common_artifact(
    *,
    artifact_kind: str,
    run_id: str,
    role: str,
    phase: str,
    epoch: int = 1,
    round_number: int = 0,
    source_path: str = "",
    payload: dict[str, Any],
    errors: list[str],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    artifact_id = f"{artifact_kind}-{_artifact_id(run_id, role, phase, epoch, round_number, payload)}"
    return {
        "artifact_id": artifact_id,
        "run_id": run_id,
        "artifact_kind": artifact_kind,
        "schema_version": 1,
        "role": role,
        "phase": phase,
        "epoch": epoch,
        "round": round_number,
        "created_at": iso_now(),
        "source_path": source_path,
        "validation_status": "valid" if not errors else "invalid",
        "validation_errors": list(errors),
        "validation_warnings": list(warnings or []),
        "payload": payload,
    }


def _first_text(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            nested = _first_text(value, "evaluator_answer", "direct_answer", "answer", "value", "final_answer")
            if nested:
                return nested
        text = clean_text(value)
        if text:
            return text
    return ""


def _numeric_value(text: str) -> float | None:
    value = clean_text(text)
    if not value:
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _looks_placeholder(text: str) -> bool:
    lowered = clean_text(text).lower()
    if not lowered:
        return True
    placeholder_markers = (
        "not available",
        "cannot determine",
        "unknown",
        "n/a",
        "no answer",
        "placeholder",
        "insufficient information to answer",
    )
    return any(marker in lowered for marker in placeholder_markers)


def _validate_answer_projection(answer_kind: str, evaluator_answer: str, full_answer: str = "") -> list[str]:
    errors: list[str] = []
    answer = clean_text(evaluator_answer)
    full = clean_text(full_answer)
    if answer_kind == "numeric_short_answer":
        if _numeric_value(answer) is None:
            errors.append("numeric_short_answer requires a numeric evaluator_answer")
    elif answer_kind == "short_text_answer":
        if _looks_placeholder(answer):
            errors.append("short_text_answer requires a concise non-placeholder evaluator_answer")
    elif answer_kind == "multi_part_research_answer":
        if _looks_placeholder(answer or full):
            errors.append("multi_part_research_answer requires a non-empty substantive answer")
    elif answer_kind == "multiple_choice":
        if not re.fullmatch(r"[A-Za-z](?:\s*(?:[,|/;]|\band\b)\s*[A-Za-z])*", answer):
            errors.append("multiple_choice requires option-label evaluator_answer")
    elif answer_kind == "structure_answer":
        if _looks_placeholder(answer):
            errors.append("structure_answer requires a non-empty structure projection")
    elif answer_kind == "generic_semantic_answer":
        if _looks_placeholder(answer or full):
            errors.append("generic_semantic_answer requires non-empty answer text")
    else:
        errors.append(f"unsupported answer_kind `{answer_kind}`")
    return errors


def validate_candidate_artifact(
    payload: dict[str, Any],
    *,
    answer_kind: str,
    run_id: str = "",
    source_path: str = "",
) -> ArtifactValidation:
    answer_kind = answer_kind if answer_kind in ANSWER_KINDS else "generic_semantic_answer"
    role = clean_text(payload.get("role") or payload.get("owner") or CANDIDATE_OWNER) or CANDIDATE_OWNER
    epoch = int(payload.get("epoch") or 1)
    round_number = int(payload.get("round") or payload.get("proposal_round") or 0)
    evaluator_answer = _first_text(payload, "evaluator_answer", "direct_answer", "answer", "value", "final_answer")
    display_answer = _first_text(payload, "display_answer") or evaluator_answer
    full_answer = _first_text(payload, "full_answer", "final_markdown", "final_text") or clean_text(payload.get("summary"))
    reasoning_summary = clean_text(payload.get("reasoning_summary") or payload.get("summary") or payload.get("justification"))
    candidate_payload = {
        "answer_kind": answer_kind,
        "evaluator_answer": evaluator_answer,
        "display_answer": display_answer,
        "full_answer": full_answer or evaluator_answer,
        "reasoning_summary": reasoning_summary,
        "submission_trace": clone_jsonish(payload.get("submission_trace") or payload.get("trace") or []),
        "claim_anchors": clone_jsonish(payload.get("claim_anchors") or []),
        "evidence_limits": clone_jsonish(payload.get("evidence_limits") or payload.get("limitations") or []),
    }
    errors = _validate_answer_projection(answer_kind, evaluator_answer, full_answer)
    if not reasoning_summary and answer_kind != "multi_part_research_answer":
        errors.append("candidate artifact is missing reasoning_summary")
    artifact = _common_artifact(
        artifact_kind="candidate",
        run_id=run_id,
        role=role,
        phase="propose",
        epoch=epoch,
        round_number=round_number,
        source_path=source_path,
        payload=candidate_payload,
        errors=errors,
    )
    return ArtifactValidation(artifact=artifact, valid=not errors, errors=errors)


def _review_item_key(*, epoch: int, round_number: int, reviewer_lane: str, item: dict[str, Any], index: int) -> str:
    item_id = clean_text(item.get("item_id") or item.get("id"))
    if not item_id:
        stable_payload = {
            "reviewer_lane": reviewer_lane,
            "target_field": item.get("target_field"),
            "severity": item.get("severity"),
            "finding": item.get("finding"),
            "requested_change": item.get("requested_change"),
            "index": index,
        }
        item_id = hashlib.sha256(json.dumps(stable_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
    return f"{epoch}:{round_number}:{reviewer_lane}:{item_id}"


def validate_review_artifact(payload: dict[str, Any], *, run_id: str = "", source_path: str = "") -> ArtifactValidation:
    reviewer_lane = clean_text(payload.get("reviewer_lane") or payload.get("role") or payload.get("reviewer"))
    epoch = int(payload.get("epoch") or 1)
    round_number = int(payload.get("round") or payload.get("review_round") or 0)
    verdict = clean_text(payload.get("verdict")).lower()
    errors: list[str] = []
    if reviewer_lane not in REVIEWER_LANES:
        errors.append("review artifact requires a fixed reviewer_lane")
    if verdict not in {"blocking", "non_blocking", "insufficient_evidence"}:
        errors.append("review artifact requires valid verdict")
    if clean_text(payload.get("target_owner") or payload.get("target")) != CANDIDATE_OWNER:
        errors.append("review artifact must target proposer-1")
    if clean_text(payload.get("target_kind") or "candidate_submission") != "candidate_submission":
        errors.append("review artifact must target candidate_submission")
    counts_for_acceptance = payload.get("counts_for_acceptance")
    synthetic = payload.get("synthetic")
    if counts_for_acceptance is not True:
        errors.append("review artifact must declare counts_for_acceptance true")
    if synthetic is not False:
        errors.append("review artifact must declare synthetic false")

    items: list[dict[str, Any]] = []
    raw_items = payload.get("review_items") or []
    if not isinstance(raw_items, list):
        errors.append("review artifact review_items must be a list")
        raw_items = []
    for index, item in enumerate(raw_items, start=1):
        item_payload = dict(item) if isinstance(item, dict) else {"finding": clean_text(item)}
        item_payload.setdefault("severity", "high" if verdict in {"blocking", "insufficient_evidence"} else "low")
        item_payload.setdefault("target_field", clean_text(item_payload.get("target_field")) or "evaluator_answer")
        item_payload["item_key"] = _review_item_key(
            epoch=epoch,
            round_number=round_number,
            reviewer_lane=reviewer_lane,
            item=item_payload,
            index=index,
        )
        item_payload["status"] = "open" if verdict in {"blocking", "insufficient_evidence"} else "non_blocking"
        items.append(item_payload)
    review_payload = {
        "target_artifact_id": clean_text(payload.get("target_artifact_id")),
        "verdict": verdict,
        "summary": clean_text(payload.get("summary")),
        "review_items": items,
        "counts_for_acceptance": counts_for_acceptance is True,
        "synthetic": synthetic is True,
    }
    artifact = _common_artifact(
        artifact_kind="review",
        run_id=run_id,
        role=reviewer_lane or "unknown",
        phase="review",
        epoch=epoch,
        round_number=round_number,
        source_path=source_path,
        payload=review_payload,
        errors=errors,
    )
    return ArtifactValidation(artifact=artifact, valid=not errors, errors=errors)


def validate_rebuttal_artifact(
    payload: dict[str, Any],
    *,
    answer_kind: str,
    run_id: str = "",
    source_path: str = "",
) -> ArtifactValidation:
    answer_kind = answer_kind if answer_kind in ANSWER_KINDS else "generic_semantic_answer"
    role = clean_text(payload.get("role") or payload.get("owner") or CANDIDATE_OWNER) or CANDIDATE_OWNER
    epoch = int(payload.get("epoch") or 1)
    round_number = int(payload.get("round") or payload.get("rebuttal_round") or 0)
    mode = clean_text(payload.get("mode")).lower()
    if not mode:
        if payload.get("concede") is True:
            mode = "concession"
        elif payload.get("updated_answer") is not None or payload.get("updated_direct_answer") is not None:
            mode = "answer_revision"
        else:
            mode = "response_only"

    errors: list[str] = []
    if mode not in {"response_only", "answer_revision", "concession"}:
        errors.append("rebuttal mode must be response_only, answer_revision, or concession")
    updated = payload.get("updated_answer")
    if not isinstance(updated, dict):
        updated = {}
    updated_evaluator = clean_text(
        updated.get("evaluator_answer")
        or updated.get("direct_answer")
        or payload.get("updated_direct_answer")
        or payload.get("direct_answer")
        or payload.get("final_answer")
    )
    updated_display = clean_text(updated.get("display_answer")) or updated_evaluator
    updated_full = clean_text(updated.get("full_answer") or payload.get("updated_full_answer")) or updated_evaluator
    if mode == "answer_revision":
        errors.extend(_validate_answer_projection(answer_kind, updated_evaluator, updated_full))
    rebuttal_payload = {
        "mode": mode,
        "concede": payload.get("concede") is True or mode == "concession",
        "response_summary": clean_text(payload.get("response_summary") or payload.get("summary")),
        "addressed_review_items": [clean_text(item) for item in (payload.get("addressed_review_items") or []) if clean_text(item)],
        "updated_answer": {
            "answer_kind": answer_kind,
            "evaluator_answer": updated_evaluator,
            "display_answer": updated_display,
            "full_answer": updated_full,
        }
        if updated_evaluator or updated_full
        else None,
        "updated_trace": clone_jsonish(payload.get("updated_trace") or payload.get("updated_submission_trace") or []),
        "remaining_open_items": clone_jsonish(payload.get("remaining_open_items") or []),
    }
    if not rebuttal_payload["response_summary"] and mode != "concession":
        errors.append("rebuttal artifact requires response_summary unless concession")
    artifact = _common_artifact(
        artifact_kind="rebuttal",
        run_id=run_id,
        role=role,
        phase="rebuttal",
        epoch=epoch,
        round_number=round_number,
        source_path=source_path,
        payload=rebuttal_payload,
        errors=errors,
    )
    return ArtifactValidation(artifact=artifact, valid=not errors, errors=errors)


def build_current_candidate_view(
    *,
    candidate_artifact: dict[str, Any],
    review_artifacts: list[dict[str, Any]],
    rebuttal_artifacts: list[dict[str, Any]],
) -> CandidateState:
    payload = clone_jsonish(candidate_artifact.get("payload") or {})
    run_id = clean_text(candidate_artifact.get("run_id"))
    errors = list(candidate_artifact.get("validation_errors") or [])
    warnings = list(candidate_artifact.get("validation_warnings") or [])
    review_items: dict[str, dict[str, Any]] = {}

    for review in review_artifacts:
        for item in ((review.get("payload") or {}).get("review_items") or []):
            if not isinstance(item, dict):
                continue
            key = clean_text(item.get("item_key"))
            if not key:
                continue
            review_items[key] = {**clone_jsonish(item), "status": item.get("status") or "open"}

    for rebuttal in rebuttal_artifacts:
        rebuttal_payload = rebuttal.get("payload") or {}
        mode = clean_text(rebuttal_payload.get("mode"))
        addressed = [clean_text(item) for item in (rebuttal_payload.get("addressed_review_items") or []) if clean_text(item)]
        if mode == "answer_revision" and isinstance(rebuttal_payload.get("updated_answer"), dict):
            updated = rebuttal_payload["updated_answer"]
            for key in ("evaluator_answer", "display_answer", "full_answer"):
                if clean_text(updated.get(key)):
                    payload[key] = clean_text(updated[key])
            if rebuttal_payload.get("updated_trace"):
                payload["submission_trace"] = clone_jsonish(rebuttal_payload["updated_trace"])
            for key in addressed:
                if key in review_items and review_items[key].get("status") == "open":
                    review_items[key]["status"] = "addressed_by_revision"
        elif mode == "response_only":
            for key in addressed:
                if key in review_items and review_items[key].get("status") == "open":
                    review_items[key]["status"] = "addressed_by_response"

    candidate_view = _common_artifact(
        artifact_kind="candidate_view",
        run_id=run_id,
        role=CANDIDATE_OWNER,
        phase="artifact_flow",
        epoch=int(candidate_artifact.get("epoch") or 1),
        round_number=int(candidate_artifact.get("round") or 0),
        payload=payload,
        errors=errors,
        warnings=warnings,
    )
    return CandidateState(candidate_view=candidate_view, review_items=review_items, validation_errors=errors, warnings=warnings)


def _atomic_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(path)
    return path


def _file_meta(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _write_manifest(path: Path, *, files: dict[str, Path], run_id: str, terminal_state: str) -> Path:
    payload = {
        "run_id": run_id,
        "schema_version": 1,
        "terminal_state": terminal_state,
        "created_at": iso_now(),
        "files": {key: _file_meta(value) for key, value in files.items() if value.is_file()},
    }
    return _atomic_json(path, payload)


def finalize_success(
    *,
    run_id: str,
    output_dir: Path,
    answer_kind: str,
    candidate_state: CandidateState,
    acceptance_status: str,
    protocol_payload: dict[str, Any] | None = None,
    degraded_execution: bool = False,
    warnings: list[str] | None = None,
) -> FinalizationResult:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    answer_kind = answer_kind if answer_kind in ANSWER_KINDS else "generic_semantic_answer"
    candidate_view = candidate_state.candidate_view
    candidate_payload = candidate_view.get("payload") or {}
    evaluator_answer = clean_text(candidate_payload.get("evaluator_answer"))
    display_answer = clean_text(candidate_payload.get("display_answer")) or evaluator_answer
    full_answer = clean_text(candidate_payload.get("full_answer")) or display_answer or evaluator_answer
    terminal_warnings = list(warnings or []) + list(candidate_state.warnings)
    open_items = [
        {**clone_jsonish(item), "status": "unresolved_at_terminal"}
        for item in candidate_state.review_items.values()
        if item.get("status") == "open"
    ]
    for item in open_items:
        key = clean_text(item.get("item_key"))
        if key in candidate_state.review_items:
            candidate_state.review_items[key]["status"] = "unresolved_at_terminal"

    final_answer_artifact = {
        "terminal_state": "completed",
        "schema_version": 1,
        "run_id": run_id,
        "answer_kind": answer_kind,
        "evaluator_answer": evaluator_answer,
        "display_answer": display_answer,
        "full_answer": full_answer,
        "source_candidate_view_id": candidate_view.get("artifact_id"),
        "acceptance_status": clean_text(acceptance_status) or "rejected",
        "review_summary": {
            "open_review_items": open_items,
            "review_items": clone_jsonish(candidate_state.review_items),
        },
        "confidence": clone_jsonish((protocol_payload or {}).get("overall_confidence") or {}),
        "degraded_execution": degraded_execution,
        "warnings": terminal_warnings,
    }
    final_path = _atomic_json(output_dir / "final_answer_artifact.json", final_answer_artifact)
    candidate_view_path = _atomic_json(output_dir / "candidate_view.json", candidate_view)
    validation_summary_path = _atomic_json(
        output_dir / "validation_summary.json",
        {
            "run_id": run_id,
            "terminal_state": "completed",
            "validation_errors": list(candidate_state.validation_errors),
            "warnings": terminal_warnings,
            "open_review_items_count": len(open_items),
            "open_review_items": open_items,
        },
    )
    qa_result = {
        "terminal_state": "completed",
        "acceptance_status": final_answer_artifact["acceptance_status"],
        "answer_kind": answer_kind,
        "final_answer": {
            "direct_answer": evaluator_answer,
            "answer": evaluator_answer,
            "value": evaluator_answer,
            "display_answer": display_answer,
            "full_answer": full_answer,
        },
        "artifact_paths": {
            "final_answer_artifact": str(final_path),
            "artifact_manifest": str(output_dir / "artifact_manifest.json"),
            "candidate_view": str(candidate_view_path),
            "validation_summary": str(validation_summary_path),
            "qa_result": str(output_dir / "qa_result.json"),
        },
    }
    if protocol_payload:
        qa_result["question"] = clean_text(protocol_payload.get("question"))
        qa_result["review_completion_status"] = clone_jsonish(protocol_payload.get("review_completion_status") or {})
        qa_result["overall_confidence"] = clone_jsonish(protocol_payload.get("overall_confidence") or {})
    qa_result_path = _atomic_json(output_dir / "qa_result.json", qa_result)
    manifest_path = _write_manifest(
        output_dir / "artifact_manifest.json",
        files={
            "final_answer_artifact": final_path,
            "candidate_view": candidate_view_path,
            "validation_summary": validation_summary_path,
            "qa_result": qa_result_path,
        },
        run_id=run_id,
        terminal_state="completed",
    )
    qa_result["artifact_paths"]["artifact_manifest"] = str(manifest_path)
    _atomic_json(qa_result_path, qa_result)
    artifact_paths = {key: str(value) for key, value in {
        "final_answer_artifact": final_path,
        "candidate_view": candidate_view_path,
        "validation_summary": validation_summary_path,
        "qa_result": qa_result_path,
        "artifact_manifest": manifest_path,
    }.items()}
    status_overlay = {
        "run_id": run_id,
        "status": "done",
        "protocol_terminal_state": "completed",
        "artifact_flow_state": "finalized",
        "benchmark_terminal_state": "completed",
        "terminal_state": "completed",
        "artifacts_output_dir": str(output_dir),
        "qa_result_path": str(qa_result_path),
        "final_answer_artifact_path": str(final_path),
        "artifact_manifest_path": str(manifest_path),
        "candidate_view_path": str(candidate_view_path),
        "artifact_paths": artifact_paths,
    }
    return FinalizationResult("completed", status_overlay, qa_result, artifact_paths)


def finalize_failure(
    *,
    run_id: str,
    output_dir: Path,
    failure_code: str,
    failure_message: str,
    last_valid_candidate_view: dict[str, Any] | None = None,
    answer_projection: dict[str, Any] | None = None,
    recovery_eligibility: dict[str, Any] | None = None,
    missing_artifacts: list[str] | None = None,
    validation_errors: list[str] | None = None,
    open_review_items: list[dict[str, Any]] | None = None,
    diagnostic_paths: list[str] | None = None,
) -> FinalizationResult:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    recovery = {
        "evaluable": False,
        "scored": False,
        "reliability": "none",
        "recovery_mode": "none",
        "reason": clean_text(failure_code),
    }
    recovery.update(recovery_eligibility or {})
    failure_artifact = {
        "terminal_state": "failed",
        "schema_version": 1,
        "run_id": run_id,
        "failure_code": clean_text(failure_code) or "artifact_finalization_failed",
        "failure_message": clean_text(failure_message),
        "last_valid_candidate_view": clone_jsonish(last_valid_candidate_view or {}),
        "answer_projection": clone_jsonish(answer_projection) if answer_projection else None,
        "recovery_eligibility": clone_jsonish(recovery),
        "missing_artifacts": list(missing_artifacts or []),
        "validation_errors": list(validation_errors or []),
        "open_review_items": clone_jsonish(open_review_items or []),
        "diagnostic_paths": list(diagnostic_paths or []),
    }
    failure_path = _atomic_json(output_dir / "failure_artifact.json", failure_artifact)
    validation_summary_path = _atomic_json(
        output_dir / "validation_summary.json",
        {
            "run_id": run_id,
            "terminal_state": "failed",
            "failure_code": failure_artifact["failure_code"],
            "validation_errors": failure_artifact["validation_errors"],
            "missing_artifacts": failure_artifact["missing_artifacts"],
            "open_review_items_count": len(failure_artifact["open_review_items"]),
        },
    )
    qa_result = {
        "terminal_state": "failed",
        "failure_code": failure_artifact["failure_code"],
        "failure_message": failure_artifact["failure_message"],
        "answer_projection": clone_jsonish(answer_projection) if answer_projection else None,
        "recovery_eligibility": clone_jsonish(recovery),
        "artifact_paths": {
            "failure_artifact": str(failure_path),
            "artifact_manifest": str(output_dir / "artifact_manifest.json"),
            "validation_summary": str(validation_summary_path),
            "qa_result": str(output_dir / "qa_result.json"),
        },
    }
    qa_result_path = _atomic_json(output_dir / "qa_result.json", qa_result)
    manifest_path = _write_manifest(
        output_dir / "artifact_manifest.json",
        files={
            "failure_artifact": failure_path,
            "validation_summary": validation_summary_path,
            "qa_result": qa_result_path,
        },
        run_id=run_id,
        terminal_state="failed",
    )
    qa_result["artifact_paths"]["artifact_manifest"] = str(manifest_path)
    _atomic_json(qa_result_path, qa_result)
    artifact_paths = {key: str(value) for key, value in {
        "failure_artifact": failure_path,
        "validation_summary": validation_summary_path,
        "qa_result": qa_result_path,
        "artifact_manifest": manifest_path,
    }.items()}
    status_overlay = {
        "run_id": run_id,
        "status": "done",
        "protocol_terminal_state": "failed",
        "artifact_flow_state": "finalization_failed",
        "benchmark_terminal_state": "failed",
        "terminal_state": "failed",
        "terminal_reason_code": failure_artifact["failure_code"],
        "terminal_reason": failure_artifact["failure_message"],
        "artifacts_output_dir": str(output_dir),
        "qa_result_path": str(qa_result_path),
        "failure_artifact_path": str(failure_path),
        "artifact_manifest_path": str(manifest_path),
        "artifact_paths": artifact_paths,
    }
    return FinalizationResult("failed", status_overlay, qa_result, artifact_paths)


def candidate_from_protocol(protocol: dict[str, Any], *, answer_kind: str, run_id: str = "") -> ArtifactValidation:
    candidate = clone_jsonish(protocol.get("candidate_submission") or {})
    final_answer = protocol.get("final_answer")
    if isinstance(final_answer, dict):
        for src, dst in (
            ("direct_answer", "direct_answer"),
            ("answer", "answer"),
            ("value", "value"),
            ("display_answer", "display_answer"),
            ("full_answer", "full_answer"),
            ("summary", "summary"),
        ):
            if final_answer.get(src) not in (None, ""):
                candidate[dst] = final_answer[src]
    elif final_answer not in (None, ""):
        candidate["direct_answer"] = final_answer
    return validate_candidate_artifact(candidate, answer_kind=answer_kind, run_id=run_id)


def finalization_from_protocol(
    *,
    protocol: dict[str, Any],
    output_dir: Path,
    run_id: str,
    answer_kind: str = "generic_semantic_answer",
) -> FinalizationResult:
    terminal_state = clean_text(protocol.get("terminal_state") or "completed")
    if terminal_state == "failed":
        return finalize_failure(
            run_id=run_id,
            output_dir=output_dir,
            failure_code=clean_text(protocol.get("failure_code") or "protocol_failed"),
            failure_message=clean_text(protocol.get("failure_reason") or protocol.get("terminal_reason") or "ChemQA protocol failed."),
            validation_errors=[],
            diagnostic_paths=[clean_text(protocol.get("terminal_failure_artifact"))] if clean_text(protocol.get("terminal_failure_artifact")) else [],
        )

    candidate = candidate_from_protocol(protocol, answer_kind=answer_kind, run_id=run_id)
    if not candidate.valid:
        return finalize_failure(
            run_id=run_id,
            output_dir=output_dir,
            failure_code="candidate_validation_failed",
            failure_message="Candidate artifact could not be projected into a valid final answer.",
            last_valid_candidate_view=candidate.artifact if candidate.artifact.get("payload") else None,
            validation_errors=candidate.errors,
        )
    candidate_state = build_current_candidate_view(
        candidate_artifact=candidate.artifact,
        review_artifacts=[],
        rebuttal_artifacts=[],
    )
    return finalize_success(
        run_id=run_id,
        output_dir=output_dir,
        answer_kind=answer_kind,
        candidate_state=candidate_state,
        acceptance_status=clean_text(protocol.get("acceptance_status") or "rejected"),
        protocol_payload=protocol,
        degraded_execution=bool((protocol.get("acceptance_decision") or {}).get("accepted_under_degraded_quorum"))
        if isinstance(protocol.get("acceptance_decision"), dict)
        else False,
        warnings=[clean_text(item) for item in protocol.get("execution_warnings") or [] if clean_text(item)],
    )
