from __future__ import annotations

import copy
from typing import Any


CHEMQA_RUN_LIFECYCLE_STATUSES = {"planned", "running", "done"}
CHEMQA_TERMINAL_STATES = {"completed", "failed", "cancelled"}
CHEMQA_ARTIFACT_FLOW_STATES = {"finalizing", "finalized", "finalization_failed"}


def deep_copy_jsonish(value: Any) -> Any:
    return copy.deepcopy(value)


def normalize_chemqa_run_status(payload: dict[str, Any] | None) -> dict[str, Any]:
    normalized = deep_copy_jsonish(payload or {})
    legacy_status = str(normalized.get("status") or "").strip()
    status = legacy_status
    terminal_state = str(normalized.get("terminal_state") or "").strip()
    artifact_flow_state = str(normalized.get("artifact_flow_state") or "").strip()
    benchmark_terminal_state = str(normalized.get("benchmark_terminal_state") or "").strip()
    terminal_reason_code = str(normalized.get("terminal_reason_code") or "").strip()
    terminal_reason = str(normalized.get("terminal_reason") or normalized.get("reason") or "").strip()

    artifact_collection = normalized.get("artifact_collection")
    if isinstance(artifact_collection, dict):
        artifact_collection_payload = deep_copy_jsonish(artifact_collection)
    else:
        artifact_collection_payload = {}
    artifact_collection_status = str(artifact_collection_payload.get("status") or "").strip()

    if artifact_flow_state == "finalizing" or benchmark_terminal_state == "running" or terminal_state == "running":
        status = "running"
        terminal_state = "running"
        if benchmark_terminal_state:
            normalized["benchmark_terminal_state"] = benchmark_terminal_state
        normalized["artifact_flow_state"] = artifact_flow_state or "finalizing"
    elif benchmark_terminal_state in {"completed", "failed", "cancelled"}:
        status = "done"
        terminal_state = benchmark_terminal_state
    elif legacy_status == "completed":
        status = "done"
        terminal_state = terminal_state or "completed"
        artifact_collection_status = artifact_collection_status or "ok"
    elif legacy_status == "completed_with_artifact_errors":
        status = "done"
        terminal_state = terminal_state or "completed"
        terminal_reason_code = terminal_reason_code or "artifact_collection_error"
        artifact_collection_status = "error"
    elif legacy_status == "stalled":
        status = "done"
        terminal_state = terminal_state or "failed"
        terminal_reason_code = terminal_reason_code or "stalled"
    elif legacy_status == "terminal_failure":
        status = "done"
        terminal_state = terminal_state or "failed"
        terminal_reason_code = terminal_reason_code or "terminal_failure"
    elif legacy_status == "failed":
        status = "done"
        terminal_state = terminal_state or "failed"
    elif legacy_status == "abandoned":
        status = "done"
        terminal_state = terminal_state or "cancelled"
        terminal_reason_code = terminal_reason_code or "abandoned"
    elif legacy_status == "cancelled":
        status = "done"
        terminal_state = terminal_state or "cancelled"
        terminal_reason_code = terminal_reason_code or "cancelled"
    elif legacy_status == "done":
        status = "done"
    elif legacy_status not in CHEMQA_RUN_LIFECYCLE_STATUSES:
        status = status or ""

    if status == "done" and not terminal_state:
        if terminal_reason_code in {"abandoned", "cancelled"}:
            terminal_state = "cancelled"
        elif artifact_collection_status == "error":
            terminal_state = "completed"

    if status == "done" and terminal_state == "completed":
        artifact_collection_status = artifact_collection_status or ("error" if normalized.get("artifact_collection_error") else "ok")

    if artifact_collection_status:
        artifact_collection_payload["status"] = artifact_collection_status
        normalized["artifact_collection"] = artifact_collection_payload
    elif "artifact_collection" in normalized and not artifact_collection_payload:
        normalized.pop("artifact_collection", None)

    normalized["status"] = status
    if terminal_state:
        normalized["terminal_state"] = terminal_state
    else:
        normalized.pop("terminal_state", None)
    if terminal_reason_code:
        normalized["terminal_reason_code"] = terminal_reason_code
    else:
        normalized.pop("terminal_reason_code", None)
    if terminal_reason:
        normalized["terminal_reason"] = terminal_reason
    elif "terminal_reason" in normalized:
        normalized.pop("terminal_reason", None)

    if legacy_status and legacy_status != status:
        normalized["legacy_status"] = legacy_status
    elif "legacy_status" in normalized and not normalized["legacy_status"]:
        normalized.pop("legacy_status", None)

    return normalized


def is_chemqa_terminal_status(payload: dict[str, Any] | None) -> bool:
    normalized = normalize_chemqa_run_status(payload)
    return (
        str(normalized.get("status") or "") == "done"
        and str(normalized.get("terminal_state") or "") in CHEMQA_TERMINAL_STATES
    )


def is_chemqa_success_status(payload: dict[str, Any] | None) -> bool:
    normalized = normalize_chemqa_run_status(payload)
    return (
        str(normalized.get("status") or "") == "done"
        and str(normalized.get("terminal_state") or "") == "completed"
    )


def normalize_run_status_value(status: Any) -> str:
    return str(getattr(status, "value", status) or "").strip()


def build_result_axes_from_runner(run_result: Any) -> dict[str, Any]:
    status = getattr(run_result, "status", None)
    normalized_status = normalize_run_status_value(status)
    runner_meta = getattr(run_result, "runner_meta", None) or {}
    raw = getattr(run_result, "raw", None) or {}
    recovery = getattr(run_result, "recovery", None)
    scored = bool(run_result.should_score())
    run_lifecycle_status = "completed" if normalized_status in {"completed", "recovered"} else "failed"

    terminal_state = runner_meta.get("terminal_state")
    if terminal_state == "completed":
        protocol_completion_status = "completed"
    elif raw.get("run_status") is not None:
        protocol_completion_status = "failed"
    else:
        protocol_completion_status = "missing"

    axes: dict[str, Any] = {
        "schema_version": 2,
        "run_lifecycle_status": run_lifecycle_status,
        "protocol_completion_status": protocol_completion_status,
        "protocol_acceptance_status": runner_meta.get("acceptance_status"),
    }

    if recovery is not None:
        recovery_mode = str(getattr(recovery, "recovery_mode", "") or "none")
        axes.update(
            answer_availability=(
                "preview_only"
                if recovery_mode == "run-status-final-answer-preview"
                else "recovered_candidate"
            ),
            answer_reliability=str(getattr(recovery, "reliability", "") or "none"),
            evaluable=bool(getattr(recovery, "evaluable", False)),
            scored=scored,
            recovery_mode=recovery_mode,
            degraded_execution=True,
        )
    else:
        status_is_completed = normalized_status == "completed"
        if normalized_status == "recovered":
            fallback_source = str(runner_meta.get("fallback_source") or "")
            answer_availability = (
                "preview_only"
                if fallback_source == "run-status-final-answer-preview"
                else "recovered_candidate"
            )
            answer_reliability = str(runner_meta.get("answer_reliability") or "").strip() or (
                "low_confidence_recovered"
                if fallback_source == "run-status-final-answer-preview"
                else "high_confidence_recovered"
            )
            axes.update(
                answer_availability=answer_availability,
                answer_reliability=answer_reliability,
                evaluable=False,
                scored=False,
                recovery_mode=str(runner_meta.get("recovery_mode") or fallback_source or "none"),
                degraded_execution=True,
            )
        elif status_is_completed:
            axes.update(
                answer_availability="native_final",
                answer_reliability="native",
                evaluable=scored,
                scored=scored,
                recovery_mode="none",
                degraded_execution=False,
            )
        else:
            axes.update(
                answer_availability="missing",
                answer_reliability="none",
                evaluable=False,
                scored=False,
                recovery_mode="none",
                degraded_execution=True,
            )

    axes["execution_error_kind"] = None if axes["scored"] else "execution_error"
    return axes
