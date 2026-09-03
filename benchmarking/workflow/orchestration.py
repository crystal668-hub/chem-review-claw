from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarking.core.convergence import ConvergencePolicy
from benchmarking.core.reporting import GroupRecordResult
from benchmarking.core.status import build_result_axes_from_runner
from benchmarking.runtime.cancellation import BenchmarkCancelledError, CancellationToken


class OrchestrationError(RuntimeError):
    pass


def build_cancelled_group_record_result(
    *,
    group: Any,
    record: Any,
    build_error_group_record_result_fn: Callable[..., GroupRecordResult],
    elapsed_seconds: float = 0.0,
    runner_meta: dict[str, Any] | None = None,
    raw: dict[str, Any] | None = None,
) -> GroupRecordResult:
    message = "Benchmark run cancelled before evaluation completed."
    entry = build_error_group_record_result_fn(
        group=group,
        record=record,
        error_message=message,
        elapsed_seconds=elapsed_seconds,
        runner_meta={**dict(runner_meta or {}), "cancellation": {"status": "cancelled"}},
        raw=dict(raw or {"status": "cancelled"}),
    )
    payload = asdict(entry)
    payload.update(
        {
            "run_lifecycle_status": "cancelled",
            "protocol_completion_status": "missing",
            "answer_availability": "missing",
            "answer_reliability": "none",
            "evaluable": False,
            "scored": False,
            "recovery_mode": "none",
            "degraded_execution": False,
            "execution_error_kind": "cancelled",
            "evaluation": {
                "eval_kind": str(getattr(record, "eval_kind", "") or ""),
                "score": None,
                "max_score": None,
                "normalized_score": None,
                "passed": None,
                "primary_metric": "cancelled",
                "primary_metric_direction": "not_applicable",
                "details": {"execution_error_kind": "cancelled"},
            },
            "error": message,
        }
    )
    return GroupRecordResult(**payload)


def ensure_compatible_runner_result(run_result: Any, *, runner_kind: str) -> None:
    missing: list[str] = []
    should_score = getattr(run_result, "should_score", None)
    if not callable(should_score):
        missing.append("callable should_score()")
    answer = getattr(run_result, "answer", None)
    if answer is None:
        missing.append("answer")
    else:
        if not hasattr(answer, "short_answer_text"):
            missing.append("answer.short_answer_text")
        if not hasattr(answer, "full_response_text"):
            missing.append("answer.full_response_text")
    if not isinstance(getattr(run_result, "runner_meta", None), dict):
        missing.append("runner_meta: dict")
    if not isinstance(getattr(run_result, "raw", None), dict):
        missing.append("raw: dict")
    if not hasattr(run_result, "status"):
        missing.append("status")
    failure = getattr(run_result, "failure", None)
    if failure is not None and not hasattr(failure, "message"):
        missing.append("failure.message")
    if missing:
        raise OrchestrationError(
            f"Runner `{runner_kind}` returned incompatible result object `{type(run_result).__name__}`; "
            f"missing/invalid fields: {', '.join(missing)}"
        )


def status_label(run_result: Any) -> str:
    status = getattr(run_result, "status", None)
    if status is None:
        return "unknown"
    return str(getattr(status, "value", status))


def run_group(
    *,
    group: Any,
    records: list[Any],
    output_root: Path,
    single_timeout: int,
    chemqa_timeout: int,
    judge: Any,
    config_path: Path,
    single_agent: str,
    chemqa_root: Path,
    chemqa_model_profile: str,
    review_rounds: int | None,
    rebuttal_rounds: int | None,
    chemqa_slot_sets: dict[str, str],
    experiment_specs: dict[str, Any],
    build_runner_fn: Callable[..., Any],
    evaluate_answer_fn: Callable[..., Any],
    build_error_group_record_result_fn: Callable[..., GroupRecordResult],
    classify_subset_fn: Callable[[Any], str],
    save_json_fn: Callable[[Path, Any], None],
    slugify_fn: Callable[..., str],
    single_convergence_policy: ConvergencePolicy | None = None,
    chemqa_convergence_policy: ConvergencePolicy | None = None,
    skill_health_summary: dict[str, Any] | None = None,
    single_timeout_retries: int = 3,
    single_timeout_retry_backoff_seconds: tuple[int | float, ...] | list[int | float] = (5, 15, 45),
    single_agent_thinking: str,
    no_timeout: bool = False,
    workspace_manager: Any | None = None,
    progress_writer: Any | None = None,
    cancellation_token: CancellationToken | None = None,
    process_registry: Any | None = None,
    pypi_cutoff: str | None = None,
) -> list[GroupRecordResult]:
    runtime_bundle_root = output_root / "input-bundles"

    def mark_cancelling() -> None:
        if progress_writer is None or cancellation_token is None:
            return
        reason = cancellation_token.reason
        progress_writer.run_cancelling(reason=reason.to_payload() if reason is not None else {})

    if cancellation_token is not None and cancellation_token.is_cancelled:
        mark_cancelling()
        group_results = [
            build_cancelled_group_record_result(
                group=group,
                record=record,
                build_error_group_record_result_fn=build_error_group_record_result_fn,
            )
            for record in records
        ]
        for entry in group_results:
            save_json_fn(output_root / "per-record" / group.id / f"{slugify_fn(entry.record_id)}.json", asdict(entry))
            if progress_writer is not None:
                progress_writer.record_cancelled(group.id, entry.record_id)
        if progress_writer is not None:
            progress_writer.group_cancelled(group.id)
        return group_results
    try:
        if group.runner == "chemqa":
            runner = build_runner_fn(
                runner_kind=group.runner,
                chemqa_root=chemqa_root,
                timeout_seconds=chemqa_timeout,
                config_path=config_path,
                slot_set=chemqa_slot_sets[group.id],
                review_rounds=review_rounds,
                rebuttal_rounds=rebuttal_rounds,
                model_profile=chemqa_model_profile,
                runtime_bundle_root=runtime_bundle_root,
                launch_workspace_root=output_root / "chemqa-launch",
                convergence_policy=chemqa_convergence_policy or ConvergencePolicy(timeout_seconds=chemqa_timeout),
                workspace_manager=workspace_manager,
                cancellation_token=cancellation_token,
                process_registry=process_registry,
            )
        else:
            runner = build_runner_fn(
                runner_kind=group.runner,
                agent_id=single_agent,
                timeout_seconds=single_timeout,
                config_path=config_path,
                runtime_bundle_root=runtime_bundle_root,
                configured_skills=tuple(experiment_specs[group.id].skill_allowlist or ()),
                skill_health_summary=skill_health_summary,
                convergence_policy=single_convergence_policy or ConvergencePolicy(timeout_seconds=single_timeout),
                timeout_retries=single_timeout_retries,
                timeout_retry_backoff_seconds=single_timeout_retry_backoff_seconds,
                benchmark_agent_thinking=single_agent_thinking,
                no_timeout=no_timeout,
                workspace_manager=workspace_manager,
                cancellation_token=cancellation_token,
                process_registry=process_registry,
                pypi_cutoff=pypi_cutoff,
            )
    except Exception as exc:
        if cancellation_token is not None and cancellation_token.is_cancelled:
            mark_cancelling()
            group_results = [
                build_cancelled_group_record_result(
                    group=group,
                    record=record,
                    build_error_group_record_result_fn=build_error_group_record_result_fn,
                )
                for record in records
            ]
            for entry in group_results:
                save_json_fn(
                    output_root / "per-record" / group.id / f"{slugify_fn(entry.record_id)}.json",
                    asdict(entry),
                )
                if progress_writer is not None:
                    progress_writer.record_cancelled(group.id, entry.record_id)
            if progress_writer is not None:
                progress_writer.group_cancelled(group.id)
            return group_results
        error_message = f"Failed to initialize runner for group `{group.id}`: {exc}"
        if progress_writer is not None:
            progress_writer.group_started(group.id)
            progress_writer.error(group_id=group.id, message=error_message)
        group_results = [
            build_error_group_record_result_fn(
                group=group,
                record=record,
                error_message=error_message,
            )
            for record in records
        ]
        for index, entry in enumerate(group_results, start=1):
            save_json_fn(output_root / "per-record" / group.id / f"{slugify_fn(entry.record_id)}.json", asdict(entry))
            if progress_writer is not None:
                progress_writer.record_started(group.id, str(entry.record_id), index=index)
                progress_writer.record_completed(group.id, str(entry.record_id), status="failed", score=0.0)
        if progress_writer is not None:
            progress_writer.group_completed(group.id, status="failed")
        return group_results

    group_results: list[GroupRecordResult] = []
    if progress_writer is not None:
        progress_writer.group_started(group.id)
    for index, record in enumerate(records, start=1):
        if cancellation_token is not None and cancellation_token.is_cancelled:
            mark_cancelling()
            entry = build_cancelled_group_record_result(
                group=group,
                record=record,
                build_error_group_record_result_fn=build_error_group_record_result_fn,
            )
            group_results.append(entry)
            save_json_fn(output_root / "per-record" / group.id / f"{slugify_fn(record.record_id)}.json", asdict(entry))
            if progress_writer is not None:
                progress_writer.record_cancelled(group.id, record.record_id)
            continue
        if progress_writer is not None:
            progress_writer.record_started(group.id, record.record_id, index=index)
        started = time.time()
        run_result: Any | None = None
        try:
            run_result = runner.run(record, group)
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
            ensure_compatible_runner_result(run_result, runner_kind=group.runner)
            axes = build_result_axes_from_runner(run_result)
            if run_result.should_score():
                answer_text = run_result.answer.full_response_text or run_result.answer.short_answer_text
                evaluation = evaluate_answer_fn(
                    record,
                    short_answer_text=run_result.answer.short_answer_text,
                    full_response_text=run_result.answer.full_response_text,
                    answer_text=answer_text,
                    judge=judge,
                )
                entry = GroupRecordResult(
                    **axes,
                    group_id=group.id,
                    group_label=group.label,
                    runner=group.runner,
                    websearch=group.websearch,
                    skills_enabled=group.skills_enabled,
                    record_id=record.record_id,
                    subset=classify_subset_fn(record),
                    dataset=record.dataset,
                    source_file=record.source_file,
                    eval_kind=record.eval_kind,
                    prompt=record.prompt,
                    reference_answer=record.reference_answer,
                    answer_text=answer_text,
                    evaluation=asdict(evaluation),
                    runner_meta=run_result.runner_meta,
                    raw=run_result.raw,
                    elapsed_seconds=time.time() - started,
                    error=None,
                    short_answer_text=run_result.answer.short_answer_text,
                    full_response_text=run_result.answer.full_response_text,
                )
            else:
                runner_meta_error = str((run_result.runner_meta or {}).get("error") or "").strip()
                failure = getattr(run_result, "failure", None)
                failure_message = str(getattr(failure, "message", "") or "").strip()
                error_message = (
                    runner_meta_error
                    or failure_message
                    or f"Record `{record.record_id}` finished in non-success terminal status `{status_label(run_result)}`"
                )
                entry = build_error_group_record_result_fn(
                    group=group,
                    record=record,
                    error_message=error_message,
                    elapsed_seconds=time.time() - started,
                    short_answer_text=run_result.answer.short_answer_text,
                    full_response_text=run_result.answer.full_response_text,
                    runner_meta=run_result.runner_meta,
                    raw=run_result.raw,
                )
                entry = GroupRecordResult(**{**asdict(entry), **axes, "error": error_message})
        except Exception as exc:
            elapsed = time.time() - started
            if isinstance(exc, BenchmarkCancelledError) or (
                cancellation_token is not None and cancellation_token.is_cancelled
            ):
                mark_cancelling()
                entry = build_cancelled_group_record_result(
                    group=group,
                    record=record,
                    build_error_group_record_result_fn=build_error_group_record_result_fn,
                    elapsed_seconds=elapsed,
                    runner_meta=(dict(run_result.runner_meta) if run_result is not None else None),
                    raw=(dict(run_result.raw) if run_result is not None else None),
                )
            elif run_result is not None:
                error_message = (
                    f"Record `{record.record_id}` judge/evaluator failed in group `{group.id}` after runner output: {exc}"
                )
                runner_meta = dict(getattr(run_result, "runner_meta", {}) or {})
                runner_meta["evaluation_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "stage": "judge_or_evaluator",
                }
                runner_meta["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                answer = getattr(run_result, "answer", None)
                entry = build_error_group_record_result_fn(
                    group=group,
                    record=record,
                    error_message=error_message,
                    elapsed_seconds=elapsed,
                    short_answer_text=str(getattr(answer, "short_answer_text", "") or ""),
                    full_response_text=str(getattr(answer, "full_response_text", "") or ""),
                    runner_meta=runner_meta,
                    raw=getattr(run_result, "raw", {}) if isinstance(getattr(run_result, "raw", None), dict) else {},
                )
            else:
                error_message = f"Record `{record.record_id}` runner failed in group `{group.id}`: {exc}"
                entry = build_error_group_record_result_fn(
                    group=group,
                    record=record,
                    error_message=error_message,
                    elapsed_seconds=elapsed,
                    runner_meta={
                        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                    },
                )
            if progress_writer is not None:
                progress_writer.error(group_id=group.id, record_id=record.record_id, message=str(exc))
        group_results.append(entry)
        save_json_fn(output_root / "per-record" / group.id / f"{slugify_fn(record.record_id)}.json", asdict(entry))
        if progress_writer is not None:
            if entry.run_lifecycle_status == "cancelled":
                progress_writer.record_cancelled(group.id, record.record_id)
            else:
                evaluation = entry.evaluation if isinstance(entry.evaluation, dict) else {}
                score = evaluation.get("normalized_score", evaluation.get("score"))
                progress_writer.record_completed(
                    group.id,
                    record.record_id,
                    status=str(entry.run_lifecycle_status or "completed"),
                    score=float(score) if isinstance(score, (int, float)) else None,
                )
    if progress_writer is not None:
        if any(item.run_lifecycle_status == "cancelled" for item in group_results):
            progress_writer.group_cancelled(group.id)
        else:
            group_status = "completed" if all(item.run_lifecycle_status == "completed" for item in group_results) else "completed_with_errors"
            progress_writer.group_completed(group.id, status=group_status)
    return group_results
