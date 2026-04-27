from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class GroupRecordResult:
    schema_version: int
    group_id: str
    group_label: str
    runner: str
    websearch: bool
    record_id: str
    subset: str
    dataset: str
    source_file: str
    eval_kind: str
    prompt: str
    reference_answer: str
    answer_text: str
    evaluation: dict[str, Any]
    runner_meta: dict[str, Any]
    raw: dict[str, Any]
    elapsed_seconds: float
    run_lifecycle_status: str
    protocol_completion_status: str
    protocol_acceptance_status: str | None
    answer_availability: str
    answer_reliability: str
    evaluable: bool
    scored: bool
    recovery_mode: str
    degraded_execution: bool
    execution_error_kind: str | None = None
    error: str | None = None
    short_answer_text: str = ""
    full_response_text: str = ""


def average_optional_metric(items: list[GroupRecordResult], key: str) -> float | None:
    values: list[float] = []
    for item in items:
        details = item.evaluation.get("details") or {}
        value = details.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def aggregate_bucket(items: list[GroupRecordResult]) -> dict[str, Any]:
    return {
        "count": len(items),
        "pass_count": sum(1 for item in items if item.evaluation["passed"]),
        "avg_score": sum(float(item.evaluation["score"]) for item in items) / len(items),
        "avg_normalized_score": sum(float(item.evaluation["normalized_score"]) for item in items) / len(items),
        "avg_elapsed_seconds": sum(float(item.elapsed_seconds) for item in items) / len(items),
        "avg_answer_accuracy": average_optional_metric(items, "answer_accuracy"),
        "avg_rpf": average_optional_metric(items, "rpf"),
    }


def aggregate_results(results: list[GroupRecordResult]) -> dict[str, Any]:
    grouped: dict[str, list[GroupRecordResult]] = {}
    for item in results:
        grouped.setdefault(item.group_id, []).append(item)

    summary_groups: dict[str, Any] = {}
    summary_group_subset: dict[str, dict[str, Any]] = {}
    for group_id, items in grouped.items():
        by_eval_kind: dict[str, list[GroupRecordResult]] = {}
        by_subset: dict[str, list[GroupRecordResult]] = {}
        for item in items:
            by_eval_kind.setdefault(item.eval_kind, []).append(item)
            by_subset.setdefault(item.subset, []).append(item)
        bucket = aggregate_bucket(items)
        summary_groups[group_id] = {
            "group_label": items[0].group_label,
            "runner": items[0].runner,
            "websearch": items[0].websearch,
            **bucket,
            "by_eval_kind": {
                eval_kind: {
                    key: value
                    for key, value in aggregate_bucket(eval_items).items()
                }
                for eval_kind, eval_items in by_eval_kind.items()
            },
            "by_subset": {
                subset: {
                    key: value
                    for key, value in aggregate_bucket(subset_items).items()
                }
                for subset, subset_items in by_subset.items()
            },
        }
        for subset, subset_items in by_subset.items():
            summary_group_subset[f"{group_id}::{subset}"] = {
                "group_id": group_id,
                "group_label": items[0].group_label,
                "runner": items[0].runner,
                "websearch": items[0].websearch,
                "subset": subset,
                **aggregate_bucket(subset_items),
            }

    return {
        "group_order": list(grouped.keys()),
        "groups": summary_groups,
        "group_subset": summary_group_subset,
    }


def build_error_group_record_result(
    *,
    group: Any,
    record: Any,
    error_message: str,
    elapsed_seconds: float = 0.0,
    answer_text: str = "",
    short_answer_text: str = "",
    full_response_text: str = "",
    runner_meta: dict[str, Any] | None = None,
    raw: dict[str, Any] | None = None,
    classify_subset_fn: Callable[[Any], str],
    normalize_answer_tracks_fn: Callable[..., tuple[str, str]],
    build_execution_error_evaluation_fn: Callable[..., Any],
    deep_copy_jsonish_fn: Callable[[Any], Any],
) -> GroupRecordResult:
    evaluation = build_execution_error_evaluation_fn(record, error_message=error_message)
    if is_dataclass(evaluation):
        evaluation_payload = asdict(evaluation)
    elif isinstance(evaluation, dict):
        evaluation_payload = deep_copy_jsonish_fn(evaluation)
    else:
        raise TypeError("build_execution_error_evaluation_fn must return a dataclass or dict payload")

    meta = deep_copy_jsonish_fn(runner_meta or {})
    meta.setdefault("error", error_message)
    payload = deep_copy_jsonish_fn(raw or {"error": error_message})
    short_text, full_text = normalize_answer_tracks_fn(
        short_answer_text=short_answer_text,
        full_response_text=full_response_text,
    )
    compatible_answer_text = answer_text or full_text or short_text
    return GroupRecordResult(
        schema_version=2,
        group_id=str(getattr(group, "id", "") or ""),
        group_label=str(getattr(group, "label", "") or ""),
        runner=str(getattr(group, "runner", "") or ""),
        websearch=bool(getattr(group, "websearch", False)),
        record_id=str(getattr(record, "record_id", "") or ""),
        subset=classify_subset_fn(record),
        dataset=str(getattr(record, "dataset", "") or ""),
        source_file=str(getattr(record, "source_file", "") or ""),
        eval_kind=str(getattr(record, "eval_kind", "") or ""),
        prompt=str(getattr(record, "prompt", "") or ""),
        reference_answer=str(getattr(record, "reference_answer", "") or ""),
        answer_text=compatible_answer_text,
        evaluation=evaluation_payload,
        runner_meta=meta,
        raw=payload,
        elapsed_seconds=elapsed_seconds,
        run_lifecycle_status="failed",
        protocol_completion_status="missing",
        protocol_acceptance_status=None,
        answer_availability="missing",
        answer_reliability="none",
        evaluable=False,
        scored=False,
        recovery_mode="none",
        degraded_execution=False,
        execution_error_kind="execution_error",
        error=error_message,
        short_answer_text=short_text,
        full_response_text=full_text,
    )


def materialize_group_failure_results(
    *,
    group: Any,
    records: list[Any],
    output_root: Path,
    error_message: str,
    save_json_fn: Callable[[Path, Any], None],
    slugify_fn: Callable[..., str],
    classify_subset_fn: Callable[[Any], str],
    normalize_answer_tracks_fn: Callable[..., tuple[str, str]],
    build_execution_error_evaluation_fn: Callable[..., Any],
    deep_copy_jsonish_fn: Callable[[Any], Any],
) -> list[GroupRecordResult]:
    group_results = [
        build_error_group_record_result(
            group=group,
            record=record,
            error_message=error_message,
            classify_subset_fn=classify_subset_fn,
            normalize_answer_tracks_fn=normalize_answer_tracks_fn,
            build_execution_error_evaluation_fn=build_execution_error_evaluation_fn,
            deep_copy_jsonish_fn=deep_copy_jsonish_fn,
        )
        for record in records
    ]
    for entry in group_results:
        save_json_fn(output_root / "per-record" / str(getattr(group, "id", "")) / f"{slugify_fn(entry.record_id)}.json", asdict(entry))
    return group_results
