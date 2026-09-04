from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from benchmarking.analysis.launcher import analysis_paths
from benchmarking.core.datasets import BenchmarkRecord
from benchmarking.core.reporting import GroupRecordResult
from benchmarking.runtime.vgb_bridge import (
    ReleaseConfig,
    VerifierGroundedRuntimeError,
    load_public_reference_answers,
    load_release_config,
)
from benchmarking.workflow.errors import BenchmarkError
from benchmarking.workflow.experiments import EXPERIMENT_GROUPS


def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def slugify(value: str, *, limit: int = 64) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    cleaned = cleaned or "item"
    if len(cleaned) <= limit:
        return cleaned
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[: limit - 9]}-{digest}".strip("-")


LEGACY_SUMMARY_CSV_FILENAMES = (
    "summary_by_group.csv",
    "summary_by_group_and_subset.csv",
)


def remove_legacy_summary_csvs(output_root: Path) -> None:
    for filename in LEGACY_SUMMARY_CSV_FILENAMES:
        path = output_root / filename
        if path.exists():
            path.unlink()


def count_per_record_outputs(output_root: Path, *, group_ids: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    per_record_root = output_root / "per-record"
    for group_id in group_ids:
        group_dir = per_record_root / group_id
        counts[group_id] = len(list(group_dir.glob("*.json"))) if group_dir.is_dir() else 0
    return counts


def pending_records_for_group(
    records: list[BenchmarkRecord],
    *,
    output_root: Path,
    group_id: str,
    merge_existing_per_record: bool,
) -> list[BenchmarkRecord]:
    if not merge_existing_per_record:
        return list(records)
    group_root = output_root / "per-record" / group_id
    return [record for record in records if not (group_root / f"{slugify(record.record_id)}.json").is_file()]


def load_group_record_result(path: Path) -> GroupRecordResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "schema_version" not in payload:
        runner_meta = payload.get("runner_meta") or {}
        raw = payload.get("raw") or {}
        evaluation = payload.get("evaluation") or {}
        primary_metric = str(evaluation.get("primary_metric") or "")
        fallback_used = bool(runner_meta.get("fallback_used"))
        fallback_source = str(runner_meta.get("fallback_source") or "")
        run_status_present = isinstance(raw.get("run_status"), dict)
        scored = bool(runner_meta.get("scored", primary_metric != "execution_error"))
        explicit_evaluable = runner_meta.get("evaluable")
        explicit_reliability = str(runner_meta.get("answer_reliability") or "").strip()
        explicit_recovery_mode = str(runner_meta.get("recovery_mode") or "").strip()
        explicit_degraded = runner_meta.get("degraded_execution")
        evaluable = bool(explicit_evaluable) if explicit_evaluable is not None else scored
        if fallback_used:
            run_lifecycle_status = "completed" if scored else "failed"
            protocol_completion_status = "failed" if run_status_present else "missing"
            recovery_mode = explicit_recovery_mode or fallback_source or "none"
            if recovery_mode == "run-status-final-answer-preview":
                answer_availability = "preview_only"
                default_reliability = "low_confidence_recovered"
            else:
                answer_availability = "recovered_candidate"
                default_reliability = "high_confidence_recovered"
            answer_reliability = explicit_reliability or default_reliability
            degraded_execution = bool(explicit_degraded) if explicit_degraded is not None else True
        elif scored:
            run_lifecycle_status = "completed"
            protocol_completion_status = "completed"
            answer_availability = "native_final"
            answer_reliability = explicit_reliability or "native"
            recovery_mode = explicit_recovery_mode or "none"
            degraded_execution = bool(explicit_degraded) if explicit_degraded is not None else False
        else:
            run_lifecycle_status = "failed"
            protocol_completion_status = "failed" if run_status_present else "missing"
            answer_availability = "missing"
            answer_reliability = explicit_reliability or "none"
            evaluable = False if explicit_evaluable is None else bool(explicit_evaluable)
            recovery_mode = explicit_recovery_mode or "none"
            degraded_execution = bool(explicit_degraded) if explicit_degraded is not None else True
        payload = {
            **payload,
            # Upconvert schema-v1 per-record payloads so historical outputs remain loadable.
            "schema_version": 3,
            "run_lifecycle_status": run_lifecycle_status,
            "protocol_completion_status": protocol_completion_status,
            "protocol_acceptance_status": None,
            "answer_availability": answer_availability,
            "answer_reliability": answer_reliability,
            "evaluable": evaluable,
            "scored": scored,
            "recovery_mode": recovery_mode,
            "degraded_execution": degraded_execution,
            "execution_error_kind": None if scored else "execution_error",
        }
    if "skills_enabled" not in payload:
        group = EXPERIMENT_GROUPS.get(str(payload.get("group_id") or ""))
        payload["skills_enabled"] = bool(getattr(group, "skills_enabled", False))
    return GroupRecordResult(**payload)



def resolve_aggregate_group_ids(
    selected_group_ids: list[str],
    *,
    output_root: Path,
    merge_existing_per_record: bool,
) -> list[str]:
    if not merge_existing_per_record:
        return list(selected_group_ids)
    present = set(selected_group_ids)
    per_record_root = output_root / "per-record"
    for group_id in EXPERIMENT_GROUPS:
        group_dir = per_record_root / group_id
        if group_dir.is_dir() and any(group_dir.glob("*.json")):
            present.add(group_id)
    return [group_id for group_id in EXPERIMENT_GROUPS if group_id in present]



def load_results_from_output_root(output_root: Path, *, group_ids: list[str]) -> list[GroupRecordResult]:
    results: list[GroupRecordResult] = []
    for group_id in group_ids:
        group_dir = output_root / "per-record" / group_id
        if not group_dir.is_dir():
            continue
        for path in sorted(group_dir.glob("*.json")):
            results.append(load_group_record_result(path))
    return results


def apply_verifier_grounded_reporting_references(
    results: list[GroupRecordResult],
    *,
    release_config: ReleaseConfig | None = None,
) -> list[GroupRecordResult]:
    property_results = [
        item
        for item in results
        if str(getattr(item, "dataset", "")).startswith("verifier_grounded_property_calculation")
    ]
    if not property_results:
        return results
    try:
        config = release_config or load_release_config()
    except VerifierGroundedRuntimeError as exc:
        raise BenchmarkError(f"Unable to load public property-calculation gold: {exc}") from exc
    dataset_tracks = {
        str(track_config.get("dataset") or ""): track
        for track, track_config in config.tracks.items()
        if track.startswith("property_calculation") and isinstance(track_config, dict)
    }
    for dataset, track in dataset_tracks.items():
        track_results = [
            item for item in property_results if str(getattr(item, "dataset", "")) == dataset
        ]
        if not track_results:
            continue
        try:
            samples = load_public_reference_answers(
                track,
                release_config=config,
            )
        except VerifierGroundedRuntimeError as exc:
            raise BenchmarkError(f"Unable to load public property-calculation gold: {exc}") from exc

        references: dict[str, str] = {}
        for sample in samples:
            task_id = str(sample.get("task_id") or "").strip()
            answer = {key: value for key, value in sample.items() if key != "task_id"}
            if task_id and answer:
                references[task_id] = json.dumps(
                    answer,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
        missing = sorted(
            {
                str(getattr(item, "record_id", "") or "")
                for item in track_results
                if str(getattr(item, "record_id", "") or "") not in references
            }
        )
        if missing:
            raise BenchmarkError(
                "Verifier-grounded property-calculation results are missing public gold for: "
                + ", ".join(missing)
            )
        for item in track_results:
            item.reference_answer = references[str(getattr(item, "record_id", "") or "")]

    unmatched_datasets = sorted(
        {
            str(getattr(item, "dataset", "") or "")
            for item in property_results
            if str(getattr(item, "dataset", "") or "") not in dataset_tracks
        }
    )
    if unmatched_datasets:
        raise BenchmarkError(
            "Verifier-grounded property-calculation datasets are missing pinned track metadata: "
            + ", ".join(unmatched_datasets)
        )
    return results


def write_wave_status(
    output_root: Path,
    *,
    wave_index: int,
    wave_group_ids: list[str],
    status: str,
    started_at: str,
    completed_at: str | None = None,
    per_record_counts: dict[str, int] | None = None,
    inter_wave_delay_seconds: int | None = None,
) -> None:
    payload: dict[str, Any] = {
        "wave_index": wave_index,
        "groups": wave_group_ids,
        "status": status,
        "started_at": started_at,
    }
    if completed_at is not None:
        payload["completed_at"] = completed_at
    if per_record_counts is not None:
        payload["per_record_counts"] = per_record_counts
    if inter_wave_delay_seconds is not None:
        payload["inter_wave_delay_seconds"] = inter_wave_delay_seconds
    save_json(output_root / "waves" / f"wave-{wave_index:02d}.json", payload)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def automated_evaluation_launch_failed(output_root: Path, exc: Exception) -> dict[str, Any]:
    analysis_dir = output_root / "analysis"
    status_path = analysis_dir / "status.json"
    return {
        "status": "launch_failed",
        "error": f"{type(exc).__name__}: {exc}",
        "analysis_dir": str(analysis_dir),
        "status_path": str(status_path),
        "input_bundle_path": str(analysis_dir / "input-bundle.json"),
        "events_path": str(analysis_dir / "codex-events.jsonl"),
        "report_path": str(analysis_dir / "report.json"),
        "markdown_report_path": str(analysis_dir / "report.md"),
    }


def automated_evaluation_skipped(output_root: Path) -> dict[str, Any]:
    return {
        "status": "skipped",
        "reason": "disabled_by_cli",
        "output_root": str(output_root),
        **analysis_paths(output_root),
    }
