from __future__ import annotations

import json
import math
import mimetypes
import os
from pathlib import Path
from typing import Any

from benchmarking.dashboard.annotations import AnnotationStore
from benchmarking.dashboard.progress import load_progress
from benchmarking.runtime import paths as runtime_paths

RUN_DISCOVERY_IGNORED_DIRECTORIES = {"legacy-workspace-archives"}
REFERENCE_PLACEHOLDER_PREFIX = "No reference answer is exposed"


class DashboardError(RuntimeError):
    pass


class RunNotFoundError(DashboardError):
    pass


class RecordNotFoundError(DashboardError):
    pass


class AssetAccessError(DashboardError):
    pass


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_load_json(path: Path) -> Any:
    try:
        return _load_json(path)
    except Exception:
        return {}


def _slug_variants(value: str) -> set[str]:
    stripped = str(value or "").strip()
    return {stripped, stripped.replace("_", "-"), stripped.replace("-", "_")}


def _dataset_from_source_file(result: dict[str, Any]) -> str:
    source_file = str(result.get("source_file") or "").strip()
    if not source_file:
        return ""
    source_path = Path(source_file)
    if source_path.parent.name != "data":
        return ""
    return source_path.parent.parent.name


def _format_number(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return ""
    return f"{float(value):.4g}"


def _dashboard_dataset_subset(result: dict[str, Any]) -> tuple[str, str]:
    dataset = str(result.get("dataset") or "")
    source_dataset = _dataset_from_source_file(result)
    if source_dataset and source_dataset != dataset:
        dataset = source_dataset
    subset = str(result.get("subset") or "")
    if dataset.startswith("verifier_grounded_"):
        # Keep historical dataset files readable while exposing the current
        # release track names in dashboard facets.
        record_id = str(result.get("record_id") or "")
        if record_id.startswith("property_calculation_advanced_"):
            subset = "property_calculation_advanced"
        elif record_id.startswith("property_calculation_basic_"):
            subset = "property_calculation_basic"
        return "vgb", subset or dataset
    return dataset, subset


def _score_label(result: dict[str, Any]) -> str:
    evaluation = result.get("evaluation") if isinstance(result.get("evaluation"), dict) else {}
    status = {key: result.get(key) for key in ("evaluable", "scored")}
    if status.get("evaluable") is False:
        return "不可评测"
    if status.get("scored") is False:
        return "未评分"
    primary_metric = str(evaluation.get("primary_metric") or "")
    if primary_metric == "verifier_score":
        return f"Verifier {_format_number(evaluation.get('normalized_score', evaluation.get('score')))}".strip()
    if primary_metric == "rubric_points":
        return f"{_format_number(evaluation.get('score'))}/{_format_number(evaluation.get('max_score'))}"
    details = evaluation.get("details") if isinstance(evaluation.get("details"), dict) else {}
    if primary_metric == "answer_accuracy" and "rpf" in details:
        return f"{'正确' if evaluation.get('passed') is True else '错误'}; RPF {_format_number(float(details.get('rpf')) * 100)}%"
    passed = evaluation.get("passed")
    if passed is True:
        return "正确"
    if passed is False:
        return "错误"
    return "已评分" if result.get("scored") else "未知"


def _outcome(result: dict[str, Any]) -> str:
    evaluation = result.get("evaluation") if isinstance(result.get("evaluation"), dict) else {}
    if result.get("evaluable") is False or result.get("scored") is False:
        return "not_scored"
    if evaluation.get("passed") is True:
        return "passed"
    if evaluation.get("passed") is False:
        return "failed"
    return "scored"


def _verifier_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    evaluation = result.get("evaluation") if isinstance(result.get("evaluation"), dict) else {}
    if evaluation.get("primary_metric") != "verifier_score" and result.get("eval_kind") != "verifier_grounded":
        return None
    details = evaluation.get("details") if isinstance(evaluation.get("details"), dict) else {}
    return {
        "status": details.get("status"),
        "failure_type": details.get("failure_type"),
        "message": details.get("message"),
        "canonical_smiles": details.get("canonical_smiles"),
        "properties": details.get("properties") or {},
        "constraint_scores": details.get("constraint_scores") or [],
        "versions": details.get("versions") or {},
    }


def _is_reference_placeholder(value: Any) -> bool:
    return str(value or "").strip().startswith(REFERENCE_PLACEHOLDER_PREFIX)


def _verifier_gold_reference(result: dict[str, Any]) -> str:
    evaluation = result.get("evaluation") if isinstance(result.get("evaluation"), dict) else {}
    details = evaluation.get("details") if isinstance(evaluation.get("details"), dict) else {}
    properties = details.get("properties") if isinstance(details.get("properties"), dict) else {}
    gold_answers = properties.get("gold_answers")
    if not isinstance(gold_answers, dict) or not gold_answers:
        return ""

    answers: list[dict[str, Any]] = []
    for property_name, raw_answer in gold_answers.items():
        if not isinstance(raw_answer, dict) or "value" not in raw_answer:
            return json.dumps(gold_answers, ensure_ascii=False, separators=(",", ":"))
        answer = {"property": property_name, "value": raw_answer["value"]}
        if "unit" in raw_answer:
            answer["unit"] = raw_answer["unit"]
        answers.append(answer)

    if len(answers) == 1:
        answer = {"answer": answers[0]["value"]}
        if "unit" in answers[0]:
            answer["unit"] = answers[0]["unit"]
        return json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
    return json.dumps({"answers": answers}, ensure_ascii=False, separators=(",", ":"))


def _resolved_reference_answer(results: list[dict[str, Any]]) -> str:
    for result in results:
        answer = str(result.get("reference_answer") or "")
        if answer and not _is_reference_placeholder(answer):
            return answer
    for result in results:
        answer = _verifier_gold_reference(result)
        if answer:
            return answer
    return str(results[0].get("reference_answer") or "") if results else ""


def _group_sort_key(group: dict[str, Any]) -> tuple[int, str]:
    group_id = str(group.get("group_id") or "")
    preferred = {
        "single_llm_skills_on": 0,
        "single_llm_skills_off": 1,
        "chemqa_skills_on": 2,
    }
    return (preferred.get(group_id, 100), group_id)


def _audit_int(audit: dict[str, Any], key: str) -> int:
    value = audit.get(key)
    return int(value) if isinstance(value, (int, float)) else 0


def _agent_duration_seconds(result: dict[str, Any]) -> float | None:
    runner_meta = result.get("runner_meta") if isinstance(result.get("runner_meta"), dict) else {}
    duration_ms = runner_meta.get("durationMs")
    if (
        isinstance(duration_ms, (int, float))
        and not isinstance(duration_ms, bool)
        and math.isfinite(float(duration_ms))
        and duration_ms >= 0
    ):
        return float(duration_ms) / 1000
    return None


def _diagnostics_payload(result: dict[str, Any], skill_audit: dict[str, Any]) -> dict[str, Any]:
    skills_enabled = bool(result.get("skills_enabled", False))
    runner_meta = result.get("runner_meta") if isinstance(result.get("runner_meta"), dict) else {}
    execution_error = runner_meta.get("execution_error")
    legacy_skill_calls = _audit_int(skill_audit, "skill_tool_call_count")
    legacy_skill_failures = _audit_int(skill_audit, "skill_tool_failure_count")
    has_exec_call_count = isinstance(skill_audit.get("exec_tool_call_count"), (int, float))
    has_exec_failure_count = isinstance(skill_audit.get("exec_tool_failure_count"), (int, float))
    exec_call_count = _audit_int(skill_audit, "exec_tool_call_count")
    exec_failure_count = _audit_int(skill_audit, "exec_tool_failure_count")
    if not has_exec_call_count:
        exec_call_count = legacy_skill_calls
    if not has_exec_failure_count:
        exec_failure_count = legacy_skill_failures
    return {
        "agent_duration_seconds": _agent_duration_seconds(result),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "openclaw_tool_call_count": skill_audit.get("openclaw_tool_call_count", skill_audit.get("tool_call_count")),
        "exec_tool_call_count": exec_call_count,
        "exec_tool_failure_count": exec_failure_count,
        "skill_tool_call_count": legacy_skill_calls if skills_enabled else 0,
        "skill_tool_failure_count": legacy_skill_failures if skills_enabled else 0,
        "coverage_checklist_present": skill_audit.get("coverage_checklist_present"),
        "execution_error": dict(execution_error) if isinstance(execution_error, dict) else None,
    }


def _workspace_isolation_payload(runner_meta: dict[str, Any]) -> dict[str, Any]:
    isolation = runner_meta.get("workspace_isolation")
    if not isinstance(isolation, dict):
        return {}
    if "adjudication" in isolation:
        return {
            key: isolation.get(key)
            for key in (
                "policy_digest",
                "audit_execution_status",
                "boundary_status",
                "contamination_status",
                "adjudication",
                "findings",
                "cleanup",
            )
        }
    legacy_status = str(isolation.get("audit_status") or "").strip()
    if not legacy_status:
        return {}
    return {
        "legacy_schema": True,
        "audit_execution_status": "unavailable" if legacy_status == "unavailable" else "complete",
        "boundary_status": "clean" if legacy_status == "clean" else "unknown",
        "contamination_status": "clear" if legacy_status == "clean" else "indeterminate",
        "adjudication": "scoreable" if legacy_status == "clean" else "non_evaluable",
        "findings": isolation.get("findings") or [],
        "cleanup": isolation.get("cleanup") or {},
    }


class BenchmarkDashboard:
    def __init__(
        self,
        *,
        run_roots: list[str | Path] | tuple[str | Path, ...] | None = None,
        annotation_store: AnnotationStore | None = None,
    ) -> None:
        self.run_roots = [
            Path(root).expanduser().resolve()
            for root in (run_roots or [runtime_paths.project_state_root / "benchmark-runs"])
        ]
        self.annotation_store = annotation_store or AnnotationStore(runtime_paths.project_state_root / "benchmark-dashboard" / "dashboard.sqlite")

    def _candidate_run_dirs(self) -> list[Path]:
        candidates: list[Path] = []
        for root in self.run_roots:
            if not root.exists():
                continue
            if self._looks_like_run(root):
                candidates.append(root)
                continue
            for current, directories, _files in os.walk(root):
                directories[:] = [
                    name for name in directories if name not in RUN_DISCOVERY_IGNORED_DIRECTORIES
                ]
                path = Path(current)
                if path == root or not self._looks_like_run(path):
                    continue
                candidates.append(path)
                directories.clear()
        return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)

    @staticmethod
    def _looks_like_run(path: Path) -> bool:
        return any((path / name).exists() for name in ("results.json", "runtime-manifest.json", "per-record", "waves", "progress"))

    def _run_dir(self, run_id: str) -> Path:
        matches = [path for path in self._candidate_run_dirs() if path.name == run_id]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RunNotFoundError(f"Ambiguous benchmark run id: {run_id}")
        raise RunNotFoundError(f"Unknown benchmark run: {run_id}")

    def _load_results(self, run_root: Path) -> list[dict[str, Any]]:
        results: dict[tuple[str, str], dict[str, Any]] = {}
        unkeyed_results: list[dict[str, Any]] = []

        def merge_result(result: dict[str, Any], *, fallback_group_id: str = "") -> None:
            group_id = str(result.get("group_id") or fallback_group_id)
            record_id = str(result.get("record_id") or "")
            if group_id and record_id:
                existing = results.get((group_id, record_id))
                if (
                    existing
                    and _is_reference_placeholder(result.get("reference_answer"))
                    and existing.get("reference_answer")
                    and not _is_reference_placeholder(existing.get("reference_answer"))
                ):
                    result = {**result, "reference_answer": existing["reference_answer"]}
                results[(group_id, record_id)] = result
            else:
                unkeyed_results.append(result)

        results_path = run_root / "results.json"
        if results_path.is_file():
            payload = _safe_load_json(results_path)
            aggregate_results = payload.get("results") if isinstance(payload, dict) else []
            if isinstance(aggregate_results, list):
                for item in aggregate_results:
                    if isinstance(item, dict):
                        merge_result(item)

        per_record_root = run_root / "per-record"
        if per_record_root.is_dir():
            for path in sorted(per_record_root.glob("*/*.json")):
                loaded = _safe_load_json(path)
                if isinstance(loaded, dict):
                    # Per-record files are written immediately and can be newer
                    # than the aggregate snapshot while a run is still active.
                    merge_result(loaded, fallback_group_id=path.parent.name)
        return [*results.values(), *unkeyed_results]

    def _run_payload(self, run_root: Path) -> dict[str, Any]:
        payload = _safe_load_json(run_root / "results.json") if (run_root / "results.json").is_file() else {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _group_ids(run_payload: dict[str, Any], results: list[dict[str, Any]], run_root: Path) -> list[str]:
        groups_payload = run_payload.get("groups") if isinstance(run_payload.get("groups"), list) else []
        group_ids = [str(group.get("id") or "") for group in groups_payload if isinstance(group, dict) and group.get("id")]
        if not group_ids:
            group_ids = sorted({str(result.get("group_id") or "") for result in results if result.get("group_id")})
        if not group_ids and (run_root / "per-record").is_dir():
            group_ids = sorted(path.name for path in (run_root / "per-record").iterdir() if path.is_dir())
        return group_ids

    def list_runs(self, *, include_hidden: bool = False) -> list[dict[str, Any]]:
        metadata = self.annotation_store.list_run_metadata()
        runs: list[dict[str, Any]] = []
        for run_root in self._candidate_run_dirs():
            run_id = run_root.name
            meta = metadata.get(run_id, {})
            if meta.get("hidden") and not include_hidden:
                continue
            payload = self._run_payload(run_root)
            results = self._load_results(run_root)
            group_ids = self._group_ids(payload, results, run_root)
            record_ids = sorted({str(result.get("record_id") or "") for result in results if result.get("record_id")})
            display_pairs = [_dashboard_dataset_subset(result) for result in results]
            datasets = sorted({dataset for dataset, _subset in display_pairs if dataset})
            subsets = sorted({subset for _dataset, subset in display_pairs if subset})
            total = int(payload.get("records") or len(record_ids)) * max(1, len(group_ids))
            progress = load_progress(run_root, expected_total=total, group_ids=group_ids)
            summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
            runs.append(
                {
                    "run_id": run_id,
                    "alias": meta.get("alias", ""),
                    "favorite": bool(meta.get("favorite", False)),
                    "hidden": bool(meta.get("hidden", False)),
                    "path": str(run_root),
                    "generated_at": payload.get("generated_at", ""),
                    "updated_at": progress.get("updated_at") or payload.get("generated_at", ""),
                    "status": progress.get("status") or ("completed" if (run_root / "results.json").is_file() else "pending"),
                    "record_count": int(payload.get("records") or len(record_ids)),
                    "group_count": len(group_ids),
                    "dataset_files": payload.get("dataset_files") or [],
                    "datasets": datasets,
                    "subsets": subsets,
                    "progress": progress,
                    "summary": summary,
                }
            )
        # Keep discovery's newest-first order within each group while pinning
        # favorited runs ahead of all other runs.
        runs.sort(key=lambda run: not run["favorite"])
        return runs

    def get_run(self, run_id: str) -> dict[str, Any]:
        run_root = self._run_dir(run_id)
        payload = self._run_payload(run_root)
        results = self._load_results(run_root)
        group_ids = self._group_ids(payload, results, run_root)
        record_ids = sorted({str(result.get("record_id") or "") for result in results if result.get("record_id")})
        total = int(payload.get("records") or len(record_ids)) * max(1, len(group_ids))
        meta = self.annotation_store.get_run_metadata(run_id) or {}
        return {
            "run_id": run_id,
            "alias": meta.get("alias", ""),
            "favorite": bool(meta.get("favorite", False)),
            "hidden": bool(meta.get("hidden", False)),
            "path": str(run_root),
            "payload": payload,
            "progress": load_progress(run_root, expected_total=total, group_ids=group_ids),
            "annotations": self.annotation_store.list_annotations(run_id=run_id),
        }

    def list_records(self, run_id: str) -> list[dict[str, Any]]:
        run_root = self._run_dir(run_id)
        results = self._load_results(run_root)
        annotations = self.annotation_store.list_annotations(run_id=run_id)
        annotation_count: dict[str, int] = {}
        for annotation in annotations:
            annotation_count[str(annotation["record_id"])] = annotation_count.get(str(annotation["record_id"]), 0) + 1
        by_record: dict[str, list[dict[str, Any]]] = {}
        for result in results:
            by_record.setdefault(str(result.get("record_id") or ""), []).append(result)
        records: list[dict[str, Any]] = []
        for record_id, items in sorted(by_record.items()):
            first = items[0]
            dataset, subset = _dashboard_dataset_subset(first)
            records.append(
                {
                    "record_id": record_id,
                    "dataset": dataset,
                    "subset": subset,
                    "eval_kind": first.get("eval_kind", ""),
                    "prompt_preview": str(first.get("prompt") or "")[:320],
                    "group_results": [
                        {
                            "group_id": item.get("group_id", ""),
                            "score_label": _score_label(item),
                            "outcome": _outcome(item),
                            "agent_duration_seconds": _agent_duration_seconds(item),
                            "elapsed_seconds": item.get("elapsed_seconds"),
                        }
                        for item in sorted(items, key=_group_sort_key)
                    ],
                    "annotation_count": annotation_count.get(record_id, 0),
                }
            )
        return records

    def _find_record_results(self, run_root: Path, record_id: str) -> list[dict[str, Any]]:
        variants = _slug_variants(record_id)
        results = [
            result
            for result in self._load_results(run_root)
            if str(result.get("record_id") or "") in variants
        ]
        if results:
            return results
        per_record_root = run_root / "per-record"
        found: list[dict[str, Any]] = []
        if per_record_root.is_dir():
            for path in sorted(per_record_root.glob("*/*.json")):
                if path.stem in variants:
                    loaded = _safe_load_json(path)
                    if isinstance(loaded, dict):
                        found.append(loaded)
        return found

    def _runtime_bundle(self, run_root: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
        for result in results:
            runner_meta = result.get("runner_meta") if isinstance(result.get("runner_meta"), dict) else {}
            bundle = runner_meta.get("runtime_bundle") if isinstance(runner_meta.get("runtime_bundle"), dict) else {}
            if bundle:
                return bundle
        return {}

    def _question_markdown(self, run_root: Path, results: list[dict[str, Any]]) -> tuple[str, str]:
        bundle = self._runtime_bundle(run_root, results)
        question_path_raw = str(bundle.get("question_markdown") or "").strip()
        if question_path_raw:
            question_path = Path(question_path_raw).expanduser().resolve()
            if self._is_within_run(run_root, question_path) and question_path.is_file():
                return question_path.read_text(encoding="utf-8", errors="replace"), str(question_path.relative_to(run_root.resolve()))
        first = results[0] if results else {}
        return str(first.get("prompt") or ""), ""

    def _assets(self, run_root: Path, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        bundle = self._runtime_bundle(run_root, results)
        assets: list[dict[str, Any]] = []
        for raw in bundle.get("image_files") or []:
            path = Path(str(raw)).expanduser().resolve()
            if not self._is_within_run(run_root, path) or not path.is_file():
                continue
            relative = str(path.relative_to(run_root.resolve()))
            assets.append(
                {
                    "relative_path": relative,
                    "url": f"/api/runs/{run_root.name}/assets/{relative}",
                    "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                }
            )
        return assets

    def get_record(self, run_id: str, record_id: str) -> dict[str, Any]:
        run_root = self._run_dir(run_id)
        results = self._find_record_results(run_root, record_id)
        if not results:
            raise RecordNotFoundError(f"Unknown record `{record_id}` in run `{run_id}`")
        results = sorted(results, key=_group_sort_key)
        first = results[0]
        reference_answer = _resolved_reference_answer(results)
        dataset, subset = _dashboard_dataset_subset(first)
        question_markdown, question_source = self._question_markdown(run_root, results)
        groups: list[dict[str, Any]] = []
        for result in results:
            runner_meta = result.get("runner_meta") if isinstance(result.get("runner_meta"), dict) else {}
            skill_audit = runner_meta.get("skill_use_audit") if isinstance(runner_meta.get("skill_use_audit"), dict) else {}
            groups.append(
                {
                    "group_id": result.get("group_id", ""),
                    "group_label": result.get("group_label", ""),
                    "runner": result.get("runner", ""),
                    "skills_enabled": result.get("skills_enabled", False),
                    "answer_text": result.get("answer_text", ""),
                    "short_answer_text": result.get("short_answer_text", ""),
                    "full_response_text": result.get("full_response_text", ""),
                    "evaluation": result.get("evaluation") or {},
                    "score_label": _score_label(result),
                    "outcome": _outcome(result),
                    "verifier": _verifier_payload(result),
                    "status_axes": {
                        "run_lifecycle_status": result.get("run_lifecycle_status"),
                        "protocol_completion_status": result.get("protocol_completion_status"),
                        "answer_availability": result.get("answer_availability"),
                        "answer_reliability": result.get("answer_reliability"),
                        "evaluable": result.get("evaluable"),
                        "scored": result.get("scored"),
                        "recovery_mode": result.get("recovery_mode"),
                        "degraded_execution": result.get("degraded_execution"),
                        "execution_error_kind": result.get("execution_error_kind"),
                        "error": result.get("error"),
                    },
                    "diagnostics": _diagnostics_payload(result, skill_audit),
                    "workspace_isolation": _workspace_isolation_payload(runner_meta),
                    "annotations": self.annotation_store.list_annotations(
                        run_id=run_id,
                        record_id=str(result.get("record_id") or record_id),
                        group_id=str(result.get("group_id") or ""),
                    ),
                }
            )
        return {
            "run_id": run_id,
            "record_id": first.get("record_id") or record_id,
            "dataset": dataset,
            "subset": subset,
            "eval_kind": first.get("eval_kind", ""),
            "prompt": first.get("prompt", ""),
            "question_markdown": question_markdown,
            "question_source": question_source,
            "reference_answer": reference_answer,
            "reference": self._reference_payload(results, reference_answer=reference_answer),
            "assets": self._assets(run_root, results),
            "groups": groups,
            "annotations": self.annotation_store.list_annotations(run_id=run_id, record_id=str(first.get("record_id") or record_id)),
        }

    @staticmethod
    def _reference_payload(
        results: list[dict[str, Any]],
        *,
        reference_answer: str,
    ) -> dict[str, Any]:
        result = results[0]
        evaluation = result.get("evaluation") if isinstance(result.get("evaluation"), dict) else {}
        details = evaluation.get("details") if isinstance(evaluation.get("details"), dict) else {}
        checkpoints = details.get("checkpoint_matches") or details.get("items") or []
        return {
            "answer": reference_answer,
            "reasoning": details.get("reference_reasoning") or "",
            "checkpoints": checkpoints,
            "judge": details.get("judge") or {},
            "available": bool(reference_answer or details.get("reference_reasoning") or checkpoints),
        }

    @staticmethod
    def _is_within_run(run_root: Path, candidate: Path) -> bool:
        root = run_root.resolve()
        try:
            candidate.resolve().relative_to(root)
            return True
        except ValueError:
            return False

    def resolve_asset(self, run_id: str, asset_path: str) -> Path:
        run_root = self._run_dir(run_id)
        candidate = (run_root / asset_path).resolve()
        if not self._is_within_run(run_root, candidate):
            raise AssetAccessError("Asset path escapes benchmark run directory")
        if not candidate.is_file():
            raise AssetAccessError(f"Asset does not exist: {asset_path}")
        return candidate
