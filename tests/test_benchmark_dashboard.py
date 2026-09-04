from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmarking.dashboard import annotations as dashboard_annotations
from benchmarking.dashboard import service as dashboard_service


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def result_payload(
    *,
    group_id: str,
    record_id: str,
    dataset: str = "demo",
    subset: str = "chembench",
    eval_kind: str = "chembench_open_ended",
    passed: bool | None = True,
    score: float = 1.0,
    primary_metric: str = "judge_accuracy",
    details: dict[str, object] | None = None,
    runner_meta: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "group_id": group_id,
        "group_label": group_id,
        "runner": "single_llm",
        "websearch": False,
        "skills_enabled": group_id.endswith("skills_on"),
        "record_id": record_id,
        "subset": subset,
        "dataset": dataset,
        "source_file": "/tmp/demo.jsonl",
        "eval_kind": eval_kind,
        "prompt": "Question body",
        "reference_answer": "Reference answer",
        "answer_text": f"FINAL ANSWER: {group_id}",
        "short_answer_text": group_id,
        "full_response_text": f"FINAL ANSWER: {group_id}",
        "evaluation": {
            "eval_kind": eval_kind,
            "score": score,
            "max_score": 1.0,
            "normalized_score": score,
            "passed": passed,
            "primary_metric": primary_metric,
            "primary_metric_direction": "higher_is_better",
            "details": details or {},
        },
        "runner_meta": runner_meta or {},
        "raw": {},
        "elapsed_seconds": 12.5,
        "run_lifecycle_status": "completed",
        "protocol_completion_status": "missing",
        "protocol_acceptance_status": None,
        "answer_availability": "native_final",
        "answer_reliability": "native",
        "evaluable": True,
        "scored": True,
        "recovery_mode": "none",
        "degraded_execution": False,
        "execution_error_kind": None,
        "error": None,
    }


def write_demo_run(root: Path, run_id: str = "benchmark-20260604-120000") -> Path:
    run_root = root / run_id
    result = result_payload(group_id="single_llm_skills_on", record_id="r1")
    write_json(
        run_root / "results.json",
        {
            "schema_version": 2,
            "generated_at": "2026-06-04T12:00:00+0800",
            "records": 1,
            "groups": [
                {
                    "id": "single_llm_skills_on",
                    "label": "skills on",
                    "runner": "single_llm",
                    "websearch": False,
                    "skills_enabled": True,
                }
            ],
            "results": [result],
            "summary": {
                "group_order": ["single_llm_skills_on"],
                "groups": {
                    "single_llm_skills_on": {
                        "count": 1,
                        "pass_count": 1,
                        "avg_normalized_score": 1.0,
                    }
                },
            },
        },
    )
    write_json(run_root / "runtime-manifest.json", {"run_groups": ["single_llm_skills_on"]})
    write_json(run_root / "per-record" / "single_llm_skills_on" / "r1.json", result)
    return run_root


def test_list_runs_reads_schema_v2_results_and_annotations(tmp_path: Path) -> None:
    run_root = write_demo_run(tmp_path)
    store = dashboard_annotations.AnnotationStore(tmp_path / "dashboard.sqlite")
    store.upsert_run_metadata(run_id=run_root.name, alias="Verifier smoke", favorite=True)

    dashboard = dashboard_service.BenchmarkDashboard(run_roots=[tmp_path], annotation_store=store)

    runs = dashboard.list_runs()

    assert [item["run_id"] for item in runs] == [run_root.name]
    assert runs[0]["alias"] == "Verifier smoke"
    assert runs[0]["favorite"] is True
    assert runs[0]["status"] == "completed"
    assert runs[0]["record_count"] == 1
    assert runs[0]["group_count"] == 1
    assert runs[0]["datasets"] == ["demo"]
    assert runs[0]["subsets"] == ["chembench"]
    assert "average_normalized_score" not in runs[0]
    assert runs[0]["progress"]["completed"] == 1
    assert runs[0]["summary"]["groups"]["single_llm_skills_on"]["avg_normalized_score"] == 1.0


def test_dashboard_prefers_runner_agent_duration_for_record_timings(tmp_path: Path) -> None:
    run_root = write_demo_run(tmp_path)
    result = result_payload(
        group_id="single_llm_skills_on",
        record_id="r1",
        runner_meta={"durationMs": 3210},
    )
    write_json(run_root / "results.json", {"records": 1, "groups": [{"id": "single_llm_skills_on"}], "results": [result]})
    write_json(run_root / "per-record" / "single_llm_skills_on" / "r1.json", result)

    dashboard = dashboard_service.BenchmarkDashboard(run_roots=[tmp_path])

    records = dashboard.list_records(run_root.name)
    record = dashboard.get_record(run_root.name, "r1")

    assert records[0]["group_results"][0]["agent_duration_seconds"] == 3.21
    assert records[0]["group_results"][0]["elapsed_seconds"] == 12.5
    assert record["groups"][0]["diagnostics"]["agent_duration_seconds"] == 3.21
    assert record["groups"][0]["diagnostics"]["elapsed_seconds"] == 12.5


@pytest.mark.parametrize("duration_ms", [None, "3210", -1, float("nan"), float("inf")])
def test_dashboard_ignores_invalid_runner_agent_duration(duration_ms: object) -> None:
    result = result_payload(
        group_id="single_llm_skills_on",
        record_id="r1",
        runner_meta={"durationMs": duration_ms},
    )

    assert dashboard_service._agent_duration_seconds(result) is None


def test_list_runs_pins_favorited_runs_and_preserves_newest_first_order(tmp_path: Path) -> None:
    older_favorite = write_demo_run(tmp_path, run_id="favorite-old")
    newer_favorite = write_demo_run(tmp_path, run_id="favorite-new")
    newest_unfavorited = write_demo_run(tmp_path, run_id="plain-new")
    oldest_unfavorited = write_demo_run(tmp_path, run_id="plain-old")

    # Make the expected discovery order deterministic instead of relying on
    # filesystem timestamp resolution.
    for path, mtime in (
        (older_favorite, 100),
        (newer_favorite, 200),
        (newest_unfavorited, 300),
        (oldest_unfavorited, 400),
    ):
        os.utime(path, (mtime, mtime))

    store = dashboard_annotations.AnnotationStore(tmp_path / "dashboard.sqlite")
    store.upsert_run_metadata(run_id=older_favorite.name, favorite=True)
    store.upsert_run_metadata(run_id=newer_favorite.name, favorite=True)

    runs = dashboard_service.BenchmarkDashboard(run_roots=[tmp_path], annotation_store=store).list_runs()

    assert [run["run_id"] for run in runs] == [
        newer_favorite.name,
        older_favorite.name,
        oldest_unfavorited.name,
        newest_unfavorited.name,
    ]


def test_dashboard_uses_source_dataset_when_persisted_result_dataset_is_run_id(tmp_path: Path) -> None:
    run_id = "temp-benchmark-20260513-141355"
    run_root = tmp_path / run_id
    result = result_payload(
        group_id="single_llm_skills_on",
        record_id="r1",
        dataset=run_id,
    )
    result["source_file"] = str(tmp_path / "temp-benchmarks" / "frontierscience" / "data" / "pool.jsonl")
    write_json(
        run_root / "results.json",
        {
            "schema_version": 2,
            "generated_at": "2026-05-13T20:44:37+0800",
            "records": 1,
            "groups": [{"id": "single_llm_skills_on"}],
            "results": [result],
            "summary": {},
        },
    )

    dashboard = dashboard_service.BenchmarkDashboard(run_roots=[tmp_path])

    assert dashboard.list_runs()[0]["datasets"] == ["frontierscience"]
    assert dashboard.list_records(run_id)[0]["dataset"] == "frontierscience"
    assert dashboard.get_record(run_id, "r1")["dataset"] == "frontierscience"


def test_list_runs_discovers_classified_run_without_descending_into_run_artifacts(tmp_path: Path) -> None:
    run_root = write_demo_run(
        tmp_path / "formal" / "verifier-grounded-rdkit" / "qwen3-7-max",
        run_id="verifier-grounded-rdkit-qwen3-7-max-20260721-120000",
    )
    write_demo_run(run_root / "recovery", run_id="snapshot")

    runs = dashboard_service.BenchmarkDashboard(run_roots=[tmp_path]).list_runs()

    assert [item["run_id"] for item in runs] == [run_root.name]
    assert runs[0]["path"] == str(run_root)


def test_list_runs_skips_legacy_workspace_archives(tmp_path: Path) -> None:
    archive_root = tmp_path / "legacy-workspace-archives"
    fake_nested_run = archive_root / "benchmark-single-skills-on-20260727T010203Z" / "workspace" / "nested"
    write_json(fake_nested_run / "results.json", {"results": []})

    runs = dashboard_service.BenchmarkDashboard(run_roots=[tmp_path]).list_runs()

    assert runs == []


def test_run_lookup_rejects_duplicate_ids_across_categories(tmp_path: Path) -> None:
    write_demo_run(tmp_path / "formal" / "demo" / "model", run_id="duplicate-run")
    write_demo_run(tmp_path / "temporary" / "demo" / "model", run_id="duplicate-run")

    dashboard = dashboard_service.BenchmarkDashboard(run_roots=[tmp_path])

    with pytest.raises(dashboard_service.RunNotFoundError, match="Ambiguous benchmark run id"):
        dashboard.get_run("duplicate-run")


def test_vgb_tracks_are_grouped_under_one_dashboard_dataset(tmp_path: Path) -> None:
    run_root = tmp_path / "vgb-run"
    payloads = [
        result_payload(
            group_id="single_llm_skills_on",
            record_id="rdkit-qed-max-001",
            dataset="verifier_grounded_rdkit",
            subset="verifier_grounded_rdkit",
            eval_kind="verifier_grounded",
        ),
        result_payload(
            group_id="single_llm_skills_on",
            record_id="xtb-gap-min-001",
            dataset="verifier_grounded_xtb_xyz",
            subset="verifier_grounded_xtb_xyz",
            eval_kind="verifier_grounded",
        ),
        result_payload(
            group_id="single_llm_skills_on",
            record_id="property_calculation_advanced_001",
            dataset="verifier_grounded_property_calculation",
            subset="verifier_grounded_property_calculation",
            eval_kind="verifier_grounded",
        ),
    ]
    write_json(
        run_root / "results.json",
        {
            "schema_version": 2,
            "generated_at": "2026-07-16T12:00:00+0800",
            "records": 3,
            "groups": [{"id": "single_llm_skills_on"}],
            "results": payloads,
            "summary": {},
        },
    )
    dashboard = dashboard_service.BenchmarkDashboard(run_roots=[tmp_path])

    runs = dashboard.list_runs()
    records = dashboard.list_records("vgb-run")
    record = dashboard.get_record("vgb-run", "rdkit-qed-max-001")

    assert runs[0]["datasets"] == ["vgb"]
    assert runs[0]["subsets"] == [
        "property_calculation_advanced",
        "verifier_grounded_rdkit",
        "verifier_grounded_xtb_xyz",
    ]
    assert {item["dataset"] for item in records} == {"vgb"}
    assert {item["subset"] for item in records} == {
        "property_calculation_advanced",
        "verifier_grounded_rdkit",
        "verifier_grounded_xtb_xyz",
    }
    assert record["dataset"] == "vgb"
    assert record["subset"] == "verifier_grounded_rdkit"


def test_record_detail_exposes_workspace_adjudication_and_findings(tmp_path: Path) -> None:
    run_root = tmp_path / "workspace-adjudication-run"
    result = result_payload(
        group_id="single_llm_skills_on",
        record_id="r-boundary",
        runner_meta={
            "workspace_isolation": {
                "policy_digest": "a" * 64,
                "audit_execution_status": "complete",
                "boundary_status": "violated",
                "contamination_status": "clear",
                "adjudication": "scoreable_degraded",
                "findings": [
                    {
                        "access_mode": "write",
                        "operation_outcome": "succeeded",
                        "policy_id": "benchmark_runtime_root",
                        "resolved_path": "/tmp/sibling/generated.py",
                    }
                ],
                "cleanup": {"failed_count": 0},
            }
        },
    )
    write_json(run_root / "results.json", {"schema_version": 3, "results": [result], "groups": []})

    record = dashboard_service.BenchmarkDashboard(run_roots=[tmp_path]).get_record(
        run_root.name,
        "r-boundary",
    )

    isolation = record["groups"][0]["workspace_isolation"]
    assert isolation["adjudication"] == "scoreable_degraded"
    assert isolation["contamination_status"] == "clear"
    assert isolation["findings"][0]["access_mode"] == "write"


def test_list_runs_reconciles_stale_progress_state_with_per_record_outputs(tmp_path: Path) -> None:
    run_root = tmp_path / "stale-progress-run"
    payloads = [
        result_payload(
            group_id=group_id,
            record_id=record_id,
            dataset="verifier_grounded_property_calculation",
            subset="verifier_grounded_property_calculation",
            eval_kind="verifier_grounded",
        )
        for group_id in ("single_llm_skills_on", "single_llm_skills_off")
        for record_id in ("property_calc_free_energy_001", "property_calc_crystal_phase_002")
    ]
    write_json(
        run_root / "results.json",
        {
            "schema_version": 2,
            "generated_at": "2026-07-16T12:00:00+0800",
            "records": 2,
            "groups": [{"id": "single_llm_skills_on"}, {"id": "single_llm_skills_off"}],
            "results": payloads,
            "summary": {},
        },
    )
    for payload in payloads:
        record_file = str(payload["record_id"]).replace("_", "-")
        write_json(run_root / "per-record" / str(payload["group_id"]) / f"{record_file}.json", payload)
    write_json(
        run_root / "progress" / "state.json",
        {
            "status": "completed",
            "total": 3,
            "completed": 3,
            "groups": {
                "single_llm_skills_on": {"completed_records": ["property_calc_crystal_phase_002"], "completed_count": 1},
                "single_llm_skills_off": {
                    "completed_records": ["property_calc_free_energy_001", "property_calc_crystal_phase_002"],
                    "completed_count": 2,
                },
            },
        },
    )
    dashboard = dashboard_service.BenchmarkDashboard(run_roots=[tmp_path])

    run = dashboard.list_runs()[0]

    assert run["progress"]["source"] == "progress_state"
    assert run["progress"]["total"] == 4
    assert run["progress"]["completed"] == 4
    assert run["progress"]["groups"]["single_llm_skills_on"]["completed_count"] == 2
    assert run["progress"]["groups"]["single_llm_skills_off"]["completed_count"] == 2


def test_dashboard_merges_newer_per_record_outputs_into_partial_aggregate(tmp_path: Path) -> None:
    run_root = tmp_path / "partial-aggregate-run"
    aggregate = result_payload(group_id="single_llm_skills_on", record_id="r1")
    newer = result_payload(group_id="single_llm_skills_on", record_id="r2", score=0.5)
    write_json(
        run_root / "results.json",
        {
            "schema_version": 3,
            "results": [aggregate],
            "summary": {},
        },
    )
    write_json(run_root / "per-record" / "single_llm_skills_on" / "r1.json", aggregate)
    write_json(run_root / "per-record" / "single_llm_skills_on" / "r2.json", newer)

    dashboard = dashboard_service.BenchmarkDashboard(run_roots=[tmp_path])

    run = dashboard.list_runs()[0]
    records = dashboard.list_records(run_root.name)

    assert run["record_count"] == 2
    assert run["progress"]["completed"] == 2
    assert [item["record_id"] for item in records] == ["r1", "r2"]
    assert dashboard.get_record(run_root.name, "r2")["groups"][0]["evaluation"]["score"] == 0.5


def test_dashboard_preserves_aggregate_reference_when_per_record_still_has_placeholder(tmp_path: Path) -> None:
    run_root = tmp_path / "property-reference-run"
    aggregate = result_payload(group_id="single_llm_skills_on", record_id="property-calc-1")
    aggregate["reference_answer"] = '{"answer":3.85,"unit":"Debye"}'
    newer = result_payload(
        group_id="single_llm_skills_on",
        record_id="property-calc-1",
        score=0.9,
    )
    newer["reference_answer"] = (
        "No reference answer is exposed; score with the pinned verifier release."
    )
    write_json(run_root / "results.json", {"schema_version": 3, "results": [aggregate]})
    write_json(run_root / "per-record" / "single_llm_skills_on" / "property-calc-1.json", newer)

    record = dashboard_service.BenchmarkDashboard(run_roots=[tmp_path]).get_record(
        run_root.name,
        "property-calc-1",
    )

    assert record["reference"]["answer"] == '{"answer":3.85,"unit":"Debye"}'
    assert record["groups"][0]["evaluation"]["score"] == 0.9


def test_dashboard_uses_verifier_gold_when_reporting_reference_is_placeholder(tmp_path: Path) -> None:
    run_root = tmp_path / "active-property-run"
    payload = result_payload(
        group_id="single_llm_skills_on",
        record_id="property-calc-easy-1",
        dataset="verifier_grounded_property_calculation_easy",
        eval_kind="verifier_grounded",
        primary_metric="verifier_score",
        details={
            "properties": {
                "gold_answers": {
                    "dipole_moment": {"value": 3.85, "unit": "Debye"},
                }
            }
        },
    )
    payload["reference_answer"] = (
        "No reference answer is exposed; score with the pinned verifier release."
    )
    write_json(run_root / "per-record" / "single_llm_skills_on" / "property-calc-easy-1.json", payload)

    record = dashboard_service.BenchmarkDashboard(run_roots=[tmp_path]).get_record(
        run_root.name,
        "property-calc-easy-1",
    )

    assert json.loads(record["reference_answer"]) == {"answer": 3.85, "unit": "Debye"}
    assert json.loads(record["reference"]["answer"]) == {"answer": 3.85, "unit": "Debye"}
    assert record["reference"]["available"] is True


def test_get_record_preserves_verifier_score_without_marking_failed(tmp_path: Path) -> None:
    verifier_details = {
        "status": "ok",
        "canonical_smiles": "CCO",
        "properties": {"qed": 0.92},
        "constraint_scores": [{"property": "qed", "score": 0.92}],
    }
    run_root = tmp_path / "verifier-run"
    payload = result_payload(
        group_id="single_llm_skills_on",
        record_id="rdkit_qed_max_001",
        eval_kind="verifier_grounded",
        passed=None,
        score=0.92,
        primary_metric="verifier_score",
        details=verifier_details,
    )
    write_json(
        run_root / "results.json",
        {
            "schema_version": 2,
            "generated_at": "2026-06-04T12:00:00+0800",
            "records": 1,
            "groups": [{"id": "single_llm_skills_on"}],
            "results": [payload],
            "summary": {},
        },
    )
    write_json(run_root / "per-record" / "single_llm_skills_on" / "rdkit-qed-max-001.json", payload)
    dashboard = dashboard_service.BenchmarkDashboard(run_roots=[tmp_path])

    record = dashboard.get_record("verifier-run", "rdkit_qed_max_001")

    group = record["groups"][0]
    assert group["score_label"] == "Verifier 0.92"
    assert group["outcome"] == "scored"
    assert group["evaluation"]["passed"] is None
    assert group["verifier"]["canonical_smiles"] == "CCO"
    assert group["verifier"]["properties"]["qed"] == 0.92


def test_get_record_relabels_legacy_skill_off_exec_diagnostics(tmp_path: Path) -> None:
    run_root = tmp_path / "legacy-exec-run"
    on_payload = result_payload(
        group_id="single_llm_skills_on",
        record_id="r1",
        runner_meta={
            "skill_use_audit": {
                "openclaw_tool_call_count": 3,
                "skill_tool_call_count": 1,
                "skill_tool_failure_count": 0,
            }
        },
    )
    off_payload = result_payload(
        group_id="single_llm_skills_off",
        record_id="r1",
        runner_meta={
            "skill_use_audit": {
                "openclaw_tool_call_count": 2,
                "skill_tool_call_count": 1,
                "skill_tool_failure_count": 1,
            }
        },
    )
    write_json(
        run_root / "results.json",
        {
            "schema_version": 2,
            "generated_at": "2026-06-04T12:00:00+0800",
            "records": 1,
            "groups": [{"id": "single_llm_skills_on"}, {"id": "single_llm_skills_off"}],
            "results": [on_payload, off_payload],
            "summary": {},
        },
    )
    dashboard = dashboard_service.BenchmarkDashboard(run_roots=[tmp_path])

    record = dashboard.get_record("legacy-exec-run", "r1")

    diagnostics_by_group = {group["group_id"]: group["diagnostics"] for group in record["groups"]}
    assert diagnostics_by_group["single_llm_skills_on"]["exec_tool_call_count"] == 1
    assert diagnostics_by_group["single_llm_skills_on"]["skill_tool_call_count"] == 1
    assert diagnostics_by_group["single_llm_skills_off"]["exec_tool_call_count"] == 1
    assert diagnostics_by_group["single_llm_skills_off"]["exec_tool_failure_count"] == 1
    assert diagnostics_by_group["single_llm_skills_off"]["skill_tool_call_count"] == 0
    assert diagnostics_by_group["single_llm_skills_off"]["skill_tool_failure_count"] == 0


def test_get_record_exposes_structured_execution_error_diagnostics(tmp_path: Path) -> None:
    run_root = tmp_path / "provider-error-run"
    execution_error = {
        "code": "provider_access_denied",
        "message": "Access to model denied.",
        "layer": "provider_authorization",
        "retryable": False,
        "source": "stderr",
        "primary_error": {
            "source": "stderr",
            "line_number": 2,
            "event_kind": "provider_request_failed",
            "status_code": 403,
            "error_code": None,
            "error_type": None,
            "message": "Access to model denied.",
            "raw": "403 Access to model denied.",
            "parser": "openclaw_raw_error",
        },
        "observed_errors": [],
    }
    payload = result_payload(
        group_id="single_llm_skills_on",
        record_id="r1",
        passed=False,
        score=0.0,
        primary_metric="execution_error",
        runner_meta={"execution_error": execution_error},
    )
    write_json(run_root / "per-record" / "single_llm_skills_on" / "r1.json", payload)
    dashboard = dashboard_service.BenchmarkDashboard(run_roots=[tmp_path])

    record = dashboard.get_record("provider-error-run", "r1")

    assert record["groups"][0]["diagnostics"]["execution_error"] == execution_error


def test_get_record_uses_runtime_bundle_question_markdown_and_assets(tmp_path: Path) -> None:
    run_root = tmp_path / "superchem-run"
    bundle_root = run_root / "input-bundles" / "superchem-r1"
    question_path = bundle_root / "question.md"
    image_path = bundle_root / "images" / "img01.png"
    question_path.parent.mkdir(parents=True)
    question_path.write_text("Question with image ![](images/img01.png)", encoding="utf-8")
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png")
    payload = result_payload(
        group_id="single_llm_skills_on",
        record_id="superchem-r1",
        eval_kind="superchem_multiple_choice_rpf",
        runner_meta={
            "runtime_bundle": {
                "question_markdown": str(question_path),
                "image_files": [str(image_path)],
            }
        },
    )
    write_json(run_root / "per-record" / "single_llm_skills_on" / "superchem-r1.json", payload)
    dashboard = dashboard_service.BenchmarkDashboard(run_roots=[tmp_path])

    record = dashboard.get_record("superchem-run", "superchem-r1")

    assert record["question_markdown"] == "Question with image ![](images/img01.png)"
    assert record["assets"][0]["relative_path"] == "input-bundles/superchem-r1/images/img01.png"
    assert dashboard.resolve_asset("superchem-run", "input-bundles/superchem-r1/images/img01.png") == image_path.resolve()
    with pytest.raises(dashboard_service.AssetAccessError):
        dashboard.resolve_asset("superchem-run", "../outside.txt")


def test_annotation_crud_is_soft_delete_and_does_not_touch_results(tmp_path: Path) -> None:
    run_root = write_demo_run(tmp_path)
    results_before = (run_root / "results.json").read_text(encoding="utf-8")
    store = dashboard_annotations.AnnotationStore(tmp_path / "dashboard.sqlite")

    created = store.create_annotation(
        run_id=run_root.name,
        record_id="r1",
        group_id="single_llm_skills_on",
        note="needs review",
        status="needs_review",
        tags=["numeric", "priority"],
        manual_verdict="uncertain",
    )
    updated = store.update_annotation(created["id"], note="checked", tags=["done"])
    store.delete_annotation(created["id"])

    assert updated["note"] == "checked"
    assert updated["tags"] == ["done"]
    assert store.list_annotations(run_id=run_root.name) == []
    assert store.list_annotations(run_id=run_root.name, include_deleted=True)[0]["deleted"] is True
    assert (run_root / "results.json").read_text(encoding="utf-8") == results_before
