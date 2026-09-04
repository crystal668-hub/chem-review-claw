from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarking.core.datasets import BenchmarkRecord, GradingSpec
from benchmarking.core.experiments import ExperimentSpec
from benchmarking.core.reporting import GroupRecordResult
from benchmarking.runtime import config_pool as runtime_config_pool
from benchmarking.runtime import judge as judge_runtime
from benchmarking.runtime.agent_workspace import (
    AttemptIdentity,
    AttemptOutcome,
    AttemptWorkspaceManager,
    WorkspaceTemplate,
)
from benchmarking.workflow import cli as benchmarking_cli
from benchmarking.workflow import (
    dataset_selection,
    experiments,
    orchestration,
    run_state,
    runner_adapters,
    runtime_config,
)
from benchmarking.workflow.errors import BenchmarkError


def test_benchmarking_cli_owns_benchmark_entrypoint_behavior() -> None:
    assert callable(benchmarking_cli.main)
    assert callable(benchmarking_cli.parse_args)
    assert experiments.EXPERIMENT_GROUPS["single_llm_skills_on"].runner == "single_llm"
    assert experiments.EXPERIMENT_GROUPS["chemqa_skills_on"].runner == "chemqa"
    assert all(group.websearch is False for group in experiments.EXPERIMENT_GROUPS.values())
    assert all(spec.websearch_enabled is False for spec in experiments.EXPERIMENT_SPECS.values())


def test_default_run_output_root_classifies_formal_run_by_dataset_and_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    temp_root = tmp_path / "temp-benchmarks"
    monkeypatch.setattr(dataset_selection.runtime_paths, "temp_benchmarks_root", temp_root)
    record = BenchmarkRecord(
        record_id="r1",
        dataset="verifier_grounded_rdkit",
        source_file=str(tmp_path / "formal" / "rdkit.jsonl"),
        prompt="question",
        eval_kind="verifier_grounded",
        reference_answer="hidden",
        payload={},
    )

    output_root = dataset_selection.default_run_output_root(
        output_dir=tmp_path / "runs",
        dataset_files=[tmp_path / "formal" / "rdkit.jsonl"],
        records=[record],
        single_agent_model="qwen/qwen3.7-max",
        timestamp="20260721-120000",
    )

    assert output_root == (
        tmp_path
        / "runs"
        / "formal"
        / "verifier-grounded-rdkit"
        / "qwen3-7-max"
        / "verifier-grounded-rdkit-qwen3-7-max-20260721-120000"
    )


def test_default_run_output_root_classifies_multi_dataset_temp_run(
    monkeypatch,
    tmp_path: Path,
) -> None:
    temp_root = tmp_path / "temp-benchmarks"
    monkeypatch.setattr(dataset_selection.runtime_paths, "temp_benchmarks_root", temp_root)
    records = [
        BenchmarkRecord(
            record_id=f"r{index}",
            dataset=dataset,
            source_file=str(temp_root / f"{dataset}.jsonl"),
            prompt="question",
            eval_kind="generic_semantic",
            reference_answer="answer",
            payload={},
        )
        for index, dataset in enumerate(("chembench", "superchem"), start=1)
    ]

    output_root = dataset_selection.default_run_output_root(
        output_dir=tmp_path / "runs",
        dataset_files=[temp_root / "chembench.jsonl", temp_root / "superchem.jsonl"],
        records=records,
        single_agent_model="openai/gpt-5.5",
        timestamp="20260721-120000",
    )

    assert output_root == (
        tmp_path
        / "runs"
        / "temporary"
        / "mixed-datasets"
        / "gpt-5-5"
        / "mixed-datasets-gpt-5-5-20260721-120000"
    )


def test_production_forbidden_path_policy_uses_explicit_runtime_roots_and_custom_dataset(
    monkeypatch,
    tmp_path,
) -> None:
    custom_dataset = tmp_path / "custom-vgb-data"
    path_values = {
        "benchmarks_root": custom_dataset,
        "temp_benchmarks_root": tmp_path / "temp-data",
        "data_root": tmp_path / "data",
        "project_state_root": tmp_path / "project" / "state",
        "project_root": tmp_path / "project",
        "agents_root": tmp_path / "agents",
        "benchmark_runtime_root": tmp_path / "benchmark-workspaces",
    }
    for name, value in path_values.items():
        monkeypatch.setattr(runtime_config.runtime_paths, name, value)

    runtime_root = path_values["benchmark_runtime_root"] / "runs"
    output_root = path_values["project_state_root"] / "benchmark-runs" / "run-a"
    roots = runtime_config.build_production_protected_roots(
        runtime_root=runtime_root,
        output_root=output_root,
    )

    counts = Counter(root.policy_id for root in roots)
    assert len(roots) == 13
    assert counts["legacy_benchmark_workspace"] == 4
    assert sum(count for policy, count in counts.items() if policy != "legacy_benchmark_workspace") == 9

    template = tmp_path / "template"
    template.mkdir()
    (template / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    manager = AttemptWorkspaceManager(
        runtime_root=runtime_root,
        output_root=output_root,
        run_id="run-a",
        invocation_id="invocation-a",
        templates={"single-v1": WorkspaceTemplate("single-v1", source_dir=template)},
        protected_roots=roots,
    )
    manifest = manager.forbidden_path_policy_manifest()
    dataset_policy = next(
        item for item in manifest["protected_roots"] if item["policy_id"] == "benchmark_dataset_root"
    )
    assert dataset_policy["path"] == str(custom_dataset.resolve(strict=False))

    lease = manager.prepare(
        AttemptIdentity(
            run_id="run-a",
            invocation_id="invocation-a",
            group_id="single_llm_skills_off",
            runner_kind="single_llm",
            agent_id="agent",
            record_id="record",
            attempt_index=0,
            session_id="session",
            template_id="single-v1",
        )
    )
    transcript = tmp_path / "custom-dataset-transcript.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "name": "exec",
                            "arguments": {"command": f"cat {custom_dataset / 'track' / 'tasks.jsonl'}"},
                        }
                    ],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    audit = manager.audit_attempt(
        lease,
        {"session_isolation": {"postflight_entry_session_file": str(transcript)}},
    )

    assert audit.status == "contaminated"
    assert audit.findings[0]["policy_id"] == "benchmark_dataset_root"
    assert audit.findings[0]["matched_root"] == dataset_policy["path"]
    manager.seal(lease, AttemptOutcome(runner_status="failed", contamination_audit=audit))


def test_reporting_references_use_public_property_gold_only(monkeypatch) -> None:
    property_result = SimpleNamespace(
        dataset="verifier_grounded_property_calculation",
        record_id="property_calc_free_energy_001",
        reference_answer="No reference answer is exposed; score with the pinned verifier release.",
    )
    rdkit_result = SimpleNamespace(
        dataset="verifier_grounded_rdkit",
        record_id="rdkit_qed_max_001",
        reference_answer="No reference answer is exposed; score with the pinned verifier release.",
    )
    easy_property_result = SimpleNamespace(
        dataset="verifier_grounded_property_calculation_easy",
        record_id="property_calc_easy_001_toluene_aqueous_solvation_free_energy",
        reference_answer="No reference answer is exposed; score with the pinned verifier release.",
    )
    release_config = SimpleNamespace(
        tracks={
            "property_calculation": {"dataset": "verifier_grounded_property_calculation"},
            "property_calculation_easy": {
                "dataset": "verifier_grounded_property_calculation_easy"
            },
            "rdkit": {"dataset": "verifier_grounded_rdkit"},
        }
    )

    def sample_answers(track, *, release_config):
        if track == "property_calculation_easy":
            return [
                {
                    "task_id": "property_calc_easy_001_toluene_aqueous_solvation_free_energy",
                    "answer": -0.82,
                    "unit": "kcal/mol",
                }
            ]
        return [
            {
                "task_id": "property_calc_free_energy_001",
                "answer": 0.258031679,
                "unit": "kJ/mol",
            },
            {
                "task_id": "property_calc_crystal_phase_002",
                "answers": [
                    {"property": "potential_energy_difference", "value": 0.079, "unit": "eV"},
                    {"property": "ambient_pressure_phase", "value": "alpha"},
                    {"property": "high_pressure_phase", "value": "beta"},
                ],
            },
        ]

    monkeypatch.setattr(
        run_state,
        "load_public_reference_answers",
        sample_answers,
    )

    run_state.apply_verifier_grounded_reporting_references(
        [property_result, easy_property_result, rdkit_result],
        release_config=release_config,
    )

    assert property_result.reference_answer == '{"answer":0.258031679,"unit":"kJ/mol"}'
    assert easy_property_result.reference_answer == '{"answer":-0.82,"unit":"kcal/mol"}'
    assert rdkit_result.reference_answer.startswith("No reference answer is exposed")


def test_reporting_references_require_every_selected_property_gold(monkeypatch) -> None:
    result = SimpleNamespace(
        dataset="verifier_grounded_property_calculation",
        record_id="property_calc_crystal_phase_002",
        reference_answer="placeholder",
    )
    release_config = SimpleNamespace(
        tracks={"property_calculation": {"dataset": "verifier_grounded_property_calculation"}}
    )
    monkeypatch.setattr(
        run_state,
        "load_public_reference_answers",
        lambda track, *, release_config: [],
    )

    with pytest.raises(BenchmarkError, match="missing public gold"):
        run_state.apply_verifier_grounded_reporting_references(
            [result],
            release_config=release_config,
        )


def test_parse_args_accepts_no_timeout_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmarking.workflow.cli", "--no-timeout"],
    )

    args = benchmarking_cli.parse_args()

    assert args.no_timeout is True


def test_parse_args_accepts_no_analysis_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmarking.workflow.cli", "--no-analysis"],
    )

    args = benchmarking_cli.parse_args()

    assert args.no_analysis is True


def test_filter_records_by_ids_preserves_requested_order() -> None:
    records = [
        BenchmarkRecord(
            record_id=record_id,
            dataset="demo",
            source_file="/tmp/demo.jsonl",
            prompt="Question?",
            reference_answer="Answer",
            eval_kind="generic_semantic",
        )
        for record_id in ("first", "second")
    ]

    selected = dataset_selection.filter_records_by_ids(records, "second,first")

    assert [record.record_id for record in selected] == ["second", "first"]


def test_filter_records_by_ids_rejects_unknown_ids() -> None:
    records = [
        BenchmarkRecord(
            record_id="known",
            dataset="demo",
            source_file="/tmp/demo.jsonl",
            prompt="Question?",
            reference_answer="Answer",
            eval_kind="generic_semantic",
        )
    ]

    with pytest.raises(BenchmarkError, match="Unknown record id"):
        dataset_selection.filter_records_by_ids(records, "missing")


def test_filter_records_by_ids_rejects_duplicate_requested_ids() -> None:
    records = [
        BenchmarkRecord(
            record_id="known",
            dataset="demo",
            source_file="/tmp/demo.jsonl",
            prompt="Question?",
            reference_answer="Answer",
            eval_kind="generic_semantic",
        )
    ]

    with pytest.raises(BenchmarkError, match="duplicate ids"):
        dataset_selection.filter_records_by_ids(records, "known,known")


def test_filter_records_by_ids_rejects_ambiguous_selected_dataset_ids() -> None:
    records = [
        BenchmarkRecord(
            record_id="shared",
            dataset=dataset,
            source_file=f"/tmp/{dataset}.jsonl",
            prompt="Question?",
            reference_answer="Answer",
            eval_kind="generic_semantic",
        )
        for dataset in ("first", "second")
    ]

    with pytest.raises(BenchmarkError, match="Ambiguous record id"):
        dataset_selection.filter_records_by_ids(records, "shared")


def test_default_web_search_preflight_skips_all_experiment_groups(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    class FakeConfigPool:
        def config_for_group(self, group):
            return tmp_path / f"{group.id}.json"

    def fake_preflight(**kwargs):
        calls.append(kwargs["agent_id"])
        return {"available": True}

    monkeypatch.setattr(benchmarking_cli, "run_web_search_preflight", fake_preflight)

    report = benchmarking_cli.run_benchmark_web_search_preflight(
        group_ids=list(experiments.EXPERIMENT_GROUPS),
        config_pool=FakeConfigPool(),
        args=type("Args", (), {"single_agent_id_override": None})(),
    )

    assert calls == []
    assert report["available"] is True
    assert report["reports"] == {}


def test_resume_filters_existing_per_record_before_runner_creation(tmp_path) -> None:
    records = [
        BenchmarkRecord(
            record_id=record_id,
            dataset="demo",
            source_file="/tmp/demo.jsonl",
            prompt="Question?",
            reference_answer="Answer",
            eval_kind="generic_semantic",
        )
        for record_id in ("existing", "pending")
    ]
    existing = tmp_path / "per-record" / "single_llm_skills_on" / "existing.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("{}\n", encoding="utf-8")

    pending = run_state.pending_records_for_group(
        records,
        output_root=tmp_path,
        group_id="single_llm_skills_on",
        merge_existing_per_record=True,
    )

    assert [record.record_id for record in pending] == ["pending"]


def test_web_search_preflight_failure_materializes_group_failure(monkeypatch, tmp_path) -> None:
    record = BenchmarkRecord(
        record_id="record-1",
        dataset="demo",
        source_file="/tmp/demo.jsonl",
        prompt="Question?",
        reference_answer="Answer",
        eval_kind="generic_semantic",
        grading=GradingSpec(
            kind="generic_semantic",
            reference_answer="Answer",
            subset="demo_subset",
        ),
    )
    run_group_calls: list[str] = []
    monkeypatch.setitem(
        experiments.EXPERIMENT_GROUPS,
        "single_llm_skills_on",
        experiments.ExperimentGroup(
            id="single_llm_skills_on",
            label=experiments.EXPERIMENT_GROUPS["single_llm_skills_on"].label,
            runner="single_llm",
            websearch=True,
            skills_enabled=True,
        ),
    )
    monkeypatch.setitem(
        experiments.EXPERIMENT_SPECS,
        "single_llm_skills_on",
        ExperimentSpec(
            id="single_llm_skills_on",
            label=experiments.EXPERIMENT_GROUPS["single_llm_skills_on"].label,
            runner_kind="single_llm",
            websearch_enabled=True,
            skills_enabled=True,
            single_agent_id=experiments.BASELINE_AGENT_IDS["single_llm_skills_on"],
            skill_allowlist=tuple(experiments.BENCHMARK_SKILLS_ALLOWLIST),
        ),
    )

    class FakeConfigPool:
        def __init__(self, *args, **kwargs) -> None:
            self.context = type("Context", (), {"experiment_specs": experiments.EXPERIMENT_SPECS})()

        def config_for_group(self, group):
            path = tmp_path / "runtime-config" / f"{group.id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            return path

        def judge_config_path(self):
            path = tmp_path / "runtime-config" / "judge.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            return path

    monkeypatch.setattr(
        benchmarking_cli,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "single_timeout": 30,
                "no_timeout": False,
                "chemqa_timeout": 30,
                "max_unchanged_status_polls": 1,
                "max_recovery_attempts": 1,
                "groups": "single_llm_skills_on",
                "benchmark_root": str(tmp_path),
                "files": None,
                "datasets": None,
                "list_datasets": False,
                "subsets": None,
                "random_count_per_subset": None,
                "random_seed": 0,
                "offset": 0,
                "limit": None,
                "print_selected_records": False,
                "exact_output_dir": str(tmp_path / "out"),
                "output_dir": str(tmp_path / "out-parent"),
                "openclaw_config": str(tmp_path / "openclaw.json"),
                "single_agent_model": "openai/gpt-5.4",
                "single_agent_thinking": "high",
                "judge_model": "openai/gpt-5.4",
                "judge_agent_thinking": "high",
                "single_agent_id_override": None,
                "judge_agent": "benchmark-judge",
                "judge_timeout": 30,
                "max_concurrent_groups": 1,
                "inter_wave_delay_seconds": 0,
                "chemqa_root": str(tmp_path / "chemqa"),
                "chemqa_model_profile": "profile",
                "review_rounds": None,
                "rebuttal_rounds": None,
                "merge_existing_per_record": False,
            },
        )(),
    )
    monkeypatch.setattr(dataset_selection, "select_dataset_files", lambda args: [tmp_path / "demo.jsonl"])
    monkeypatch.setattr(dataset_selection, "load_records", lambda paths: [record])
    monkeypatch.setattr(benchmarking_cli, "check_all_skill_health", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        benchmarking_cli,
        "summarize_skill_health",
        lambda reports: {"available_skill_count": 0, "unavailable_skill_count": 0, "available_skills": [], "unavailable_skills": []},
    )
    monkeypatch.setattr(runtime_config_pool, "ConfigPool", FakeConfigPool)
    monkeypatch.setattr(
        benchmarking_cli,
        "run_benchmark_web_search_preflight",
        lambda **kwargs: {
            "enabled": True,
            "provider": "duckduckgo",
            "available": False,
            "reports": {"single_llm_skills_on": {"available": False, "error": "fetch failed"}},
        },
    )
    monkeypatch.setattr(runner_adapters, "run_pending_cleanroom_cleanup", lambda: [])
    monkeypatch.setattr(
        benchmarking_cli,
        "launch_automated_evaluation",
        lambda output_root: {"status": "launched", "output_root": str(output_root)},
    )

    def fake_run_group(**kwargs):
        run_group_calls.append(kwargs["group"].id)
        return []

    monkeypatch.setattr(orchestration, "run_group", fake_run_group)

    assert benchmarking_cli.main() == 0

    assert run_group_calls == []
    result_path = tmp_path / "out" / "per-record" / "single_llm_skills_on" / "record-1.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert "web_search preflight failed" in payload["error"]
    progress = json.loads((tmp_path / "out" / "progress" / "state.json").read_text(encoding="utf-8"))
    assert progress["completed"] == 1
    assert progress["groups"]["single_llm_skills_on"]["status"] == "failed"
    assert progress["groups"]["single_llm_skills_on"]["completed_records"] == ["record-1"]


def test_main_launches_automated_evaluation_after_results_are_written(monkeypatch, tmp_path) -> None:
    record = BenchmarkRecord(
        record_id="record-1",
        dataset="demo",
        source_file="/tmp/demo.jsonl",
        prompt="Question?",
        reference_answer="Answer",
        eval_kind="generic_semantic",
        grading=GradingSpec(
            kind="generic_semantic",
            reference_answer="Answer",
            subset="demo_subset",
        ),
    )
    launched: list[Path] = []

    class FakeConfigPool:
        def __init__(self, *args, **kwargs) -> None:
            self.context = type("Context", (), {"experiment_specs": experiments.EXPERIMENT_SPECS})()

        def config_for_group(self, group):
            path = tmp_path / "out" / "runtime-config" / f"{group.id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            return path

        def judge_config_path(self):
            path = tmp_path / "out" / "runtime-config" / "judge.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            return path

    def fake_run_group(**kwargs):
        return [
            GroupRecordResult(
                schema_version=2,
                group_id="single_llm_skills_off",
                group_label="single off",
                runner="single_llm",
                websearch=False,
                record_id="record-1",
                subset="demo_subset",
                dataset="demo",
                source_file="/tmp/demo.jsonl",
                eval_kind="generic_semantic",
                prompt="Question?",
                reference_answer="Answer",
                answer_text="FINAL ANSWER: Answer",
                evaluation={
                    "eval_kind": "generic_semantic",
                    "score": 1.0,
                    "max_score": 1.0,
                    "normalized_score": 1.0,
                    "passed": True,
                    "primary_metric": "semantic_match",
                    "primary_metric_direction": "higher_is_better",
                    "details": {},
                },
                runner_meta={},
                raw={},
                elapsed_seconds=0.1,
                run_lifecycle_status="completed",
                protocol_completion_status="completed",
                protocol_acceptance_status=None,
                answer_availability="native_final",
                answer_reliability="native",
                evaluable=True,
                scored=True,
                recovery_mode="none",
                degraded_execution=False,
                skills_enabled=False,
                execution_error_kind=None,
                error=None,
                short_answer_text="Answer",
                full_response_text="FINAL ANSWER: Answer",
            )
        ]

    def fake_launch(output_root):
        output_path = Path(output_root)
        assert (output_path / "results.json").is_file()
        assert (output_path / "runtime-manifest.json").is_file()
        assert not (output_path / "summary_by_group.csv").exists()
        assert not (output_path / "summary_by_group_and_subset.csv").exists()
        launched.append(output_path)
        return {
            "status": "launched",
            "status_path": str(output_path / "analysis" / "status.json"),
            "report_path": str(output_path / "analysis" / "report.json"),
        }

    output_path = tmp_path / "out"
    output_path.mkdir()
    (output_path / "summary_by_group.csv").write_text("stale\n", encoding="utf-8")
    (output_path / "summary_by_group_and_subset.csv").write_text("stale\n", encoding="utf-8")

    monkeypatch.setattr(
        benchmarking_cli,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "single_timeout": 30,
                "no_timeout": False,
                "chemqa_timeout": 30,
                "max_unchanged_status_polls": 1,
                "max_recovery_attempts": 1,
                "groups": "single_llm_skills_off",
                "benchmark_root": str(tmp_path),
                "files": None,
                "datasets": None,
                "list_datasets": False,
                "subsets": None,
                "random_count_per_subset": None,
                "random_seed": 0,
                "offset": 0,
                "limit": None,
                "print_selected_records": False,
                "exact_output_dir": str(tmp_path / "out"),
                "output_dir": str(tmp_path / "out-parent"),
                "openclaw_config": str(tmp_path / "openclaw.json"),
                "single_agent_model": "openai/gpt-5.4",
                "single_agent_thinking": "medium",
                "judge_model": "openai/gpt-5.4",
                "judge_agent_thinking": "minimal",
                "single_agent_id_override": None,
                "judge_agent": "benchmark-judge",
                "judge_timeout": 30,
                "max_concurrent_groups": 1,
                "inter_wave_delay_seconds": 0,
                "chemqa_root": str(tmp_path / "chemqa"),
                "chemqa_model_profile": "profile",
                "review_rounds": None,
                "rebuttal_rounds": None,
                "merge_existing_per_record": False,
            },
        )(),
    )
    monkeypatch.setattr(dataset_selection, "select_dataset_files", lambda args: [tmp_path / "demo.jsonl"])
    monkeypatch.setattr(dataset_selection, "load_records", lambda paths: [record])
    monkeypatch.setattr(benchmarking_cli, "check_all_skill_health", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        benchmarking_cli,
        "summarize_skill_health",
        lambda reports: {"available_skill_count": 0, "unavailable_skill_count": 0, "available_skills": [], "unavailable_skills": []},
    )
    monkeypatch.setattr(runtime_config_pool, "ConfigPool", FakeConfigPool)
    monkeypatch.setattr(judge_runtime, "JudgeClient", lambda **kwargs: object())
    monkeypatch.setattr(
        benchmarking_cli,
        "run_benchmark_web_search_preflight",
        lambda **kwargs: {
            "enabled": True,
            "provider": "duckduckgo",
            "available": True,
            "reports": {"single_llm_skills_off": {"available": True}},
        },
    )
    monkeypatch.setattr(runner_adapters, "run_pending_cleanroom_cleanup", lambda: [])
    monkeypatch.setattr(orchestration, "run_group", fake_run_group)
    monkeypatch.setattr(benchmarking_cli, "launch_automated_evaluation", fake_launch)

    assert benchmarking_cli.main() == 0

    assert launched == [tmp_path / "out"]
    manifest = json.loads((tmp_path / "out" / "runtime-manifest.json").read_text(encoding="utf-8"))
    assert manifest["automated_evaluation"]["status"] == "launched"
    assert manifest["groups"]["single_llm_skills_off"]["single_agent_thinking"] == "medium"
    assert manifest["groups"]["single_llm_skills_off"]["group"]["websearch"] is False
    assert manifest["judge"]["thinking"] == "minimal"
    assert manifest["timeout_mode"] == "bounded"
    results = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    assert results["timeout_mode"] == "bounded"
    assert results["run_groups"][0]["websearch"] is False


def test_main_skips_automated_evaluation_when_no_analysis_is_set(monkeypatch, tmp_path) -> None:
    record = BenchmarkRecord(
        record_id="record-1",
        dataset="demo",
        source_file="/tmp/demo.jsonl",
        prompt="Question?",
        reference_answer="Answer",
        eval_kind="generic_semantic",
        grading=GradingSpec(
            kind="generic_semantic",
            reference_answer="Answer",
            subset="demo_subset",
        ),
    )
    launched: list[Path] = []

    class FakeConfigPool:
        def __init__(self, *args, **kwargs) -> None:
            self.context = type("Context", (), {"experiment_specs": experiments.EXPERIMENT_SPECS})()

        def config_for_group(self, group):
            path = tmp_path / "out" / "runtime-config" / f"{group.id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            return path

        def judge_config_path(self):
            path = tmp_path / "out" / "runtime-config" / "judge.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            return path

    def fake_run_group(**kwargs):
        return [
            GroupRecordResult(
                schema_version=2,
                group_id="single_llm_skills_off",
                group_label="single off",
                runner="single_llm",
                websearch=False,
                record_id="record-1",
                subset="demo_subset",
                dataset="demo",
                source_file="/tmp/demo.jsonl",
                eval_kind="generic_semantic",
                prompt="Question?",
                reference_answer="Answer",
                answer_text="FINAL ANSWER: Answer",
                evaluation={
                    "eval_kind": "generic_semantic",
                    "score": 1.0,
                    "max_score": 1.0,
                    "normalized_score": 1.0,
                    "passed": True,
                    "primary_metric": "semantic_match",
                    "primary_metric_direction": "higher_is_better",
                    "details": {},
                },
                runner_meta={},
                raw={},
                elapsed_seconds=0.1,
                run_lifecycle_status="completed",
                protocol_completion_status="completed",
                protocol_acceptance_status=None,
                answer_availability="native_final",
                answer_reliability="native",
                evaluable=True,
                scored=True,
                recovery_mode="none",
                degraded_execution=False,
                skills_enabled=False,
                execution_error_kind=None,
                error=None,
                short_answer_text="Answer",
                full_response_text="FINAL ANSWER: Answer",
            )
        ]

    monkeypatch.setattr(
        benchmarking_cli,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "single_timeout": 30,
                "no_timeout": False,
                "no_analysis": True,
                "chemqa_timeout": 30,
                "max_unchanged_status_polls": 1,
                "max_recovery_attempts": 1,
                "groups": "single_llm_skills_off",
                "benchmark_root": str(tmp_path),
                "files": None,
                "datasets": None,
                "list_datasets": False,
                "subsets": None,
                "random_count_per_subset": None,
                "random_seed": 0,
                "offset": 0,
                "limit": None,
                "print_selected_records": False,
                "exact_output_dir": str(tmp_path / "out"),
                "output_dir": str(tmp_path / "out-parent"),
                "openclaw_config": str(tmp_path / "openclaw.json"),
                "single_agent_model": "openai/gpt-5.4",
                "single_agent_thinking": "medium",
                "judge_model": "openai/gpt-5.4",
                "judge_agent_thinking": "minimal",
                "single_agent_id_override": None,
                "judge_agent": "benchmark-judge",
                "judge_timeout": 30,
                "max_concurrent_groups": 1,
                "inter_wave_delay_seconds": 0,
                "chemqa_root": str(tmp_path / "chemqa"),
                "chemqa_model_profile": "profile",
                "review_rounds": None,
                "rebuttal_rounds": None,
                "merge_existing_per_record": False,
            },
        )(),
    )
    monkeypatch.setattr(dataset_selection, "select_dataset_files", lambda args: [tmp_path / "demo.jsonl"])
    monkeypatch.setattr(dataset_selection, "load_records", lambda paths: [record])
    monkeypatch.setattr(benchmarking_cli, "check_all_skill_health", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        benchmarking_cli,
        "summarize_skill_health",
        lambda reports: {"available_skill_count": 0, "unavailable_skill_count": 0, "available_skills": [], "unavailable_skills": []},
    )
    monkeypatch.setattr(runtime_config_pool, "ConfigPool", FakeConfigPool)
    monkeypatch.setattr(judge_runtime, "JudgeClient", lambda **kwargs: object())
    monkeypatch.setattr(
        benchmarking_cli,
        "run_benchmark_web_search_preflight",
        lambda **kwargs: {
            "enabled": True,
            "provider": "duckduckgo",
            "available": True,
            "reports": {"single_llm_skills_off": {"available": True}},
        },
    )
    monkeypatch.setattr(runner_adapters, "run_pending_cleanroom_cleanup", lambda: [])
    monkeypatch.setattr(orchestration, "run_group", fake_run_group)
    monkeypatch.setattr(benchmarking_cli, "launch_automated_evaluation", lambda output_root: launched.append(Path(output_root)))

    assert benchmarking_cli.main() == 0

    assert launched == []
    manifest = json.loads((tmp_path / "out" / "runtime-manifest.json").read_text(encoding="utf-8"))
    assert manifest["automated_evaluation"]["status"] == "skipped"
    assert manifest["automated_evaluation"]["reason"] == "disabled_by_cli"
    assert manifest["automated_evaluation"]["status_path"] == str(tmp_path / "out" / "analysis" / "status.json")
    results = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    assert results["records"] == 1
    assert results["results"][0]["record_id"] == "record-1"


def test_main_ignores_automated_evaluation_launch_failure(monkeypatch, tmp_path) -> None:
    record = BenchmarkRecord(
        record_id="record-1",
        dataset="demo",
        source_file="/tmp/demo.jsonl",
        prompt="Question?",
        reference_answer="Answer",
        eval_kind="generic_semantic",
        grading=GradingSpec(kind="generic_semantic", reference_answer="Answer", subset="demo_subset"),
    )

    class FakeConfigPool:
        def __init__(self, *args, **kwargs) -> None:
            self.context = type("Context", (), {"experiment_specs": experiments.EXPERIMENT_SPECS})()

        def config_for_group(self, group):
            path = tmp_path / "out" / "runtime-config" / f"{group.id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            return path

        def judge_config_path(self):
            path = tmp_path / "out" / "runtime-config" / "judge.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            return path

    def fake_run_group(**kwargs):
        return []

    monkeypatch.setattr(
        benchmarking_cli,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "single_timeout": 30,
                "no_timeout": False,
                "chemqa_timeout": 30,
                "max_unchanged_status_polls": 1,
                "max_recovery_attempts": 1,
                "groups": "single_llm_skills_off",
                "benchmark_root": str(tmp_path),
                "files": None,
                "datasets": None,
                "list_datasets": False,
                "subsets": None,
                "random_count_per_subset": None,
                "random_seed": 0,
                "offset": 0,
                "limit": None,
                "print_selected_records": False,
                "exact_output_dir": str(tmp_path / "out"),
                "output_dir": str(tmp_path / "out-parent"),
                "openclaw_config": str(tmp_path / "openclaw.json"),
                "single_agent_model": "openai/gpt-5.4",
                "single_agent_thinking": "high",
                "judge_model": "openai/gpt-5.4",
                "judge_agent_thinking": "high",
                "single_agent_id_override": None,
                "judge_agent": "benchmark-judge",
                "judge_timeout": 30,
                "max_concurrent_groups": 1,
                "inter_wave_delay_seconds": 0,
                "chemqa_root": str(tmp_path / "chemqa"),
                "chemqa_model_profile": "profile",
                "review_rounds": None,
                "rebuttal_rounds": None,
                "merge_existing_per_record": False,
            },
        )(),
    )
    monkeypatch.setattr(dataset_selection, "select_dataset_files", lambda args: [tmp_path / "demo.jsonl"])
    monkeypatch.setattr(dataset_selection, "load_records", lambda paths: [record])
    monkeypatch.setattr(benchmarking_cli, "check_all_skill_health", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        benchmarking_cli,
        "summarize_skill_health",
        lambda reports: {"available_skill_count": 0, "unavailable_skill_count": 0, "available_skills": [], "unavailable_skills": []},
    )
    monkeypatch.setattr(runtime_config_pool, "ConfigPool", FakeConfigPool)
    monkeypatch.setattr(judge_runtime, "JudgeClient", lambda **kwargs: object())
    monkeypatch.setattr(
        benchmarking_cli,
        "run_benchmark_web_search_preflight",
        lambda **kwargs: {
            "enabled": True,
            "provider": "duckduckgo",
            "available": True,
            "reports": {"single_llm_skills_off": {"available": True}},
        },
    )
    monkeypatch.setattr(runner_adapters, "run_pending_cleanroom_cleanup", lambda: [])
    monkeypatch.setattr(orchestration, "run_group", fake_run_group)
    monkeypatch.setattr(benchmarking_cli, "launch_automated_evaluation", lambda output_root: (_ for _ in ()).throw(RuntimeError("boom")))

    assert benchmarking_cli.main() == 0

    manifest = json.loads((tmp_path / "out" / "runtime-manifest.json").read_text(encoding="utf-8"))
    assert manifest["automated_evaluation"]["status"] == "launch_failed"
    assert "boom" in manifest["automated_evaluation"]["error"]
