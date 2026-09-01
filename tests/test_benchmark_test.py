from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest import mock

from benchmarking.core.answer_processing import (
    extract_candidate_short_answer,
    normalize_answer_tracks,
)
from benchmarking.core.contracts import (
    AnswerPayload,
    FailureInfo,
    RecoveryInfo,
    RunnerResult,
    RunStatus,
)
from benchmarking.core.convergence import ConvergencePolicy, extract_final_answer_line
from benchmarking.core.datasets import (
    BenchmarkRecord,
    GradingSpec,
    classify_subset,
    load_records,
)
from benchmarking.core.experiments import ExperimentSpec
from benchmarking.core.reporting import (
    GroupRecordResult,
    aggregate_results,
    materialize_group_failure_results,
)
from benchmarking.core.reporting import (
    build_error_group_record_result as shared_build_error_group_record_result,
)
from benchmarking.core.status import (
    is_chemqa_terminal_status,
    normalize_chemqa_run_status,
)
from benchmarking.runtime import bundles as runtime_bundles
from benchmarking.runtime import config_pool as runtime_config_pool
from benchmarking.runtime import judge as judge_runtime
from benchmarking.runtime import paths as runtime_paths
from benchmarking.runtime import subprocess_utils
from benchmarking.runtime.cleanroom import CleanroomRuntime
from benchmarking.runtime.workspace_policy import ContaminationAudit
from benchmarking.scoring import registry as scoring_evaluation
from benchmarking.scoring.evaluators import chembench, frontierscience, superchem
from benchmarking.scoring.results import (
    EvaluationResult,
    build_execution_error_evaluation,
)
from benchmarking.skills.tree import load_chemistry_skill_inventory
from benchmarking.workflow import cli as benchmark_test
from benchmarking.workflow import (
    dataset_selection,
    experiments,
    orchestration,
    run_state,
    runner_adapters,
    runtime_config,
)
from benchmarking.workflow.chemqa_response import (
    build_chemqa_full_response,
    build_chemqa_response_from_submission,
)
from benchmarking.workflow.errors import BenchmarkError
from benchmarking.workflow.prompts import build_single_llm_prompt


@contextmanager
def patched_benchmark_runtime_paths() -> Iterator[None]:
    original_agents_root = runtime_paths.agents_root
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        runtime_paths.agents_root = root / "agents"
        try:
            yield
        finally:
            runtime_paths.agents_root = original_agents_root


class JudgeStub:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def evaluate_json(self, prompt: str) -> dict[str, object]:
        self.prompts.append(prompt)
        return dict(self.payload)


def build_error_result_for_test(**kwargs: object) -> GroupRecordResult:
    return shared_build_error_group_record_result(
        **kwargs,
        classify_subset_fn=classify_subset,
        normalize_answer_tracks_fn=normalize_answer_tracks,
        build_execution_error_evaluation_fn=build_execution_error_evaluation,
        deep_copy_jsonish_fn=subprocess_utils.deep_copy_jsonish,
    )


def materialize_failure_results_for_test(**kwargs: object) -> list[GroupRecordResult]:
    return materialize_group_failure_results(
        **kwargs,
        save_json_fn=run_state.save_json,
        slugify_fn=run_state.slugify,
        classify_subset_fn=classify_subset,
        normalize_answer_tracks_fn=normalize_answer_tracks,
        build_execution_error_evaluation_fn=build_execution_error_evaluation,
        deep_copy_jsonish_fn=subprocess_utils.deep_copy_jsonish,
    )


def run_group_for_test(**kwargs: object) -> list[GroupRecordResult]:
    experiment_specs = kwargs.pop("experiment_specs", experiments.EXPERIMENT_SPECS)
    kwargs.setdefault("single_agent_thinking", experiments.DEFAULT_SINGLE_AGENT_THINKING)
    return orchestration.run_group(
        **kwargs,
        chemqa_slot_sets=experiments.CHEMQA_SLOT_SETS,
        experiment_specs=experiment_specs,
        build_runner_fn=runner_adapters.build_runner,
        evaluate_answer_fn=scoring_evaluation.evaluate_record,
        build_error_group_record_result_fn=build_error_result_for_test,
        classify_subset_fn=classify_subset,
        save_json_fn=run_state.save_json,
        slugify_fn=run_state.slugify,
    )


class BenchmarkTestModuleTests(unittest.TestCase):
    def _single_llm_record(self, *, eval_kind: str = "superchem_multiple_choice_rpf") -> object:
        dataset = "superchem" if eval_kind == "superchem_multiple_choice_rpf" else "chembench"
        reference_answer = "B" if eval_kind == "superchem_multiple_choice_rpf" else "5"
        return BenchmarkRecord(
            record_id="demo",
            dataset=dataset,
            source_file="/tmp/demo.jsonl",
            eval_kind=eval_kind,
            prompt="Question?",
            reference_answer=reference_answer,
            payload={},
        )

    def _single_llm_completed_process(
        self,
        command: list[str],
        *,
        text: str,
        meta: dict[str, object] | None = None,
        is_error: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        payload = {
            "result": {
                "payloads": [{"text": text, **({"isError": True} if is_error else {})}],
                "meta": {
                    "stdout_diagnostics": {"schema_valid": True},
                    "session_isolation": {"session_isolation_ok": True},
                    **dict(meta or {}),
                },
            }
        }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    def test_default_experiment_groups_are_three_skills_groups(self) -> None:
        self.assertEqual(
            ["single_llm_skills_on", "single_llm_skills_off", "chemqa_skills_on"],
            list(experiments.EXPERIMENT_GROUPS),
        )
        self.assertTrue(all(not group.websearch for group in experiments.EXPERIMENT_GROUPS.values()))
        self.assertTrue(all(not spec.websearch_enabled for spec in experiments.EXPERIMENT_SPECS.values()))
        self.assertTrue(experiments.EXPERIMENT_GROUPS["single_llm_skills_on"].skills_enabled)
        self.assertFalse(experiments.EXPERIMENT_GROUPS["single_llm_skills_off"].skills_enabled)
        self.assertTrue(experiments.EXPERIMENT_GROUPS["chemqa_skills_on"].skills_enabled)

    def test_effective_experiment_specs_filter_unavailable_skills(self) -> None:
        health_reports = {
            "rdkit": {"available": True},
            "paper-access": {"available": False, "unavailable_reasons": [{"kind": "missing_dependency", "name": "bs4"}]},
        }
        specs = {
            "single_llm_skills_on": ExperimentSpec(
                id="single_llm_skills_on",
                label="demo",
                runner_kind="single_llm",
                websearch_enabled=True,
                skills_enabled=True,
                single_agent_id="benchmark-single-skills-on",
                skill_allowlist=("rdkit", "paper-access"),
            )
        }

        effective = experiments.build_effective_experiment_specs(specs, skill_health_reports=health_reports)

        self.assertEqual(("rdkit",), effective["single_llm_skills_on"].skill_allowlist)

    def test_benchmark_skills_allowlist_comes_from_skill_tree(self) -> None:
        inventory_skills = [
            str(entry["skill"])
            for entry in load_chemistry_skill_inventory().get("skills", [])
        ]

        self.assertEqual(inventory_skills, experiments.BENCHMARK_SKILLS_ALLOWLIST)
        self.assertEqual(86, len(experiments.BENCHMARK_SKILLS_ALLOWLIST))
        self.assertIn("act-like-a-chemist", experiments.BENCHMARK_SKILLS_ALLOWLIST)
        self.assertIn("chem-calculator", experiments.BENCHMARK_SKILLS_ALLOWLIST)
        self.assertIn("pymatgen", experiments.BENCHMARK_SKILLS_ALLOWLIST)
        self.assertIn("paper-retrieval", experiments.BENCHMARK_SKILLS_ALLOWLIST)
        self.assertIn("paper-access", experiments.BENCHMARK_SKILLS_ALLOWLIST)
        self.assertIn("paper-parse", experiments.BENCHMARK_SKILLS_ALLOWLIST)
        self.assertIn("paper-rerank", experiments.BENCHMARK_SKILLS_ALLOWLIST)
        self.assertNotIn("benchmark-cleanroom", experiments.BENCHMARK_SKILLS_ALLOWLIST)
        self.assertNotIn("chemqa-review", experiments.BENCHMARK_SKILLS_ALLOWLIST)
        self.assertNotIn("debateclaw-v1", experiments.BENCHMARK_SKILLS_ALLOWLIST)

    def test_single_llm_runner_does_not_use_record_scoped_skill_config(self) -> None:
        source = Path("benchmarking/workflow/runners/single_llm.py").read_text(encoding="utf-8")

        self.assertNotIn("selected_skills", source)
        self.assertNotIn("config_for_record", source)
        self.assertNotIn("SkillPlan", source)

    def test_parse_args_accepts_single_agent_id_override_and_rejects_removed_flags(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "benchmarking.workflow.cli",
                "--single-agent-id-override",
                "custom-single-agent",
            ],
        ):
            args = benchmark_test.parse_args()
        self.assertEqual("custom-single-agent", args.single_agent_id_override)

        with mock.patch.object(sys, "argv", ["benchmarking.workflow.cli", "--keep-temp-configs"]):
            with self.assertRaises(SystemExit):
                benchmark_test.parse_args()

        with mock.patch.object(sys, "argv", ["benchmarking.workflow.cli", "--single-agent", "custom"]):
            with self.assertRaises(SystemExit):
                benchmark_test.parse_args()

    def test_parse_args_accepts_convergence_policy_flags_and_rejects_finalization_grace_flag(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "benchmarking.workflow.cli",
                "--single-timeout",
                "900",
                "--max-unchanged-status-polls",
                "1",
                "--max-recovery-attempts",
                "1",
            ],
        ):
            args = benchmark_test.parse_args()

        self.assertFalse(hasattr(args, "finalization_grace_seconds"))
        self.assertEqual(1, args.max_unchanged_status_polls)
        self.assertEqual(1, args.max_recovery_attempts)

        with mock.patch.object(sys, "argv", ["benchmarking.workflow.cli", "--finalization-grace-seconds", "60"]):
            with self.assertRaises(SystemExit):
                benchmark_test.parse_args()

    def test_parse_args_accepts_single_timeout_retry_flags(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "benchmarking.workflow.cli",
                "--single-timeout-retries",
                "2",
                "--single-timeout-retry-backoff-seconds",
                "1,3",
            ],
        ):
            args = benchmark_test.parse_args()

        self.assertEqual(2, args.single_timeout_retries)
        self.assertEqual("1,3", args.single_timeout_retry_backoff_seconds)

    def test_parse_args_accepts_thinking_overrides_and_rejects_invalid_values(self) -> None:
        with mock.patch.object(sys, "argv", ["benchmarking.workflow.cli"]):
            args = benchmark_test.parse_args()
        self.assertEqual("high", args.single_agent_thinking)
        self.assertEqual("high", args.judge_agent_thinking)

        with mock.patch.object(
            sys,
            "argv",
            [
                "benchmarking.workflow.cli",
                "--single-agent-thinking",
                "medium",
                "--judge-agent-thinking",
                "minimal",
            ],
        ):
            args = benchmark_test.parse_args()
        self.assertEqual("medium", args.single_agent_thinking)
        self.assertEqual("minimal", args.judge_agent_thinking)

        with mock.patch.object(
            sys,
            "argv",
            [
                "benchmarking.workflow.cli",
                "--single-agent-thinking",
                "adaptive",
                "--judge-agent-thinking",
                "adaptive",
            ],
        ):
            args = benchmark_test.parse_args()
        self.assertEqual("adaptive", args.single_agent_thinking)
        self.assertEqual("adaptive", args.judge_agent_thinking)

        with mock.patch.object(sys, "argv", ["benchmarking.workflow.cli", "--single-agent-thinking", "extreme"]):
            with self.assertRaises(SystemExit):
                benchmark_test.parse_args()

    def test_parse_args_accepts_subsets_filter(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "benchmarking.workflow.cli",
                "--subsets",
                "frontierscience_Research,superchem_multimodal",
            ],
        ):
            args = benchmark_test.parse_args()

        self.assertEqual("frontierscience_Research,superchem_multimodal", args.subsets)

    def test_main_single_agent_override_applies_via_experiment_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "chembench" / "data" / "bench.jsonl"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "demo-record",
                        "prompt": "Question?",
                        "answer": "42",
                        "eval_kind": "chembench_open_ended",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            base_config = root / "openclaw.json"
            base_config.write_text(json.dumps({"agents": {"list": []}}, ensure_ascii=False), encoding="utf-8")
            output_root = root / "out"
            captured: dict[str, str] = {}

            class DummyConfigPool:
                def __init__(self, **_: object) -> None:
                    pass

                def config_for_group(self, group: object) -> Path:
                    path = output_root / "runtime-config" / f"{getattr(group, 'id', 'group')}-openclaw.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}", encoding="utf-8")
                    return path

                def judge_config_path(self) -> Path:
                    path = output_root / "runtime-config" / "benchmark-judge-openclaw.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}", encoding="utf-8")
                    return path

            def fake_run_group(**kwargs):
                captured["single_agent"] = kwargs["single_agent"]
                captured["single_policy_timeout"] = str(kwargs["single_convergence_policy"].timeout_seconds)
                return []

            argv = [
                "benchmarking.workflow.cli",
                "--benchmark-root",
                str(root),
                "--openclaw-config",
                str(base_config),
                "--exact-output-dir",
                str(output_root),
                "--groups",
                "single_llm_skills_off",
                "--single-agent-id-override",
                "custom-single-agent",
            ]

            with mock.patch.object(runtime_config_pool, "ConfigPool", DummyConfigPool), \
                mock.patch.object(judge_runtime, "JudgeClient", return_value=object()), \
                mock.patch.object(benchmark_test, "check_all_skill_health", return_value={}), \
                mock.patch.object(
                    benchmark_test,
                    "summarize_skill_health",
                    return_value={"available_skill_count": 0, "unavailable_skill_count": 0, "available_skills": [], "unavailable_skills": []},
                ), \
                mock.patch.object(
                    benchmark_test,
                    "run_benchmark_web_search_preflight",
                    return_value={
                        "enabled": True,
                        "provider": "duckduckgo",
                        "available": True,
                        "reports": {"single_llm_skills_off": {"available": True}},
                    },
                ), \
                mock.patch.object(orchestration, "run_group", side_effect=fake_run_group), \
                mock.patch.object(
                    benchmark_test,
                    "launch_automated_evaluation",
                    return_value={"status": "launched", "output_root": str(output_root)},
                ), \
                mock.patch.object(sys, "argv", argv):
                exit_code = benchmark_test.main()

            self.assertEqual(0, exit_code)
            self.assertEqual("custom-single-agent", captured.get("single_agent"))
            self.assertEqual("7200", captured.get("single_policy_timeout"))

    def test_main_print_selected_records_filters_by_subsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fs_path = root / "frontierscience" / "data" / "frontierscience.jsonl"
            superchem_path = root / "superchem" / "data" / "superchem.jsonl"
            fs_path.parent.mkdir(parents=True, exist_ok=True)
            superchem_path.parent.mkdir(parents=True, exist_ok=True)
            fs_path.write_text(
                "\n".join(
                    json.dumps(payload, ensure_ascii=False)
                    for payload in [
                        {
                            "id": "fs-olympiad",
                            "problem": "Olympiad problem?",
                            "answer": "A",
                            "eval_kind": "frontierscience_olympiad",
                            "track": "olympiad",
                        },
                        {
                            "id": "fs-research",
                            "problem": "Research problem?",
                            "answer": "B",
                            "eval_kind": "frontierscience_research",
                            "track": "research",
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            superchem_path.write_text(
                json.dumps(
                    {
                        "id": "superchem-mm",
                        "question": "SuperChem problem?",
                        "answer": "C",
                        "eval_kind": "superchem_multiple_choice_rpf",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            argv = [
                "benchmarking.workflow.cli",
                "--benchmark-root",
                str(root),
                "--subsets",
                "frontierscience_Research,superchem_multimodal",
                "--print-selected-records",
            ]
            stream = io.StringIO()
            with mock.patch.object(sys, "argv", argv), redirect_stdout(stream):
                exit_code = benchmark_test.main()

            self.assertEqual(0, exit_code)
            selected = json.loads(stream.getvalue())
            self.assertEqual(["fs-research", "superchem-mm"], [item["record_id"] for item in selected])
            self.assertEqual(
                ["frontierscience_Research", "superchem_multimodal"],
                [item["subset"] for item in selected],
            )

    def test_filter_records_by_subsets_rejects_unknown_subset(self) -> None:
        records = [
            BenchmarkRecord(
                record_id="chem-1",
                dataset="chembench",
                source_file="/tmp/chembench.jsonl",
                eval_kind="chembench_open_ended",
                prompt="Q",
                reference_answer="A",
                payload={},
            )
        ]

        with self.assertRaisesRegex(BenchmarkError, "Unknown subset"):
            dataset_selection.filter_records_by_subsets(records, "hle_chemistry")

    def test_current_python_prefers_virtualenv_python(self) -> None:
        original_virtual_env = os.environ.get("VIRTUAL_ENV")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                venv_root = Path(tmpdir) / ".venv"
                python_path = venv_root / "bin" / "python"
                python_path.parent.mkdir(parents=True, exist_ok=True)
                python_path.write_text("", encoding="utf-8")
                os.environ["VIRTUAL_ENV"] = str(venv_root)
                self.assertEqual(str(python_path), subprocess_utils.current_python())
        finally:
            if original_virtual_env is None:
                os.environ.pop("VIRTUAL_ENV", None)
            else:
                os.environ["VIRTUAL_ENV"] = original_virtual_env


    def test_extract_final_answer_line_prefers_explicit_marker(self) -> None:
        text = "reasoning\nFINAL ANSWER: 42\n"
        self.assertEqual("42", extract_final_answer_line(text))
        self.assertEqual("42", extract_candidate_short_answer(text))

    def test_hle_evaluator_registered_for_benchmark_dispatch(self) -> None:
        record = BenchmarkRecord(
            record_id="hle-demo",
            dataset="hle",
            source_file="/tmp/hle.jsonl",
            eval_kind="hle",
            prompt="Which reagent oxidizes a primary alcohol to an aldehyde?",
            reference_answer="PCC",
            payload={"answer_type": "short-answer", "category": "Chemistry"},
        )
        judge = JudgeStub(
            {
                "extracted_final_answer": "PCC",
                "reasoning": "The answer matches.",
                "correct": "yes",
                "confidence": 90,
            }
        )

        result = scoring_evaluation.evaluate_record(
            record,
            short_answer_text="PCC",
            full_response_text="Explanation: short\nAnswer: PCC\nConfidence: 90%",
            judge=judge,
        )

        self.assertTrue(result.passed)
        self.assertEqual("hle_judge_accuracy", result.primary_metric)

    def test_random_subset_sampling_includes_hle_chemistry(self) -> None:
        records = [
            BenchmarkRecord(
                record_id="hle-demo",
                dataset="hle",
                source_file="/tmp/hle.jsonl",
                eval_kind="hle",
                prompt="Which reagent oxidizes a primary alcohol to an aldehyde?",
                reference_answer="PCC",
                payload={"answer_type": "short-answer", "category": "Chemistry"},
            )
        ]

        sampled = dataset_selection.sample_records_per_subset(records, per_subset_count=1, seed=7)

        self.assertEqual(["hle-demo"], [record.record_id for record in sampled])

    def test_runner_result_should_score_when_recovery_is_evaluable(self) -> None:
        result = RunnerResult(
            status=RunStatus.RECOVERED,
            answer=AnswerPayload(
                short_answer_text="CCO",
                full_response_text="FINAL ANSWER: CCO",
            ),
            raw={"run_status": {"status": "done", "terminal_state": "failed"}},
            runner_meta={"run_id": "demo-run"},
            recovery=RecoveryInfo(
                source="candidate_submission",
                scored=True,
                evaluable=True,
                reliability="high_confidence_recovered",
                recovery_mode="candidate_submission",
                details={
                    "evaluable": True,
                    "reliability": "high_confidence_recovered",
                    "recovery_mode": "candidate_submission",
                },
            ),
        )

        self.assertTrue(result.should_score())

    def test_parse_frontierscience_research_rubric(self) -> None:
        rubric = """
Points: 1.0, Item: First criterion
more detail
Points: 0.5, Item: Second criterion
""".strip()
        items = frontierscience.parse_frontierscience_research_rubric(rubric)
        self.assertEqual(2, len(items))
        self.assertEqual(1.0, items[0]["points"])
        self.assertIn("First criterion", items[0]["description"])
        self.assertIn("more detail", items[0]["description"])
        self.assertEqual(0.5, items[1]["points"])


    def test_build_group_waves_batches_in_selected_order(self) -> None:
        waves = benchmark_test.build_group_waves(
            ["single_llm_skills_on", "single_llm_skills_off", "chemqa_skills_on"],
            max_concurrent_groups=2,
        )
        self.assertEqual(
            [["single_llm_skills_on", "single_llm_skills_off"], ["chemqa_skills_on"]],
            waves,
        )

    def test_build_group_waves_respects_max_concurrent_groups(self) -> None:
        waves = benchmark_test.build_group_waves(
            ["single_llm_skills_on", "single_llm_skills_off", "chemqa_skills_on"],
            max_concurrent_groups=1,
        )
        self.assertEqual(
            [["single_llm_skills_on"], ["single_llm_skills_off"], ["chemqa_skills_on"]],
            waves,
        )

    def test_resolve_aggregate_group_ids_includes_existing_group_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            existing = root / "per-record" / "single_llm_skills_on"
            existing.mkdir(parents=True, exist_ok=True)
            (existing / "demo.json").write_text("{}\n", encoding="utf-8")
            aggregate = run_state.resolve_aggregate_group_ids(
                ["chemqa_skills_on", "single_llm_skills_off"],
                output_root=root,
                merge_existing_per_record=True,
            )
            self.assertEqual(
                ["single_llm_skills_on", "single_llm_skills_off", "chemqa_skills_on"],
                aggregate,
            )

    def test_load_results_from_output_root_reads_existing_per_record_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            group_dir = root / "per-record" / "single_llm_skills_on"
            group_dir.mkdir(parents=True, exist_ok=True)
            payload = GroupRecordResult(
                schema_version=2,
                group_id="single_llm_skills_on",
                group_label="单一 LLM + 启用 websearch plugin",
                runner="single_llm",
                websearch=True,
                record_id="demo-record",
                subset="chembench",
                dataset="chembench",
                source_file="/tmp/demo.jsonl",
                eval_kind="chembench_open_ended",
                prompt="Q",
                reference_answer="A",
                answer_text="A",
                evaluation={"score": 1},
                runner_meta={},
                raw={},
                elapsed_seconds=1.0,
                run_lifecycle_status="completed",
                protocol_completion_status="completed",
                protocol_acceptance_status=None,
                answer_availability="native_final",
                answer_reliability="native",
                evaluable=True,
                scored=True,
                recovery_mode="none",
                degraded_execution=False,
                execution_error_kind=None,
                error=None,
                short_answer_text="A",
                full_response_text="FINAL ANSWER: A",
            )
            (group_dir / "demo-record.json").write_text(json.dumps(asdict(payload)), encoding="utf-8")
            loaded = run_state.load_results_from_output_root(root, group_ids=["single_llm_skills_on"])
            self.assertEqual(1, len(loaded))
            self.assertEqual("demo-record", loaded[0].record_id)
            self.assertEqual("A", loaded[0].short_answer_text)

    def test_load_results_from_output_root_upconverts_legacy_per_record_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            group_dir = root / "per-record" / "single_llm_skills_on"
            group_dir.mkdir(parents=True, exist_ok=True)
            legacy_payload = {
                "group_id": "single_llm_skills_on",
                "group_label": "单一 LLM + 启用 websearch plugin",
                "runner": "single_llm",
                "websearch": True,
                "record_id": "legacy-record",
                "subset": "chembench",
                "dataset": "chembench",
                "source_file": "/tmp/demo.jsonl",
                "eval_kind": "chembench_open_ended",
                "prompt": "Q",
                "reference_answer": "A",
                "answer_text": "A",
                "evaluation": {
                    "eval_kind": "chembench_open_ended",
                    "score": 1.0,
                    "max_score": 1.0,
                    "normalized_score": 1.0,
                    "passed": True,
                    "primary_metric": "exact_str_match",
                    "primary_metric_direction": "higher_is_better",
                    "details": {},
                },
                "runner_meta": {},
                "raw": {},
                "elapsed_seconds": 1.0,
                "error": None,
                "short_answer_text": "A",
                "full_response_text": "FINAL ANSWER: A",
            }
            (group_dir / "legacy-record.json").write_text(json.dumps(legacy_payload), encoding="utf-8")
            loaded = run_state.load_results_from_output_root(root, group_ids=["single_llm_skills_on"])
            self.assertEqual(1, len(loaded))
            entry = loaded[0]
            self.assertEqual("legacy-record", entry.record_id)
            self.assertEqual(3, entry.schema_version)
            self.assertEqual("completed", entry.run_lifecycle_status)
            self.assertEqual("completed", entry.protocol_completion_status)
            self.assertIsNone(entry.protocol_acceptance_status)
            self.assertEqual("native_final", entry.answer_availability)
            self.assertEqual("native", entry.answer_reliability)
            self.assertTrue(entry.evaluable)
            self.assertTrue(entry.scored)
            self.assertEqual("none", entry.recovery_mode)
            self.assertFalse(entry.degraded_execution)
            self.assertIsNone(entry.execution_error_kind)

    def test_load_results_from_output_root_upconverts_legacy_per_record_recovery_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            group_dir = root / "per-record" / "chemqa_skills_on"
            group_dir.mkdir(parents=True, exist_ok=True)
            legacy_payload = {
                "group_id": "chemqa_skills_on",
                "group_label": "ChemQA + 禁用 websearch plugin",
                "runner": "chemqa",
                "websearch": False,
                "record_id": "legacy-recovery-record",
                "subset": "chembench",
                "dataset": "chembench",
                "source_file": "/tmp/demo.jsonl",
                "eval_kind": "chembench_open_ended",
                "prompt": "Q",
                "reference_answer": "A",
                "answer_text": "FINAL ANSWER: fallback-answer",
                "evaluation": {
                    "eval_kind": "chembench_open_ended",
                    "score": 1.0,
                    "max_score": 1.0,
                    "normalized_score": 1.0,
                    "passed": True,
                    "primary_metric": "exact_str_match",
                    "primary_metric_direction": "higher_is_better",
                    "details": {},
                },
                "runner_meta": {
                    "fallback_used": True,
                    "fallback_source": "proposer-1-proposal",
                },
                "raw": {"run_status": {"status": "done", "terminal_state": "failed"}},
                "elapsed_seconds": 1.0,
                "error": None,
                "short_answer_text": "fallback-answer",
                "full_response_text": "FINAL ANSWER: fallback-answer",
            }
            (group_dir / "legacy-recovery-record.json").write_text(json.dumps(legacy_payload), encoding="utf-8")
            loaded = run_state.load_results_from_output_root(root, group_ids=["chemqa_skills_on"])
            self.assertEqual(1, len(loaded))
            entry = loaded[0]
            self.assertEqual("completed", entry.run_lifecycle_status)
            self.assertEqual("failed", entry.protocol_completion_status)
            self.assertEqual("recovered_candidate", entry.answer_availability)
            self.assertEqual("high_confidence_recovered", entry.answer_reliability)
            self.assertTrue(entry.evaluable)
            self.assertTrue(entry.scored)
            self.assertEqual("proposer-1-proposal", entry.recovery_mode)
            self.assertTrue(entry.degraded_execution)

    def test_load_results_from_output_root_legacy_per_record_prefers_explicit_recovery_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            group_dir = root / "per-record" / "chemqa_skills_on"
            group_dir.mkdir(parents=True, exist_ok=True)
            legacy_payload = {
                "group_id": "chemqa_skills_on",
                "group_label": "ChemQA + 禁用 websearch plugin",
                "runner": "chemqa",
                "websearch": False,
                "record_id": "legacy-preview-record",
                "subset": "chembench",
                "dataset": "chembench",
                "source_file": "/tmp/demo.jsonl",
                "eval_kind": "chembench_open_ended",
                "prompt": "Q",
                "reference_answer": "A",
                "answer_text": "FINAL ANSWER: fallback-answer",
                "evaluation": {
                    "eval_kind": "chembench_open_ended",
                    "score": 0.0,
                    "max_score": 1.0,
                    "normalized_score": 0.0,
                    "passed": False,
                    "primary_metric": "execution_error",
                    "primary_metric_direction": "higher_is_better",
                    "details": {},
                },
                "runner_meta": {
                    "fallback_used": True,
                    "fallback_source": "run-status-final-answer-preview",
                    "evaluable": False,
                    "scored": False,
                    "answer_reliability": "none",
                    "recovery_mode": "run-status-final-answer-preview",
                    "degraded_execution": True,
                },
                "raw": {"run_status": {"status": "done", "terminal_state": "failed"}},
                "elapsed_seconds": 1.0,
                "error": "preview fallback not scoreable",
                "short_answer_text": "fallback-answer",
                "full_response_text": "FINAL ANSWER: fallback-answer",
            }
            (group_dir / "legacy-preview-record.json").write_text(json.dumps(legacy_payload), encoding="utf-8")
            loaded = run_state.load_results_from_output_root(root, group_ids=["chemqa_skills_on"])
            self.assertEqual(1, len(loaded))
            entry = loaded[0]
            self.assertEqual("failed", entry.run_lifecycle_status)
            self.assertEqual("failed", entry.protocol_completion_status)
            self.assertEqual("preview_only", entry.answer_availability)
            self.assertEqual("none", entry.answer_reliability)
            self.assertFalse(entry.evaluable)
            self.assertFalse(entry.scored)
            self.assertEqual("run-status-final-answer-preview", entry.recovery_mode)
            self.assertTrue(entry.degraded_execution)
            self.assertEqual("execution_error", entry.execution_error_kind)

    def test_build_run_scoped_config_payload_uses_explicit_single_and_judge_models(self) -> None:
        base = {
            "agents": {"list": []},
            "tools": {"web": {"search": {"enabled": False}}},
            "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
        }
        group = experiments.EXPERIMENT_GROUPS["single_llm_skills_on"]
        with patched_benchmark_runtime_paths():
            payload = runtime_config.build_run_scoped_config_payload(
                base,
                group=group,
                single_agent_model="qwen3.5-plus",
                judge_model="su8/gpt-5.4",
            )
        agents = {entry["id"]: entry for entry in payload["agents"]["list"]}
        self.assertEqual("qwen3.5-plus", agents["benchmark-single-skills-on"]["model"])
        self.assertEqual("su8/gpt-5.4", agents["benchmark-judge"]["model"])
        self.assertEqual(experiments.BENCHMARK_SKILLS_ALLOWLIST, agents["benchmark-single-skills-on"]["skills"])
        self.assertNotIn("skills", agents["benchmark-judge"])
        self.assertNotIn("thinking", agents["benchmark-single-skills-on"])
        self.assertNotIn("thinking", agents["benchmark-judge"])
        self.assertFalse(payload["tools"]["web"]["search"]["enabled"])
        self.assertFalse(payload["plugins"]["entries"]["duckduckgo"]["enabled"])
        self.assertIn(str(runtime_paths.skills_root), payload["skills"]["load"]["extraDirs"])

    def test_default_judge_model_uses_openai_gpt_55(self) -> None:
        self.assertEqual("openai/gpt-5.5", experiments.DEFAULT_JUDGE_MODEL)

    def test_build_run_scoped_config_payload_disables_single_llm_skills_off(self) -> None:
        base = {
            "agents": {"list": []},
            "tools": {"web": {"search": {"enabled": False}}},
            "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
        }
        group = experiments.EXPERIMENT_GROUPS["single_llm_skills_off"]
        with patched_benchmark_runtime_paths():
            payload = runtime_config.build_run_scoped_config_payload(
                base,
                group=group,
                single_agent_model="qwen3.5-plus",
                judge_model="su8/gpt-5.4",
            )
        agents = {entry["id"]: entry for entry in payload["agents"]["list"]}
        self.assertEqual([], agents["benchmark-single-skills-off"]["skills"])
        self.assertFalse(payload["tools"]["web"]["search"]["enabled"])
        self.assertFalse(payload["plugins"]["entries"]["duckduckgo"]["enabled"])

    def test_build_run_scoped_config_payload_benchmark_judge_runtime_uses_judge_model(self) -> None:
        base = {
            "agents": {"list": []},
            "tools": {"web": {"search": {"enabled": False}}},
            "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
        }
        group = experiments.ExperimentGroup(
            id="benchmark-judge-runtime",
            label="benchmark judge runtime",
            runner="single_llm",
            websearch=False,
        )
        with patched_benchmark_runtime_paths():
            payload = runtime_config.build_run_scoped_config_payload(
                base,
                group=group,
                single_agent_model="qwen3.5-plus",
                judge_model="su8/gpt-5.4",
            )
        agents = {entry["id"]: entry for entry in payload["agents"]["list"]}
        self.assertEqual("su8/gpt-5.4", agents["benchmark-judge"]["model"])
        self.assertNotIn("thinking", agents["benchmark-judge"])

    def test_build_run_scoped_config_payload_chemqa_uses_single_model_for_all_slots(self) -> None:
        base = {
            "agents": {"list": []},
            "tools": {"web": {"search": {"enabled": False}}},
            "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
        }
        group = experiments.EXPERIMENT_GROUPS["chemqa_skills_on"]
        with patched_benchmark_runtime_paths():
            payload = runtime_config.build_run_scoped_config_payload(
                base,
                group=group,
                single_agent_model="qwen3.5-plus",
                judge_model="su8/gpt-5.4",
            )
        agents = {entry["id"]: entry for entry in payload["agents"]["list"]}
        self.assertEqual("su8/gpt-5.4", agents["benchmark-judge"]["model"])
        self.assertEqual("qwen3.5-plus", agents["debateA-coordinator"]["model"])
        self.assertEqual(experiments.BENCHMARK_SKILLS_ALLOWLIST, agents["debateA-coordinator"]["skills"])
        self.assertNotIn("skills", agents["benchmark-judge"])
        self.assertNotIn("thinking", agents["debateA-coordinator"])
        for slot in ["debateA-1", "debateA-2", "debateA-3", "debateA-4", "debateA-5"]:
            self.assertEqual("qwen3.5-plus", agents[slot]["model"])
            self.assertEqual(experiments.BENCHMARK_SKILLS_ALLOWLIST, agents[slot]["skills"])
            self.assertNotIn("thinking", agents[slot])
        self.assertFalse(payload["tools"]["web"]["search"]["enabled"])
        self.assertFalse(payload["plugins"]["entries"]["duckduckgo"]["enabled"])

    def test_build_run_scoped_config_payload_chemqa_uses_benchmark_workspace_roots(self) -> None:
        base = {
            "agents": {"list": []},
            "tools": {"web": {"search": {"enabled": False}}},
            "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
        }
        group = experiments.EXPERIMENT_GROUPS["chemqa_skills_on"]
        with patched_benchmark_runtime_paths():
            payload = runtime_config.build_run_scoped_config_payload(
                base,
                group=group,
                single_agent_model="qwen3.5-plus",
                judge_model="su8/gpt-5.4",
            )
        agents = {entry["id"]: entry for entry in payload["agents"]["list"]}
        coordinator_workspace = Path(agents["debateA-coordinator"]["workspace"])
        proposer_workspace = Path(agents["debateA-1"]["workspace"])
        self.assertIn("runs", coordinator_workspace.parts)
        self.assertIn("active", coordinator_workspace.parts)
        self.assertNotEqual(coordinator_workspace, proposer_workspace)
        self.assertNotIn(
            str(runtime_paths.benchmark_runtime_root / "chemqa_skills_on"),
            str(coordinator_workspace),
        )

    def test_build_run_scoped_config_payload_raises_benchmark_error_when_agents_list_invalid(self) -> None:
        base = {
            "agents": {"list": {}},
            "tools": {"web": {"search": {"enabled": False}}},
            "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
        }
        group = experiments.EXPERIMENT_GROUPS["single_llm_skills_on"]

        with patched_benchmark_runtime_paths():
            with self.assertRaises(BenchmarkError):
                runtime_config.build_run_scoped_config_payload(
                    base,
                    group=group,
                    single_agent_model="qwen3.5-plus",
                    judge_model="su8/gpt-5.4",
                )

    def test_normalize_chemqa_run_status_maps_completed_with_artifact_errors(self) -> None:
        payload = normalize_chemqa_run_status({"status": "completed_with_artifact_errors"})
        self.assertEqual("done", payload["status"])
        self.assertEqual("completed", payload["terminal_state"])
        self.assertEqual("artifact_collection_error", payload["terminal_reason_code"])
        self.assertEqual("error", payload["artifact_collection"]["status"])
        self.assertEqual("completed_with_artifact_errors", payload["legacy_status"])

    def test_normalize_chemqa_run_status_maps_stalled(self) -> None:
        payload = normalize_chemqa_run_status({"status": "stalled", "phase": "review"})
        self.assertEqual("done", payload["status"])
        self.assertEqual("failed", payload["terminal_state"])
        self.assertEqual("stalled", payload["terminal_reason_code"])
        self.assertEqual("stalled", payload["legacy_status"])

    def test_normalize_chemqa_run_status_maps_terminal_failure(self) -> None:
        payload = normalize_chemqa_run_status({"status": "terminal_failure", "reason": "boom"})
        self.assertEqual("done", payload["status"])
        self.assertEqual("failed", payload["terminal_state"])
        self.assertEqual("terminal_failure", payload["terminal_reason_code"])
        self.assertEqual("boom", payload["terminal_reason"])
        self.assertEqual("terminal_failure", payload["legacy_status"])

    def test_normalize_chemqa_run_status_keeps_artifact_finalizing_non_terminal(self) -> None:
        payload = normalize_chemqa_run_status(
            {
                "status": "done",
                "protocol_terminal_state": "completed",
                "artifact_flow_state": "finalizing",
                "benchmark_terminal_state": "running",
                "terminal_state": "running",
            }
        )
        self.assertEqual("running", payload["status"])
        self.assertEqual("completed", payload["protocol_terminal_state"])
        self.assertEqual("finalizing", payload["artifact_flow_state"])
        self.assertEqual("running", payload["benchmark_terminal_state"])
        self.assertFalse(is_chemqa_terminal_status(payload))

    def test_normalize_chemqa_run_status_maps_abandoned(self) -> None:
        payload = normalize_chemqa_run_status({"status": "abandoned"})
        self.assertEqual("done", payload["status"])
        self.assertEqual("cancelled", payload["terminal_state"])
        self.assertEqual("abandoned", payload["terminal_reason_code"])
        self.assertEqual("abandoned", payload["legacy_status"])

    def test_chemqa_wait_for_terminal_status_accepts_new_done_state(self) -> None:
        runner = runner_adapters.ChemQARunner.__new__(runner_adapters.ChemQARunner)
        runner._read_run_status = lambda _run_id: {"status": "done", "terminal_state": "failed", "terminal_reason_code": "stalled"}
        payload = runner_adapters.ChemQARunner._wait_for_terminal_status(runner, "demo-run", timeout_seconds=1)
        self.assertEqual("done", payload["status"])
        self.assertEqual("failed", payload["terminal_state"])

    def test_chemqa_wait_for_terminal_status_accepts_legacy_terminal_failure(self) -> None:
        runner = runner_adapters.ChemQARunner.__new__(runner_adapters.ChemQARunner)
        runner._read_run_status = lambda _run_id: normalize_chemqa_run_status({"status": "terminal_failure", "phase": "review"})
        payload = runner_adapters.ChemQARunner._wait_for_terminal_status(runner, "demo-run", timeout_seconds=1)
        self.assertEqual("done", payload["status"])
        self.assertEqual("failed", payload["terminal_state"])
        self.assertEqual("terminal_failure", payload["legacy_status"])

    def test_chemqa_wait_for_terminal_status_timeout_on_half_initialized_runner_raises_benchmark_error(self) -> None:
        runner = runner_adapters.ChemQARunner.__new__(runner_adapters.ChemQARunner)
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.chemqa_root = Path(tmpdir)
            original_time = time.time
            original_sleep = time.sleep
            times = iter([100.0, 99.0, 99.5, 101.5])
            try:
                time.time = lambda: next(times)
                time.sleep = lambda _seconds: None
                with self.assertRaises(BenchmarkError):
                    runner_adapters.ChemQARunner._wait_for_terminal_status(runner, "demo-run", timeout_seconds=1)
            finally:
                time.time = original_time
                time.sleep = original_sleep

    def test_chemqa_wait_for_terminal_status_attempts_recovery_on_stagnant_status(self) -> None:
        runner = runner_adapters.ChemQARunner.__new__(runner_adapters.ChemQARunner)
        runner.convergence_policy = ConvergencePolicy(
            timeout_seconds=10,
            max_unchanged_status_polls=1,
            max_recovery_attempts=2,
        )
        statuses = iter(
            [
                {"status": "planned", "updated_at": "2026-04-28T15:51:48Z"},
                {"status": "planned", "updated_at": "2026-04-28T15:51:48Z"},
                {"status": "done", "terminal_state": "failed", "terminal_reason_code": "terminal_failure"},
            ]
        )
        recovery_calls: list[tuple[str, dict[str, object]]] = []
        runner._read_run_status = lambda _run_id: next(statuses)
        runner._recover_stalled_run = lambda run_id, last_status: recovery_calls.append((run_id, dict(last_status))) or {"status": "running"}
        original_time = time.time
        original_sleep = time.sleep
        times = iter([100.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        try:
            time.time = lambda: next(times)
            time.sleep = lambda _seconds: None
            payload = runner_adapters.ChemQARunner._wait_for_terminal_status(runner, "demo-run", timeout_seconds=10)
        finally:
            time.time = original_time
            time.sleep = original_sleep

        self.assertEqual("done", payload["status"])
        self.assertEqual(
            [("demo-run", {"status": "planned", "updated_at": "2026-04-28T15:51:48Z"})],
            recovery_calls,
        )

    def test_build_chemqa_response_from_submission_uses_direct_answer(self) -> None:
        short_text, full_text = build_chemqa_response_from_submission(
            final_submission={
                "direct_answer": "3-(trifluoromethyl)benzamide",
                "summary": "Candidate summary.",
                "submission_trace": [{"step": "structure_proposal", "status": "success", "detail": "Picked the best matching structure."}],
            }
        )
        self.assertEqual("3-(trifluoromethyl)benzamide", short_text)
        self.assertIn("FINAL ANSWER: 3-(trifluoromethyl)benzamide", full_text)

    def test_chemqa_runner_builds_fallback_from_proposer_one_submission_when_stalled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            team_dir = Path(tmpdir) / "team"
            proposal_path = team_dir / "debate" / "artifacts" / "proposals" / "epoch-001" / "proposer-1.md"
            proposal_path.parent.mkdir(parents=True, exist_ok=True)
            proposal_path.write_text(
                "\n".join(
                    [
                        "artifact_kind: candidate_submission",
                        "artifact_contract_version: react-reviewed-v2",
                        "phase: propose",
                        "owner: proposer-1",
                        "direct_answer: 3-(trifluoromethyl)benzamide",
                        "summary: Candidate survived proposer-main reasoning.",
                    ]
                ),
                encoding="utf-8",
            )
            runner = runner_adapters.ChemQARunner.__new__(runner_adapters.ChemQARunner)
            runner._candidate_protocol_dirs = lambda _run_id, _run_status: [team_dir]
            short_text, full_text, meta = runner_adapters.ChemQARunner._build_candidate_submission_fallback(
                runner,
                "demo-run",
                {"status": "stalled", "phase": "review"},
            )
            self.assertEqual("3-(trifluoromethyl)benzamide", short_text)
            self.assertIn("FINAL ANSWER: 3-(trifluoromethyl)benzamide", full_text)
            self.assertEqual("proposer-1-proposal", meta["fallback_source"])
            self.assertEqual(str(proposal_path.resolve()), str(Path(meta["proposal_path"]).resolve()))

    def test_candidate_protocol_dirs_include_only_active_managed_coordinator_workspace(self) -> None:
        runner = runner_adapters.ChemQARunner.__new__(runner_adapters.ChemQARunner)
        runner.chemqa_root = Path("/tmp/chemqa-root")
        runner.slot_set = "A"
        runner._active_slot_workspaces = {
            "debateA-coordinator": Path("/tmp/managed/run/invocation/active/chemqa/debateA-coordinator")
        }
        legacy_protocol = (
            runtime_paths.benchmark_runtime_root
            / "chemqa_skills_on"
            / "debateA-coordinator"
            / "chemqa_review_protocol.yaml"
        )
        candidates = runner_adapters.ChemQARunner._candidate_protocol_dirs(
            runner,
            "demo-run",
            {"workspace_protocol_path": str(legacy_protocol)},
        )
        self.assertIn(
            runner._active_slot_workspaces["debateA-coordinator"],
            candidates,
        )
        self.assertNotIn(legacy_protocol.parent, candidates)
        self.assertNotIn(
            runtime_paths.benchmark_runtime_root / "chemqa_skills_on" / "debateA-coordinator",
            candidates,
        )

    def test_evaluate_chembench_open_ended_numeric_match_uses_judge(self) -> None:
        judge = JudgeStub({"correct": True, "score": 1.0, "rationale": "matches"})
        record = BenchmarkRecord(
            record_id="demo",
            dataset="chembench",
            source_file="/tmp/demo.jsonl",
            eval_kind="chembench_open_ended",
            prompt="What is 2+2?",
            reference_answer="4",
            payload={"target": "4", "preferred_score": "mae"},
        )
        result = chembench.evaluate_chembench_open_ended(
            record,
            short_answer_text="wrong-short-answer",
            full_response_text="Reasoning\nFINAL ANSWER: 4",
            answer_text="Reasoning\nFINAL ANSWER: 4",
            judge=judge,
        )
        self.assertTrue(result.passed)
        self.assertEqual(1.0, result.score)
        self.assertEqual(1.0, result.normalized_score)
        self.assertEqual("judge", result.details["method"])
        self.assertIn("Reasoning\nFINAL ANSWER: 4", judge.prompts[0])
        self.assertNotIn("wrong-short-answer", judge.prompts[0])

    def test_load_records_uses_problem_field_for_frontierscience(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "frontierscience" / "data" / "frontierscience_chemistry_pool.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "id": "fs-demo",
                        "problem": "Solve me",
                        "answer": "42",
                        "eval_kind": "frontierscience_olympiad",
                        "track": "olympiad",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            records = load_records([path])
            self.assertEqual(1, len(records))
            self.assertEqual("Solve me", records[0].prompt)
            self.assertEqual("42", records[0].reference_answer)

    def test_evaluate_frontierscience_olympiad_always_uses_judge(self) -> None:
        judge = JudgeStub({"correct": True, "score": 1.0, "rationale": "matches"})
        record = BenchmarkRecord(
            record_id="fs-demo",
            dataset="frontierscience",
            source_file="/tmp/frontierscience.jsonl",
            eval_kind="frontierscience_olympiad",
            prompt="What is 6 x 7?",
            reference_answer="42",
            payload={"track": "olympiad"},
        )
        result = frontierscience.evaluate_frontierscience_olympiad(
            record,
            short_answer_text="42",
            full_response_text="FINAL ANSWER: 42",
            judge=judge,
        )
        self.assertTrue(result.passed)
        self.assertEqual("judge", result.details["method"])
        self.assertEqual(1, len(judge.prompts))

    def test_evaluate_answer_uses_generic_semantic_fallback(self) -> None:
        judge = JudgeStub({"correct": True, "score": 1.0, "rationale": "full answer matches"})
        record = BenchmarkRecord(
            record_id="generic-demo",
            dataset="customset",
            source_file="/tmp/custom.jsonl",
            eval_kind="custom_eval_kind",
            prompt="Name the molecule.",
            reference_answer="benzene",
            payload={},
        )
        result = scoring_evaluation.evaluate_record(
            record,
            short_answer_text="wrong-short-answer",
            full_response_text="Full answer contains benzene.",
            answer_text="Full answer contains benzene.",
            judge=judge,
        )
        self.assertTrue(result.passed)
        self.assertEqual("semantic_match", result.primary_metric)
        self.assertEqual("judge", result.details["method"])
        self.assertIn("Full answer contains benzene.", judge.prompts[0])
        self.assertNotIn("wrong-short-answer", judge.prompts[0])

    def test_load_records_malformed_json_propagates_decode_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "chembench" / "data" / "broken.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"id":"broken","prompt":"Q"\n', encoding="utf-8")

            with self.assertRaises(json.JSONDecodeError):
                load_records([path])

    def test_superchem_valid_options_uses_grading_config_before_payload(self) -> None:
        record = BenchmarkRecord(
            record_id="superchem-demo",
            dataset="superchem",
            source_file="/tmp/superchem.jsonl",
            prompt="Q",
            grading=GradingSpec(
                kind="superchem_multiple_choice_rpf",
                reference_answer="A",
                subset="superchem_multimodal",
                config={"options": {"A": "x", "C": "y"}},
            ),
            raw_payload={},
        )
        self.assertEqual(("A", "C"), superchem.superchem_valid_options(record))

    def test_classify_subset(self) -> None:
        chembench_record = BenchmarkRecord(
            record_id="c1",
            dataset="chembench",
            source_file="/tmp/chembench.jsonl",
            eval_kind="chembench_open_ended",
            prompt="Q",
            reference_answer="A",
            payload={},
        )
        olympiad_record = BenchmarkRecord(
            record_id="f1",
            dataset="frontierscience",
            source_file="/tmp/frontierscience.jsonl",
            eval_kind="frontierscience_olympiad",
            prompt="Q",
            reference_answer="A",
            payload={"track": "olympiad"},
        )
        research_record = BenchmarkRecord(
            record_id="f2",
            dataset="frontierscience",
            source_file="/tmp/frontierscience.jsonl",
            eval_kind="frontierscience_research",
            prompt="Q",
            reference_answer="A",
            payload={"track": "research"},
        )
        self.assertEqual("chembench", classify_subset(chembench_record))
        self.assertEqual("frontierscience_Olympiad", classify_subset(olympiad_record))
        self.assertEqual("frontierscience_Research", classify_subset(research_record))

    def test_classify_subset_superchem(self) -> None:
        legacy_text_record = BenchmarkRecord(
            record_id="s1",
            dataset="superchem",
            source_file="/tmp/superchem.jsonl",
            eval_kind="superchem_multiple_choice_rpf",
            prompt="Q",
            reference_answer="A",
            payload={"modality": "text_only"},
        )
        multimodal_record = BenchmarkRecord(
            record_id="s2",
            dataset="superchem",
            source_file="/tmp/superchem.jsonl",
            eval_kind="superchem_multiple_choice_rpf",
            prompt="Q",
            reference_answer="A",
            payload={"modality": "multimodal"},
        )
        self.assertEqual("superchem_multimodal", classify_subset(legacy_text_record))
        self.assertEqual("superchem_multimodal", classify_subset(multimodal_record))

    def test_build_single_llm_prompt_exposes_neutral_catalog_only_for_skills_on(self) -> None:
        record = BenchmarkRecord(
            record_id="fs-1",
            dataset="frontierscience",
            source_file="/tmp/frontierscience.jsonl",
            eval_kind="frontierscience_olympiad",
            prompt="Calculate the pH of a buffer from the supplied concentrations.",
            reference_answer="4.7",
            payload={"track": "olympiad"},
        )

        skills_on = build_single_llm_prompt(
            record,
            websearch_enabled=True,
            skills_enabled=True,
            input_bundle=None,
        )
        skills_off = build_single_llm_prompt(
            record,
            websearch_enabled=True,
            skills_enabled=False,
            input_bundle=None,
        )

        self.assertIn("Chemistry skill catalog:", skills_on)
        self.assertIn(
            "The catalog describes available capabilities; whether and how to use a skill is your choice.",
            skills_on,
        )
        self.assertIn("act-like-a-chemist", skills_on)
        self.assertIn("paper-pipeline", skills_on)
        self.assertTrue(skills_on.endswith(record.prompt))
        self.assertEqual(record.prompt, skills_off)
        for prompt in (skills_on, skills_off):
            self.assertNotIn("Read act-like-a-chemist first", prompt)
            self.assertNotIn("Atomic Coverage Checklist", prompt)
            self.assertNotIn("Do not use OpenClaw skills", prompt)

    def test_build_single_llm_prompt_only_adds_time_budget_not_coverage_sop(self) -> None:
        record = BenchmarkRecord(
            record_id="fs-1",
            dataset="frontierscience",
            source_file="/tmp/frontierscience.jsonl",
            eval_kind="frontierscience_olympiad",
            prompt="Calculate the pH of a buffer from the supplied concentrations.",
            reference_answer="4.7",
            payload={"track": "olympiad"},
        )

        prompt = build_single_llm_prompt(
            record,
            websearch_enabled=True,
            skills_enabled=True,
            input_bundle=None,
            time_budget_seconds=900,
        )

        self.assertIn("Time budget: 900 seconds", prompt)
        self.assertIn("Chemistry skill catalog:", prompt)
        self.assertIn("act-like-a-chemist", prompt)
        self.assertNotIn("Atomic Coverage Checklist", prompt)
        self.assertNotIn("Read act-like-a-chemist first", prompt)
        self.assertNotIn("Do not skip task-relevant derivation steps", prompt)
        self.assertNotIn("FINAL ANSWER", prompt)

    def test_sample_records_per_subset_draws_requested_count(self) -> None:
        records = []
        for idx in range(3):
            records.append(
                BenchmarkRecord(
                    record_id=f"chem-{idx}",
                    dataset="chembench",
                    source_file="/tmp/chembench.jsonl",
                    eval_kind="chembench_open_ended",
                    prompt="Q",
                    reference_answer="A",
                    payload={},
                )
            )
            records.append(
                BenchmarkRecord(
                    record_id=f"oly-{idx}",
                    dataset="frontierscience",
                    source_file="/tmp/frontierscience.jsonl",
                    eval_kind="frontierscience_olympiad",
                    prompt="Q",
                    reference_answer="A",
                    payload={"track": "olympiad"},
                )
            )
            records.append(
                BenchmarkRecord(
                    record_id=f"res-{idx}",
                    dataset="frontierscience",
                    source_file="/tmp/frontierscience.jsonl",
                    eval_kind="frontierscience_research",
                    prompt="Q",
                    reference_answer="A",
                    payload={"track": "research"},
                )
            )
        sampled = dataset_selection.sample_records_per_subset(records, per_subset_count=2, seed=7)
        self.assertEqual(6, len(sampled))
        counts = {subset: 0 for subset in dataset_selection.SUBSET_ORDER}
        for record in sampled:
            counts[classify_subset(record)] += 1
        self.assertEqual(2, counts["chembench"])
        self.assertEqual(2, counts["frontierscience_Olympiad"])
        self.assertEqual(2, counts["frontierscience_Research"])

    def test_sample_records_per_subset_samples_superchem_multimodal_only(self) -> None:
        records = [
            BenchmarkRecord(
                record_id="s1-mm",
                dataset="superchem",
                source_file="/tmp/superchem.jsonl",
                eval_kind="superchem_multiple_choice_rpf",
                prompt="Q1",
                reference_answer="A",
                payload={"modality": "multimodal", "source_uuid": "uuid-1"},
            ),
            BenchmarkRecord(
                record_id="s2-mm",
                dataset="superchem",
                source_file="/tmp/superchem.jsonl",
                eval_kind="superchem_multiple_choice_rpf",
                prompt="Q2",
                reference_answer="B",
                payload={"modality": "multimodal", "source_uuid": "uuid-2"},
            ),
        ]
        sampled = dataset_selection.sample_records_per_subset(records, per_subset_count=1, seed=3)
        self.assertEqual(1, len(sampled))
        self.assertEqual(
            {"superchem_multimodal"},
            {classify_subset(record) for record in sampled},
        )
        self.assertEqual(1, len({record.payload["source_uuid"] for record in sampled}))

    def test_print_selected_records_outputs_json(self) -> None:
        records = [
            BenchmarkRecord(
                record_id="chem-1",
                dataset="chembench",
                source_file="/tmp/chembench.jsonl",
                eval_kind="chembench_open_ended",
                prompt="What is the answer?",
                reference_answer="5",
                payload={},
            )
        ]
        stream = io.StringIO()
        with redirect_stdout(stream):
            dataset_selection.print_selected_records(records)
        payload = json.loads(stream.getvalue())
        self.assertEqual("chem-1", payload[0]["record_id"])
        self.assertEqual("chembench", payload[0]["subset"])
        self.assertEqual("chembench_open_ended", payload[0]["eval_kind"])

    def test_apply_offset_limit_preserves_existing_behavior(self) -> None:
        records = [
            BenchmarkRecord(
                record_id=f"r{idx}",
                dataset="chembench",
                source_file="/tmp/demo.jsonl",
                eval_kind="chembench_open_ended",
                prompt="Q",
                reference_answer="A",
                payload={},
            )
            for idx in range(10)
        ]
        sliced = dataset_selection.apply_offset_limit(records, offset=3, limit=4)
        self.assertEqual(["r3", "r4", "r5", "r6"], [record.record_id for record in sliced])

    def test_parse_superchem_option_answer_handles_common_formats(self) -> None:
        valid_options = ("A", "B", "C", "D")
        self.assertEqual("B", superchem.parse_superchem_option_answer("FINAL ANSWER: B", valid_options=valid_options))
        self.assertEqual("A|D", superchem.parse_superchem_option_answer("Option A and D are correct.", valid_options=valid_options))
        self.assertEqual(
            "B|C",
            superchem.parse_superchem_option_answer('{"answer": ["C", "B"]}', valid_options=valid_options),
        )

    def test_ensure_runtime_bundle_copies_superchem_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            data_dir = temp_dir / "data"
            source_image = temp_dir / "assets" / "source.png"
            source_image.parent.mkdir(parents=True)
            source_image.write_bytes(b"image-bytes")
            record = BenchmarkRecord(
                record_id="superchem-demo-mm",
                dataset="superchem",
                source_file=str(data_dir / "superchem.jsonl"),
                eval_kind="superchem_multiple_choice_rpf",
                prompt="Question prompt",
                reference_answer="B",
                payload={
                    "source_uuid": "uuid-demo",
                    "modality": "multimodal",
                    "question": "Question prompt",
                    "options": {"A": "foo", "B": "bar"},
                    "question_image_paths": ["../assets/source.png"],
                    "option_image_paths": {},
                },
            )
            bundle = runtime_bundles.ensure_runtime_bundle(record, bundle_root=temp_dir / "bundles")
            assert bundle is not None
            self.assertTrue(bundle.question_markdown.is_file())
            self.assertIn("Local images to inspect", bundle.question_markdown.read_text(encoding="utf-8"))
            self.assertEqual(1, len(bundle.image_files))
            self.assertTrue(bundle.image_files[0].is_file())
            self.assertEqual(b"image-bytes", bundle.image_files[0].read_bytes())

    def test_ensure_runtime_bundle_prunes_superchem_question_images_to_visible_locator(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            data_dir = temp_dir / "data"
            assets_dir = temp_dir / "assets"
            locator = "/media/uploads/question-visible.png"
            visible_image = assets_dir / runtime_bundles._superchem_asset_cache_relative_path(locator)
            visible_image.parent.mkdir(parents=True)
            visible_image.write_bytes(b"visible")
            noisy_paths: list[str] = []
            for index in range(120):
                noisy = assets_dir / "_shared" / "noise" / f"unused-{index:03d}.png"
                noisy.parent.mkdir(parents=True, exist_ok=True)
                noisy.write_bytes(b"noise")
                noisy_paths.append(os.path.relpath(noisy, start=data_dir).replace(os.sep, "/"))
            visible_relpath = os.path.relpath(visible_image, start=data_dir).replace(os.sep, "/")
            record = BenchmarkRecord(
                record_id="superchem-visible-question-mm",
                dataset="superchem",
                source_file=str(data_dir / "superchem.jsonl"),
                eval_kind="superchem_multiple_choice_rpf",
                prompt=f"Question ![q]({locator})",
                reference_answer="A",
                payload={
                    "source_uuid": "uuid-visible-question",
                    "modality": "multimodal",
                    "question": f"Question ![q]({locator})",
                    "options": {"A": "answer"},
                    "question_image_paths": noisy_paths[:60] + [visible_relpath] + noisy_paths[60:],
                    "option_image_paths": {},
                },
            )

            bundle = runtime_bundles.ensure_runtime_bundle(record, bundle_root=temp_dir / "bundles")

            assert bundle is not None
            markdown = bundle.question_markdown.read_text(encoding="utf-8")
            self.assertEqual(1, len(bundle.image_files))
            self.assertEqual(b"visible", bundle.image_files[0].read_bytes())
            self.assertNotIn(locator, markdown)
            self.assertIn("](images/img01.png)", markdown)
            self.assertEqual(1, markdown.count("- images/img"))

    def test_ensure_runtime_bundle_ignores_shared_option_bucket_and_rewrites_visible_options(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            data_dir = temp_dir / "data"
            assets_dir = temp_dir / "assets"
            locator_b = "/media/uploads/option-b.png"
            locator_c = "https://superchem.pku.edu.cn/media/uploads/option-c.jpg"
            option_b = assets_dir / runtime_bundles._superchem_asset_cache_relative_path(locator_b)
            option_c = assets_dir / runtime_bundles._superchem_asset_cache_relative_path(locator_c)
            option_b.parent.mkdir(parents=True)
            option_c.parent.mkdir(parents=True)
            option_b.write_bytes(b"option-b")
            option_c.write_bytes(b"option-c")
            shared_paths: list[str] = []
            for index in range(150):
                noisy = assets_dir / "_shared" / "shared" / f"unused-{index:03d}.png"
                noisy.parent.mkdir(parents=True, exist_ok=True)
                noisy.write_bytes(b"noise")
                shared_paths.append(os.path.relpath(noisy, start=data_dir).replace(os.sep, "/"))
            record = BenchmarkRecord(
                record_id="superchem-visible-options-mm",
                dataset="superchem",
                source_file=str(data_dir / "superchem.jsonl"),
                eval_kind="superchem_multiple_choice_rpf",
                prompt="Question",
                reference_answer="B",
                payload={
                    "source_uuid": "uuid-visible-options",
                    "modality": "multimodal",
                    "question": "Question",
                    "options": {
                        "A": "plain",
                        "B": f"<MultiModal>![b]({locator_b})</MultiModal>",
                        "C": f"<MultiModal>![c]({locator_c})</MultiModal>",
                    },
                    "question_image_paths": [],
                    "option_image_paths": {
                        "_shared": shared_paths,
                        "B": [os.path.relpath(option_b, start=data_dir).replace(os.sep, "/")],
                        "C": [os.path.relpath(option_c, start=data_dir).replace(os.sep, "/")],
                    },
                },
            )

            bundle = runtime_bundles.ensure_runtime_bundle(record, bundle_root=temp_dir / "bundles")

            assert bundle is not None
            markdown = bundle.question_markdown.read_text(encoding="utf-8")
            self.assertEqual(2, len(bundle.image_files))
            self.assertEqual([b"option-b", b"option-c"], [path.read_bytes() for path in bundle.image_files])
            self.assertNotIn(locator_b, markdown)
            self.assertNotIn(locator_c, markdown)
            self.assertIn("](images/img01.png)", markdown)
            self.assertIn("](images/img02.jpg)", markdown)
            self.assertEqual(2, markdown.count("- images/img"))
            self.assertNotIn("unused-", markdown)

    def test_ensure_runtime_bundle_fails_when_visible_superchem_locator_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            locator = "/media/uploads/missing-visible.png"
            record = BenchmarkRecord(
                record_id="superchem-missing-visible-mm",
                dataset="superchem",
                source_file=str(temp_dir / "data" / "superchem.jsonl"),
                eval_kind="superchem_multiple_choice_rpf",
                prompt=f"Question ![q]({locator})",
                reference_answer="A",
                payload={
                    "source_uuid": "uuid-missing-visible",
                    "modality": "multimodal",
                    "question": f"Question ![q]({locator})",
                    "options": {"A": "answer"},
                    "question_image_paths": [],
                    "option_image_paths": {},
                },
            )

            with self.assertRaisesRegex(runtime_bundles.RuntimeBundleError, "missing-visible"):
                runtime_bundles.ensure_runtime_bundle(record, bundle_root=temp_dir / "bundles")

    def test_ensure_runtime_bundle_rejects_superchem_absolute_asset_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            source_image = temp_dir / "assets" / "source.png"
            source_image.parent.mkdir(parents=True)
            source_image.write_bytes(b"image-bytes")
            record = BenchmarkRecord(
                record_id="superchem-absolute-mm",
                dataset="superchem",
                source_file=str(temp_dir / "data" / "superchem.jsonl"),
                eval_kind="superchem_multiple_choice_rpf",
                prompt="Question prompt",
                reference_answer="B",
                payload={
                    "source_uuid": "uuid-demo",
                    "modality": "multimodal",
                    "question": "Question prompt",
                    "options": {"A": "foo", "B": "bar"},
                    "question_image_paths": [source_image.as_posix()],
                    "option_image_paths": {},
                },
            )
            with self.assertRaises(runtime_bundles.RuntimeBundleError):
                runtime_bundles.ensure_runtime_bundle(record, bundle_root=temp_dir / "bundles")

    def test_ensure_runtime_bundle_fails_when_superchem_multimodal_images_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            record = BenchmarkRecord(
                record_id="superchem-missing-mm",
                dataset="superchem",
                source_file="/tmp/superchem.jsonl",
                eval_kind="superchem_multiple_choice_rpf",
                prompt="Question prompt",
                reference_answer="B",
                payload={
                    "source_uuid": "uuid-demo",
                    "modality": "multimodal",
                    "question": "Question prompt",
                    "options": {"A": "foo", "B": "bar"},
                    "question_image_paths": ["/missing/source.png"],
                    "option_image_paths": {},
                },
            )
            with self.assertRaises(runtime_bundles.RuntimeBundleError):
                runtime_bundles.ensure_runtime_bundle(record, bundle_root=temp_dir / "bundles")

    def test_ensure_runtime_bundle_materializes_hle_base64_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            data_uri = "data:image/png;base64," + base64.b64encode(b"png-bytes").decode("ascii")
            record = BenchmarkRecord(
                record_id="hle-chemistry-demo",
                dataset="hle",
                source_file="/tmp/hle.jsonl",
                eval_kind="hle",
                prompt="Using the provided information, identify the step.",
                reference_answer="Final step",
                payload={
                    "question": "Using the provided information, identify the step.",
                    "answer": "Final step",
                    "image": data_uri,
                },
            )
            bundle = runtime_bundles.ensure_runtime_bundle(record, bundle_root=temp_dir / "bundles")
            assert bundle is not None
            self.assertEqual(1, len(bundle.image_files))
            self.assertEqual(b"png-bytes", bundle.image_files[0].read_bytes())
            question_text = bundle.question_markdown.read_text(encoding="utf-8")
            self.assertIn("# HLE Benchmark Record", question_text)
            self.assertIn("images/hle-image-01.png", question_text)
            prompt = build_single_llm_prompt(
                record,
                websearch_enabled=True,
                skills_enabled=False,
                input_bundle=bundle,
            )
            self.assertIn("Read the question bundle file first", prompt)
            self.assertIn("Inspect the local image files referenced in the bundle", prompt)

    def test_build_chemqa_full_response_uses_final_submission_rationale(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            final_submission = {
                "direct_answer": "F",
                "summary": "Probe A needs esterase cleavage; probe B does not.",
                "submission_trace": [
                    {"step": "structural-analysis", "status": "success", "detail": "Identified acetate esters and thiourea."}
                ],
                "evidence_limits": ["No literature retrieval was run."],
                "claim_anchors": [{"anchor": "claim-1", "claim": "A requires enzymatic activation."}],
            }
            final_submission_path = temp_dir / "final_submission.json"
            final_submission_path.write_text(json.dumps(final_submission), encoding="utf-8")
            qa_result = {
                "final_answer": "F",
                "artifact_paths": {
                    "final_submission": str(final_submission_path),
                },
            }
            short_text, full_text = build_chemqa_full_response(qa_result=qa_result)
            self.assertEqual("F", short_text)
            self.assertIn("Summary:", full_text)
            self.assertIn("Probe A needs esterase cleavage", full_text)
            self.assertIn("Reasoning / submission trace:", full_text)
            self.assertIn("FINAL ANSWER: F", full_text)

    def test_build_chemqa_full_response_uses_canonical_final_answer_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            final_artifact_path = temp_dir / "final_answer_artifact.json"
            final_artifact_path.write_text(
                json.dumps(
                    {
                        "terminal_state": "completed",
                        "answer_kind": "multi_part_research_answer",
                        "evaluator_answer": "Catalysis proceeds through the two-step pathway.",
                        "display_answer": "Two-step catalytic pathway",
                        "full_answer": "Detailed pathway rationale.",
                    }
                ),
                encoding="utf-8",
            )
            qa_result = {
                "terminal_state": "completed",
                "artifact_paths": {"final_answer_artifact": str(final_artifact_path)},
            }

            short_text, full_text = build_chemqa_full_response(qa_result=qa_result)

            self.assertEqual("Catalysis proceeds through the two-step pathway.", short_text)
            self.assertEqual("Detailed pathway rationale.", full_text)

    def test_build_chemqa_full_response_uses_final_artifact_evaluator_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "final_answer_artifact.json"
            path.write_text(
                json.dumps(
                    {
                        "evaluator_answer": "7.59 μg",
                        "display_answer": "7.59 μg",
                        "full_answer": "Long derivation ending in 7.59 μg.",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            short, full = build_chemqa_full_response(
                qa_result={"artifact_paths": {"final_answer_artifact": str(path)}}
            )

        self.assertEqual("7.59 μg", short)
        self.assertEqual("Long derivation ending in 7.59 μg.", full)

    def test_build_chemqa_full_response_rejected_blob_does_not_return_blob_as_short_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            rejection_blob = {
                "accepted_owner": "",
                "answer": None,
                "direct_answer": None,
                "summary": "No candidate submission achieved acceptance.",
            }
            final_answer_path = temp_dir / "final_answer.md"
            final_answer_path.write_text(json.dumps(rejection_blob, ensure_ascii=False, indent=2), encoding="utf-8")
            qa_result = {
                "final_answer": json.dumps(rejection_blob, ensure_ascii=False, indent=2),
                "acceptance_status": "rejected",
                "terminal_state": "completed",
                "artifact_paths": {
                    "final_answer": str(final_answer_path),
                },
            }

            short_text, full_text = build_chemqa_full_response(qa_result=qa_result)

            self.assertEqual("", short_text)
            self.assertIn("No candidate submission achieved acceptance.", full_text)
            self.assertNotIn("FINAL ANSWER:", full_text)

    def test_evaluate_superchem_multiple_choice_rpf(self) -> None:
        record = BenchmarkRecord(
            record_id="superchem-1",
            dataset="superchem",
            source_file="/tmp/superchem.jsonl",
            eval_kind="superchem_multiple_choice_rpf",
            prompt="Select the best answer.",
            reference_answer="B",
            payload={
                "options": {"A": "opt-a", "B": "opt-b"},
                "reference_reasoning": (
                    "<Checkpoint id='1' weight='2.0'>Use the first principle.</Checkpoint>"
                    "<Checkpoint id='2'>Confirm the reagent identity.</Checkpoint>"
                ),
            },
        )
        judge = JudgeStub(
            {
                "answer_correct": True,
                "items": [
                    {"index": 1, "matched": True, "rationale": "covered"},
                    {"index": 2, "matched": False, "rationale": "missing"},
                ],
                "summary": "partial",
            }
        )
        result = superchem.evaluate_superchem_multiple_choice_rpf(
            record,
            short_answer_text="A",
            full_response_text="Reasoning\nFINAL ANSWER: B",
            answer_text="Reasoning\nFINAL ANSWER: B",
            judge=judge,
        )
        self.assertTrue(result.passed)
        self.assertEqual(1.0, result.score)
        self.assertAlmostEqual(2.0 / 3.0, result.details["rpf"])
        self.assertEqual(1.0, result.details["answer_accuracy"])
        self.assertEqual("Reasoning\nFINAL ANSWER: B", result.details["candidate_answer_text"])
        self.assertEqual(2, len(result.details["checkpoint_matches"]))
        self.assertEqual(1, len(judge.prompts))
        self.assertIn("Reasoning", judge.prompts[0])
        self.assertNotIn("FINAL ANSWER: A", judge.prompts[0])

    def test_aggregate_results_groups_by_experiment(self) -> None:
        sample = [
            GroupRecordResult(
                schema_version=2,
                group_id="g1",
                group_label="Group 1",
                runner="single_llm",
                websearch=False,
                record_id="r1",
                subset="chembench",
                dataset="d1",
                source_file="/tmp/a.jsonl",
                eval_kind="chembench_open_ended",
                prompt="Q1",
                reference_answer="1",
                answer_text="1",
                evaluation={
                    "eval_kind": "chembench_open_ended",
                    "score": 1.0,
                    "max_score": 1.0,
                    "normalized_score": 1.0,
                    "passed": True,
                    "primary_metric": "exact_str_match",
                    "primary_metric_direction": "higher_is_better",
                    "details": {},
                },
                runner_meta={
                    "skill_use_audit": {
                        "skills_enabled": True,
                        "tool_call_count": 5,
                        "openclaw_tool_call_count": 5,
                        "tool_failure_count": 3,
                        "openclaw_tool_failure_count": 3,
                        "skill_tool_call_count": 2,
                        "skill_tool_failure_count": 1,
                        "missing_skill_doc_read_count": 1,
                        "request_shape_error_count": 2,
                        "coverage_checklist_present": True,
                        "skill_tool_executed": True,
                        "model_declared_skip": False,
                        "no_tool_call": False,
                        "no_skill_tool_call": False,
                    },
                    "session_isolation": {
                        "session_isolation_ok": True,
                        "preflight_removed_stale_main_entry": True,
                    },
                    "workspace_isolation": {
                        "preflight_ok": True,
                        "audit_status": "clean",
                        "archive_ok": True,
                        "contaminated": False,
                    },
                },
                raw={},
                elapsed_seconds=2.0,
                run_lifecycle_status="completed",
                protocol_completion_status="completed",
                protocol_acceptance_status=None,
                answer_availability="native_final",
                answer_reliability="native",
                evaluable=True,
                scored=True,
                recovery_mode="none",
                degraded_execution=False,
                skills_enabled=True,
                execution_error_kind=None,
            ),
            GroupRecordResult(
                schema_version=2,
                group_id="g1",
                group_label="Group 1",
                runner="single_llm",
                websearch=False,
                record_id="r2",
                subset="chembench",
                dataset="d1",
                source_file="/tmp/a.jsonl",
                eval_kind="chembench_open_ended",
                prompt="Q2",
                reference_answer="2",
                answer_text="3",
                evaluation={
                    "eval_kind": "chembench_open_ended",
                    "score": 0.0,
                    "max_score": 1.0,
                    "normalized_score": 0.0,
                    "passed": False,
                    "primary_metric": "exact_str_match",
                    "primary_metric_direction": "higher_is_better",
                    "details": {},
                },
                runner_meta={
                    "skill_use_audit": {
                        "skills_enabled": True,
                        "tool_call_count": 3,
                        "openclaw_tool_call_count": 3,
                        "tool_failure_count": 1,
                        "openclaw_tool_failure_count": 1,
                        "skill_tool_call_count": 0,
                        "skill_tool_failure_count": 0,
                        "missing_skill_doc_read_count": 0,
                        "request_shape_error_count": 1,
                        "coverage_checklist_present": False,
                        "skill_tool_executed": False,
                        "model_declared_skip": True,
                        "no_tool_call": False,
                        "no_skill_tool_call": True,
                    },
                    "session_isolation": {
                        "session_isolation_ok": False,
                        "preflight_removed_stale_main_entry": False,
                        "postflight_entry_session_id": "old-session",
                    },
                    "workspace_isolation": {
                        "preflight_ok": True,
                        "audit_status": "contaminated",
                        "archive_ok": False,
                        "contaminated": True,
                    },
                },
                raw={},
                elapsed_seconds=4.0,
                run_lifecycle_status="completed",
                protocol_completion_status="completed",
                protocol_acceptance_status=None,
                answer_availability="native_final",
                answer_reliability="native",
                evaluable=True,
                scored=True,
                recovery_mode="none",
                degraded_execution=False,
                skills_enabled=True,
                execution_error_kind=None,
            ),
        ]
        summary = aggregate_results(sample)
        self.assertEqual(2, summary["groups"]["g1"]["count"])
        self.assertEqual(1, summary["groups"]["g1"]["pass_count"])
        self.assertEqual(3.0, summary["groups"]["g1"]["avg_elapsed_seconds"])
        self.assertEqual(0.5, summary["groups"]["g1"]["avg_normalized_score"])
        self.assertEqual(1, summary["groups"]["g1"]["skill_tool_executed_count"])
        self.assertEqual(1, summary["groups"]["g1"]["skill_model_declared_skip_count"])
        self.assertEqual(1, summary["groups"]["g1"]["skill_no_tool_call_count"])
        self.assertEqual(0, summary["groups"]["g1"]["exec_tool_call_total"])
        self.assertEqual(0, summary["groups"]["g1"]["exec_tool_failure_total"])
        self.assertEqual(2, summary["groups"]["g1"]["skill_tool_call_total"])
        self.assertEqual(1, summary["groups"]["g1"]["skill_tool_failure_total"])
        self.assertEqual(8, summary["groups"]["g1"]["openclaw_tool_call_total"])
        self.assertEqual(4, summary["groups"]["g1"]["openclaw_tool_failure_total"])
        self.assertEqual(1, summary["groups"]["g1"]["missing_skill_doc_read_total"])
        self.assertEqual(3, summary["groups"]["g1"]["request_shape_error_total"])
        self.assertEqual(1, summary["groups"]["g1"]["coverage_checklist_present_count"])
        self.assertEqual(1, summary["groups"]["g1"]["session_isolation_ok_count"])
        self.assertEqual(1, summary["groups"]["g1"]["session_isolation_failed_count"])
        self.assertEqual(1, summary["groups"]["g1"]["session_contaminated_count"])
        self.assertEqual(1, summary["groups"]["g1"]["workspace_isolation_ok_count"])
        self.assertEqual(1, summary["groups"]["g1"]["workspace_isolation_failed_count"])
        self.assertEqual(0, summary["groups"]["g1"]["workspace_contaminated_count"])
        self.assertEqual(1, summary["groups"]["g1"]["contamination_indeterminate_count"])
        self.assertEqual(1, summary["groups"]["g1"]["workspace_archive_failed_count"])

        sample[0].runner_meta["workspace_isolation"]["audit_status"] = "unavailable"
        unavailable_summary = aggregate_results(sample)
        self.assertEqual(0, unavailable_summary["groups"]["g1"]["workspace_isolation_ok_count"])
        self.assertEqual(2, unavailable_summary["groups"]["g1"]["workspace_isolation_failed_count"])

    def test_aggregate_results_tracks_evaluable_and_degraded_counts(self) -> None:
        sample = [
            GroupRecordResult(
                schema_version=2,
                group_id="g1",
                group_label="Group 1",
                runner="chemqa",
                websearch=True,
                record_id="r1",
                subset="chembench",
                dataset="d1",
                source_file="/tmp/a.jsonl",
                eval_kind="chembench_open_ended",
                prompt="Q1",
                reference_answer="1",
                answer_text="1",
                evaluation={
                    "eval_kind": "chembench_open_ended",
                    "score": 1.0,
                    "max_score": 1.0,
                    "normalized_score": 1.0,
                    "passed": True,
                    "primary_metric": "exact_str_match",
                    "primary_metric_direction": "higher_is_better",
                    "details": {},
                },
                runner_meta={},
                raw={},
                elapsed_seconds=2.0,
                run_lifecycle_status="completed",
                protocol_completion_status="failed",
                protocol_acceptance_status="rejected",
                answer_availability="recovered_candidate",
                answer_reliability="high_confidence_recovered",
                evaluable=True,
                scored=True,
                recovery_mode="candidate_submission",
                degraded_execution=True,
                execution_error_kind=None,
            ),
            GroupRecordResult(
                schema_version=2,
                group_id="g1",
                group_label="Group 1",
                runner="chemqa",
                websearch=True,
                record_id="r2",
                subset="chembench",
                dataset="d1",
                source_file="/tmp/a.jsonl",
                eval_kind="chembench_open_ended",
                prompt="Q2",
                reference_answer="2",
                answer_text="",
                evaluation={
                    "eval_kind": "chembench_open_ended",
                    "score": 0.0,
                    "max_score": 1.0,
                    "normalized_score": 0.0,
                    "passed": False,
                    "primary_metric": "execution_error",
                    "primary_metric_direction": "higher_is_better",
                    "details": {},
                },
                runner_meta={},
                raw={},
                elapsed_seconds=4.0,
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
                error="missing answer",
            ),
        ]
        summary = aggregate_results(sample)
        bucket = summary["groups"]["g1"]
        self.assertEqual(1, bucket["run_completed_count"])
        self.assertEqual(1, bucket["run_failed_count"])
        self.assertEqual(0, bucket["protocol_completed_count"])
        self.assertEqual(1, bucket["protocol_failed_count"])
        self.assertEqual(1, bucket["evaluable_count"])
        self.assertEqual(1, bucket["scored_count"])
        self.assertEqual(1, bucket["recovered_evaluable_count"])
        self.assertEqual(0, bucket["native_evaluable_count"])
        self.assertEqual(1, bucket["non_evaluable_count"])
        self.assertEqual(1, bucket["degraded_execution_count"])

    def test_aggregate_results_includes_superchem_metrics(self) -> None:
        sample = [
            GroupRecordResult(
                schema_version=2,
                group_id="g1",
                group_label="Group 1",
                runner="single_llm",
                websearch=False,
                record_id="superchem-1",
                subset="superchem_multimodal",
                dataset="superchem",
                source_file="/tmp/superchem.jsonl",
                eval_kind="superchem_multiple_choice_rpf",
                prompt="Q1",
                reference_answer="B",
                answer_text="B",
                evaluation={
                    "eval_kind": "superchem_multiple_choice_rpf",
                    "score": 1.0,
                    "max_score": 1.0,
                    "normalized_score": 1.0,
                    "passed": True,
                    "primary_metric": "answer_accuracy",
                    "primary_metric_direction": "higher_is_better",
                    "details": {"answer_accuracy": 1.0, "rpf": 0.75},
                },
                runner_meta={},
                raw={},
                elapsed_seconds=5.0,
                run_lifecycle_status="completed",
                protocol_completion_status="completed",
                protocol_acceptance_status=None,
                answer_availability="native_final",
                answer_reliability="native",
                evaluable=True,
                scored=True,
                recovery_mode="none",
                degraded_execution=False,
                execution_error_kind=None,
            )
        ]
        summary = aggregate_results(sample)
        self.assertEqual(1.0, summary["groups"]["g1"]["avg_answer_accuracy"])
        self.assertEqual(0.75, summary["groups"]["g1"]["avg_rpf"])

    def test_aggregate_results_includes_hle_calibration_error(self) -> None:
        sample = [
            GroupRecordResult(
                schema_version=2,
                group_id="g1",
                group_label="Group 1",
                runner="single_llm",
                websearch=False,
                record_id="hle-1",
                subset="hle_chemistry",
                dataset="hle",
                source_file="/tmp/hle.jsonl",
                eval_kind="hle",
                prompt="Q1",
                reference_answer="A",
                answer_text="A",
                evaluation={
                    "eval_kind": "hle",
                    "score": 1.0,
                    "max_score": 1.0,
                    "normalized_score": 1.0,
                    "passed": True,
                    "primary_metric": "hle_judge_accuracy",
                    "primary_metric_direction": "higher_is_better",
                    "details": {"confidence": 80},
                },
                runner_meta={},
                raw={},
                elapsed_seconds=5.0,
                run_lifecycle_status="completed",
                protocol_completion_status="completed",
                protocol_acceptance_status=None,
                answer_availability="native_final",
                answer_reliability="native",
                evaluable=True,
                scored=True,
                recovery_mode="none",
                degraded_execution=False,
                execution_error_kind=None,
            ),
            GroupRecordResult(
                schema_version=2,
                group_id="g1",
                group_label="Group 1",
                runner="single_llm",
                websearch=False,
                record_id="hle-2",
                subset="hle_chemistry",
                dataset="hle",
                source_file="/tmp/hle.jsonl",
                eval_kind="hle",
                prompt="Q2",
                reference_answer="B",
                answer_text="C",
                evaluation={
                    "eval_kind": "hle",
                    "score": 0.0,
                    "max_score": 1.0,
                    "normalized_score": 0.0,
                    "passed": False,
                    "primary_metric": "hle_judge_accuracy",
                    "primary_metric_direction": "higher_is_better",
                    "details": {"confidence": 60},
                },
                runner_meta={},
                raw={},
                elapsed_seconds=5.0,
                run_lifecycle_status="completed",
                protocol_completion_status="completed",
                protocol_acceptance_status=None,
                answer_availability="native_final",
                answer_reliability="native",
                evaluable=True,
                scored=True,
                recovery_mode="none",
                degraded_execution=False,
                execution_error_kind=None,
            ),
            GroupRecordResult(
                schema_version=2,
                group_id="g1",
                group_label="Group 1",
                runner="single_llm",
                websearch=False,
                record_id="chembench-1",
                subset="chembench",
                dataset="chembench",
                source_file="/tmp/chembench.jsonl",
                eval_kind="chembench_open_ended",
                prompt="Q3",
                reference_answer="D",
                answer_text="D",
                evaluation={
                    "eval_kind": "chembench_open_ended",
                    "score": 1.0,
                    "max_score": 1.0,
                    "normalized_score": 1.0,
                    "passed": True,
                    "primary_metric": "judge_accuracy",
                    "primary_metric_direction": "higher_is_better",
                    "details": {"confidence": 0},
                },
                runner_meta={},
                raw={},
                elapsed_seconds=5.0,
                run_lifecycle_status="completed",
                protocol_completion_status="completed",
                protocol_acceptance_status=None,
                answer_availability="native_final",
                answer_reliability="native",
                evaluable=True,
                scored=True,
                recovery_mode="none",
                degraded_execution=False,
                execution_error_kind=None,
            ),
        ]

        summary = aggregate_results(sample)

        self.assertAlmostEqual(
            ((0.8 - 1.0) ** 2 + (0.6 - 0.0) ** 2) ** 0.5 / (2 ** 0.5),
            summary["groups"]["g1"]["hle_calibration_rmse"],
        )
        self.assertAlmostEqual(
            summary["groups"]["g1"]["hle_calibration_rmse"],
            summary["groups"]["g1"]["by_eval_kind"]["hle"]["hle_calibration_rmse"],
        )
        self.assertAlmostEqual(
            summary["groups"]["g1"]["hle_calibration_rmse"],
            summary["group_subset"]["g1::hle_chemistry"]["hle_calibration_rmse"],
        )
        self.assertIsNone(summary["groups"]["g1"]["by_eval_kind"]["chembench_open_ended"]["hle_calibration_rmse"])
        self.assertIsNone(summary["group_subset"]["g1::chembench"]["hle_calibration_rmse"])

    def test_results_json_keeps_legacy_top_level_shape(self) -> None:
        sample = [
            GroupRecordResult(
                schema_version=2,
                group_id="g1",
                group_label="Group 1",
                runner="single_llm",
                websearch=False,
                record_id="r1",
                subset="chembench",
                dataset="d1",
                source_file="/tmp/a.jsonl",
                eval_kind="chembench_open_ended",
                prompt="Q1",
                reference_answer="1",
                answer_text="1",
                evaluation={
                    "eval_kind": "chembench_open_ended",
                    "score": 1.0,
                    "max_score": 1.0,
                    "normalized_score": 1.0,
                    "passed": True,
                    "primary_metric": "exact_str_match",
                    "primary_metric_direction": "higher_is_better",
                    "details": {},
                },
                runner_meta={},
                raw={},
                elapsed_seconds=2.0,
                run_lifecycle_status="completed",
                protocol_completion_status="completed",
                protocol_acceptance_status=None,
                answer_availability="native_final",
                answer_reliability="native",
                evaluable=True,
                scored=True,
                recovery_mode="none",
                degraded_execution=False,
                execution_error_kind=None,
            ),
            GroupRecordResult(
                schema_version=2,
                group_id="g1",
                group_label="Group 1",
                runner="single_llm",
                websearch=False,
                record_id="r2",
                subset="chembench",
                dataset="d1",
                source_file="/tmp/a.jsonl",
                eval_kind="chembench_open_ended",
                prompt="Q2",
                reference_answer="2",
                answer_text="3",
                evaluation={
                    "eval_kind": "chembench_open_ended",
                    "score": 0.0,
                    "max_score": 1.0,
                    "normalized_score": 0.0,
                    "passed": False,
                    "primary_metric": "exact_str_match",
                    "primary_metric_direction": "higher_is_better",
                    "details": {},
                },
                runner_meta={},
                raw={},
                elapsed_seconds=4.0,
                run_lifecycle_status="completed",
                protocol_completion_status="completed",
                protocol_acceptance_status=None,
                answer_availability="native_final",
                answer_reliability="native",
                evaluable=True,
                scored=True,
                recovery_mode="none",
                degraded_execution=False,
                execution_error_kind=None,
            ),
        ]
        summary = aggregate_results(sample)
        self.assertEqual("benchmarking.core.reporting", GroupRecordResult.__module__)
        self.assertEqual("benchmarking.core.reporting", aggregate_results.__module__)
        self.assertEqual(["group_order", "groups", "group_subset"], list(summary.keys()))
        self.assertEqual(["g1"], summary["group_order"])
        self.assertIn("g1", summary["groups"])
        self.assertIn("g1::chembench", summary["group_subset"])

    def test_results_json_payload_adds_schema_version_without_dropping_legacy_keys(self) -> None:
        sample = [
            GroupRecordResult(
                schema_version=2,
                group_id="g1",
                group_label="Group 1",
                runner="single_llm",
                websearch=False,
                record_id="r1",
                subset="chembench",
                dataset="d1",
                source_file="/tmp/a.jsonl",
                eval_kind="chembench_open_ended",
                prompt="Q1",
                reference_answer="1",
                answer_text="1",
                evaluation={
                    "eval_kind": "chembench_open_ended",
                    "score": 1.0,
                    "max_score": 1.0,
                    "normalized_score": 1.0,
                    "passed": True,
                    "primary_metric": "exact_str_match",
                    "primary_metric_direction": "higher_is_better",
                    "details": {},
                },
                runner_meta={},
                raw={},
                elapsed_seconds=2.0,
                run_lifecycle_status="completed",
                protocol_completion_status="completed",
                protocol_acceptance_status=None,
                answer_availability="native_final",
                answer_reliability="native",
                evaluable=True,
                scored=True,
                recovery_mode="none",
                degraded_execution=False,
                execution_error_kind=None,
            )
        ]
        summary = aggregate_results(sample)
        payload = {
            "schema_version": 2,
            "status_axes_description": {
                "evaluable": "whether a record has a trustworthy scoreable answer",
            },
            "results": [asdict(item) for item in sample],
            "summary": summary,
            "errors": [],
        }
        self.assertEqual(2, payload["schema_version"])
        self.assertIn("results", payload)
        self.assertIn("summary", payload)
        self.assertIn("errors", payload)

    def test_group_record_result_includes_evaluability_axes(self) -> None:
        result = GroupRecordResult(
            schema_version=2,
            group_id="g1",
            group_label="Group 1",
            runner="single_llm",
            websearch=False,
            record_id="r1",
            subset="chembench",
            dataset="d1",
            source_file="/tmp/a.jsonl",
            eval_kind="chembench_open_ended",
            prompt="Q1",
            reference_answer="1",
            answer_text="1",
            evaluation={
                "eval_kind": "chembench_open_ended",
                "score": 1.0,
                "max_score": 1.0,
                "normalized_score": 1.0,
                "passed": True,
                "primary_metric": "exact_str_match",
                "primary_metric_direction": "higher_is_better",
                "details": {},
            },
            runner_meta={},
            raw={},
            elapsed_seconds=2.0,
            run_lifecycle_status="completed",
            protocol_completion_status="completed",
            protocol_acceptance_status=None,
            answer_availability="native_final",
            answer_reliability="native",
            evaluable=True,
            scored=True,
            recovery_mode="none",
            degraded_execution=False,
            execution_error_kind=None,
            error=None,
        )
        self.assertEqual(2, result.schema_version)
        self.assertTrue(result.evaluable)
        self.assertTrue(result.scored)

    def test_judge_client_invokes_openclaw_with_configured_thinking(self) -> None:
        captured: dict[str, object] = {}
        original_run_subprocess = subprocess_utils.run_subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            agent_dir = root / "agents" / "benchmark-judge" / "agent"
            agent_dir.mkdir(parents=True, exist_ok=True)
            config_path = root / "openclaw.json"
            config_path.write_text(
                json.dumps(
                    {
                        "agents": {
                            "list": [
                                {
                                    "id": "benchmark-judge",
                                    "agentDir": str(agent_dir),
                                    "model": "openai/gpt-5.4",
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            store_path = root / "agents" / "benchmark-judge" / "sessions" / "sessions.json"
            store_path.parent.mkdir(parents=True, exist_ok=True)
            store_path.write_text("{}", encoding="utf-8")

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                captured["command"] = list(command)
                captured["env"] = dict(env or {})
                session_id = command[command.index("--session-id") + 1]
                store_path.write_text(
                    json.dumps(
                        {
                            "agent:benchmark-judge:main": {
                                "sessionId": session_id,
                                "sessionFile": str(store_path.parent / f"{session_id}.jsonl"),
                                "modelProvider": "openai",
                                "model": "gpt-5.4",
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps({"result": {"payloads": [{"text": '{"items": [], "summary": "ok"}'}], "meta": {}}}),
                    stderr="",
                )

            try:
                subprocess_utils.run_subprocess = fake_run_subprocess
                client = judge_runtime.JudgeClient(
                    judge_agent="benchmark-judge",
                    timeout_seconds=30,
                    config_path=config_path,
                    thinking="minimal",
                )
                payload = client.evaluate_json("score this")
                self.assertEqual([], payload["items"])
                command = captured["command"]
                assert isinstance(command, list)
                self.assertIn("--thinking", command)
                self.assertEqual("minimal", command[command.index("--thinking") + 1])
                isolation = client.last_workspace_isolation
                self.assertTrue(isolation["archive_ok"])
                self.assertTrue(Path(isolation["archive_manifest"]).is_file())
                self.assertFalse(Path(isolation["active_workspace"]).exists())
                env = captured["env"]
                assert isinstance(env, dict)
                self.assertIn("BENCHMARK_WORKSPACE_DIR", env)
                self.assertIn("BENCHMARK_SKILL_SCRATCH_DIR", env)
            finally:
                subprocess_utils.run_subprocess = original_run_subprocess

    def test_judge_client_rejects_contaminated_verdict_after_collection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "openclaw.json"
            config_path.write_text('{"agents":{"list":[]}}\n', encoding="utf-8")
            completed = subprocess.CompletedProcess(
                ["openclaw"],
                0,
                stdout=json.dumps(
                    {"result": {"payloads": [{"text": '{"items": [], "summary": "must-discard"}'}], "meta": {}}}
                ),
                stderr="",
            )
            client = judge_runtime.JudgeClient(
                judge_agent="benchmark-judge",
                timeout_seconds=30,
                config_path=config_path,
                contamination_auditor=lambda **_kwargs: ContaminationAudit(
                    status="contaminated",
                    findings=(
                        {
                            "rule_id": "forbidden_path",
                            "policy_id": "benchmark_runtime_root",
                            "tool_name": "read",
                            "candidate_source": "read.path",
                            "resolved_path": "/tmp/benchmark-runtime/old/verdict.json",
                            "matched_root": "/tmp/benchmark-runtime",
                            "command_excerpt": "../old/verdict.json",
                        },
                    ),
                ),
            )
            session_audit = {
                "requested_session_id": "",
                "postflight_entry_session_id": "",
                "postflight_entry_session_file": "",
                "session_isolation_ok": True,
            }
            with mock.patch.object(judge_runtime, "reset_agent_main_session_if_stale", return_value=session_audit):
                with mock.patch.object(judge_runtime, "inspect_postflight_session", return_value=session_audit):
                    with mock.patch.object(subprocess_utils, "run_subprocess", return_value=completed):
                        with self.assertRaisesRegex(judge_runtime.JudgeError, "contamination"):
                            client.evaluate_json("score this")

            self.assertTrue(client.last_workspace_isolation["archive_ok"])
            self.assertEqual("confirmed", client.last_workspace_isolation["contamination_status"])
            self.assertEqual("non_evaluable", client.last_workspace_isolation["adjudication"])

    def test_judge_client_clears_stale_main_session_before_openclaw_call(self) -> None:
        captured: dict[str, object] = {}
        original_run_subprocess = subprocess_utils.run_subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            agent_dir = root / "agents" / "benchmark-judge" / "agent"
            agent_dir.mkdir(parents=True, exist_ok=True)
            config_path = root / "openclaw.json"
            config_path.write_text(
                json.dumps(
                    {
                        "agents": {
                            "list": [
                                {
                                    "id": "benchmark-judge",
                                    "agentDir": str(agent_dir),
                                    "model": "openai/gpt-5.4",
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            store_path = root / "agents" / "benchmark-judge" / "sessions" / "sessions.json"
            store_path.parent.mkdir(parents=True, exist_ok=True)
            store_path.write_text(
                json.dumps(
                    {
                        "agent:benchmark-judge:main": {
                            "sessionId": "old-judge-session",
                            "sessionFile": str(store_path.parent / "old-judge-session.jsonl"),
                            "modelProvider": "openai",
                            "model": "gpt-5.4",
                        }
                    }
                ),
                encoding="utf-8",
            )

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                captured["store_seen_by_openclaw"] = json.loads(store_path.read_text(encoding="utf-8"))
                session_id = command[command.index("--session-id") + 1]
                store_path.write_text(
                    json.dumps(
                        {
                            "agent:benchmark-judge:main": {
                                "sessionId": session_id,
                                "sessionFile": str(store_path.parent / f"{session_id}.jsonl"),
                                "modelProvider": "openai",
                                "model": "gpt-5.4",
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps({"result": {"payloads": [{"text": '{"items": [], "summary": "ok"}'}], "meta": {}}}),
                    stderr="",
                )

            try:
                subprocess_utils.run_subprocess = fake_run_subprocess
                client = judge_runtime.JudgeClient(
                    judge_agent="benchmark-judge",
                    timeout_seconds=30,
                    config_path=config_path,
                )
                payload = client.evaluate_json("score this")
                self.assertEqual("ok", payload["summary"])
                store_seen = captured["store_seen_by_openclaw"]
                assert isinstance(store_seen, dict)
                self.assertNotIn("agent:benchmark-judge:main", store_seen)
            finally:
                subprocess_utils.run_subprocess = original_run_subprocess

    def test_judge_client_rejects_postflight_session_mismatch_before_parsing_reply(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            agent_dir = root / "agents" / "benchmark-judge" / "agent"
            agent_dir.mkdir(parents=True, exist_ok=True)
            config_path = root / "openclaw.json"
            config_path.write_text(
                json.dumps(
                    {
                        "agents": {
                            "list": [
                                {
                                    "id": "benchmark-judge",
                                    "agentDir": str(agent_dir),
                                    "model": "openai/gpt-5.4",
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            store_path = root / "agents" / "benchmark-judge" / "sessions" / "sessions.json"
            store_path.parent.mkdir(parents=True, exist_ok=True)
            store_path.write_text("{}", encoding="utf-8")

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                store_path.write_text(
                    json.dumps(
                        {
                            "agent:benchmark-judge:main": {
                                "sessionId": "old-judge-session",
                                "sessionFile": str(store_path.parent / "old-judge-session.jsonl"),
                                "modelProvider": "openai",
                                "model": "gpt-5.4",
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps({"result": {"payloads": [{"text": '{"untrusted": true}'}], "meta": {}}}),
                    stderr="",
                )

            try:
                subprocess_utils.run_subprocess = fake_run_subprocess
                client = judge_runtime.JudgeClient(
                    judge_agent="benchmark-judge",
                    timeout_seconds=30,
                    config_path=config_path,
                )
                with self.assertRaises(judge_runtime.JudgeError) as ctx:
                    client.evaluate_json("score this")
                message = str(ctx.exception)
                self.assertIn("benchmark-judge-", message)
                self.assertIn("old-judge-session", message)
            finally:
                subprocess_utils.run_subprocess = original_run_subprocess

    def test_judge_client_checks_postflight_before_parsing_bad_judge_stdout(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            agent_dir = root / "agents" / "benchmark-judge" / "agent"
            agent_dir.mkdir(parents=True, exist_ok=True)
            config_path = root / "openclaw.json"
            config_path.write_text(
                json.dumps(
                    {
                        "agents": {
                            "list": [
                                {
                                    "id": "benchmark-judge",
                                    "agentDir": str(agent_dir),
                                    "model": "openai/gpt-5.4",
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            store_path = root / "agents" / "benchmark-judge" / "sessions" / "sessions.json"
            store_path.parent.mkdir(parents=True, exist_ok=True)
            store_path.write_text("{}", encoding="utf-8")

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                store_path.write_text(
                    json.dumps(
                        {
                            "agent:benchmark-judge:main": {
                                "sessionId": "old-judge-session",
                                "sessionFile": str(store_path.parent / "old-judge-session.jsonl"),
                                "modelProvider": "openai",
                                "model": "gpt-5.4",
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, stdout="not-json", stderr="")

            try:
                subprocess_utils.run_subprocess = fake_run_subprocess
                client = judge_runtime.JudgeClient(
                    judge_agent="benchmark-judge",
                    timeout_seconds=30,
                    config_path=config_path,
                )
                with self.assertRaises(judge_runtime.JudgeError) as ctx:
                    client.evaluate_json("score this")
                message = str(ctx.exception)
                self.assertIn("session isolation failed", message)
                self.assertIn("old-judge-session", message)
                self.assertNotIn("JSON decode failed", message)
            finally:
                subprocess_utils.run_subprocess = original_run_subprocess

    def test_single_llm_runner_invokes_wrapper_with_configured_thinking(self) -> None:
        captured: dict[str, object] = {}
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                captured["command"] = list(command)
                captured["env"] = dict(env or {})
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "result": {
                                "payloads": [{"text": "Reasoning\nFINAL ANSWER: 5"}],
                                "meta": {
                                    "toolSummary": {"calls": 1, "tools": ["exec"], "failures": 0},
                                    "convergence": {"tool_call_count": 1, "tool_names": ["exec"]},
                                    "session_isolation": {
                                        "requested_session_id": "benchmark-single_llm_skills_on-demo-abc12345",
                                        "agent_id": "benchmark-single-skills-on",
                                        "session_store_path": "/tmp/sessions.json",
                                        "preflight_removed_stale_main_entry": True,
                                        "preflight_previous_session_id": "old-session",
                                        "postflight_entry_session_id": "benchmark-single_llm_skills_on-demo-abc12345",
                                        "postflight_entry_session_file": "/tmp/benchmark-single_llm_skills_on-demo-abc12345.jsonl",
                                        "session_isolation_ok": True,
                                    },
                                },
                            }
                        }
                    ),
                    stderr="",
                )

            subprocess_utils.run_subprocess = fake_run_subprocess
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=30,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
                configured_skills=("chem-calculator", "paper-retrieval"),
                benchmark_agent_thinking="medium",
            )
            record = BenchmarkRecord(
                record_id="demo",
                dataset="chembench",
                source_file="/tmp/demo.jsonl",
                eval_kind="chembench_open_ended",
                prompt="What is 2+3?",
                reference_answer="5",
                payload={},
            )
            out = runner.run(record, experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])
            self.assertEqual("5", out.short_answer_text)
            command = captured["command"]
            assert isinstance(command, list)
            self.assertNotEqual("openclaw", command[0])
            self.assertTrue(any(str(part).endswith("single_llm_openclaw_wrapper.py") for part in command))
            self.assertIn("--thinking", command)
            self.assertEqual("medium", command[command.index("--thinking") + 1])
            self.assertIn("--agent", command)
            self.assertEqual("benchmark-single-skills-on", command[command.index("--agent") + 1])
            self.assertIn("--eval-kind", command)
            self.assertEqual("chembench_open_ended", command[command.index("--eval-kind") + 1])
            self.assertNotIn("--finalization-grace-seconds", command)
            audit = out.runner_meta["skill_use_audit"]
            self.assertEqual(2, audit["available_skill_count"])
            self.assertTrue(audit["skill_tool_executed"])
            self.assertEqual(1, audit["tool_call_count"])
            self.assertTrue(out.runner_meta["session_isolation"]["session_isolation_ok"])
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_marks_session_isolation_failure_unscored(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "result": {
                                "payloads": [{"text": "Contaminated response\nFINAL ANSWER: 5"}],
                                "meta": {
                                    "session_isolation": {
                                        "requested_session_id": "benchmark-single_llm_skills_on-demo-new",
                                        "agent_id": "benchmark-single-skills-on",
                                        "session_store_path": "/tmp/sessions.json",
                                        "preflight_removed_stale_main_entry": False,
                                        "preflight_previous_session_id": "",
                                        "postflight_entry_session_id": "old-session",
                                        "postflight_entry_session_file": "/tmp/old-session.jsonl",
                                        "session_isolation_ok": False,
                                    }
                                },
                            }
                        }
                    ),
                    stderr="",
                )

            subprocess_utils.run_subprocess = fake_run_subprocess
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=30,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
                configured_skills=("chem-calculator",),
            )
            record = BenchmarkRecord(
                record_id="demo",
                dataset="chembench",
                source_file="/tmp/demo.jsonl",
                eval_kind="chembench_open_ended",
                prompt="What is 2+3?",
                reference_answer="5",
                payload={},
            )

            out = runner.run(record, experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.FAILED, out.status)
            self.assertFalse(out.should_score())
            self.assertIsNotNone(out.failure)
            assert out.failure is not None
            self.assertEqual("session_isolation_failed", out.failure.code)
            self.assertIn("old-session", out.failure.message)
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_rejects_invalid_stdout_payloads_without_scoring_empty_answer(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "result": {
                                "payloads": [],
                                "meta": {
                                    "stdout_diagnostics": {
                                        "schema_valid": False,
                                        "reason": "missing_payloads",
                                        "invalid_stdout_payload": {"query": "tool args"},
                                    },
                                    "session_isolation": {"session_isolation_ok": True},
                                },
                            }
                        }
                    ),
                    stderr="",
                )

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=30,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
            )
            record = BenchmarkRecord(
                record_id="demo",
                dataset="chembench",
                source_file="/tmp/demo.jsonl",
                eval_kind="chembench_open_ended",
                prompt="What is 2+3?",
                reference_answer="5",
                payload={},
            )

            out = runner.run(record, experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.FAILED, out.status)
            assert out.failure is not None
            self.assertEqual("agent_result_contract_invalid", out.failure.code)
            self.assertEqual("", out.answer.full_response_text)
            self.assertFalse(out.should_score())
            self.assertEqual("missing_payloads", out.runner_meta["stdout_diagnostics"]["reason"])
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_marks_openclaw_timeout_payload_unscored(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            timeout_text = (
                "Request timed out before a response was generated. "
                "Please try again, or increase `agents.defaults.timeoutSeconds` in your config."
            )

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "result": {
                                "payloads": [{"text": timeout_text}],
                                "meta": {
                                    "aborted": True,
                                    "durationMs": 910104,
                                    "livenessState": "blocked",
                                    "stdout_diagnostics": {
                                        "schema_valid": True,
                                        "payload_count": 1,
                                        "parse_mode": "embedded_agent_result",
                                    },
                                    "session_isolation": {"session_isolation_ok": True},
                                },
                            }
                        }
                    ),
                    stderr="",
                )

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
                timeout_retries=0,
            )
            record = BenchmarkRecord(
                record_id="demo",
                dataset="chembench",
                source_file="/tmp/demo.jsonl",
                eval_kind="chembench_open_ended",
                prompt="What is 2+3?",
                reference_answer="5",
                payload={},
            )

            out = runner.run(record, experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.FAILED, out.status)
            assert out.failure is not None
            self.assertEqual("agent_response_timeout", out.failure.code)
            self.assertEqual("", out.answer.full_response_text)
            self.assertFalse(out.should_score())
            self.assertTrue(out.runner_meta["agent_timeout_detected"])
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_rejects_short_llm_request_timeout_sentinel(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "result": {
                                "payloads": [{"text": "LLM request timed out."}],
                                "meta": {
                                    "stdout_diagnostics": {"schema_valid": True},
                                    "session_isolation": {"session_isolation_ok": True},
                                },
                            }
                        }
                    ),
                    stderr="",
                )

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
                timeout_retries=0,
            )
            record = BenchmarkRecord(
                record_id="demo",
                dataset="superchem",
                source_file="/tmp/demo.jsonl",
                eval_kind="superchem_multiple_choice_rpf",
                prompt="Choose.",
                reference_answer="A",
                payload={},
            )

            out = runner.run(record, experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.FAILED, out.status)
            assert out.failure is not None
            self.assertEqual("agent_response_timeout", out.failure.code)
            self.assertEqual("", out.answer.full_response_text)
            self.assertFalse(out.should_score())
            self.assertEqual("LLM request timed out.", out.runner_meta["candidate_answer_contract"]["raw_text"])
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_retries_timeout_sentinel_then_succeeds_with_backoff(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            calls: list[list[str]] = []
            sleeps: list[float] = []

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                calls.append(command)
                if len(calls) == 1:
                    return self._single_llm_completed_process(
                        command,
                        text="LLM request timed out.",
                        meta={"aborted": True, "livenessState": "blocked"},
                    )
                return self._single_llm_completed_process(command, text="Visible reason.\nFINAL ANSWER: B")

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
                timeout_retry_backoff_seconds=(5, 15, 45),
                sleep_fn=sleeps.append,
            )

            out = runner.run(self._single_llm_record(), experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.COMPLETED, out.status)
            self.assertEqual("B", out.short_answer_text)
            self.assertEqual(2, len(calls))
            self.assertEqual([5], sleeps)
            initial_session_id = calls[0][calls[0].index("--session-id") + 1]
            retry_session_id = calls[1][calls[1].index("--session-id") + 1]
            self.assertEqual(f"{initial_session_id}-retry1", retry_session_id)
            retry_meta = out.runner_meta["timeout_retry"]
            self.assertTrue(retry_meta["triggered"])
            self.assertEqual(1, retry_meta["retries_used"])
            self.assertFalse(retry_meta["exhausted"])
            self.assertEqual(2, retry_meta["attempts"])
            self.assertEqual("agent_response_timeout", retry_meta["attempt_history"][0]["failure_code"])
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_retries_replay_invalid_only_with_timeout_prompt_error(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            calls: list[list[str]] = []

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                calls.append(command)
                if len(calls) == 1:
                    return self._single_llm_completed_process(
                        command,
                        text="Agent couldn't generate a response.",
                        meta={
                            "replayInvalid": True,
                            "livenessState": "abandoned",
                            "convergence": {
                                "prompt_error_count": 1,
                                "latest_prompt_error": "HTTP 504 gateway timeout",
                                "latest_prompt_error_is_timeout": True,
                            },
                        },
                        is_error=True,
                    )
                return self._single_llm_completed_process(command, text="Visible reason.\nFINAL ANSWER: B")

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
                sleep_fn=lambda seconds: None,
            )

            out = runner.run(self._single_llm_record(), experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.COMPLETED, out.status)
            self.assertEqual(2, len(calls))
            self.assertTrue(out.runner_meta["timeout_retry"]["triggered"])
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_does_not_retry_plain_replay_invalid_without_timeout_evidence(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            calls: list[list[str]] = []

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                calls.append(command)
                return self._single_llm_completed_process(
                    command,
                    text="Agent couldn't generate a response.",
                    meta={"replayInvalid": True, "livenessState": "abandoned"},
                    is_error=True,
                )

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
                sleep_fn=lambda seconds: None,
            )

            out = runner.run(self._single_llm_record(), experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.FAILED, out.status)
            assert out.failure is not None
            self.assertEqual("agent_response_unavailable", out.failure.code)
            self.assertEqual(1, len(calls))
            self.assertFalse(out.runner_meta["timeout_retry"]["triggered"])
            diagnostics = out.failure.details["replay_invalid_diagnostics"]
            self.assertEqual("replay_invalid", diagnostics["reason"])
            self.assertEqual("abandoned", diagnostics["livenessState"])
            self.assertEqual("agent_response_unavailable", out.runner_meta["agent_error"]["kind"])
            self.assertEqual(diagnostics, out.runner_meta["agent_error"]["replay_invalid_diagnostics"])
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_keeps_complete_replay_invalid_answer_native(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                return self._single_llm_completed_process(
                    command,
                    text="Visible verification.\nFINAL ANSWER: CCO",
                    meta={
                        "replayInvalid": True,
                        "stopReason": "stop",
                        "completion": {"finishReason": "stop"},
                        "livenessState": "working",
                        "convergence": {
                            "transcript_answer_recovered": False,
                            "replay_invalid_diagnostics": {"reason": "replay_invalid"},
                        },
                    },
                )

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
            )

            out = runner.run(
                self._single_llm_record(eval_kind="verifier_grounded"),
                experiments.EXPERIMENT_GROUPS["single_llm_skills_on"],
            )

            self.assertEqual(RunStatus.COMPLETED, out.status)
            self.assertIsNone(out.recovery)
            self.assertTrue(out.should_score())
            self.assertEqual("CCO", out.short_answer_text)
            self.assertFalse(out.runner_meta.get("degraded_execution", False))
            self.assertNotIn("agent_error", out.runner_meta)
            self.assertEqual("replay_invalid", out.runner_meta["convergence"]["replay_invalid_diagnostics"]["reason"])
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_retries_structured_meta_timeout(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            calls: list[list[str]] = []

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                calls.append(command)
                if len(calls) == 1:
                    return self._single_llm_completed_process(
                        command,
                        text="",
                        meta={"error": {"kind": "timeout", "message": "provider deadline exceeded"}},
                    )
                return self._single_llm_completed_process(command, text="Visible reason.\nFINAL ANSWER: B")

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
                sleep_fn=lambda seconds: None,
            )

            out = runner.run(self._single_llm_record(), experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.COMPLETED, out.status)
            self.assertEqual(2, len(calls))
            self.assertTrue(out.runner_meta["timeout_retry"]["triggered"])
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_retries_subprocess_timeout_expired(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            calls: list[list[str]] = []

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                calls.append(command)
                if len(calls) == 1:
                    raise subprocess.TimeoutExpired(command, timeout=930)
                return self._single_llm_completed_process(command, text="Visible reason.\nFINAL ANSWER: B")

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
                sleep_fn=lambda seconds: None,
            )

            out = runner.run(self._single_llm_record(), experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.COMPLETED, out.status)
            self.assertEqual(2, len(calls))
            self.assertTrue(out.runner_meta["timeout_retry"]["triggered"])
            self.assertEqual("TimeoutExpired", out.runner_meta["timeout_retry"]["attempt_history"][0]["exception_type"])
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_classifies_openclaw_config_error_without_timeout_retry(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            calls: list[list[str]] = []
            stderr = (
                "Error: agent: failed to apply resolved secret assignment at "
                "models.providers.qwen.apiKey (Path segment does not exist at models.providers.qwen.)."
            )

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                calls.append(command)
                self.assertIn("--timeout", command)
                return subprocess.CompletedProcess(command, 1, stdout="", stderr=stderr)

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
                sleep_fn=lambda seconds: None,
            )

            out = runner.run(self._single_llm_record(), experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.FAILED, out.status)
            assert out.failure is not None
            self.assertEqual("openclaw_config_secret_assignment_error", out.failure.code)
            self.assertEqual("openclaw_config", out.failure.details["layer"])
            self.assertFalse(out.failure.details["retryable"])
            self.assertEqual(1, len(calls))
            self.assertFalse(out.runner_meta["timeout_retry"]["triggered"])
            self.assertEqual("openclaw_config_secret_assignment_error", out.runner_meta["execution_error"]["code"])
            self.assertEqual("openclaw_config", out.runner_meta["execution_error"]["layer"])
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_retries_openclaw_subprocess_provider_timeout(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            calls: list[list[str]] = []

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                calls.append(command)
                if len(calls) == 1:
                    return subprocess.CompletedProcess(
                        command,
                        1,
                        stdout="",
                        stderr="Provider request failed: HTTP 504 gateway timeout",
                    )
                return self._single_llm_completed_process(command, text="Visible reason.\nFINAL ANSWER: B")

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
                sleep_fn=lambda seconds: None,
            )

            out = runner.run(self._single_llm_record(), experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.COMPLETED, out.status)
            self.assertEqual(2, len(calls))
            self.assertTrue(out.runner_meta["timeout_retry"]["triggered"])
            self.assertEqual(
                "provider_timeout",
                out.runner_meta["timeout_retry"]["attempt_history"][0]["failure_code"],
            )
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_retries_http_and_transport_timeout_family(self) -> None:
        retryable_errors = [
            "HTTP 408 request timeout",
            "HTTP 504 gateway timeout",
            "ETIMEDOUT while waiting for model",
            "ECONNABORTED socket hang up",
            "context deadline exceeded",
        ]
        for error_text in retryable_errors:
            with self.subTest(error_text=error_text):
                original_run_subprocess = subprocess_utils.run_subprocess
                original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
                try:
                    runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
                    calls: list[list[str]] = []

                    def fake_run_subprocess(
                        command: list[str], *, env=None, cwd=None, timeout=None, calls=calls, error_text=error_text
                    ):
                        calls.append(command)
                        if len(calls) == 1:
                            return self._single_llm_completed_process(command, text=error_text)
                        return self._single_llm_completed_process(command, text="Visible reason.\nFINAL ANSWER: B")

                    subprocess_utils.run_subprocess = fake_run_subprocess
                    runner = runner_adapters.SingleLLMRunner(
                        agent_id="benchmark-single-skills-on",
                        timeout_seconds=900,
                        config_path=Path("/tmp/single.json"),
                        runtime_bundle_root=Path("/tmp"),
                        sleep_fn=lambda seconds: None,
                    )

                    out = runner.run(
                        self._single_llm_record(),
                        experiments.EXPERIMENT_GROUPS["single_llm_skills_on"],
                    )

                    self.assertEqual(RunStatus.COMPLETED, out.status)
                    self.assertEqual(2, len(calls))
                    self.assertTrue(out.runner_meta["timeout_retry"]["triggered"])
                finally:
                    subprocess_utils.run_subprocess = original_run_subprocess
                    runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_does_not_retry_non_model_timeout_family(self) -> None:
        non_retryable_errors = [
            "approval timeout while waiting for user",
            "tool timeout running shell command",
            "maximum context length exceeded",
            "401 auth failed",
            "billing hard limit reached",
            "rate limit exceeded",
            "image size too large",
            "role ordering is invalid",
            "The computed value is 500 but the final marker is missing.",
        ]
        for error_text in non_retryable_errors:
            with self.subTest(error_text=error_text):
                original_run_subprocess = subprocess_utils.run_subprocess
                original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
                try:
                    runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
                    calls: list[list[str]] = []

                    def fake_run_subprocess(
                        command: list[str], *, env=None, cwd=None, timeout=None, calls=calls, error_text=error_text
                    ):
                        calls.append(command)
                        return self._single_llm_completed_process(command, text=error_text)

                    subprocess_utils.run_subprocess = fake_run_subprocess
                    runner = runner_adapters.SingleLLMRunner(
                        agent_id="benchmark-single-skills-on",
                        timeout_seconds=900,
                        config_path=Path("/tmp/single.json"),
                        runtime_bundle_root=Path("/tmp"),
                        sleep_fn=lambda seconds: None,
                    )

                    out = runner.run(
                        self._single_llm_record(),
                        experiments.EXPERIMENT_GROUPS["single_llm_skills_on"],
                    )

                    self.assertEqual(RunStatus.FAILED, out.status)
                    self.assertEqual(1, len(calls))
                    self.assertFalse(out.runner_meta["timeout_retry"]["triggered"])
                finally:
                    subprocess_utils.run_subprocess = original_run_subprocess
                    runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_exhausts_timeout_retries_with_metadata(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            calls: list[list[str]] = []
            sleeps: list[float] = []

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                calls.append(command)
                return self._single_llm_completed_process(
                    command,
                    text="LLM request timed out.",
                    meta={"aborted": True, "livenessState": "blocked"},
                )

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
                timeout_retries=3,
                timeout_retry_backoff_seconds=(5, 15, 45),
                sleep_fn=sleeps.append,
            )

            out = runner.run(self._single_llm_record(), experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.FAILED, out.status)
            assert out.failure is not None
            self.assertEqual("agent_response_timeout", out.failure.code)
            self.assertEqual(4, len(calls))
            self.assertEqual([5, 15, 45], sleeps)
            retry_meta = out.runner_meta["timeout_retry"]
            self.assertTrue(retry_meta["triggered"])
            self.assertTrue(retry_meta["exhausted"])
            self.assertEqual(4, retry_meta["attempts"])
            self.assertEqual(3, retry_meta["retries_used"])
            self.assertEqual([5, 15, 45], retry_meta["backoff_seconds"])
            self.assertEqual(4, len(retry_meta["attempt_history"]))
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_subprocess_timeout_covers_finalization_rescue(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            timeouts: list[float | None] = []

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                timeouts.append(timeout)
                return self._single_llm_completed_process(command, text="Visible reason.\nFINAL ANSWER: B")

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
                convergence_policy=ConvergencePolicy(
                    timeout_seconds=900,
                ),
            )

            out = runner.run(self._single_llm_record(), experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.COMPLETED, out.status)
            self.assertEqual([1020], timeouts)
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_classifies_stream_read_error_before_answer_contract(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "result": {
                                "payloads": [{"text": "stream_read_error", "isError": True}],
                                "meta": {
                                    "stopReason": "error",
                                    "completion": {"finishReason": "error"},
                                    "livenessState": "blocked",
                                    "stdout_diagnostics": {"schema_valid": True},
                                    "session_isolation": {"session_isolation_ok": True},
                                },
                            }
                        }
                    ),
                    stderr="",
                )

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
            )
            record = BenchmarkRecord(
                record_id="superchem-demo",
                dataset="superchem",
                source_file="/tmp/demo.jsonl",
                eval_kind="superchem_multiple_choice_rpf",
                prompt="Choose.",
                reference_answer="B",
                payload={},
            )

            out = runner.run(record, experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.FAILED, out.status)
            assert out.failure is not None
            self.assertEqual("agent_stream_read_error", out.failure.code)
            self.assertEqual("", out.answer.full_response_text)
            self.assertFalse(out.should_score())
            self.assertEqual("agent_stream_read_error", out.runner_meta["agent_error"]["kind"])
            self.assertNotIn("candidate_answer_contract", out.runner_meta)
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_classifies_openclaw_no_response_fallback(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            fallback_text = (
                "⚠️ Agent couldn't generate a response. Note: some tool actions may have already been executed — "
                "please verify before retrying."
            )

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "result": {
                                "payloads": [{"text": fallback_text, "isError": True}],
                                "meta": {
                                    "replayInvalid": True,
                                    "livenessState": "abandoned",
                                    "stdout_diagnostics": {"schema_valid": True},
                                    "session_isolation": {"session_isolation_ok": True},
                                },
                            }
                        }
                    ),
                    stderr="",
                )

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
            )
            record = BenchmarkRecord(
                record_id="superchem-demo",
                dataset="superchem",
                source_file="/tmp/demo.jsonl",
                eval_kind="superchem_multiple_choice_rpf",
                prompt="Choose.",
                reference_answer="B",
                payload={},
            )

            out = runner.run(record, experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.FAILED, out.status)
            assert out.failure is not None
            self.assertEqual("agent_response_unavailable", out.failure.code)
            self.assertEqual("agent_response_unavailable", out.runner_meta["agent_error"]["kind"])
            self.assertIn("replay_invalid_diagnostics", out.runner_meta["agent_error"])
            self.assertFalse(out.should_score())
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_marks_finalization_rescue_as_recovered(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "result": {
                                "payloads": [{"text": "Visible reason.\nFINAL ANSWER: B"}],
                                "meta": {
                                    "stopReason": "error",
                                    "completion": {"finishReason": "error"},
                                    "convergence": {
                                        "agent_error_payload_detected": True,
                                        "agent_error_kind": "agent_stream_read_error",
                                        "finalization_rescue_attempted": True,
                                        "finalization_rescue_succeeded": True,
                                        "recovery_source": "single-llm-finalization-rescue",
                                    },
                                    "stdout_diagnostics": {"schema_valid": True},
                                    "session_isolation": {"session_isolation_ok": True},
                                },
                            }
                        }
                    ),
                    stderr="",
                )

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
            )
            record = BenchmarkRecord(
                record_id="superchem-demo",
                dataset="superchem",
                source_file="/tmp/demo.jsonl",
                eval_kind="superchem_multiple_choice_rpf",
                prompt="Choose.",
                reference_answer="B",
                payload={},
            )

            out = runner.run(record, experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.RECOVERED, out.status)
            self.assertTrue(out.should_score())
            self.assertEqual("Visible reason.\nFINAL ANSWER: B", out.answer.full_response_text)
            assert out.recovery is not None
            self.assertEqual("single-llm-finalization-rescue", out.recovery.source)
            self.assertEqual("single-llm-finalization-rescue", out.runner_meta["recovery_mode"])
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_marks_research_wide_rescue_as_recovered(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "result": {
                                "payloads": [
                                    {
                                        "text": (
                                            "## FINAL ANSWER\n"
                                            "The rescue answer covers the research protocol, evidence, and conclusion."
                                        )
                                    }
                                ],
                                "meta": {
                                    "stopReason": "error",
                                    "completion": {"finishReason": "error"},
                                    "convergence": {
                                        "agent_error_payload_detected": True,
                                        "agent_error_kind": "agent_stream_read_error",
                                        "finalization_rescue_attempted": True,
                                        "finalization_rescue_succeeded": True,
                                        "recovery_source": "single-llm-finalization-rescue",
                                    },
                                    "stdout_diagnostics": {"schema_valid": True},
                                    "session_isolation": {"session_isolation_ok": True},
                                },
                            }
                        }
                    ),
                    stderr="",
                )

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
            )
            record = BenchmarkRecord(
                record_id="research-demo",
                dataset="frontierscience",
                source_file="/tmp/demo.jsonl",
                eval_kind="frontierscience_research",
                prompt="Explain the research result.",
                reference_answer="rubric",
                payload={"track": "research"},
            )

            out = runner.run(record, experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.RECOVERED, out.status)
            self.assertTrue(out.should_score())
            self.assertIn("## FINAL ANSWER", out.answer.full_response_text)
            assert out.recovery is not None
            self.assertEqual("single-llm-finalization-rescue", out.recovery.source)
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_accepts_native_research_conclusion_with_blocked_meta(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            research_answer = (
                "## Evidence ledger\n"
                "The requested source-specific claims are checked above.\n\n"
                "## Supported conclusion\n"
                "The answer covers the synthesis protocol, mechanism, spectra, and reactivity evidence."
            )

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "result": {
                                "payloads": [{"text": research_answer}],
                                "meta": {
                                    "livenessState": "blocked",
                                    "stdout_diagnostics": {"schema_valid": True},
                                    "session_isolation": {"session_isolation_ok": True},
                                },
                            }
                        }
                    ),
                    stderr="",
                )

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
            )
            record = BenchmarkRecord(
                record_id="research-demo",
                dataset="frontierscience",
                source_file="/tmp/demo.jsonl",
                eval_kind="frontierscience_research",
                prompt="Explain the research result.",
                reference_answer="rubric",
                payload={"track": "research"},
            )

            out = runner.run(record, experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.COMPLETED, out.status)
            self.assertTrue(out.should_score())
            self.assertEqual(research_answer, out.answer.full_response_text)
            self.assertNotIn("agent_error", out.runner_meta)
            self.assertFalse(out.runner_meta["candidate_answer_contract"]["has_research_final_marker"])
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_reports_research_final_marker_metadata(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            research_answer = (
                "## Evidence ledger\n"
                "The requested source-specific claims are checked above.\n\n"
                "## FINAL RESEARCH ANSWER\n"
                "The answer covers the synthesis protocol, mechanism, spectra, and reactivity evidence."
            )

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "result": {
                                "payloads": [{"text": research_answer}],
                                "meta": {
                                    "stdout_diagnostics": {"schema_valid": True},
                                    "session_isolation": {"session_isolation_ok": True},
                                },
                            }
                        }
                    ),
                    stderr="",
                )

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
            )
            record = BenchmarkRecord(
                record_id="research-demo",
                dataset="frontierscience",
                source_file="/tmp/demo.jsonl",
                eval_kind="frontierscience_research",
                prompt="Explain the research result.",
                reference_answer="rubric",
                payload={"track": "research"},
            )

            out = runner.run(record, experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.COMPLETED, out.status)
            self.assertTrue(out.should_score())
            self.assertTrue(out.runner_meta["candidate_answer_contract"]["has_research_final_marker"])
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_rejects_superchem_response_without_final_answer_marker(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "result": {
                                "payloads": [{"text": "The spectrum appears most consistent with option B."}],
                                "meta": {
                                    "stdout_diagnostics": {"schema_valid": True},
                                    "session_isolation": {"session_isolation_ok": True},
                                },
                            }
                        }
                    ),
                    stderr="",
                )

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
            )
            record = BenchmarkRecord(
                record_id="superchem-demo",
                dataset="superchem",
                source_file="/tmp/demo.jsonl",
                eval_kind="superchem_multiple_choice_rpf",
                prompt="Choose.",
                reference_answer="B",
                payload={},
            )

            out = runner.run(record, experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.FAILED, out.status)
            assert out.failure is not None
            self.assertEqual("candidate_answer_contract_invalid", out.failure.code)
            self.assertEqual("", out.answer.full_response_text)
            self.assertFalse(out.should_score())
            self.assertIn("short_answer_text", out.failure.details["missing_fields"])
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_accepts_markdown_final_answer_marker(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "result": {
                                "payloads": [{"text": "Visible reasoning.\n**FINAL ANSWER:** B"}],
                                "meta": {
                                    "stdout_diagnostics": {"schema_valid": True},
                                    "session_isolation": {"session_isolation_ok": True},
                                },
                            }
                        }
                    ),
                    stderr="",
                )

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
            )
            record = BenchmarkRecord(
                record_id="superchem-demo",
                dataset="superchem",
                source_file="/tmp/demo.jsonl",
                eval_kind="superchem_multiple_choice_rpf",
                prompt="Choose.",
                reference_answer="B",
                payload={},
            )

            out = runner.run(record, experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.COMPLETED, out.status)
            self.assertTrue(out.should_score())
            self.assertEqual("B", out.short_answer_text)
            self.assertEqual("Visible reasoning.\n**FINAL ANSWER:** B", out.full_response_text)
            self.assertTrue(out.runner_meta["candidate_answer_contract"]["has_final_answer_marker"])
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_rejects_hle_response_without_answer_field(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "result": {
                                "payloads": [{"text": "Explanation: estimated from the figure\nConfidence: 60%"}],
                                "meta": {
                                    "stdout_diagnostics": {"schema_valid": True},
                                    "session_isolation": {"session_isolation_ok": True},
                                },
                            }
                        }
                    ),
                    stderr="",
                )

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
            )
            record = BenchmarkRecord(
                record_id="hle-demo",
                dataset="hle",
                source_file="/tmp/hle.jsonl",
                eval_kind="hle",
                prompt="Question?",
                reference_answer="273",
                payload={},
            )

            out = runner.run(record, experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.FAILED, out.status)
            assert out.failure is not None
            self.assertEqual("candidate_answer_contract_invalid", out.failure.code)
            self.assertFalse(out.should_score())
            self.assertIn("Answer:", out.failure.message)
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_returns_recovered_for_transcript_answer(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "result": {
                                "payloads": [{"text": "Explanation: ok\nAnswer: 273\nConfidence: 60%"}],
                                "meta": {
                                    "aborted": True,
                                    "durationMs": 910104,
                                    "livenessState": "blocked",
                                    "convergence": {
                                        "transcript_answer_recovered": True,
                                        "tool_call_count": 8,
                                        "assistant_turn_count": 12,
                                    },
                                    "session_isolation": {"session_isolation_ok": True},
                                },
                            }
                        }
                    ),
                    stderr="",
                )

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
            )
            record = BenchmarkRecord(
                record_id="hle-demo",
                dataset="hle",
                source_file="/tmp/demo.jsonl",
                eval_kind="hle",
                prompt="Question?",
                reference_answer="273",
                payload={},
            )

            out = runner.run(record, experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.RECOVERED, out.status)
            self.assertTrue(out.should_score())
            self.assertIsNotNone(out.recovery)
            assert out.recovery is not None
            self.assertEqual("single-llm-session-transcript", out.recovery.source)
            self.assertEqual("Explanation: ok\nAnswer: 273\nConfidence: 60%", out.full_response_text)
            self.assertIn("convergence_policy", out.runner_meta)
            self.assertEqual(8, out.runner_meta["convergence"]["tool_call_count"])
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_single_llm_runner_does_not_score_recovered_answer_when_session_isolation_fails(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        try:
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "result": {
                                "payloads": [{"text": "Explanation: ok\nAnswer: 273\nConfidence: 60%"}],
                                "meta": {
                                    "aborted": True,
                                    "livenessState": "blocked",
                                    "convergence": {"transcript_answer_recovered": True},
                                    "session_isolation": {
                                        "requested_session_id": "session-new",
                                        "postflight_entry_session_id": "session-old",
                                        "session_isolation_ok": False,
                                    },
                                },
                            }
                        }
                    ),
                    stderr="",
                )

            subprocess_utils.run_subprocess = fake_run_subprocess
            runner = runner_adapters.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=900,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
            )
            record = BenchmarkRecord(
                record_id="hle-demo",
                dataset="hle",
                source_file="/tmp/demo.jsonl",
                eval_kind="hle",
                prompt="Question?",
                reference_answer="273",
                payload={},
            )

            out = runner.run(record, experiments.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(RunStatus.FAILED, out.status)
            self.assertFalse(out.should_score())
            assert out.failure is not None
            self.assertEqual("session_isolation_failed", out.failure.code)
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_chemqa_runner_uses_run_scoped_writable_template_and_command_map_dirs(self) -> None:
        captured: dict[str, object] = {}
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        original_wait_for_terminal_status = runner_adapters.ChemQARunner._wait_for_terminal_status
        original_ensure_artifacts = runner_adapters.ChemQARunner._ensure_artifacts
        original_invoke_cleanroom_cleanup = CleanroomRuntime.invoke_cleanroom_cleanup
        try:
            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                captured["command"] = list(command)
                captured["env"] = dict(env or {})
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps({"run_id": "demo", "launch_mode": "run", "launched": {"returncode": 0}}),
                    stderr="",
                )

            subprocess_utils.run_subprocess = fake_run_subprocess
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            CleanroomRuntime.invoke_cleanroom_cleanup = lambda self, manifest_path: {"status": "cleaned", "manifest_path": str(manifest_path)}
            runner_adapters.ChemQARunner._wait_for_terminal_status = lambda self, run_id, timeout_seconds: {
                "status": "done",
                "terminal_state": "completed",
                "terminal_reason_code": "",
                "artifact_collection": {},
            }

            def fake_ensure_artifacts(self, run_id, *, env, run_status, wait_seconds=120, poll_seconds=5):
                qa_result_path = self.launch_workspace_root / "qa_result.json"
                qa_result_path.write_text(
                    json.dumps(
                        {
                            "final_answer": "c1ccccc1",
                            "artifact_paths": {},
                            "acceptance_status": "accepted",
                            "terminal_state": "completed",
                        }
                    ),
                    encoding="utf-8",
                )
                return qa_result_path

            runner_adapters.ChemQARunner._ensure_artifacts = fake_ensure_artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                launch_root = Path(tmpdir) / "chemqa-launch"
                runner = runner_adapters.ChemQARunner(
                    chemqa_root=Path(tmpdir) / "chemqa-root",
                    timeout_seconds=30,
                    config_path=Path(tmpdir) / "config.json",
                    slot_set="A",
                    review_rounds=None,
                    rebuttal_rounds=None,
                    model_profile="profile-x",
                    runtime_bundle_root=Path(tmpdir) / "bundles",
                    launch_workspace_root=launch_root,
                )
                record = BenchmarkRecord(
                    record_id="chembench-0001",
                    dataset="chembench",
                    source_file="/tmp/demo.jsonl",
                    eval_kind="chembench_open_ended",
                    prompt="Calculate the value.",
                    reference_answer="42",
                    payload={},
                )
                out = runner.run(record, experiments.EXPERIMENT_GROUPS["chemqa_skills_on"])
                self.assertEqual("c1ccccc1", out.short_answer_text)
                command = captured["command"]
                assert isinstance(command, list)
                self.assertIn("--template-dir", command)
                self.assertIn("--command-map-dir", command)
                template_dir = Path(command[command.index("--template-dir") + 1])
                command_map_dir = Path(command[command.index("--command-map-dir") + 1])
                self.assertTrue(str(template_dir).startswith(str(launch_root)))
                self.assertTrue(str(command_map_dir).startswith(str(launch_root)))
                self.assertEqual("templates", template_dir.name)
                self.assertEqual("command-maps", command_map_dir.name)
                self.assertEqual(".clawteam", template_dir.parent.name)
                self.assertNotEqual(str(Path.home() / ".clawteam" / "templates"), str(template_dir))
                env = captured["env"]
                assert isinstance(env, dict)
                self.assertEqual(str(launch_root / "chemqa_skills_on" / "chembench-0001" / "home"), env["HOME"])
                self.assertEqual(str(runner_adapters.DEFAULT_OPENCLAW_ENV_FILE), env["OPENCLAW_ENV_FILE"])
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle
            CleanroomRuntime.invoke_cleanroom_cleanup = original_invoke_cleanroom_cleanup
            runner_adapters.ChemQARunner._wait_for_terminal_status = original_wait_for_terminal_status
            runner_adapters.ChemQARunner._ensure_artifacts = original_ensure_artifacts

    def test_chemqa_runner_archives_completed_artifacts_under_output_root(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        original_wait_for_terminal_status = runner_adapters.ChemQARunner._wait_for_terminal_status
        original_ensure_artifacts = runner_adapters.ChemQARunner._ensure_artifacts
        original_invoke_cleanroom_cleanup = CleanroomRuntime.invoke_cleanroom_cleanup
        try:
            subprocess_utils.run_subprocess = lambda command, *, env=None, cwd=None, timeout=None: subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"run_id": "demo", "launch_mode": "run", "launched": {"returncode": 0}}),
                stderr="",
            )
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            CleanroomRuntime.invoke_cleanroom_cleanup = lambda self, manifest_path: {"status": "cleaned", "manifest_path": str(manifest_path)}
            runner_adapters.ChemQARunner._wait_for_terminal_status = lambda self, run_id, timeout_seconds: {
                "status": "done",
                "terminal_state": "completed",
                "terminal_reason_code": "",
                "artifact_collection": {"status": "ok"},
            }

            def fake_ensure_artifacts(self, run_id, *, env, run_status, wait_seconds=120, poll_seconds=5):
                scratch_dir = self.chemqa_root / "generated" / "artifacts" / run_id
                scratch_dir.mkdir(parents=True, exist_ok=True)
                protocol_dir = self.chemqa_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id
                protocol_dir.mkdir(parents=True, exist_ok=True)
                (protocol_dir / "chemqa_review_protocol.yaml").write_text(
                    "question: Demo\nacceptance_status: accepted\nterminal_state: completed\nfinal_answer: c1ccccc1\n",
                    encoding="utf-8",
                )
                qa_result_path = scratch_dir / "qa_result.json"
                qa_result_path.write_text(
                    json.dumps(
                        {
                            "final_answer": "c1ccccc1",
                            "artifact_paths": {
                                "qa_result": str(qa_result_path),
                                "final_answer": str(scratch_dir / "final_answer.md"),
                            },
                            "acceptance_status": "accepted",
                            "terminal_state": "completed",
                        }
                    ),
                    encoding="utf-8",
                )
                (scratch_dir / "final_answer.md").write_text("c1ccccc1\n", encoding="utf-8")
                return qa_result_path

            runner_adapters.ChemQARunner._ensure_artifacts = fake_ensure_artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                output_root = Path(tmpdir) / "benchmark-output"
                launch_root = output_root / "chemqa-launch"
                runner = runner_adapters.ChemQARunner(
                    chemqa_root=Path(tmpdir) / "chemqa-root",
                    timeout_seconds=30,
                    config_path=Path(tmpdir) / "config.json",
                    slot_set="A",
                    review_rounds=None,
                    rebuttal_rounds=None,
                    model_profile="profile-x",
                    runtime_bundle_root=Path(tmpdir) / "bundles",
                    launch_workspace_root=launch_root,
                )
                record = BenchmarkRecord(
                    record_id="chembench-0001",
                    dataset="chembench",
                    source_file="/tmp/demo.jsonl",
                    eval_kind="chembench_open_ended",
                    prompt="Calculate the value.",
                    reference_answer="42",
                    payload={},
                )

                out = runner.run(record, experiments.EXPERIMENT_GROUPS["chemqa_skills_on"])

                self.assertEqual(RunStatus.COMPLETED, out.status)
                archive_dir = output_root / "artifacts" / "chemqa_skills_on" / "chembench-0001" / out.runner_meta["run_id"]
                self.assertEqual(str(archive_dir), out.runner_meta["archive_dir"])
                self.assertEqual(str(archive_dir / "qa_result.json"), out.runner_meta["qa_result_path"])
                self.assertEqual(str(archive_dir / "chemqa_review_protocol.yaml"), out.runner_meta["archived_protocol_path"])
                self.assertEqual("ok", out.runner_meta["artifact_archive_status"])
                self.assertTrue((archive_dir / "qa_result.json").is_file())
                self.assertTrue((archive_dir / "chemqa_review_protocol.yaml").is_file())
                self.assertTrue((archive_dir / "final_answer.md").is_file())
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle
            CleanroomRuntime.invoke_cleanroom_cleanup = original_invoke_cleanroom_cleanup
            runner_adapters.ChemQARunner._wait_for_terminal_status = original_wait_for_terminal_status
            runner_adapters.ChemQARunner._ensure_artifacts = original_ensure_artifacts

    def test_chemqa_runner_uses_canonical_qa_result_path_from_terminal_status(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        original_wait_for_terminal_status = runner_adapters.ChemQARunner._wait_for_terminal_status
        original_ensure_artifacts = runner_adapters.ChemQARunner._ensure_artifacts
        original_invoke_cleanroom_cleanup = CleanroomRuntime.invoke_cleanroom_cleanup
        try:
            subprocess_utils.run_subprocess = lambda command, *, env=None, cwd=None, timeout=None: subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"run_id": "demo", "launch_mode": "run", "launched": {"returncode": 0}}),
                stderr="",
            )
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            CleanroomRuntime.invoke_cleanroom_cleanup = lambda self, manifest_path: {"status": "cleaned", "manifest_path": str(manifest_path)}

            with tempfile.TemporaryDirectory() as tmpdir:
                canonical_dir = Path(tmpdir) / "chemqa-root" / "generated" / "artifacts" / "status-run"
                canonical_dir.mkdir(parents=True, exist_ok=True)
                qa_result_path = canonical_dir / "qa_result.json"
                final_artifact_path = canonical_dir / "final_answer_artifact.json"
                manifest_path = canonical_dir / "artifact_manifest.json"
                final_artifact_path.write_text(
                    json.dumps(
                        {
                            "terminal_state": "completed",
                            "answer_kind": "numeric_short_answer",
                            "evaluator_answer": "7.59",
                            "display_answer": "7.59 micrograms",
                            "full_answer": "FINAL ANSWER: 7.59",
                        }
                    ),
                    encoding="utf-8",
                )
                manifest_path.write_text(json.dumps({"files": {}}), encoding="utf-8")
                qa_result_path.write_text(
                    json.dumps(
                        {
                            "terminal_state": "completed",
                            "acceptance_status": "accepted",
                            "answer_kind": "numeric_short_answer",
                            "final_answer": {"direct_answer": "7.59", "answer": "7.59", "value": "7.59", "full_answer": "FINAL ANSWER: 7.59"},
                            "artifact_paths": {
                                "qa_result": str(qa_result_path),
                                "final_answer_artifact": str(final_artifact_path),
                                "artifact_manifest": str(manifest_path),
                            },
                        }
                    ),
                    encoding="utf-8",
                )

                runner_adapters.ChemQARunner._wait_for_terminal_status = lambda self, run_id, timeout_seconds: {
                    "run_id": run_id,
                    "status": "done",
                    "terminal_state": "completed",
                    "benchmark_terminal_state": "completed",
                    "artifact_flow_state": "finalized",
                    "qa_result_path": str(qa_result_path),
                    "final_answer_artifact_path": str(final_artifact_path),
                    "artifact_manifest_path": str(manifest_path),
                }

                def fail_if_called(self, run_id, *, env, run_status, wait_seconds=120, poll_seconds=5):
                    raise AssertionError("_ensure_artifacts should not rebuild when canonical qa_result_path is readable")

                runner_adapters.ChemQARunner._ensure_artifacts = fail_if_called
                output_root = Path(tmpdir) / "benchmark-output"
                runner = runner_adapters.ChemQARunner(
                    chemqa_root=Path(tmpdir) / "chemqa-root",
                    timeout_seconds=30,
                    config_path=Path(tmpdir) / "config.json",
                    slot_set="A",
                    review_rounds=None,
                    rebuttal_rounds=None,
                    model_profile="profile-x",
                    runtime_bundle_root=Path(tmpdir) / "bundles",
                    launch_workspace_root=output_root / "chemqa-launch",
                )
                record = BenchmarkRecord(
                    record_id="chembench-0001",
                    dataset="chembench",
                    source_file="/tmp/demo.jsonl",
                    eval_kind="chembench_open_ended",
                    prompt="How much product?",
                    reference_answer="7.59",
                    payload={},
                )

                out = runner.run(record, experiments.EXPERIMENT_GROUPS["chemqa_skills_on"])

                self.assertEqual(RunStatus.COMPLETED, out.status)
                self.assertEqual("7.59", out.short_answer_text)
                self.assertEqual(str(output_root / "artifacts" / "chemqa_skills_on" / "chembench-0001" / out.runner_meta["run_id"] / "qa_result.json"), out.runner_meta["qa_result_path"])
                self.assertIn("final_answer_artifact", out.runner_meta["archived_artifact_paths"])
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle
            CleanroomRuntime.invoke_cleanroom_cleanup = original_invoke_cleanroom_cleanup
            runner_adapters.ChemQARunner._wait_for_terminal_status = original_wait_for_terminal_status
            runner_adapters.ChemQARunner._ensure_artifacts = original_ensure_artifacts

    def test_chemqa_runner_archives_protocol_and_rebuilds_qa_result_for_failed_terminal_run(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        original_wait_for_terminal_status = runner_adapters.ChemQARunner._wait_for_terminal_status
        original_build_candidate_submission_fallback = runner_adapters.ChemQARunner._build_candidate_submission_fallback
        original_collect_artifacts = runner_adapters.ChemQARunner._collect_artifacts_from_source
        original_invoke_cleanroom_cleanup = CleanroomRuntime.invoke_cleanroom_cleanup
        try:
            subprocess_utils.run_subprocess = lambda command, *, env=None, cwd=None, timeout=None: subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"run_id": "demo", "launch_mode": "run", "launched": {"returncode": 0}}),
                stderr="",
            )
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            CleanroomRuntime.invoke_cleanroom_cleanup = lambda self, manifest_path: {"status": "cleaned", "manifest_path": str(manifest_path)}
            runner_adapters.ChemQARunner._wait_for_terminal_status = lambda self, run_id, timeout_seconds: {
                "status": "done",
                "terminal_state": "failed",
                "terminal_reason_code": "lane_stalled",
                "artifact_collection": {"status": "error"},
                "protocol_path": str(self.chemqa_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id / "chemqa_review_protocol.yaml"),
            }
            runner_adapters.ChemQARunner._build_candidate_submission_fallback = lambda self, run_id, run_status: None

            def fake_collect_artifacts(self, *, source_dir, output_dir, env):
                output_dir.mkdir(parents=True, exist_ok=True)
                qa_result_path = output_dir / "qa_result.json"
                qa_result_path.write_text(
                    json.dumps(
                        {
                            "final_answer": "",
                            "artifact_paths": {"qa_result": str(qa_result_path)},
                            "acceptance_status": "rejected",
                            "terminal_state": "failed",
                        }
                    ),
                    encoding="utf-8",
                )
                (output_dir / "final_answer.md").write_text("No accepted answer.\n", encoding="utf-8")

            runner_adapters.ChemQARunner._collect_artifacts_from_source = fake_collect_artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                output_root = Path(tmpdir) / "benchmark-output"
                launch_root = output_root / "chemqa-launch"
                chemqa_root = Path(tmpdir) / "chemqa-root"
                runner = runner_adapters.ChemQARunner(
                    chemqa_root=chemqa_root,
                    timeout_seconds=30,
                    config_path=Path(tmpdir) / "config.json",
                    slot_set="A",
                    review_rounds=None,
                    rebuttal_rounds=None,
                    model_profile="profile-x",
                    runtime_bundle_root=Path(tmpdir) / "bundles",
                    launch_workspace_root=launch_root,
                )
                record = BenchmarkRecord(
                    record_id="chembench-0001",
                    dataset="chembench",
                    source_file="/tmp/demo.jsonl",
                    eval_kind="chembench_open_ended",
                    prompt="Calculate the value.",
                    reference_answer="42",
                    payload={},
                )
                run_id = "benchmark-chemqa_skills_on-chembench-0001-20260424-000000"
                protocol_dir = chemqa_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id
                protocol_dir.mkdir(parents=True, exist_ok=True)
                (protocol_dir / "chemqa_review_protocol.yaml").write_text(
                    "question: Demo\nacceptance_status: rejected\nterminal_state: failed\nfailure_reason: lane stalled\nfinal_answer: \"\"\n",
                    encoding="utf-8",
                )
                runner._now_stamp = lambda: "20260424-000000"

                out = runner.run(record, experiments.EXPERIMENT_GROUPS["chemqa_skills_on"])

                self.assertEqual(RunStatus.FAILED, out.status)
                archive_dir = output_root / "artifacts" / "chemqa_skills_on" / "chembench-0001" / run_id
                self.assertEqual(str(archive_dir), out.runner_meta["archive_dir"])
                self.assertEqual(str(archive_dir / "chemqa_review_protocol.yaml"), out.runner_meta["archived_protocol_path"])
                self.assertEqual("ok", out.runner_meta["artifact_archive_status"])
                self.assertTrue((archive_dir / "chemqa_review_protocol.yaml").is_file())
                self.assertTrue((archive_dir / "qa_result.json").is_file())
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle
            CleanroomRuntime.invoke_cleanroom_cleanup = original_invoke_cleanroom_cleanup
            runner_adapters.ChemQARunner._wait_for_terminal_status = original_wait_for_terminal_status
            runner_adapters.ChemQARunner._build_candidate_submission_fallback = original_build_candidate_submission_fallback
            runner_adapters.ChemQARunner._collect_artifacts_from_source = original_collect_artifacts

    def test_chemqa_runner_failed_terminal_with_candidate_fallback_returns_scored_recovered_result(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        original_wait_for_terminal_status = runner_adapters.ChemQARunner._wait_for_terminal_status
        original_collect_artifacts = runner_adapters.ChemQARunner._collect_artifacts_from_source
        original_invoke_cleanroom_cleanup = CleanroomRuntime.invoke_cleanroom_cleanup
        try:
            subprocess_utils.run_subprocess = lambda command, *, env=None, cwd=None, timeout=None: subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"run_id": "demo", "launch_mode": "run", "launched": {"returncode": 0}}),
                stderr="",
            )
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            CleanroomRuntime.invoke_cleanroom_cleanup = lambda self, manifest_path: {"status": "cleaned", "manifest_path": str(manifest_path)}
            runner_adapters.ChemQARunner._wait_for_terminal_status = lambda self, run_id, timeout_seconds: {
                "status": "done",
                "terminal_state": "failed",
                "terminal_reason_code": "stalled",
                "artifact_collection": {},
                "protocol_path": str(self.chemqa_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id / "chemqa_review_protocol.yaml"),
            }

            def fake_collect_artifacts(self, *, source_dir, output_dir, env):
                output_dir.mkdir(parents=True, exist_ok=True)
                _ = (self, source_dir, env)

            runner_adapters.ChemQARunner._collect_artifacts_from_source = fake_collect_artifacts

            with tempfile.TemporaryDirectory() as tmpdir:
                output_root = Path(tmpdir) / "benchmark-output"
                launch_root = output_root / "chemqa-launch"
                chemqa_root = Path(tmpdir) / "chemqa-root"
                runner = runner_adapters.ChemQARunner(
                    chemqa_root=chemqa_root,
                    timeout_seconds=30,
                    config_path=Path(tmpdir) / "config.json",
                    slot_set="A",
                    review_rounds=None,
                    rebuttal_rounds=None,
                    model_profile="profile-x",
                    runtime_bundle_root=Path(tmpdir) / "bundles",
                    launch_workspace_root=launch_root,
                )
                record = BenchmarkRecord(
                    record_id="chembench-0001",
                    dataset="chembench",
                    source_file="/tmp/demo.jsonl",
                    eval_kind="chembench_open_ended",
                    prompt="Return ethanol.",
                    reference_answer="CCO",
                    payload={},
                )
                run_id = "benchmark-chemqa_skills_on-chembench-0001-20260427-000000"
                protocol_dir = chemqa_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id
                protocol_dir.mkdir(parents=True, exist_ok=True)
                (protocol_dir / "chemqa_review_protocol.yaml").write_text(
                    "\n".join(
                        [
                            "artifact_kind: coordinator_protocol",
                            "artifact_contract_version: react-reviewed-v2",
                            "terminal_state: failed",
                            "acceptance_status: failed",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                proposal_path = protocol_dir / "debate" / "artifacts" / "proposals" / "epoch-001" / "proposer-1.md"
                proposal_path.parent.mkdir(parents=True, exist_ok=True)
                proposal_path.write_text(
                    "\n".join(
                        [
                            "artifact_kind: candidate_submission",
                            "artifact_contract_version: react-reviewed-v2",
                            "phase: propose",
                            "owner: proposer-1",
                            "direct_answer: CCO",
                            "summary: recovered answer",
                            "submission_trace:",
                            "  - step: structural_reasoning",
                            "    status: success",
                            "    detail: reconstructed from proposer artifact",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                runner._now_stamp = lambda: "20260427-000000"

                out = runner.run(record, experiments.EXPERIMENT_GROUPS["chemqa_skills_on"])

                self.assertEqual(RunStatus.RECOVERED, out.status)
                self.assertEqual("CCO", out.short_answer_text)
                self.assertIn("FINAL ANSWER: CCO", out.full_response_text)
                self.assertTrue(out.should_score())
                self.assertTrue(out.recovery.scored)
                self.assertEqual("candidate_submission", out.recovery.source)
                self.assertTrue(out.runner_meta["fallback_used"])
                self.assertEqual("proposer-1-proposal", out.runner_meta["fallback_source"])
                self.assertIn("proposal_path", out.raw["fallback"])
                self.assertEqual(str(proposal_path.resolve()), str(Path(out.raw["fallback"]["proposal_path"]).resolve()))
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle
            CleanroomRuntime.invoke_cleanroom_cleanup = original_invoke_cleanroom_cleanup
            runner_adapters.ChemQARunner._wait_for_terminal_status = original_wait_for_terminal_status
            runner_adapters.ChemQARunner._collect_artifacts_from_source = original_collect_artifacts

    def test_chemqa_runner_failed_terminal_uses_failure_artifact_answer_projection(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        original_wait_for_terminal_status = runner_adapters.ChemQARunner._wait_for_terminal_status
        original_build_candidate_submission_fallback = runner_adapters.ChemQARunner._build_candidate_submission_fallback
        original_collect_artifacts = runner_adapters.ChemQARunner._collect_artifacts_from_source
        original_invoke_cleanroom_cleanup = CleanroomRuntime.invoke_cleanroom_cleanup
        try:
            subprocess_utils.run_subprocess = lambda command, *, env=None, cwd=None, timeout=None: subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"run_id": "demo", "launch_mode": "run", "launched": {"returncode": 0}}),
                stderr="",
            )
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            CleanroomRuntime.invoke_cleanroom_cleanup = lambda self, manifest_path: {"status": "cleaned", "manifest_path": str(manifest_path)}
            runner_adapters.ChemQARunner._build_candidate_submission_fallback = lambda self, run_id, run_status: None
            runner_adapters.ChemQARunner._wait_for_terminal_status = lambda self, run_id, timeout_seconds: {
                "status": "done",
                "terminal_state": "failed",
                "terminal_reason_code": "protocol_stalled",
                "artifact_flow_state": "finalization_failed",
                "benchmark_terminal_state": "failed",
                "failure_artifact_path": str(self.chemqa_root / "generated" / "artifacts" / run_id / "failure_artifact.json"),
                "qa_result_path": str(self.chemqa_root / "generated" / "artifacts" / run_id / "qa_result.json"),
            }

            def fake_collect_artifacts(self, *, source_dir, output_dir, env):
                output_dir.mkdir(parents=True, exist_ok=True)
                _ = (self, source_dir, env)

            runner_adapters.ChemQARunner._collect_artifacts_from_source = fake_collect_artifacts

            with tempfile.TemporaryDirectory() as tmpdir:
                output_root = Path(tmpdir) / "benchmark-output"
                chemqa_root = Path(tmpdir) / "chemqa-root"
                runner = runner_adapters.ChemQARunner(
                    chemqa_root=chemqa_root,
                    timeout_seconds=30,
                    config_path=Path(tmpdir) / "config.json",
                    slot_set="A",
                    review_rounds=None,
                    rebuttal_rounds=None,
                    model_profile="profile-x",
                    runtime_bundle_root=Path(tmpdir) / "bundles",
                    launch_workspace_root=output_root / "chemqa-launch",
                )
                record = BenchmarkRecord(
                    record_id="superchem-0001",
                    dataset="superchem",
                    source_file="/tmp/demo.jsonl",
                    eval_kind="superchem_multiple_choice_rpf",
                    prompt="Pick one.",
                    reference_answer="B",
                    payload={"options": {"A": "wrong", "B": "right"}},
                )
                run_id = "benchmark-chemqa_skills_on-superchem-0001-20260427-000000"
                artifact_dir = chemqa_root / "generated" / "artifacts" / run_id
                artifact_dir.mkdir(parents=True, exist_ok=True)
                (artifact_dir / "failure_artifact.json").write_text(
                    json.dumps(
                        {
                            "terminal_state": "failed",
                            "failure_code": "protocol_stalled",
                            "failure_message": "review phase stalled",
                            "answer_projection": {
                                "answer_kind": "multiple_choice",
                                "evaluator_answer": "B",
                                "display_answer": "B",
                                "full_answer": "Recovered from current candidate view.",
                            },
                            "recovery_eligibility": {
                                "evaluable": True,
                                "scored": True,
                                "reliability": "high_confidence_recovered",
                                "recovery_mode": "failure_artifact_answer_projection",
                                "reason": "last_valid_candidate_view",
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                (artifact_dir / "qa_result.json").write_text(
                    json.dumps(
                        {
                            "terminal_state": "failed",
                            "failure_code": "protocol_stalled",
                            "answer_projection": {
                                "answer_kind": "multiple_choice",
                                "evaluator_answer": "B",
                                "display_answer": "B",
                                "full_answer": "Recovered from current candidate view.",
                            },
                            "recovery_eligibility": {
                                "evaluable": True,
                                "scored": True,
                                "reliability": "high_confidence_recovered",
                                "recovery_mode": "failure_artifact_answer_projection",
                                "reason": "last_valid_candidate_view",
                            },
                            "artifact_paths": {"failure_artifact": str(artifact_dir / "failure_artifact.json")},
                        }
                    ),
                    encoding="utf-8",
                )
                protocol_dir = chemqa_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id
                protocol_dir.mkdir(parents=True, exist_ok=True)
                (protocol_dir / "chemqa_review_protocol.yaml").write_text(
                    "artifact_kind: coordinator_protocol\nterminal_state: failed\nacceptance_status: failed\nfailure_reason: stalled\n",
                    encoding="utf-8",
                )
                runner._now_stamp = lambda: "20260427-000000"

                out = runner.run(record, experiments.EXPERIMENT_GROUPS["chemqa_skills_on"])

                self.assertEqual(RunStatus.RECOVERED, out.status)
                self.assertEqual("B", out.short_answer_text)
                self.assertEqual("failure_artifact", out.recovery.source)
                self.assertEqual("failure_artifact_answer_projection", out.recovery.recovery_mode)
                self.assertTrue(out.runner_meta["fallback_used"])
                self.assertEqual("failure_artifact_answer_projection", out.runner_meta["recovery_mode"])
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle
            CleanroomRuntime.invoke_cleanroom_cleanup = original_invoke_cleanroom_cleanup
            runner_adapters.ChemQARunner._wait_for_terminal_status = original_wait_for_terminal_status
            runner_adapters.ChemQARunner._build_candidate_submission_fallback = original_build_candidate_submission_fallback
            runner_adapters.ChemQARunner._collect_artifacts_from_source = original_collect_artifacts

    def test_chemqa_runner_failed_terminal_with_final_answer_preview_stays_failed_and_unscored(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        original_wait_for_terminal_status = runner_adapters.ChemQARunner._wait_for_terminal_status
        original_collect_artifacts = runner_adapters.ChemQARunner._collect_artifacts_from_source
        original_invoke_cleanroom_cleanup = CleanroomRuntime.invoke_cleanroom_cleanup
        try:
            subprocess_utils.run_subprocess = lambda command, *, env=None, cwd=None, timeout=None: subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"run_id": "demo", "launch_mode": "run", "launched": {"returncode": 0}}),
                stderr="",
            )
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            CleanroomRuntime.invoke_cleanroom_cleanup = lambda self, manifest_path: {"status": "cleaned", "manifest_path": str(manifest_path)}
            runner_adapters.ChemQARunner._wait_for_terminal_status = lambda self, run_id, timeout_seconds: {
                "status": "done",
                "terminal_state": "failed",
                "terminal_reason_code": "stalled",
                "artifact_collection": {},
                "final_answer_preview": "CCO",
                "protocol_path": str(self.chemqa_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id / "chemqa_review_protocol.yaml"),
            }

            def fake_collect_artifacts(self, *, source_dir, output_dir, env):
                output_dir.mkdir(parents=True, exist_ok=True)
                _ = (self, source_dir, env)

            runner_adapters.ChemQARunner._collect_artifacts_from_source = fake_collect_artifacts

            with tempfile.TemporaryDirectory() as tmpdir:
                output_root = Path(tmpdir) / "benchmark-output"
                launch_root = output_root / "chemqa-launch"
                chemqa_root = Path(tmpdir) / "chemqa-root"
                runner = runner_adapters.ChemQARunner(
                    chemqa_root=chemqa_root,
                    timeout_seconds=30,
                    config_path=Path(tmpdir) / "config.json",
                    slot_set="A",
                    review_rounds=None,
                    rebuttal_rounds=None,
                    model_profile="profile-x",
                    runtime_bundle_root=Path(tmpdir) / "bundles",
                    launch_workspace_root=launch_root,
                )
                record = BenchmarkRecord(
                    record_id="chembench-0001",
                    dataset="chembench",
                    source_file="/tmp/demo.jsonl",
                    eval_kind="chembench_open_ended",
                    prompt="Return ethanol.",
                    reference_answer="CCO",
                    payload={},
                )
                run_id = "benchmark-chemqa_skills_on-chembench-0001-20260427-000000"
                protocol_dir = chemqa_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id
                protocol_dir.mkdir(parents=True, exist_ok=True)
                (protocol_dir / "chemqa_review_protocol.yaml").write_text(
                    "\n".join(
                        [
                            "artifact_kind: coordinator_protocol",
                            "artifact_contract_version: react-reviewed-v2",
                            "terminal_state: failed",
                            "acceptance_status: failed",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                runner._now_stamp = lambda: "20260427-000000"

                out = runner.run(record, experiments.EXPERIMENT_GROUPS["chemqa_skills_on"])

                self.assertEqual(RunStatus.FAILED, out.status)
                self.assertFalse(out.should_score())
                self.assertTrue(out.runner_meta["fallback_used"])
                self.assertEqual("run-status-final-answer-preview", out.runner_meta["fallback_source"])
                self.assertIs(out.runner_meta["evaluable"], False)
                self.assertIs(out.runner_meta["scored"], False)
                self.assertEqual("low_confidence_recovered", out.runner_meta["answer_reliability"])
                self.assertEqual("preview_requires_strict_validation", out.runner_meta["recovery_reason"])
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle
            CleanroomRuntime.invoke_cleanroom_cleanup = original_invoke_cleanroom_cleanup
            runner_adapters.ChemQARunner._wait_for_terminal_status = original_wait_for_terminal_status
            runner_adapters.ChemQARunner._collect_artifacts_from_source = original_collect_artifacts

    def test_chemqa_runner_reconciles_failed_run_status_with_completed_archived_rejection(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        original_wait_for_terminal_status = runner_adapters.ChemQARunner._wait_for_terminal_status
        original_collect_artifacts = runner_adapters.ChemQARunner._collect_artifacts_from_source
        original_invoke_cleanroom_cleanup = CleanroomRuntime.invoke_cleanroom_cleanup
        try:
            subprocess_utils.run_subprocess = lambda command, *, env=None, cwd=None, timeout=None: subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"run_id": "demo", "launch_mode": "run", "launched": {"returncode": 0}}),
                stderr="",
            )
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            CleanroomRuntime.invoke_cleanroom_cleanup = lambda self, manifest_path: {"status": "cleaned", "manifest_path": str(manifest_path)}
            runner_adapters.ChemQARunner._wait_for_terminal_status = lambda self, run_id, timeout_seconds: {
                "status": "done",
                "terminal_state": "failed",
                "terminal_reason_code": "stalled",
                "artifact_collection": {},
                "protocol_path": str(self.chemqa_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id / "chemqa_review_protocol.yaml"),
            }

            def fake_collect_artifacts(self, *, source_dir, output_dir, env):
                output_dir.mkdir(parents=True, exist_ok=True)
                qa_result_path = output_dir / "qa_result.json"
                qa_result_path.write_text(
                    json.dumps(
                        {
                            "final_answer": "",
                            "artifact_paths": {
                                "qa_result": str(qa_result_path),
                                "final_answer": str(output_dir / "final_answer.md"),
                            },
                            "acceptance_status": "rejected",
                            "terminal_state": "completed",
                        }
                    ),
                    encoding="utf-8",
                )
                (output_dir / "final_answer.md").write_text("No accepted answer.\n", encoding="utf-8")

            runner_adapters.ChemQARunner._collect_artifacts_from_source = fake_collect_artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                output_root = Path(tmpdir) / "benchmark-output"
                launch_root = output_root / "chemqa-launch"
                chemqa_root = Path(tmpdir) / "chemqa-root"
                runner = runner_adapters.ChemQARunner(
                    chemqa_root=chemqa_root,
                    timeout_seconds=30,
                    config_path=Path(tmpdir) / "config.json",
                    slot_set="A",
                    review_rounds=None,
                    rebuttal_rounds=None,
                    model_profile="profile-x",
                    runtime_bundle_root=Path(tmpdir) / "bundles",
                    launch_workspace_root=launch_root,
                )
                record = BenchmarkRecord(
                    record_id="chembench-0001",
                    dataset="chembench",
                    source_file="/tmp/demo.jsonl",
                    eval_kind="chembench_open_ended",
                    prompt="Calculate the value.",
                    reference_answer="42",
                    payload={},
                )
                run_id = "benchmark-chemqa_skills_on-chembench-0001-20260424-000000"
                protocol_dir = chemqa_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id
                protocol_dir.mkdir(parents=True, exist_ok=True)
                (protocol_dir / "chemqa_review_protocol.yaml").write_text(
                    "question: Demo\nacceptance_status: rejected\nterminal_state: completed\nfinal_answer: \"\"\n",
                    encoding="utf-8",
                )
                runner._now_stamp = lambda: "20260424-000000"

                out = runner.run(record, experiments.EXPERIMENT_GROUPS["chemqa_skills_on"])

                self.assertEqual(RunStatus.COMPLETED, out.status)
                self.assertEqual("", out.short_answer_text)
                self.assertIn("No accepted answer", out.full_response_text)
                self.assertEqual("rejected", out.runner_meta["acceptance_status"])
                self.assertEqual("completed", out.runner_meta["terminal_state"])
                self.assertEqual("stalled", out.runner_meta["terminal_reason_code"])
                archive_dir = output_root / "artifacts" / "chemqa_skills_on" / "chembench-0001" / run_id
                self.assertTrue((archive_dir / "chemqa_review_protocol.yaml").is_file())
                self.assertTrue((archive_dir / "qa_result.json").is_file())
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle
            CleanroomRuntime.invoke_cleanroom_cleanup = original_invoke_cleanroom_cleanup
            runner_adapters.ChemQARunner._wait_for_terminal_status = original_wait_for_terminal_status
            runner_adapters.ChemQARunner._collect_artifacts_from_source = original_collect_artifacts

    def test_chemqa_runner_reconciled_rejected_run_does_not_expose_blob_as_short_answer(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        original_wait_for_terminal_status = runner_adapters.ChemQARunner._wait_for_terminal_status
        original_collect_artifacts = runner_adapters.ChemQARunner._collect_artifacts_from_source
        original_invoke_cleanroom_cleanup = CleanroomRuntime.invoke_cleanroom_cleanup
        try:
            subprocess_utils.run_subprocess = lambda command, *, env=None, cwd=None, timeout=None: subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"run_id": "demo", "launch_mode": "run", "launched": {"returncode": 0}}),
                stderr="",
            )
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            CleanroomRuntime.invoke_cleanroom_cleanup = lambda self, manifest_path: {"status": "cleaned", "manifest_path": str(manifest_path)}
            runner_adapters.ChemQARunner._wait_for_terminal_status = lambda self, run_id, timeout_seconds: {
                "status": "done",
                "terminal_state": "failed",
                "terminal_reason_code": "stalled",
                "artifact_collection": {},
                "protocol_path": str(self.chemqa_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id / "chemqa_review_protocol.yaml"),
            }

            def fake_collect_artifacts(self, *, source_dir, output_dir, env):
                output_dir.mkdir(parents=True, exist_ok=True)
                rejection_blob = {
                    "accepted_owner": "",
                    "answer": None,
                    "direct_answer": None,
                    "summary": "No candidate submission achieved acceptance.",
                }
                qa_result_path = output_dir / "qa_result.json"
                qa_result_path.write_text(
                    json.dumps(
                        {
                            "final_answer": json.dumps(rejection_blob, ensure_ascii=False, indent=2),
                            "artifact_paths": {
                                "qa_result": str(qa_result_path),
                                "final_answer": str(output_dir / "final_answer.md"),
                            },
                            "acceptance_status": "rejected",
                            "terminal_state": "completed",
                        }
                    ),
                    encoding="utf-8",
                )
                (output_dir / "final_answer.md").write_text(
                    json.dumps(rejection_blob, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

            runner_adapters.ChemQARunner._collect_artifacts_from_source = fake_collect_artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                output_root = Path(tmpdir) / "benchmark-output"
                launch_root = output_root / "chemqa-launch"
                chemqa_root = Path(tmpdir) / "chemqa-root"
                runner = runner_adapters.ChemQARunner(
                    chemqa_root=chemqa_root,
                    timeout_seconds=30,
                    config_path=Path(tmpdir) / "config.json",
                    slot_set="A",
                    review_rounds=None,
                    rebuttal_rounds=None,
                    model_profile="profile-x",
                    runtime_bundle_root=Path(tmpdir) / "bundles",
                    launch_workspace_root=launch_root,
                )
                record = BenchmarkRecord(
                    record_id="chembench-0001",
                    dataset="chembench",
                    source_file="/tmp/demo.jsonl",
                    eval_kind="chembench_open_ended",
                    prompt="Calculate the value.",
                    reference_answer="42",
                    payload={},
                )
                run_id = "benchmark-chemqa_skills_on-chembench-0001-20260424-000000"
                protocol_dir = chemqa_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id
                protocol_dir.mkdir(parents=True, exist_ok=True)
                (protocol_dir / "chemqa_review_protocol.yaml").write_text(
                    "question: Demo\nacceptance_status: rejected\nterminal_state: completed\nfinal_answer: \"\"\n",
                    encoding="utf-8",
                )
                runner._now_stamp = lambda: "20260424-000000"

                out = runner.run(record, experiments.EXPERIMENT_GROUPS["chemqa_skills_on"])

                self.assertEqual(RunStatus.COMPLETED, out.status)
                self.assertEqual("", out.short_answer_text)
                self.assertIn("No candidate submission achieved acceptance.", out.full_response_text)
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle
            CleanroomRuntime.invoke_cleanroom_cleanup = original_invoke_cleanroom_cleanup
            runner_adapters.ChemQARunner._wait_for_terminal_status = original_wait_for_terminal_status
            runner_adapters.ChemQARunner._collect_artifacts_from_source = original_collect_artifacts

    def test_chemqa_runner_archives_repeated_runs_into_distinct_run_id_directories(self) -> None:
        original_run_subprocess = subprocess_utils.run_subprocess
        original_ensure_runtime_bundle = runtime_bundles.ensure_runtime_bundle
        original_wait_for_terminal_status = runner_adapters.ChemQARunner._wait_for_terminal_status
        original_ensure_artifacts = runner_adapters.ChemQARunner._ensure_artifacts
        original_invoke_cleanroom_cleanup = CleanroomRuntime.invoke_cleanroom_cleanup
        try:
            subprocess_utils.run_subprocess = lambda command, *, env=None, cwd=None, timeout=None: subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"run_id": "demo", "launch_mode": "run", "launched": {"returncode": 0}}),
                stderr="",
            )
            runtime_bundles.ensure_runtime_bundle = lambda record, bundle_root: None
            CleanroomRuntime.invoke_cleanroom_cleanup = lambda self, manifest_path: {"status": "cleaned", "manifest_path": str(manifest_path)}
            runner_adapters.ChemQARunner._wait_for_terminal_status = lambda self, run_id, timeout_seconds: {
                "status": "done",
                "terminal_state": "completed",
                "terminal_reason_code": "",
                "artifact_collection": {"status": "ok"},
            }

            def fake_ensure_artifacts(self, run_id, *, env, run_status, wait_seconds=120, poll_seconds=5):
                scratch_dir = self.chemqa_root / "generated" / "artifacts" / run_id
                scratch_dir.mkdir(parents=True, exist_ok=True)
                protocol_dir = self.chemqa_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id
                protocol_dir.mkdir(parents=True, exist_ok=True)
                (protocol_dir / "chemqa_review_protocol.yaml").write_text(
                    "question: Demo\nacceptance_status: accepted\nterminal_state: completed\nfinal_answer: c1ccccc1\n",
                    encoding="utf-8",
                )
                qa_result_path = scratch_dir / "qa_result.json"
                qa_result_path.write_text(
                    json.dumps(
                        {
                            "final_answer": "c1ccccc1",
                            "artifact_paths": {"qa_result": str(qa_result_path)},
                            "acceptance_status": "accepted",
                            "terminal_state": "completed",
                        }
                    ),
                    encoding="utf-8",
                )
                return qa_result_path

            runner_adapters.ChemQARunner._ensure_artifacts = fake_ensure_artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                output_root = Path(tmpdir) / "benchmark-output"
                launch_root = output_root / "chemqa-launch"
                runner = runner_adapters.ChemQARunner(
                    chemqa_root=Path(tmpdir) / "chemqa-root",
                    timeout_seconds=30,
                    config_path=Path(tmpdir) / "config.json",
                    slot_set="A",
                    review_rounds=None,
                    rebuttal_rounds=None,
                    model_profile="profile-x",
                    runtime_bundle_root=Path(tmpdir) / "bundles",
                    launch_workspace_root=launch_root,
                )
                record = BenchmarkRecord(
                    record_id="chembench-0001",
                    dataset="chembench",
                    source_file="/tmp/demo.jsonl",
                    eval_kind="chembench_open_ended",
                    prompt="Calculate the value.",
                    reference_answer="42",
                    payload={},
                )
                stamps = iter(["20260424-000001", "20260424-000002"])
                runner._now_stamp = lambda: next(stamps)

                out1 = runner.run(record, experiments.EXPERIMENT_GROUPS["chemqa_skills_on"])
                out2 = runner.run(record, experiments.EXPERIMENT_GROUPS["chemqa_skills_on"])

                self.assertNotEqual(out1.runner_meta["run_id"], out2.runner_meta["run_id"])
                archive1 = Path(out1.runner_meta["archive_dir"])
                archive2 = Path(out2.runner_meta["archive_dir"])
                self.assertNotEqual(archive1, archive2)
                self.assertTrue((archive1 / "qa_result.json").is_file())
                self.assertTrue((archive2 / "qa_result.json").is_file())
        finally:
            subprocess_utils.run_subprocess = original_run_subprocess
            runtime_bundles.ensure_runtime_bundle = original_ensure_runtime_bundle
            CleanroomRuntime.invoke_cleanroom_cleanup = original_invoke_cleanroom_cleanup
            runner_adapters.ChemQARunner._wait_for_terminal_status = original_wait_for_terminal_status
            runner_adapters.ChemQARunner._ensure_artifacts = original_ensure_artifacts

    def test_run_group_continues_after_record_failure(self) -> None:
        records = [
            BenchmarkRecord(
                record_id="r1",
                dataset="chembench",
                source_file="/tmp/demo.jsonl",
                eval_kind="chembench_open_ended",
                prompt="What is 2+2?",
                reference_answer="4",
                payload={"target": "4"},
            ),
            BenchmarkRecord(
                record_id="r2",
                dataset="chembench",
                source_file="/tmp/demo.jsonl",
                eval_kind="chembench_open_ended",
                prompt="What is 2+3?",
                reference_answer="5",
                payload={"target": "5"},
            ),
        ]

        class StubSingleRunner:
            def __init__(self, **_: object) -> None:
                pass

            def run(self, record: object, group: object) -> object:
                _ = group
                if record.record_id == "r1":
                    raise RuntimeError("boom")
                return RunnerResult(
                    status=RunStatus.COMPLETED,
                    answer=AnswerPayload(
                        short_answer_text="5",
                        full_response_text="Reasoning\nFINAL ANSWER: 5",
                    ),
                    raw={},
                    runner_meta={},
                )

        original_runner = runner_adapters.SingleLLMRunner
        runner_adapters.SingleLLMRunner = StubSingleRunner
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                results = run_group_for_test(
                    group=experiments.EXPERIMENT_GROUPS["single_llm_skills_off"],
                    records=records,
                    output_root=Path(tmpdir),
                    single_timeout=10,
                    chemqa_timeout=10,
                    judge=JudgeStub({"correct": True, "score": 1.0, "rationale": "matches"}),
                    config_path=Path(tmpdir) / "cfg.json",
                    single_agent="benchmark-single-skills-off",
                    chemqa_root=Path(tmpdir),
                    chemqa_model_profile="unused",
                    review_rounds=None,
                    rebuttal_rounds=None,
                )
                self.assertEqual(2, len(results))
                self.assertIsNotNone(results[0].error)
                self.assertFalse(results[0].evaluation["passed"])
                self.assertTrue(results[1].evaluation["passed"])
                self.assertTrue((Path(tmpdir) / "per-record" / "single_llm_skills_off" / "r1.json").exists())
                self.assertTrue((Path(tmpdir) / "per-record" / "single_llm_skills_off" / "r2.json").exists())
        finally:
            runner_adapters.SingleLLMRunner = original_runner

    def test_run_group_passes_single_timeout_retry_options_to_runner(self) -> None:
        record = BenchmarkRecord(
            record_id="r1",
            dataset="chembench",
            source_file="/tmp/demo.jsonl",
            eval_kind="chembench_open_ended",
            prompt="What is 2+2?",
            reference_answer="4",
            payload={"target": "4"},
        )
        captured: dict[str, object] = {}

        class StubSingleRunner:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

            def run(self, record: object, group: object) -> object:
                _ = record, group
                return RunnerResult(
                    status=RunStatus.COMPLETED,
                    answer=AnswerPayload(
                        short_answer_text="4",
                        full_response_text="Reasoning\nFINAL ANSWER: 4",
                    ),
                    raw={},
                    runner_meta={},
                )

        original_runner = runner_adapters.SingleLLMRunner
        runner_adapters.SingleLLMRunner = StubSingleRunner
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                results = run_group_for_test(
                    group=experiments.EXPERIMENT_GROUPS["single_llm_skills_off"],
                    records=[record],
                    output_root=Path(tmpdir),
                    single_timeout=10,
                    chemqa_timeout=10,
                    judge=JudgeStub({"correct": True, "score": 1.0, "rationale": "matches"}),
                    config_path=Path(tmpdir) / "cfg.json",
                    single_agent="benchmark-single-skills-off",
                    chemqa_root=Path(tmpdir),
                    chemqa_model_profile="unused",
                    review_rounds=None,
                    rebuttal_rounds=None,
                    single_agent_thinking="low",
                    single_timeout_retries=2,
                    single_timeout_retry_backoff_seconds=(1, 3),
                )

            self.assertEqual(1, len(results))
            self.assertEqual("low", captured["benchmark_agent_thinking"])
            self.assertEqual(2, captured["timeout_retries"])
            self.assertEqual((1, 3), captured["timeout_retry_backoff_seconds"])
        finally:
            runner_adapters.SingleLLMRunner = original_runner

    def test_run_group_marks_unscored_recovery_as_execution_error(self) -> None:
        record = BenchmarkRecord(
            record_id="recovered-record",
            dataset="chembench",
            source_file="/tmp/demo.jsonl",
            eval_kind="chembench_open_ended",
            prompt="Q",
            reference_answer="A",
            payload={},
        )
        recovered_result = RunnerResult(
            status=RunStatus.RECOVERED,
            answer=AnswerPayload(
                short_answer_text="fallback-answer",
                full_response_text="FINAL ANSWER: fallback-answer",
            ),
            raw={"run_status": {"status": "done", "terminal_state": "failed"}},
            runner_meta={
                "run_id": "demo-run",
                "fallback_used": True,
                "fallback_source": "proposer-1-proposal",
                "error": "ChemQA run `demo-run` terminated as failed (reason=stalled)",
            },
            recovery=RecoveryInfo(
                source="proposer-1-proposal",
                scored=False,
                details={"reason": "stalled_review"},
            ),
        )

        class StubRunner:
            def run(self, record: object, group: object) -> RunnerResult:
                self.called_with = (record, group)
                return recovered_result

        stub_runner = StubRunner()
        original_build_runner = runner_adapters.build_runner
        original_evaluate_answer = scoring_evaluation.evaluate_record
        try:
            runner_adapters.build_runner = lambda **kwargs: stub_runner

            def fail_evaluate_answer(*args, **kwargs):
                raise AssertionError("evaluate_answer should not be called for unscored recovery")

            scoring_evaluation.evaluate_record = fail_evaluate_answer
            with tempfile.TemporaryDirectory() as tmpdir:
                results = run_group_for_test(
                    group=experiments.EXPERIMENT_GROUPS["chemqa_skills_on"],
                    records=[record],
                    output_root=Path(tmpdir),
                    single_timeout=10,
                    chemqa_timeout=10,
                    judge=object(),
                    config_path=Path(tmpdir) / "cfg.json",
                    single_agent="benchmark-single-skills-off",
                    chemqa_root=Path(tmpdir),
                    chemqa_model_profile="unused",
                    review_rounds=None,
                    rebuttal_rounds=None,
                )
            self.assertEqual(1, len(results))
            entry = results[0]
            self.assertEqual("execution_error", entry.evaluation["primary_metric"])
            self.assertFalse(entry.evaluation["passed"])
            self.assertEqual("fallback-answer", entry.short_answer_text)
            self.assertEqual("FINAL ANSWER: fallback-answer", entry.full_response_text)
            self.assertEqual("FINAL ANSWER: fallback-answer", entry.answer_text)
            self.assertEqual("demo-run", entry.runner_meta["run_id"])
            self.assertEqual("proposer-1-proposal", entry.runner_meta["fallback_source"])
            self.assertEqual({"status": "done", "terminal_state": "failed"}, entry.raw["run_status"])
            self.assertEqual("ChemQA run `demo-run` terminated as failed (reason=stalled)", entry.error)
        finally:
            runner_adapters.build_runner = original_build_runner
            scoring_evaluation.evaluate_record = original_evaluate_answer

    def test_run_group_failed_result_axes_for_non_recovery(self) -> None:
        record = BenchmarkRecord(
            record_id="failed-record",
            dataset="chembench",
            source_file="/tmp/demo.jsonl",
            eval_kind="chembench_open_ended",
            prompt="Q",
            reference_answer="A",
            payload={},
        )
        failed_result = RunnerResult(
            status=RunStatus.FAILED,
            answer=AnswerPayload(short_answer_text="", full_response_text=""),
            raw={"run_status": {"status": "done", "terminal_state": "failed"}},
            runner_meta={"run_id": "demo-run"},
            failure=FailureInfo(code="failed", message="runner failed"),
        )

        class StubRunner:
            def run(self, record: object, group: object) -> RunnerResult:
                return failed_result

        original_build_runner = runner_adapters.build_runner
        original_evaluate_answer = scoring_evaluation.evaluate_record
        try:
            runner_adapters.build_runner = lambda **kwargs: StubRunner()

            def fail_evaluate_answer(*args, **kwargs):
                raise AssertionError("evaluate_answer should not be called for failed non-recovery results")

            scoring_evaluation.evaluate_record = fail_evaluate_answer
            with tempfile.TemporaryDirectory() as tmpdir:
                results = run_group_for_test(
                    group=experiments.EXPERIMENT_GROUPS["chemqa_skills_on"],
                    records=[record],
                    output_root=Path(tmpdir),
                    single_timeout=10,
                    chemqa_timeout=10,
                    judge=object(),
                    config_path=Path(tmpdir) / "cfg.json",
                    single_agent="unused",
                    chemqa_root=Path(tmpdir),
                    chemqa_model_profile="unused",
                    review_rounds=None,
                    rebuttal_rounds=None,
                )
            self.assertEqual(1, len(results))
            entry = results[0]
            self.assertEqual("failed", entry.run_lifecycle_status)
            self.assertEqual("failed", entry.protocol_completion_status)
            self.assertEqual("missing", entry.answer_availability)
            self.assertEqual("none", entry.answer_reliability)
            self.assertFalse(entry.evaluable)
            self.assertFalse(entry.scored)
            self.assertEqual("none", entry.recovery_mode)
            self.assertTrue(entry.degraded_execution)
            self.assertEqual("execution_error", entry.execution_error_kind)
            self.assertEqual("runner failed", entry.error)
        finally:
            runner_adapters.build_runner = original_build_runner
            scoring_evaluation.evaluate_record = original_evaluate_answer

    def test_run_group_scores_evaluable_recovery(self) -> None:
        record = BenchmarkRecord(
            record_id="recovered-record",
            dataset="chembench",
            source_file="/tmp/demo.jsonl",
            eval_kind="chembench_open_ended",
            prompt="Q",
            reference_answer="fallback-answer",
            payload={},
        )
        recovered_result = RunnerResult(
            status=RunStatus.RECOVERED,
            answer=AnswerPayload(
                short_answer_text="fallback-answer",
                full_response_text="FINAL ANSWER: fallback-answer",
            ),
            raw={"run_status": {"status": "done", "terminal_state": "failed"}},
            runner_meta={
                "run_id": "demo-run",
                "fallback_used": True,
                "fallback_source": "proposer-1-proposal",
                "error": "ChemQA run `demo-run` terminated as failed (reason=stalled)",
            },
            recovery=RecoveryInfo(
                source="candidate_submission",
                scored=True,
                evaluable=True,
                reliability="high_confidence_recovered",
                recovery_mode="candidate_submission",
                details={
                    "evaluable": True,
                    "reliability": "high_confidence_recovered",
                    "recovery_mode": "candidate_submission",
                },
            ),
        )

        class StubRunner:
            def run(self, record: object, group: object) -> RunnerResult:
                return recovered_result

        original_build_runner = runner_adapters.build_runner
        original_evaluate_answer = scoring_evaluation.evaluate_record
        judge = object()
        try:
            runner_adapters.build_runner = lambda **kwargs: StubRunner()

            def fake_evaluate_answer(
                actual_record: object,
                *,
                short_answer_text: str,
                full_response_text: str,
                answer_text: str,
                judge: object,
            ) -> EvaluationResult:
                self.assertIs(record, actual_record)
                self.assertEqual("fallback-answer", short_answer_text)
                self.assertEqual("FINAL ANSWER: fallback-answer", full_response_text)
                self.assertEqual("FINAL ANSWER: fallback-answer", answer_text)
                self.assertIs(judge, judge_obj)
                return EvaluationResult(
                    eval_kind="chembench_open_ended",
                    score=1.0,
                    max_score=1.0,
                    normalized_score=1.0,
                    passed=True,
                    primary_metric="exact_str_match",
                    primary_metric_direction="higher_is_better",
                    details={},
                )

            judge_obj = judge
            scoring_evaluation.evaluate_record = fake_evaluate_answer
            with tempfile.TemporaryDirectory() as tmpdir:
                results = run_group_for_test(
                    group=experiments.EXPERIMENT_GROUPS["chemqa_skills_on"],
                    records=[record],
                    output_root=Path(tmpdir),
                    single_timeout=10,
                    chemqa_timeout=10,
                    judge=judge,
                    config_path=Path(tmpdir) / "cfg.json",
                    single_agent="unused",
                    chemqa_root=Path(tmpdir),
                    chemqa_model_profile="unused",
                    review_rounds=None,
                    rebuttal_rounds=None,
                )
            entry = results[0]
            self.assertIsNone(entry.error)
            self.assertTrue(entry.evaluation["passed"])
            self.assertEqual("fallback-answer", entry.short_answer_text)
            self.assertEqual("FINAL ANSWER: fallback-answer", entry.full_response_text)
            self.assertEqual("proposer-1-proposal", entry.runner_meta["fallback_source"])
            self.assertEqual({"status": "done", "terminal_state": "failed"}, entry.raw["run_status"])
        finally:
            runner_adapters.build_runner = original_build_runner
            scoring_evaluation.evaluate_record = original_evaluate_answer

    def test_run_group_accepts_structural_result_object_for_unscored_recovery(self) -> None:
        record = BenchmarkRecord(
            record_id="structural-recovery-record",
            dataset="chembench",
            source_file="/tmp/demo.jsonl",
            eval_kind="chembench_open_ended",
            prompt="Q",
            reference_answer="A",
            payload={},
        )

        class FakeAnswer:
            short_answer_text = "fallback-answer"
            full_response_text = "FINAL ANSWER: fallback-answer"

        class FakeStatus:
            value = "recovered"

        class FakeResult:
            status = FakeStatus()
            answer = FakeAnswer()
            raw = {"run_status": {"status": "done", "terminal_state": "failed"}}
            runner_meta = {
                "run_id": "demo-run",
                "fallback_used": True,
                "fallback_source": "test-double",
                "error": "ChemQA run `demo-run` terminated as failed (reason=stalled)",
            }
            failure = None

            def should_score(self) -> bool:
                return False

        class StubRunner:
            def run(self, record: object, group: object) -> object:
                self.called_with = (record, group)
                return FakeResult()

        stub_runner = StubRunner()
        original_build_runner = runner_adapters.build_runner
        original_evaluate_answer = scoring_evaluation.evaluate_record
        try:
            runner_adapters.build_runner = lambda **kwargs: stub_runner

            def fail_evaluate_answer(*args, **kwargs):
                raise AssertionError("evaluate_answer should not be called for unscored recovery")

            scoring_evaluation.evaluate_record = fail_evaluate_answer
            with tempfile.TemporaryDirectory() as tmpdir:
                results = run_group_for_test(
                    group=experiments.EXPERIMENT_GROUPS["chemqa_skills_on"],
                    records=[record],
                    output_root=Path(tmpdir),
                    single_timeout=10,
                    chemqa_timeout=10,
                    judge=object(),
                    config_path=Path(tmpdir) / "cfg.json",
                    single_agent="benchmark-single-skills-off",
                    chemqa_root=Path(tmpdir),
                    chemqa_model_profile="unused",
                    review_rounds=None,
                    rebuttal_rounds=None,
                )
            self.assertEqual(1, len(results))
            entry = results[0]
            self.assertEqual("execution_error", entry.evaluation["primary_metric"])
            self.assertFalse(entry.evaluation["passed"])
            self.assertEqual("fallback-answer", entry.short_answer_text)
            self.assertEqual("FINAL ANSWER: fallback-answer", entry.full_response_text)
            self.assertEqual("FINAL ANSWER: fallback-answer", entry.answer_text)
            self.assertEqual("demo-run", entry.runner_meta["run_id"])
            self.assertEqual("test-double", entry.runner_meta["fallback_source"])
            self.assertEqual({"status": "done", "terminal_state": "failed"}, entry.raw["run_status"])
            self.assertEqual("ChemQA run `demo-run` terminated as failed (reason=stalled)", entry.error)
            self.assertEqual("completed", entry.run_lifecycle_status)
            self.assertEqual("failed", entry.protocol_completion_status)
            self.assertEqual("recovered_candidate", entry.answer_availability)
            self.assertEqual("high_confidence_recovered", entry.answer_reliability)
            self.assertEqual("test-double", entry.recovery_mode)
            self.assertTrue(entry.degraded_execution)
        finally:
            runner_adapters.build_runner = original_build_runner
            scoring_evaluation.evaluate_record = original_evaluate_answer

    def test_run_group_structural_unscored_recovery_without_failure_attr_uses_runner_meta_error(self) -> None:
        record = BenchmarkRecord(
            record_id="structural-omitted-failure-record",
            dataset="chembench",
            source_file="/tmp/demo.jsonl",
            eval_kind="chembench_open_ended",
            prompt="Q",
            reference_answer="A",
            payload={},
        )

        class FakeAnswer:
            short_answer_text = "fallback-answer"
            full_response_text = "FINAL ANSWER: fallback-answer"

        class FakeStatus:
            value = "recovered"

        class FakeResult:
            status = FakeStatus()
            answer = FakeAnswer()
            raw = {"run_status": {"status": "done", "terminal_state": "failed"}}
            runner_meta = {
                "run_id": "demo-run",
                "fallback_used": True,
                "fallback_source": "test-double",
                "error": "ChemQA run `demo-run` terminated as failed (reason=stalled)",
            }

            def should_score(self) -> bool:
                return False

        class StubRunner:
            def run(self, record: object, group: object) -> object:
                self.called_with = (record, group)
                return FakeResult()

        stub_runner = StubRunner()
        original_build_runner = runner_adapters.build_runner
        original_evaluate_answer = scoring_evaluation.evaluate_record
        try:
            runner_adapters.build_runner = lambda **kwargs: stub_runner

            def fail_evaluate_answer(*args, **kwargs):
                raise AssertionError("evaluate_answer should not be called for unscored recovery")

            scoring_evaluation.evaluate_record = fail_evaluate_answer
            with tempfile.TemporaryDirectory() as tmpdir:
                results = run_group_for_test(
                    group=experiments.EXPERIMENT_GROUPS["chemqa_skills_on"],
                    records=[record],
                    output_root=Path(tmpdir),
                    single_timeout=10,
                    chemqa_timeout=10,
                    judge=object(),
                    config_path=Path(tmpdir) / "cfg.json",
                    single_agent="benchmark-single-skills-off",
                    chemqa_root=Path(tmpdir),
                    chemqa_model_profile="unused",
                    review_rounds=None,
                    rebuttal_rounds=None,
                )
            self.assertEqual(1, len(results))
            entry = results[0]
            self.assertEqual("execution_error", entry.evaluation["primary_metric"])
            self.assertFalse(entry.evaluation["passed"])
            self.assertEqual("fallback-answer", entry.short_answer_text)
            self.assertEqual("FINAL ANSWER: fallback-answer", entry.full_response_text)
            self.assertEqual("FINAL ANSWER: fallback-answer", entry.answer_text)
            self.assertEqual("demo-run", entry.runner_meta["run_id"])
            self.assertEqual("test-double", entry.runner_meta["fallback_source"])
            self.assertEqual({"status": "done", "terminal_state": "failed"}, entry.raw["run_status"])
            self.assertEqual("ChemQA run `demo-run` terminated as failed (reason=stalled)", entry.error)
        finally:
            runner_adapters.build_runner = original_build_runner
            scoring_evaluation.evaluate_record = original_evaluate_answer

    def test_materialize_group_failure_results_writes_error_entries(self) -> None:
        records = [
            BenchmarkRecord(
                record_id="r1",
                dataset="chembench",
                source_file="/tmp/demo.jsonl",
                eval_kind="chembench_open_ended",
                prompt="Q1",
                reference_answer="A",
                payload={},
            ),
            BenchmarkRecord(
                record_id="r2",
                dataset="chembench",
                source_file="/tmp/demo.jsonl",
                eval_kind="chembench_open_ended",
                prompt="Q2",
                reference_answer="B",
                payload={},
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            results = materialize_failure_results_for_test(
                group=experiments.EXPERIMENT_GROUPS["chemqa_skills_on"],
                records=records,
                output_root=Path(tmpdir),
                error_message="group crashed",
            )
            self.assertEqual(2, len(results))
            self.assertTrue(all(item.error == "group crashed" for item in results))
            self.assertTrue((Path(tmpdir) / "per-record" / "chemqa_skills_on" / "r1.json").exists())
            self.assertTrue((Path(tmpdir) / "per-record" / "chemqa_skills_on" / "r2.json").exists())

    def test_benchmark_test_build_error_group_record_result_preserves_legacy_compatibility(self) -> None:
        record = BenchmarkRecord(
            record_id="demo-record",
            dataset="frontierscience",
            source_file="/tmp/frontier.jsonl",
            prompt="Question?",
            grading=GradingSpec(
                kind="frontierscience_research",
                reference_answer="42",
                subset="frontierscience_Research",
                config={"track": "research"},
            ),
            raw_payload={"track": "research"},
        )
        entry = build_error_result_for_test(
            group=experiments.EXPERIMENT_GROUPS["single_llm_skills_off"],
            record=record,
            error_message="boom",
            full_response_text="Reasoning\nFinal conclusion",
        )
        self.assertEqual("frontierscience_Research", entry.subset)
        self.assertEqual("Final conclusion", entry.short_answer_text)
        self.assertEqual("Reasoning\nFinal conclusion", entry.full_response_text)
        self.assertEqual("Reasoning\nFinal conclusion", entry.answer_text)

    def test_shared_reporting_build_error_group_record_result_requires_explicit_dependencies(self) -> None:
        record = BenchmarkRecord(
            record_id="demo-record",
            dataset="chembench",
            source_file="/tmp/demo.jsonl",
            eval_kind="chembench_open_ended",
            prompt="Q",
            reference_answer="A",
            payload={},
        )
        with self.assertRaises(TypeError):
            shared_build_error_group_record_result(
                group=experiments.EXPERIMENT_GROUPS["single_llm_skills_off"],
                record=record,
                error_message="group crashed",
            )


if __name__ == "__main__":
    unittest.main()
