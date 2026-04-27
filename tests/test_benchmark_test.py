from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from contextlib import redirect_stdout
from pathlib import Path
from typing import Iterator
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmark_test.py"
SPEC = importlib.util.spec_from_file_location("benchmark_test", MODULE_PATH)
benchmark_test = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = benchmark_test
SPEC.loader.exec_module(benchmark_test)

try:
    from benchmarking.contracts import AnswerPayload, RecoveryInfo, RunnerResult, RunStatus
    from benchmarking.reporting import build_error_group_record_result as shared_build_error_group_record_result
except ModuleNotFoundError as exc:
    if exc.name != "benchmarking":
        raise
    from workspace.benchmarking.contracts import AnswerPayload, RecoveryInfo, RunnerResult, RunStatus
    from workspace.benchmarking.reporting import build_error_group_record_result as shared_build_error_group_record_result


@contextmanager
def patched_benchmark_runtime_paths() -> Iterator[None]:
    original_baseline_root = benchmark_test.BASELINE_WORKSPACE_ROOT
    original_chemqa_roots = benchmark_test.CHEMQA_WORKSPACE_ROOTS
    original_agents_root = benchmark_test.runtime_paths.agents_root
    original_load_slot_agents_template = benchmark_test.load_slot_agents_template
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        benchmark_test.BASELINE_WORKSPACE_ROOT = root / "benchmark-runtime"
        benchmark_test.CHEMQA_WORKSPACE_ROOTS = {
            "A": benchmark_test.BASELINE_WORKSPACE_ROOT / "chemqa_web_on",
            "B": benchmark_test.BASELINE_WORKSPACE_ROOT / "chemqa_web_off",
        }
        benchmark_test.runtime_paths.agents_root = root / "agents"
        benchmark_test.load_slot_agents_template = lambda: "# test slot template\n"
        try:
            yield
        finally:
            benchmark_test.BASELINE_WORKSPACE_ROOT = original_baseline_root
            benchmark_test.CHEMQA_WORKSPACE_ROOTS = original_chemqa_roots
            benchmark_test.runtime_paths.agents_root = original_agents_root
            benchmark_test.load_slot_agents_template = original_load_slot_agents_template


class JudgeStub:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def evaluate_json(self, prompt: str) -> dict[str, object]:
        self.prompts.append(prompt)
        return dict(self.payload)


class BenchmarkTestModuleTests(unittest.TestCase):
    def test_parse_args_accepts_single_agent_id_override_and_rejects_removed_flags(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "benchmark_test.py",
                "--single-agent-id-override",
                "custom-single-agent",
            ],
        ):
            args = benchmark_test.parse_args()
        self.assertEqual("custom-single-agent", args.single_agent_id_override)

        with mock.patch.object(sys, "argv", ["benchmark_test.py", "--keep-temp-configs"]):
            with self.assertRaises(SystemExit):
                benchmark_test.parse_args()

        with mock.patch.object(sys, "argv", ["benchmark_test.py", "--single-agent", "custom"]):
            with self.assertRaises(SystemExit):
                benchmark_test.parse_args()

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
                return []

            argv = [
                "benchmark_test.py",
                "--benchmark-root",
                str(root),
                "--openclaw-config",
                str(base_config),
                "--exact-output-dir",
                str(output_root),
                "--groups",
                "single_llm_web_off",
                "--single-agent-id-override",
                "custom-single-agent",
            ]

            with mock.patch.object(benchmark_test, "ConfigPool", DummyConfigPool), \
                mock.patch.object(benchmark_test, "JudgeClient", return_value=object()), \
                mock.patch.object(benchmark_test, "run_group", side_effect=fake_run_group), \
                mock.patch.object(sys, "argv", argv):
                exit_code = benchmark_test.main()

            self.assertEqual(0, exit_code)
            self.assertEqual("custom-single-agent", captured.get("single_agent"))

    def test_current_python_prefers_virtualenv_python(self) -> None:
        original_virtual_env = os.environ.get("VIRTUAL_ENV")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                venv_root = Path(tmpdir) / ".venv"
                python_path = venv_root / "bin" / "python"
                python_path.parent.mkdir(parents=True, exist_ok=True)
                python_path.write_text("", encoding="utf-8")
                os.environ["VIRTUAL_ENV"] = str(venv_root)
                self.assertEqual(str(python_path), benchmark_test.current_python())
        finally:
            if original_virtual_env is None:
                os.environ.pop("VIRTUAL_ENV", None)
            else:
                os.environ["VIRTUAL_ENV"] = original_virtual_env

    def test_invoke_cleanroom_cleanup_uses_current_python(self) -> None:
        original_current_python = benchmark_test.current_python
        original_run_subprocess = benchmark_test.run_subprocess
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                captured: dict[str, object] = {}
                manifest_path = Path(tmpdir) / "demo.manifest.json"
                manifest_path.write_text("{}", encoding="utf-8")

                benchmark_test.current_python = lambda: "/tmp/fake-venv/bin/python"

                def fake_run_subprocess(command, **kwargs):
                    captured["command"] = list(command)
                    return benchmark_test.subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=json.dumps({"success": True}),
                        stderr="",
                    )

                benchmark_test.run_subprocess = fake_run_subprocess
                payload = benchmark_test.invoke_cleanroom_cleanup(manifest_path=manifest_path)
                self.assertTrue(payload["success"])
                self.assertEqual("/tmp/fake-venv/bin/python", captured["command"][0])
        finally:
            benchmark_test.current_python = original_current_python
            benchmark_test.run_subprocess = original_run_subprocess

    def test_cleanup_manifest_path_uses_cleanroom_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            path = benchmark_test.cleanup_manifest_path(output_root, "demo-run")
            self.assertEqual((output_root / "cleanroom" / "manifests" / "demo-run.manifest.json").resolve(), path.resolve())

    def test_register_and_unregister_pending_cleanup_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "demo.manifest.json"
            benchmark_test.register_pending_cleanup_manifest(manifest_path)
            self.assertIn(manifest_path, benchmark_test.iter_pending_cleanup_manifests())
            benchmark_test.unregister_pending_cleanup_manifest(manifest_path)
            self.assertNotIn(manifest_path, benchmark_test.iter_pending_cleanup_manifests())

    def test_extract_final_answer_line_prefers_explicit_marker(self) -> None:
        text = "reasoning\nFINAL ANSWER: 42\n"
        self.assertEqual("42", benchmark_test.extract_final_answer_line(text))
        self.assertEqual("42", benchmark_test.extract_candidate_short_answer(text))

    def test_runner_result_should_score_when_recovery_is_evaluable(self) -> None:
        result = benchmark_test.RunnerResult(
            status=benchmark_test.RunStatus.RECOVERED,
            answer=benchmark_test.AnswerPayload(
                short_answer_text="CCO",
                full_response_text="FINAL ANSWER: CCO",
            ),
            raw={"run_status": {"status": "done", "terminal_state": "failed"}},
            runner_meta={"run_id": "demo-run"},
            recovery=benchmark_test.RecoveryInfo(
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
        items = benchmark_test.parse_frontierscience_research_rubric(rubric)
        self.assertEqual(2, len(items))
        self.assertEqual(1.0, items[0]["points"])
        self.assertIn("First criterion", items[0]["description"])
        self.assertIn("more detail", items[0]["description"])
        self.assertEqual(0.5, items[1]["points"])

    def test_build_temp_openclaw_config_payload_toggles_websearch(self) -> None:
        base = {
            "tools": {"web": {"search": {"enabled": False}}},
            "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
        }
        enabled = benchmark_test.build_temp_openclaw_config_payload(base, enable_websearch=True)
        disabled = benchmark_test.build_temp_openclaw_config_payload(base, enable_websearch=False)
        self.assertIs(True, enabled["tools"]["web"]["search"]["enabled"])
        self.assertIs(True, enabled["plugins"]["entries"]["duckduckgo"]["enabled"])
        self.assertIs(False, disabled["tools"]["web"]["search"]["enabled"])
        self.assertIs(False, disabled["plugins"]["entries"]["duckduckgo"]["enabled"])

    def test_build_group_waves_batches_web_on_then_web_off(self) -> None:
        waves = benchmark_test.build_group_waves(
            ["chemqa_web_on", "chemqa_web_off", "single_llm_web_on", "single_llm_web_off"],
            max_concurrent_groups=2,
        )
        self.assertEqual(
            [["chemqa_web_on", "single_llm_web_on"], ["chemqa_web_off", "single_llm_web_off"]],
            waves,
        )

    def test_build_group_waves_respects_max_concurrent_groups(self) -> None:
        waves = benchmark_test.build_group_waves(
            ["chemqa_web_on", "chemqa_web_off", "single_llm_web_on", "single_llm_web_off"],
            max_concurrent_groups=1,
        )
        self.assertEqual(
            [["chemqa_web_on"], ["single_llm_web_on"], ["chemqa_web_off"], ["single_llm_web_off"]],
            waves,
        )

    def test_resolve_aggregate_group_ids_includes_existing_group_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            existing = root / "per-record" / "single_llm_web_on"
            existing.mkdir(parents=True, exist_ok=True)
            (existing / "demo.json").write_text("{}\n", encoding="utf-8")
            aggregate = benchmark_test.resolve_aggregate_group_ids(
                ["chemqa_web_on", "chemqa_web_off", "single_llm_web_off"],
                output_root=root,
                merge_existing_per_record=True,
            )
            self.assertEqual(
                ["chemqa_web_on", "chemqa_web_off", "single_llm_web_on", "single_llm_web_off"],
                aggregate,
            )

    def test_load_results_from_output_root_reads_existing_per_record_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            group_dir = root / "per-record" / "single_llm_web_on"
            group_dir.mkdir(parents=True, exist_ok=True)
            payload = benchmark_test.GroupRecordResult(
                group_id="single_llm_web_on",
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
                error=None,
                short_answer_text="A",
                full_response_text="FINAL ANSWER: A",
            )
            (group_dir / "demo-record.json").write_text(json.dumps(benchmark_test.asdict(payload)), encoding="utf-8")
            loaded = benchmark_test.load_results_from_output_root(root, group_ids=["single_llm_web_on"])
            self.assertEqual(1, len(loaded))
            self.assertEqual("demo-record", loaded[0].record_id)
            self.assertEqual("A", loaded[0].short_answer_text)

    def test_build_run_scoped_config_payload_uses_explicit_single_and_judge_models(self) -> None:
        base = {
            "agents": {"list": []},
            "tools": {"web": {"search": {"enabled": False}}},
            "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
        }
        group = benchmark_test.EXPERIMENT_GROUPS["single_llm_web_on"]
        with patched_benchmark_runtime_paths():
            payload = benchmark_test.build_run_scoped_config_payload(
                base,
                group=group,
                single_agent_model="qwen3.5-plus",
                judge_model="su8/gpt-5.4",
            )
        agents = {entry["id"]: entry for entry in payload["agents"]["list"]}
        self.assertEqual("qwen3.5-plus", agents["benchmark-single-web-on"]["model"])
        self.assertEqual("su8/gpt-5.4", agents["benchmark-judge"]["model"])
        self.assertNotIn("thinking", agents["benchmark-single-web-on"])
        self.assertNotIn("thinking", agents["benchmark-judge"])

    def test_build_run_scoped_config_payload_benchmark_judge_runtime_uses_judge_model(self) -> None:
        base = {
            "agents": {"list": []},
            "tools": {"web": {"search": {"enabled": False}}},
            "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
        }
        group = benchmark_test.ExperimentGroup(
            id="benchmark-judge-runtime",
            label="benchmark judge runtime",
            runner="single_llm",
            websearch=False,
        )
        with patched_benchmark_runtime_paths():
            payload = benchmark_test.build_run_scoped_config_payload(
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
        group = benchmark_test.EXPERIMENT_GROUPS["chemqa_web_off"]
        with patched_benchmark_runtime_paths():
            payload = benchmark_test.build_run_scoped_config_payload(
                base,
                group=group,
                single_agent_model="qwen3.5-plus",
                judge_model="su8/gpt-5.4",
            )
        agents = {entry["id"]: entry for entry in payload["agents"]["list"]}
        self.assertEqual("su8/gpt-5.4", agents["benchmark-judge"]["model"])
        self.assertEqual("qwen3.5-plus", agents["debateB-coordinator"]["model"])
        self.assertNotIn("thinking", agents["debateB-coordinator"])
        for slot in ["debateB-1", "debateB-2", "debateB-3", "debateB-4", "debateB-5"]:
            self.assertEqual("qwen3.5-plus", agents[slot]["model"])
            self.assertNotIn("thinking", agents[slot])

    def test_build_run_scoped_config_payload_chemqa_uses_benchmark_workspace_roots(self) -> None:
        base = {
            "agents": {"list": []},
            "tools": {"web": {"search": {"enabled": False}}},
            "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
        }
        group = benchmark_test.EXPERIMENT_GROUPS["chemqa_web_on"]
        with patched_benchmark_runtime_paths():
            payload = benchmark_test.build_run_scoped_config_payload(
                base,
                group=group,
                single_agent_model="qwen3.5-plus",
                judge_model="su8/gpt-5.4",
            )
            expected_root = benchmark_test.BASELINE_WORKSPACE_ROOT / "chemqa_web_on"
        agents = {entry["id"]: entry for entry in payload["agents"]["list"]}
        self.assertEqual(str((expected_root / "debateA-coordinator").resolve()), agents["debateA-coordinator"]["workspace"])
        self.assertEqual(str((expected_root / "debateA-1").resolve()), agents["debateA-1"]["workspace"])

    def test_build_run_scoped_config_payload_raises_benchmark_error_when_agents_list_invalid(self) -> None:
        base = {
            "agents": {"list": {}},
            "tools": {"web": {"search": {"enabled": False}}},
            "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
        }
        group = benchmark_test.EXPERIMENT_GROUPS["single_llm_web_on"]

        with patched_benchmark_runtime_paths():
            with self.assertRaises(benchmark_test.BenchmarkError):
                benchmark_test.build_run_scoped_config_payload(
                    base,
                    group=group,
                    single_agent_model="qwen3.5-plus",
                    judge_model="su8/gpt-5.4",
                )

    def test_normalize_chemqa_run_status_maps_completed_with_artifact_errors(self) -> None:
        payload = benchmark_test.normalize_chemqa_run_status({"status": "completed_with_artifact_errors"})
        self.assertEqual("done", payload["status"])
        self.assertEqual("completed", payload["terminal_state"])
        self.assertEqual("artifact_collection_error", payload["terminal_reason_code"])
        self.assertEqual("error", payload["artifact_collection"]["status"])
        self.assertEqual("completed_with_artifact_errors", payload["legacy_status"])

    def test_normalize_chemqa_run_status_maps_stalled(self) -> None:
        payload = benchmark_test.normalize_chemqa_run_status({"status": "stalled", "phase": "review"})
        self.assertEqual("done", payload["status"])
        self.assertEqual("failed", payload["terminal_state"])
        self.assertEqual("stalled", payload["terminal_reason_code"])
        self.assertEqual("stalled", payload["legacy_status"])

    def test_normalize_chemqa_run_status_maps_terminal_failure(self) -> None:
        payload = benchmark_test.normalize_chemqa_run_status({"status": "terminal_failure", "reason": "boom"})
        self.assertEqual("done", payload["status"])
        self.assertEqual("failed", payload["terminal_state"])
        self.assertEqual("terminal_failure", payload["terminal_reason_code"])
        self.assertEqual("boom", payload["terminal_reason"])
        self.assertEqual("terminal_failure", payload["legacy_status"])

    def test_normalize_chemqa_run_status_maps_abandoned(self) -> None:
        payload = benchmark_test.normalize_chemqa_run_status({"status": "abandoned"})
        self.assertEqual("done", payload["status"])
        self.assertEqual("cancelled", payload["terminal_state"])
        self.assertEqual("abandoned", payload["terminal_reason_code"])
        self.assertEqual("abandoned", payload["legacy_status"])

    def test_chemqa_wait_for_terminal_status_accepts_new_done_state(self) -> None:
        runner = benchmark_test.ChemQARunner.__new__(benchmark_test.ChemQARunner)
        runner._read_run_status = lambda _run_id: {"status": "done", "terminal_state": "failed", "terminal_reason_code": "stalled"}
        payload = benchmark_test.ChemQARunner._wait_for_terminal_status(runner, "demo-run", timeout_seconds=1)
        self.assertEqual("done", payload["status"])
        self.assertEqual("failed", payload["terminal_state"])

    def test_chemqa_wait_for_terminal_status_accepts_legacy_terminal_failure(self) -> None:
        runner = benchmark_test.ChemQARunner.__new__(benchmark_test.ChemQARunner)
        runner._read_run_status = lambda _run_id: benchmark_test.normalize_chemqa_run_status({"status": "terminal_failure", "phase": "review"})
        payload = benchmark_test.ChemQARunner._wait_for_terminal_status(runner, "demo-run", timeout_seconds=1)
        self.assertEqual("done", payload["status"])
        self.assertEqual("failed", payload["terminal_state"])
        self.assertEqual("terminal_failure", payload["legacy_status"])

    def test_chemqa_wait_for_terminal_status_timeout_on_half_initialized_runner_raises_benchmark_error(self) -> None:
        runner = benchmark_test.ChemQARunner.__new__(benchmark_test.ChemQARunner)
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.chemqa_root = Path(tmpdir)
            original_time = benchmark_test.time.time
            original_sleep = benchmark_test.time.sleep
            times = iter([100.0, 99.0, 101.5])
            try:
                benchmark_test.time.time = lambda: next(times)
                benchmark_test.time.sleep = lambda _seconds: None
                with self.assertRaises(benchmark_test.BenchmarkError):
                    benchmark_test.ChemQARunner._wait_for_terminal_status(runner, "demo-run", timeout_seconds=1)
            finally:
                benchmark_test.time.time = original_time
                benchmark_test.time.sleep = original_sleep

    def test_build_chemqa_response_from_submission_uses_direct_answer(self) -> None:
        short_text, full_text = benchmark_test.build_chemqa_response_from_submission(
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
            runner = benchmark_test.ChemQARunner.__new__(benchmark_test.ChemQARunner)
            runner._candidate_protocol_dirs = lambda _run_id, _run_status: [team_dir]
            short_text, full_text, meta = benchmark_test.ChemQARunner._build_candidate_submission_fallback(
                runner,
                "demo-run",
                {"status": "stalled", "phase": "review"},
            )
            self.assertEqual("3-(trifluoromethyl)benzamide", short_text)
            self.assertIn("FINAL ANSWER: 3-(trifluoromethyl)benzamide", full_text)
            self.assertEqual("proposer-1-proposal", meta["fallback_source"])
            self.assertEqual(str(proposal_path.resolve()), str(Path(meta["proposal_path"]).resolve()))

    def test_candidate_protocol_dirs_include_new_benchmark_coordinator_workspace(self) -> None:
        runner = benchmark_test.ChemQARunner.__new__(benchmark_test.ChemQARunner)
        runner.chemqa_root = Path("/tmp/chemqa-root")
        runner.slot_set = "B"
        candidates = benchmark_test.ChemQARunner._candidate_protocol_dirs(runner, "demo-run", {})
        self.assertIn(
            benchmark_test.BASELINE_WORKSPACE_ROOT / "chemqa_web_off" / "debateB-coordinator",
            candidates,
        )

    def test_evaluate_chembench_open_ended_numeric_match(self) -> None:
        record = benchmark_test.BenchmarkRecord(
            record_id="demo",
            dataset="chembench",
            source_file="/tmp/demo.jsonl",
            eval_kind="chembench_open_ended",
            prompt="What is 2+2?",
            reference_answer="4",
            payload={"target": "4", "preferred_score": "mae"},
        )
        result = benchmark_test.evaluate_chembench_open_ended(
            record,
            short_answer_text="4",
            full_response_text="Reasoning\nFINAL ANSWER: 4",
        )
        self.assertTrue(result.passed)
        self.assertEqual(0.0, result.score)
        self.assertEqual(1.0, result.normalized_score)
        self.assertEqual(0.0, result.details["mae"])

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
            records = benchmark_test.load_records([path])
            self.assertEqual(1, len(records))
            self.assertEqual("Solve me", records[0].prompt)
            self.assertEqual("42", records[0].reference_answer)

    def test_evaluate_frontierscience_olympiad_heuristic_match(self) -> None:
        judge = JudgeStub({"correct": False})
        record = benchmark_test.BenchmarkRecord(
            record_id="fs-demo",
            dataset="frontierscience",
            source_file="/tmp/frontierscience.jsonl",
            eval_kind="frontierscience_olympiad",
            prompt="What is 6 x 7?",
            reference_answer="42",
            payload={"track": "olympiad"},
        )
        result = benchmark_test.evaluate_frontierscience_olympiad(
            record,
            short_answer_text="42",
            full_response_text="FINAL ANSWER: 42",
            judge=judge,
        )
        self.assertTrue(result.passed)
        self.assertEqual("heuristic", result.details["method"])
        self.assertEqual([], judge.prompts)

    def test_evaluate_answer_uses_generic_semantic_fallback(self) -> None:
        judge = JudgeStub({"correct": False})
        record = benchmark_test.BenchmarkRecord(
            record_id="generic-demo",
            dataset="customset",
            source_file="/tmp/custom.jsonl",
            eval_kind="custom_eval_kind",
            prompt="Name the molecule.",
            reference_answer="benzene",
            payload={},
        )
        result = benchmark_test.evaluate_answer(
            record,
            short_answer_text="benzene",
            full_response_text="FINAL ANSWER: benzene",
            judge=judge,
        )
        self.assertTrue(result.passed)
        self.assertEqual("semantic_match", result.primary_metric)
        self.assertEqual("heuristic", result.details["method"])
        self.assertEqual([], judge.prompts)

    def test_load_records_malformed_json_propagates_decode_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "chembench" / "data" / "broken.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"id":"broken","prompt":"Q"\n', encoding="utf-8")

            with self.assertRaises(json.JSONDecodeError):
                benchmark_test.load_records([path])

    def test_superchem_valid_options_uses_grading_config_before_payload(self) -> None:
        record = benchmark_test.BenchmarkRecord(
            record_id="superchem-demo",
            dataset="superchem",
            source_file="/tmp/superchem.jsonl",
            prompt="Q",
            grading=benchmark_test.GradingSpec(
                kind="superchem_multiple_choice_rpf",
                reference_answer="A",
                subset="superchem_multimodal",
                config={"options": {"A": "x", "C": "y"}},
            ),
            raw_payload={},
        )
        self.assertEqual(("A", "C"), benchmark_test.superchem_valid_options(record))

    def test_classify_subset(self) -> None:
        chembench_record = benchmark_test.BenchmarkRecord(
            record_id="c1",
            dataset="chembench",
            source_file="/tmp/chembench.jsonl",
            eval_kind="chembench_open_ended",
            prompt="Q",
            reference_answer="A",
            payload={},
        )
        olympiad_record = benchmark_test.BenchmarkRecord(
            record_id="f1",
            dataset="frontierscience",
            source_file="/tmp/frontierscience.jsonl",
            eval_kind="frontierscience_olympiad",
            prompt="Q",
            reference_answer="A",
            payload={"track": "olympiad"},
        )
        research_record = benchmark_test.BenchmarkRecord(
            record_id="f2",
            dataset="frontierscience",
            source_file="/tmp/frontierscience.jsonl",
            eval_kind="frontierscience_research",
            prompt="Q",
            reference_answer="A",
            payload={"track": "research"},
        )
        self.assertEqual("chembench", benchmark_test.classify_subset(chembench_record))
        self.assertEqual("frontierscience_Olympiad", benchmark_test.classify_subset(olympiad_record))
        self.assertEqual("frontierscience_Research", benchmark_test.classify_subset(research_record))

    def test_classify_subset_superchem(self) -> None:
        legacy_text_record = benchmark_test.BenchmarkRecord(
            record_id="s1",
            dataset="superchem",
            source_file="/tmp/superchem.jsonl",
            eval_kind="superchem_multiple_choice_rpf",
            prompt="Q",
            reference_answer="A",
            payload={"modality": "text_only"},
        )
        multimodal_record = benchmark_test.BenchmarkRecord(
            record_id="s2",
            dataset="superchem",
            source_file="/tmp/superchem.jsonl",
            eval_kind="superchem_multiple_choice_rpf",
            prompt="Q",
            reference_answer="A",
            payload={"modality": "multimodal"},
        )
        self.assertEqual("superchem_multimodal", benchmark_test.classify_subset(legacy_text_record))
        self.assertEqual("superchem_multimodal", benchmark_test.classify_subset(multimodal_record))

    def test_classify_subset_conformabench(self) -> None:
        record = benchmark_test.BenchmarkRecord(
            record_id="cb-1",
            dataset="conformabench",
            source_file="/tmp/conformabench.jsonl",
            eval_kind="conformabench_constructive",
            prompt="Q",
            reference_answer="A",
            payload={},
        )
        self.assertEqual("conformabench", benchmark_test.classify_subset(record))

    def test_build_single_llm_prompt_conformabench_requires_smiles_final_line(self) -> None:
        record = benchmark_test.BenchmarkRecord(
            record_id="cb-1",
            dataset="conformabench",
            source_file="/tmp/conformabench.jsonl",
            eval_kind="conformabench_constructive",
            prompt="Design a molecule.",
            reference_answer="Points: 1.0, Item: ok",
            payload={},
        )
        prompt = benchmark_test.build_single_llm_prompt(record, websearch_enabled=False, input_bundle=None)
        self.assertIn("FINAL ANSWER: <SMILES>", prompt)

    def test_build_chemqa_goal_conformabench_requires_smiles_final_line(self) -> None:
        record = benchmark_test.BenchmarkRecord(
            record_id="cb-1",
            dataset="conformabench",
            source_file="/tmp/conformabench.jsonl",
            eval_kind="conformabench_constructive",
            prompt="Design a molecule.",
            reference_answer="Points: 1.0, Item: ok",
            payload={},
        )
        goal = benchmark_test.build_chemqa_goal(record, websearch_enabled=True, input_bundle=None)
        self.assertIn("FINAL ANSWER: <SMILES>", goal)

    def test_evaluate_conformabench_constructive_handles_current_rdkit_environment(self) -> None:
        record = benchmark_test.BenchmarkRecord(
            record_id="conformabench-0001",
            dataset="conformabench",
            source_file=str(
                Path(__file__).resolve().parents[2] / "benchmarks" / "conformabench" / "data" / "conformabench_pool.jsonl"
            ),
            eval_kind="conformabench_constructive",
            prompt="Design one molecule.",
            reference_answer="Points: 1.0, Item: Submitted a chemically valid molecule in parseable SMILES form.",
            payload={"hidden_judge_spec_ref": "conformabench-0001"},
        )
        try:
            import rdkit  # noqa: F401
        except ModuleNotFoundError:
            with self.assertRaises(benchmark_test.BenchmarkError) as exc:
                benchmark_test.evaluate_answer(
                    record,
                    short_answer_text="c1ccccc1O",
                    full_response_text="Reasoning\nFINAL ANSWER: c1ccccc1O",
                    judge=JudgeStub({}),
                )
            self.assertIn("optional `rdkit` dependency", str(exc.exception))
            return

        hidden_path = benchmark_test.resolve_hidden_judge_spec_path(record.source_file, str(record.payload.get("hidden_judge_spec_ref") or ""))
        if not hidden_path.is_file():
            self.skipTest(f"Hidden judge spec fixture not present: {hidden_path}")

        result = benchmark_test.evaluate_answer(
            record,
            short_answer_text="Nc1ccccc1O",
            full_response_text="Reasoning\nFINAL ANSWER: Nc1ccccc1O",
            judge=JudgeStub({}),
        )
        self.assertEqual("rdkit_gate_pass", result.primary_metric)
        self.assertEqual("conformabench_rdkit_gate", result.details["method"])
        self.assertIn("canonical_smiles", result.details)
        self.assertIn("topology_predicates", result.details)
        self.assertIn("seed_runs", result.details)

    def test_evaluate_conformabench_constructive_invokes_submission_once(self) -> None:
        record = benchmark_test.BenchmarkRecord(
            record_id="conformabench-0002",
            dataset="conformabench",
            source_file="/tmp/conformabench/data/conformabench_pool.jsonl",
            eval_kind="conformabench_constructive",
            prompt="Design one molecule.",
            reference_answer="Points: 1.0, Item: Submitted a chemically valid molecule in parseable SMILES form.",
            payload={"hidden_judge_spec_ref": "conformabench-0002"},
        )
        hidden_path = Path("/tmp/conformabench/items/conformabench-0002/hidden_judge_spec.yaml")
        gate_payload = {"passed": False, "canonical_smiles": "c1ccccc1"}
        with mock.patch.object(benchmark_test, "ensure_rdkit_available", return_value=None), \
            mock.patch.object(benchmark_test, "resolve_hidden_judge_spec_path", return_value=hidden_path), \
            mock.patch.object(benchmark_test, "load_hidden_judge_spec", return_value={"normalization": {}}), \
            mock.patch.object(benchmark_test, "evaluate_conformabench_submission", return_value=gate_payload) as evaluate_mock:
            result = benchmark_test.evaluate_answer(
                record,
                short_answer_text="c1ccccc1",
                full_response_text="Reasoning\nFINAL ANSWER: c1ccccc1",
                judge=JudgeStub({}),
            )

        evaluate_mock.assert_called_once_with(final_answer_smiles="c1ccccc1", hidden_spec={"normalization": {}})
        self.assertFalse(result.passed)

    def test_sample_records_per_subset_draws_requested_count(self) -> None:
        records = []
        for idx in range(3):
            records.append(
                benchmark_test.BenchmarkRecord(
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
                benchmark_test.BenchmarkRecord(
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
                benchmark_test.BenchmarkRecord(
                    record_id=f"res-{idx}",
                    dataset="frontierscience",
                    source_file="/tmp/frontierscience.jsonl",
                    eval_kind="frontierscience_research",
                    prompt="Q",
                    reference_answer="A",
                    payload={"track": "research"},
                )
            )
        sampled = benchmark_test.sample_records_per_subset(records, per_subset_count=2, seed=7)
        self.assertEqual(6, len(sampled))
        counts = {subset: 0 for subset in benchmark_test.SUBSET_ORDER}
        for record in sampled:
            counts[benchmark_test.classify_subset(record)] += 1
        self.assertEqual(2, counts["chembench"])
        self.assertEqual(2, counts["frontierscience_Olympiad"])
        self.assertEqual(2, counts["frontierscience_Research"])

    def test_sample_records_per_subset_samples_superchem_multimodal_only(self) -> None:
        records = [
            benchmark_test.BenchmarkRecord(
                record_id="s1-mm",
                dataset="superchem",
                source_file="/tmp/superchem.jsonl",
                eval_kind="superchem_multiple_choice_rpf",
                prompt="Q1",
                reference_answer="A",
                payload={"modality": "multimodal", "source_uuid": "uuid-1"},
            ),
            benchmark_test.BenchmarkRecord(
                record_id="s2-mm",
                dataset="superchem",
                source_file="/tmp/superchem.jsonl",
                eval_kind="superchem_multiple_choice_rpf",
                prompt="Q2",
                reference_answer="B",
                payload={"modality": "multimodal", "source_uuid": "uuid-2"},
            ),
        ]
        sampled = benchmark_test.sample_records_per_subset(records, per_subset_count=1, seed=3)
        self.assertEqual(1, len(sampled))
        self.assertEqual(
            {"superchem_multimodal"},
            {benchmark_test.classify_subset(record) for record in sampled},
        )
        self.assertEqual(1, len({record.payload["source_uuid"] for record in sampled}))

    def test_print_selected_records_outputs_json(self) -> None:
        records = [
            benchmark_test.BenchmarkRecord(
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
            benchmark_test.print_selected_records(records)
        payload = json.loads(stream.getvalue())
        self.assertEqual("chem-1", payload[0]["record_id"])
        self.assertEqual("chembench", payload[0]["subset"])
        self.assertEqual("chembench_open_ended", payload[0]["eval_kind"])

    def test_apply_offset_limit_preserves_existing_behavior(self) -> None:
        records = [
            benchmark_test.BenchmarkRecord(
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
        sliced = benchmark_test.apply_offset_limit(records, offset=3, limit=4)
        self.assertEqual(["r3", "r4", "r5", "r6"], [record.record_id for record in sliced])

    def test_parse_superchem_option_answer_handles_common_formats(self) -> None:
        valid_options = ("A", "B", "C", "D")
        self.assertEqual("B", benchmark_test.parse_superchem_option_answer("FINAL ANSWER: B", valid_options=valid_options))
        self.assertEqual("A|D", benchmark_test.parse_superchem_option_answer("Option A and D are correct.", valid_options=valid_options))
        self.assertEqual(
            "B|C",
            benchmark_test.parse_superchem_option_answer('{"answer": ["C", "B"]}', valid_options=valid_options),
        )

    def test_ensure_runtime_bundle_copies_superchem_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            source_image = temp_dir / "source.png"
            source_image.write_bytes(b"image-bytes")
            record = benchmark_test.BenchmarkRecord(
                record_id="superchem-demo-mm",
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
                    "question_image_paths": [str(source_image)],
                    "option_image_paths": {},
                },
            )
            bundle = benchmark_test.ensure_runtime_bundle(record, bundle_root=temp_dir / "bundles")
            assert bundle is not None
            self.assertTrue(bundle.question_markdown.is_file())
            self.assertIn("Local images to inspect", bundle.question_markdown.read_text(encoding="utf-8"))
            self.assertEqual(1, len(bundle.image_files))
            self.assertTrue(bundle.image_files[0].is_file())
            self.assertEqual(b"image-bytes", bundle.image_files[0].read_bytes())

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
            short_text, full_text = benchmark_test.build_chemqa_full_response(qa_result=qa_result)
            self.assertEqual("F", short_text)
            self.assertIn("Summary:", full_text)
            self.assertIn("Probe A needs esterase cleavage", full_text)
            self.assertIn("Reasoning / submission trace:", full_text)
            self.assertIn("FINAL ANSWER: F", full_text)

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

            short_text, full_text = benchmark_test.build_chemqa_full_response(qa_result=qa_result)

            self.assertEqual("", short_text)
            self.assertIn("No candidate submission achieved acceptance.", full_text)
            self.assertNotIn("FINAL ANSWER:", full_text)

    def test_evaluate_superchem_multiple_choice_rpf(self) -> None:
        record = benchmark_test.BenchmarkRecord(
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
                "items": [
                    {"index": 1, "matched": True, "rationale": "covered"},
                    {"index": 2, "matched": False, "rationale": "missing"},
                ],
                "summary": "partial",
            }
        )
        result = benchmark_test.evaluate_superchem_multiple_choice_rpf(
            record,
            short_answer_text="B",
            full_response_text="Reasoning\nFINAL ANSWER: B",
            judge=judge,
        )
        self.assertTrue(result.passed)
        self.assertEqual(1.0, result.score)
        self.assertAlmostEqual(2.0 / 3.0, result.details["rpf"])
        self.assertEqual("B", result.details["parsed_prediction"])
        self.assertEqual(2, len(result.details["checkpoint_matches"]))
        self.assertEqual(1, len(judge.prompts))
        self.assertIn("Reasoning", judge.prompts[0])

    def test_aggregate_results_groups_by_experiment(self) -> None:
        sample = [
            benchmark_test.GroupRecordResult(
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
            ),
            benchmark_test.GroupRecordResult(
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
            ),
        ]
        summary = benchmark_test.aggregate_results(sample)
        self.assertEqual(2, summary["groups"]["g1"]["count"])
        self.assertEqual(1, summary["groups"]["g1"]["pass_count"])
        self.assertEqual(3.0, summary["groups"]["g1"]["avg_elapsed_seconds"])
        self.assertEqual(0.5, summary["groups"]["g1"]["avg_normalized_score"])

    def test_aggregate_results_includes_superchem_metrics(self) -> None:
        sample = [
            benchmark_test.GroupRecordResult(
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
            )
        ]
        summary = benchmark_test.aggregate_results(sample)
        self.assertEqual(1.0, summary["groups"]["g1"]["avg_answer_accuracy"])
        self.assertEqual(0.75, summary["groups"]["g1"]["avg_rpf"])

    def test_results_json_keeps_legacy_top_level_shape(self) -> None:
        sample = [
            benchmark_test.GroupRecordResult(
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
            ),
            benchmark_test.GroupRecordResult(
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
            ),
        ]
        summary = benchmark_test.aggregate_results(sample)
        self.assertEqual("benchmark_test", benchmark_test.GroupRecordResult.__module__)
        self.assertEqual("benchmarking.reporting", benchmark_test.aggregate_results.__module__)
        self.assertEqual(["group_order", "groups", "group_subset"], list(summary.keys()))
        self.assertEqual(["g1"], summary["group_order"])
        self.assertIn("g1", summary["groups"])
        self.assertIn("g1::chembench", summary["group_subset"])

    def test_export_csv_reports_writes_summary_files(self) -> None:
        summary = {
            "groups": {
                "g1": {
                    "runner": "single_llm",
                    "websearch": False,
                    "count": 2,
                    "pass_count": 1,
                    "avg_normalized_score": 0.5,
                    "avg_answer_accuracy": 1.0,
                    "avg_rpf": 0.75,
                }
            },
            "group_subset": {
                "g1::chembench": {
                    "group_id": "g1",
                    "runner": "single_llm",
                    "websearch": False,
                    "subset": "chembench",
                    "count": 2,
                    "pass_count": 1,
                    "avg_normalized_score": 0.5,
                    "avg_answer_accuracy": None,
                    "avg_rpf": None,
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark_test.export_csv_reports(root, summary, ["g1"])
            self.assertTrue((root / "summary_by_group.csv").exists())
            self.assertTrue((root / "summary_by_group_and_subset.csv").exists())

            with (root / "summary_by_group.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [
                    "group_id",
                    "runner",
                    "websearch",
                    "count",
                    "pass_count",
                    "avg_normalized_score",
                    "avg_answer_accuracy",
                    "avg_rpf",
                ],
                list(rows[0].keys()),
            )
            self.assertEqual("g1", rows[0]["group_id"])
            self.assertEqual("single_llm", rows[0]["runner"])
            self.assertEqual("0.5", rows[0]["avg_normalized_score"])
            self.assertEqual("1.0", rows[0]["avg_answer_accuracy"])
            self.assertEqual("0.75", rows[0]["avg_rpf"])

            with (root / "summary_by_group_and_subset.csv").open(newline="", encoding="utf-8") as handle:
                subset_rows = list(csv.DictReader(handle))
            self.assertEqual("chembench", subset_rows[0]["subset"])

    def test_judge_client_invokes_openclaw_with_high_thinking(self) -> None:
        captured: dict[str, object] = {}
        original_run_subprocess = benchmark_test.run_subprocess
        try:
            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                captured["command"] = list(command)
                captured["env"] = dict(env or {})
                return benchmark_test.subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps({"result": {"payloads": [{"text": '{"items": [], "summary": "ok"}'}], "meta": {}}}),
                    stderr="",
                )

            benchmark_test.run_subprocess = fake_run_subprocess
            client = benchmark_test.JudgeClient(
                judge_agent="benchmark-judge",
                timeout_seconds=30,
                config_path=Path("/tmp/judge.json"),
            )
            payload = client.evaluate_json("score this")
            self.assertEqual([], payload["items"])
            command = captured["command"]
            assert isinstance(command, list)
            self.assertIn("--thinking", command)
            self.assertEqual("high", command[command.index("--thinking") + 1])
        finally:
            benchmark_test.run_subprocess = original_run_subprocess

    def test_single_llm_runner_invokes_openclaw_with_high_thinking(self) -> None:
        captured: dict[str, object] = {}
        original_run_subprocess = benchmark_test.run_subprocess
        original_ensure_runtime_bundle = benchmark_test.ensure_runtime_bundle
        try:
            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                captured["command"] = list(command)
                captured["env"] = dict(env or {})
                return benchmark_test.subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps({"result": {"payloads": [{"text": 'Reasoning\nFINAL ANSWER: 5'}], "meta": {}}}),
                    stderr="",
                )

            benchmark_test.run_subprocess = fake_run_subprocess
            benchmark_test.ensure_runtime_bundle = lambda record, bundle_root: None
            runner = benchmark_test.SingleLLMRunner(
                agent_id="benchmark-single-web-on",
                timeout_seconds=30,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
            )
            record = benchmark_test.BenchmarkRecord(
                record_id="demo",
                dataset="chembench",
                source_file="/tmp/demo.jsonl",
                eval_kind="chembench_open_ended",
                prompt="What is 2+3?",
                reference_answer="5",
                payload={},
            )
            out = runner.run(record, benchmark_test.EXPERIMENT_GROUPS["single_llm_web_on"])
            self.assertEqual("5", out.short_answer_text)
            command = captured["command"]
            assert isinstance(command, list)
            self.assertIn("--thinking", command)
            self.assertEqual("high", command[command.index("--thinking") + 1])
        finally:
            benchmark_test.run_subprocess = original_run_subprocess
            benchmark_test.ensure_runtime_bundle = original_ensure_runtime_bundle

    def test_chemqa_runner_uses_run_scoped_writable_template_and_command_map_dirs(self) -> None:
        captured: dict[str, object] = {}
        original_run_subprocess = benchmark_test.run_subprocess
        original_ensure_runtime_bundle = benchmark_test.ensure_runtime_bundle
        original_wait_for_terminal_status = benchmark_test.ChemQARunner._wait_for_terminal_status
        original_ensure_artifacts = benchmark_test.ChemQARunner._ensure_artifacts
        original_invoke_cleanroom_cleanup = benchmark_test.invoke_cleanroom_cleanup
        try:
            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                captured["command"] = list(command)
                captured["env"] = dict(env or {})
                return benchmark_test.subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps({"run_id": "demo", "launch_mode": "run", "launched": {"returncode": 0}}),
                    stderr="",
                )

            benchmark_test.run_subprocess = fake_run_subprocess
            benchmark_test.ensure_runtime_bundle = lambda record, bundle_root: None
            benchmark_test.invoke_cleanroom_cleanup = lambda manifest_path: {"status": "cleaned", "manifest_path": str(manifest_path)}
            benchmark_test.ChemQARunner._wait_for_terminal_status = lambda self, run_id, timeout_seconds: {
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

            benchmark_test.ChemQARunner._ensure_artifacts = fake_ensure_artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                launch_root = Path(tmpdir) / "chemqa-launch"
                runner = benchmark_test.ChemQARunner(
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
                record = benchmark_test.BenchmarkRecord(
                    record_id="conformabench-0001",
                    dataset="conformabench",
                    source_file="/tmp/demo.jsonl",
                    eval_kind="conformabench_constructive",
                    prompt="Design a molecule.",
                    reference_answer="Points: 1.0, Item: ok",
                    payload={},
                )
                out = runner.run(record, benchmark_test.EXPERIMENT_GROUPS["chemqa_web_on"])
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
                self.assertEqual(str(launch_root / "chemqa_web_on" / "conformabench-0001" / "home"), env["HOME"])
                self.assertEqual(str(benchmark_test.DEFAULT_OPENCLAW_ENV_FILE), env["OPENCLAW_ENV_FILE"])
        finally:
            benchmark_test.run_subprocess = original_run_subprocess
            benchmark_test.ensure_runtime_bundle = original_ensure_runtime_bundle
            benchmark_test.invoke_cleanroom_cleanup = original_invoke_cleanroom_cleanup
            benchmark_test.ChemQARunner._wait_for_terminal_status = original_wait_for_terminal_status
            benchmark_test.ChemQARunner._ensure_artifacts = original_ensure_artifacts

    def test_chemqa_runner_archives_completed_artifacts_under_output_root(self) -> None:
        original_run_subprocess = benchmark_test.run_subprocess
        original_ensure_runtime_bundle = benchmark_test.ensure_runtime_bundle
        original_wait_for_terminal_status = benchmark_test.ChemQARunner._wait_for_terminal_status
        original_ensure_artifacts = benchmark_test.ChemQARunner._ensure_artifacts
        original_invoke_cleanroom_cleanup = benchmark_test.invoke_cleanroom_cleanup
        try:
            benchmark_test.run_subprocess = lambda command, *, env=None, cwd=None, timeout=None: benchmark_test.subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"run_id": "demo", "launch_mode": "run", "launched": {"returncode": 0}}),
                stderr="",
            )
            benchmark_test.ensure_runtime_bundle = lambda record, bundle_root: None
            benchmark_test.invoke_cleanroom_cleanup = lambda manifest_path: {"status": "cleaned", "manifest_path": str(manifest_path)}
            benchmark_test.ChemQARunner._wait_for_terminal_status = lambda self, run_id, timeout_seconds: {
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

            benchmark_test.ChemQARunner._ensure_artifacts = fake_ensure_artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                output_root = Path(tmpdir) / "benchmark-output"
                launch_root = output_root / "chemqa-launch"
                runner = benchmark_test.ChemQARunner(
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
                record = benchmark_test.BenchmarkRecord(
                    record_id="conformabench-0001",
                    dataset="conformabench",
                    source_file="/tmp/demo.jsonl",
                    eval_kind="conformabench_constructive",
                    prompt="Design a molecule.",
                    reference_answer="Points: 1.0, Item: ok",
                    payload={},
                )

                out = runner.run(record, benchmark_test.EXPERIMENT_GROUPS["chemqa_web_on"])

                self.assertEqual(benchmark_test.RunStatus.COMPLETED, out.status)
                archive_dir = output_root / "artifacts" / "chemqa_web_on" / "conformabench-0001" / out.runner_meta["run_id"]
                self.assertEqual(str(archive_dir), out.runner_meta["archive_dir"])
                self.assertEqual(str(archive_dir / "qa_result.json"), out.runner_meta["qa_result_path"])
                self.assertEqual(str(archive_dir / "chemqa_review_protocol.yaml"), out.runner_meta["archived_protocol_path"])
                self.assertEqual("ok", out.runner_meta["artifact_archive_status"])
                self.assertTrue((archive_dir / "qa_result.json").is_file())
                self.assertTrue((archive_dir / "chemqa_review_protocol.yaml").is_file())
                self.assertTrue((archive_dir / "final_answer.md").is_file())
        finally:
            benchmark_test.run_subprocess = original_run_subprocess
            benchmark_test.ensure_runtime_bundle = original_ensure_runtime_bundle
            benchmark_test.invoke_cleanroom_cleanup = original_invoke_cleanroom_cleanup
            benchmark_test.ChemQARunner._wait_for_terminal_status = original_wait_for_terminal_status
            benchmark_test.ChemQARunner._ensure_artifacts = original_ensure_artifacts

    def test_chemqa_runner_archives_protocol_and_rebuilds_qa_result_for_failed_terminal_run(self) -> None:
        original_run_subprocess = benchmark_test.run_subprocess
        original_ensure_runtime_bundle = benchmark_test.ensure_runtime_bundle
        original_wait_for_terminal_status = benchmark_test.ChemQARunner._wait_for_terminal_status
        original_build_candidate_submission_fallback = benchmark_test.ChemQARunner._build_candidate_submission_fallback
        original_collect_artifacts = benchmark_test.ChemQARunner._collect_artifacts_from_source
        original_invoke_cleanroom_cleanup = benchmark_test.invoke_cleanroom_cleanup
        try:
            benchmark_test.run_subprocess = lambda command, *, env=None, cwd=None, timeout=None: benchmark_test.subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"run_id": "demo", "launch_mode": "run", "launched": {"returncode": 0}}),
                stderr="",
            )
            benchmark_test.ensure_runtime_bundle = lambda record, bundle_root: None
            benchmark_test.invoke_cleanroom_cleanup = lambda manifest_path: {"status": "cleaned", "manifest_path": str(manifest_path)}
            benchmark_test.ChemQARunner._wait_for_terminal_status = lambda self, run_id, timeout_seconds: {
                "status": "done",
                "terminal_state": "failed",
                "terminal_reason_code": "lane_stalled",
                "artifact_collection": {"status": "error"},
                "protocol_path": str(self.chemqa_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id / "chemqa_review_protocol.yaml"),
            }
            benchmark_test.ChemQARunner._build_candidate_submission_fallback = lambda self, run_id, run_status: None

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

            benchmark_test.ChemQARunner._collect_artifacts_from_source = fake_collect_artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                output_root = Path(tmpdir) / "benchmark-output"
                launch_root = output_root / "chemqa-launch"
                chemqa_root = Path(tmpdir) / "chemqa-root"
                runner = benchmark_test.ChemQARunner(
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
                record = benchmark_test.BenchmarkRecord(
                    record_id="conformabench-0001",
                    dataset="conformabench",
                    source_file="/tmp/demo.jsonl",
                    eval_kind="conformabench_constructive",
                    prompt="Design a molecule.",
                    reference_answer="Points: 1.0, Item: ok",
                    payload={},
                )
                run_id = "benchmark-chemqa_web_on-conformabench-0001-20260424-000000"
                protocol_dir = chemqa_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id
                protocol_dir.mkdir(parents=True, exist_ok=True)
                (protocol_dir / "chemqa_review_protocol.yaml").write_text(
                    "question: Demo\nacceptance_status: rejected\nterminal_state: failed\nfailure_reason: lane stalled\nfinal_answer: \"\"\n",
                    encoding="utf-8",
                )
                runner._now_stamp = lambda: "20260424-000000"

                out = runner.run(record, benchmark_test.EXPERIMENT_GROUPS["chemqa_web_on"])

                self.assertEqual(benchmark_test.RunStatus.FAILED, out.status)
                archive_dir = output_root / "artifacts" / "chemqa_web_on" / "conformabench-0001" / run_id
                self.assertEqual(str(archive_dir), out.runner_meta["archive_dir"])
                self.assertEqual(str(archive_dir / "chemqa_review_protocol.yaml"), out.runner_meta["archived_protocol_path"])
                self.assertEqual("ok", out.runner_meta["artifact_archive_status"])
                self.assertTrue((archive_dir / "chemqa_review_protocol.yaml").is_file())
                self.assertTrue((archive_dir / "qa_result.json").is_file())
        finally:
            benchmark_test.run_subprocess = original_run_subprocess
            benchmark_test.ensure_runtime_bundle = original_ensure_runtime_bundle
            benchmark_test.invoke_cleanroom_cleanup = original_invoke_cleanroom_cleanup
            benchmark_test.ChemQARunner._wait_for_terminal_status = original_wait_for_terminal_status
            benchmark_test.ChemQARunner._build_candidate_submission_fallback = original_build_candidate_submission_fallback
            benchmark_test.ChemQARunner._collect_artifacts_from_source = original_collect_artifacts

    def test_chemqa_runner_failed_terminal_with_candidate_fallback_returns_scored_recovered_result(self) -> None:
        original_run_subprocess = benchmark_test.run_subprocess
        original_ensure_runtime_bundle = benchmark_test.ensure_runtime_bundle
        original_wait_for_terminal_status = benchmark_test.ChemQARunner._wait_for_terminal_status
        original_collect_artifacts = benchmark_test.ChemQARunner._collect_artifacts_from_source
        original_invoke_cleanroom_cleanup = benchmark_test.invoke_cleanroom_cleanup
        try:
            benchmark_test.run_subprocess = lambda command, *, env=None, cwd=None, timeout=None: benchmark_test.subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"run_id": "demo", "launch_mode": "run", "launched": {"returncode": 0}}),
                stderr="",
            )
            benchmark_test.ensure_runtime_bundle = lambda record, bundle_root: None
            benchmark_test.invoke_cleanroom_cleanup = lambda manifest_path: {"status": "cleaned", "manifest_path": str(manifest_path)}
            benchmark_test.ChemQARunner._wait_for_terminal_status = lambda self, run_id, timeout_seconds: {
                "status": "done",
                "terminal_state": "failed",
                "terminal_reason_code": "stalled",
                "artifact_collection": {},
                "protocol_path": str(self.chemqa_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id / "chemqa_review_protocol.yaml"),
            }

            def fake_collect_artifacts(self, *, source_dir, output_dir, env):
                output_dir.mkdir(parents=True, exist_ok=True)
                _ = (self, source_dir, env)

            benchmark_test.ChemQARunner._collect_artifacts_from_source = fake_collect_artifacts

            with tempfile.TemporaryDirectory() as tmpdir:
                output_root = Path(tmpdir) / "benchmark-output"
                launch_root = output_root / "chemqa-launch"
                chemqa_root = Path(tmpdir) / "chemqa-root"
                runner = benchmark_test.ChemQARunner(
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
                record = benchmark_test.BenchmarkRecord(
                    record_id="chembench-0001",
                    dataset="chembench",
                    source_file="/tmp/demo.jsonl",
                    eval_kind="chembench_open_ended",
                    prompt="Return ethanol.",
                    reference_answer="CCO",
                    payload={},
                )
                run_id = "benchmark-chemqa_web_on-chembench-0001-20260427-000000"
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

                out = runner.run(record, benchmark_test.EXPERIMENT_GROUPS["chemqa_web_on"])

                self.assertEqual(benchmark_test.RunStatus.RECOVERED, out.status)
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
            benchmark_test.run_subprocess = original_run_subprocess
            benchmark_test.ensure_runtime_bundle = original_ensure_runtime_bundle
            benchmark_test.invoke_cleanroom_cleanup = original_invoke_cleanroom_cleanup
            benchmark_test.ChemQARunner._wait_for_terminal_status = original_wait_for_terminal_status
            benchmark_test.ChemQARunner._collect_artifacts_from_source = original_collect_artifacts

    def test_chemqa_runner_failed_terminal_with_final_answer_preview_stays_failed_and_unscored(self) -> None:
        original_run_subprocess = benchmark_test.run_subprocess
        original_ensure_runtime_bundle = benchmark_test.ensure_runtime_bundle
        original_wait_for_terminal_status = benchmark_test.ChemQARunner._wait_for_terminal_status
        original_collect_artifacts = benchmark_test.ChemQARunner._collect_artifacts_from_source
        original_invoke_cleanroom_cleanup = benchmark_test.invoke_cleanroom_cleanup
        try:
            benchmark_test.run_subprocess = lambda command, *, env=None, cwd=None, timeout=None: benchmark_test.subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"run_id": "demo", "launch_mode": "run", "launched": {"returncode": 0}}),
                stderr="",
            )
            benchmark_test.ensure_runtime_bundle = lambda record, bundle_root: None
            benchmark_test.invoke_cleanroom_cleanup = lambda manifest_path: {"status": "cleaned", "manifest_path": str(manifest_path)}
            benchmark_test.ChemQARunner._wait_for_terminal_status = lambda self, run_id, timeout_seconds: {
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

            benchmark_test.ChemQARunner._collect_artifacts_from_source = fake_collect_artifacts

            with tempfile.TemporaryDirectory() as tmpdir:
                output_root = Path(tmpdir) / "benchmark-output"
                launch_root = output_root / "chemqa-launch"
                chemqa_root = Path(tmpdir) / "chemqa-root"
                runner = benchmark_test.ChemQARunner(
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
                record = benchmark_test.BenchmarkRecord(
                    record_id="chembench-0001",
                    dataset="chembench",
                    source_file="/tmp/demo.jsonl",
                    eval_kind="chembench_open_ended",
                    prompt="Return ethanol.",
                    reference_answer="CCO",
                    payload={},
                )
                run_id = "benchmark-chemqa_web_on-chembench-0001-20260427-000000"
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

                out = runner.run(record, benchmark_test.EXPERIMENT_GROUPS["chemqa_web_on"])

                self.assertEqual(benchmark_test.RunStatus.FAILED, out.status)
                self.assertFalse(out.should_score())
                self.assertTrue(out.runner_meta["fallback_used"])
                self.assertEqual("run-status-final-answer-preview", out.runner_meta["fallback_source"])
                self.assertIs(out.runner_meta["evaluable"], False)
                self.assertIs(out.runner_meta["scored"], False)
                self.assertEqual("low_confidence_recovered", out.runner_meta["answer_reliability"])
                self.assertEqual("preview_requires_strict_validation", out.runner_meta["recovery_reason"])
        finally:
            benchmark_test.run_subprocess = original_run_subprocess
            benchmark_test.ensure_runtime_bundle = original_ensure_runtime_bundle
            benchmark_test.invoke_cleanroom_cleanup = original_invoke_cleanroom_cleanup
            benchmark_test.ChemQARunner._wait_for_terminal_status = original_wait_for_terminal_status
            benchmark_test.ChemQARunner._collect_artifacts_from_source = original_collect_artifacts

    def test_chemqa_runner_reconciles_failed_run_status_with_completed_archived_rejection(self) -> None:
        original_run_subprocess = benchmark_test.run_subprocess
        original_ensure_runtime_bundle = benchmark_test.ensure_runtime_bundle
        original_wait_for_terminal_status = benchmark_test.ChemQARunner._wait_for_terminal_status
        original_collect_artifacts = benchmark_test.ChemQARunner._collect_artifacts_from_source
        original_invoke_cleanroom_cleanup = benchmark_test.invoke_cleanroom_cleanup
        try:
            benchmark_test.run_subprocess = lambda command, *, env=None, cwd=None, timeout=None: benchmark_test.subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"run_id": "demo", "launch_mode": "run", "launched": {"returncode": 0}}),
                stderr="",
            )
            benchmark_test.ensure_runtime_bundle = lambda record, bundle_root: None
            benchmark_test.invoke_cleanroom_cleanup = lambda manifest_path: {"status": "cleaned", "manifest_path": str(manifest_path)}
            benchmark_test.ChemQARunner._wait_for_terminal_status = lambda self, run_id, timeout_seconds: {
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

            benchmark_test.ChemQARunner._collect_artifacts_from_source = fake_collect_artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                output_root = Path(tmpdir) / "benchmark-output"
                launch_root = output_root / "chemqa-launch"
                chemqa_root = Path(tmpdir) / "chemqa-root"
                runner = benchmark_test.ChemQARunner(
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
                record = benchmark_test.BenchmarkRecord(
                    record_id="conformabench-0001",
                    dataset="conformabench",
                    source_file="/tmp/demo.jsonl",
                    eval_kind="conformabench_constructive",
                    prompt="Design a molecule.",
                    reference_answer="Points: 1.0, Item: ok",
                    payload={},
                )
                run_id = "benchmark-chemqa_web_on-conformabench-0001-20260424-000000"
                protocol_dir = chemqa_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id
                protocol_dir.mkdir(parents=True, exist_ok=True)
                (protocol_dir / "chemqa_review_protocol.yaml").write_text(
                    "question: Demo\nacceptance_status: rejected\nterminal_state: completed\nfinal_answer: \"\"\n",
                    encoding="utf-8",
                )
                runner._now_stamp = lambda: "20260424-000000"

                out = runner.run(record, benchmark_test.EXPERIMENT_GROUPS["chemqa_web_on"])

                self.assertEqual(benchmark_test.RunStatus.COMPLETED, out.status)
                self.assertEqual("", out.short_answer_text)
                self.assertIn("No accepted answer", out.full_response_text)
                self.assertEqual("rejected", out.runner_meta["acceptance_status"])
                self.assertEqual("completed", out.runner_meta["terminal_state"])
                self.assertEqual("stalled", out.runner_meta["terminal_reason_code"])
                archive_dir = output_root / "artifacts" / "chemqa_web_on" / "conformabench-0001" / run_id
                self.assertTrue((archive_dir / "chemqa_review_protocol.yaml").is_file())
                self.assertTrue((archive_dir / "qa_result.json").is_file())
        finally:
            benchmark_test.run_subprocess = original_run_subprocess
            benchmark_test.ensure_runtime_bundle = original_ensure_runtime_bundle
            benchmark_test.invoke_cleanroom_cleanup = original_invoke_cleanroom_cleanup
            benchmark_test.ChemQARunner._wait_for_terminal_status = original_wait_for_terminal_status
            benchmark_test.ChemQARunner._collect_artifacts_from_source = original_collect_artifacts

    def test_chemqa_runner_reconciled_rejected_run_does_not_expose_blob_as_short_answer(self) -> None:
        original_run_subprocess = benchmark_test.run_subprocess
        original_ensure_runtime_bundle = benchmark_test.ensure_runtime_bundle
        original_wait_for_terminal_status = benchmark_test.ChemQARunner._wait_for_terminal_status
        original_collect_artifacts = benchmark_test.ChemQARunner._collect_artifacts_from_source
        original_invoke_cleanroom_cleanup = benchmark_test.invoke_cleanroom_cleanup
        try:
            benchmark_test.run_subprocess = lambda command, *, env=None, cwd=None, timeout=None: benchmark_test.subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"run_id": "demo", "launch_mode": "run", "launched": {"returncode": 0}}),
                stderr="",
            )
            benchmark_test.ensure_runtime_bundle = lambda record, bundle_root: None
            benchmark_test.invoke_cleanroom_cleanup = lambda manifest_path: {"status": "cleaned", "manifest_path": str(manifest_path)}
            benchmark_test.ChemQARunner._wait_for_terminal_status = lambda self, run_id, timeout_seconds: {
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

            benchmark_test.ChemQARunner._collect_artifacts_from_source = fake_collect_artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                output_root = Path(tmpdir) / "benchmark-output"
                launch_root = output_root / "chemqa-launch"
                chemqa_root = Path(tmpdir) / "chemqa-root"
                runner = benchmark_test.ChemQARunner(
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
                record = benchmark_test.BenchmarkRecord(
                    record_id="conformabench-0001",
                    dataset="conformabench",
                    source_file="/tmp/demo.jsonl",
                    eval_kind="conformabench_constructive",
                    prompt="Design a molecule.",
                    reference_answer="Points: 1.0, Item: ok",
                    payload={},
                )
                run_id = "benchmark-chemqa_web_on-conformabench-0001-20260424-000000"
                protocol_dir = chemqa_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id
                protocol_dir.mkdir(parents=True, exist_ok=True)
                (protocol_dir / "chemqa_review_protocol.yaml").write_text(
                    "question: Demo\nacceptance_status: rejected\nterminal_state: completed\nfinal_answer: \"\"\n",
                    encoding="utf-8",
                )
                runner._now_stamp = lambda: "20260424-000000"

                out = runner.run(record, benchmark_test.EXPERIMENT_GROUPS["chemqa_web_on"])

                self.assertEqual(benchmark_test.RunStatus.COMPLETED, out.status)
                self.assertEqual("", out.short_answer_text)
                self.assertIn("No candidate submission achieved acceptance.", out.full_response_text)
        finally:
            benchmark_test.run_subprocess = original_run_subprocess
            benchmark_test.ensure_runtime_bundle = original_ensure_runtime_bundle
            benchmark_test.invoke_cleanroom_cleanup = original_invoke_cleanroom_cleanup
            benchmark_test.ChemQARunner._wait_for_terminal_status = original_wait_for_terminal_status
            benchmark_test.ChemQARunner._collect_artifacts_from_source = original_collect_artifacts

    def test_chemqa_runner_archives_repeated_runs_into_distinct_run_id_directories(self) -> None:
        original_run_subprocess = benchmark_test.run_subprocess
        original_ensure_runtime_bundle = benchmark_test.ensure_runtime_bundle
        original_wait_for_terminal_status = benchmark_test.ChemQARunner._wait_for_terminal_status
        original_ensure_artifacts = benchmark_test.ChemQARunner._ensure_artifacts
        original_invoke_cleanroom_cleanup = benchmark_test.invoke_cleanroom_cleanup
        try:
            benchmark_test.run_subprocess = lambda command, *, env=None, cwd=None, timeout=None: benchmark_test.subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"run_id": "demo", "launch_mode": "run", "launched": {"returncode": 0}}),
                stderr="",
            )
            benchmark_test.ensure_runtime_bundle = lambda record, bundle_root: None
            benchmark_test.invoke_cleanroom_cleanup = lambda manifest_path: {"status": "cleaned", "manifest_path": str(manifest_path)}
            benchmark_test.ChemQARunner._wait_for_terminal_status = lambda self, run_id, timeout_seconds: {
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

            benchmark_test.ChemQARunner._ensure_artifacts = fake_ensure_artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                output_root = Path(tmpdir) / "benchmark-output"
                launch_root = output_root / "chemqa-launch"
                runner = benchmark_test.ChemQARunner(
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
                record = benchmark_test.BenchmarkRecord(
                    record_id="conformabench-0001",
                    dataset="conformabench",
                    source_file="/tmp/demo.jsonl",
                    eval_kind="conformabench_constructive",
                    prompt="Design a molecule.",
                    reference_answer="Points: 1.0, Item: ok",
                    payload={},
                )
                stamps = iter(["20260424-000001", "20260424-000002"])
                runner._now_stamp = lambda: next(stamps)

                out1 = runner.run(record, benchmark_test.EXPERIMENT_GROUPS["chemqa_web_on"])
                out2 = runner.run(record, benchmark_test.EXPERIMENT_GROUPS["chemqa_web_on"])

                self.assertNotEqual(out1.runner_meta["run_id"], out2.runner_meta["run_id"])
                archive1 = Path(out1.runner_meta["archive_dir"])
                archive2 = Path(out2.runner_meta["archive_dir"])
                self.assertNotEqual(archive1, archive2)
                self.assertTrue((archive1 / "qa_result.json").is_file())
                self.assertTrue((archive2 / "qa_result.json").is_file())
        finally:
            benchmark_test.run_subprocess = original_run_subprocess
            benchmark_test.ensure_runtime_bundle = original_ensure_runtime_bundle
            benchmark_test.invoke_cleanroom_cleanup = original_invoke_cleanroom_cleanup
            benchmark_test.ChemQARunner._wait_for_terminal_status = original_wait_for_terminal_status
            benchmark_test.ChemQARunner._ensure_artifacts = original_ensure_artifacts

    def test_run_group_continues_after_record_failure(self) -> None:
        records = [
            benchmark_test.BenchmarkRecord(
                record_id="r1",
                dataset="chembench",
                source_file="/tmp/demo.jsonl",
                eval_kind="chembench_open_ended",
                prompt="What is 2+2?",
                reference_answer="4",
                payload={"target": "4"},
            ),
            benchmark_test.BenchmarkRecord(
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
                if getattr(record, "record_id") == "r1":
                    raise RuntimeError("boom")
                return benchmark_test.RunOutput(
                    short_answer_text="5",
                    full_response_text="Reasoning\nFINAL ANSWER: 5",
                    raw={},
                    runner_meta={},
                )

        original_runner = benchmark_test.SingleLLMRunner
        benchmark_test.SingleLLMRunner = StubSingleRunner
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                results = benchmark_test.run_group(
                    group=benchmark_test.EXPERIMENT_GROUPS["single_llm_web_off"],
                    records=records,
                    output_root=Path(tmpdir),
                    single_timeout=10,
                    chemqa_timeout=10,
                    judge=JudgeStub({}),
                    config_path=Path(tmpdir) / "cfg.json",
                    single_agent="benchmark-single-web-off",
                    chemqa_root=Path(tmpdir),
                    chemqa_model_profile="unused",
                    review_rounds=None,
                    rebuttal_rounds=None,
                )
                self.assertEqual(2, len(results))
                self.assertIsNotNone(results[0].error)
                self.assertFalse(results[0].evaluation["passed"])
                self.assertTrue(results[1].evaluation["passed"])
                self.assertTrue((Path(tmpdir) / "per-record" / "single_llm_web_off" / "r1.json").exists())
                self.assertTrue((Path(tmpdir) / "per-record" / "single_llm_web_off" / "r2.json").exists())
        finally:
            benchmark_test.SingleLLMRunner = original_runner

    def test_run_group_marks_unscored_recovery_as_execution_error(self) -> None:
        record = benchmark_test.BenchmarkRecord(
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
        original_build_runner = getattr(benchmark_test, "build_runner", None)
        original_evaluate_answer = benchmark_test.evaluate_answer
        try:
            benchmark_test.build_runner = lambda **kwargs: stub_runner

            def fail_evaluate_answer(*args, **kwargs):
                raise AssertionError("evaluate_answer should not be called for unscored recovery")

            benchmark_test.evaluate_answer = fail_evaluate_answer
            with tempfile.TemporaryDirectory() as tmpdir:
                results = benchmark_test.run_group(
                    group=benchmark_test.EXPERIMENT_GROUPS["chemqa_web_off"],
                    records=[record],
                    output_root=Path(tmpdir),
                    single_timeout=10,
                    chemqa_timeout=10,
                    judge=object(),
                    config_path=Path(tmpdir) / "cfg.json",
                    single_agent="benchmark-single-web-off",
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
            if original_build_runner is None:
                delattr(benchmark_test, "build_runner")
            else:
                benchmark_test.build_runner = original_build_runner
            benchmark_test.evaluate_answer = original_evaluate_answer

    def test_run_group_scores_evaluable_recovery(self) -> None:
        record = benchmark_test.BenchmarkRecord(
            record_id="recovered-record",
            dataset="chembench",
            source_file="/tmp/demo.jsonl",
            eval_kind="chembench_open_ended",
            prompt="Q",
            reference_answer="fallback-answer",
            payload={},
        )
        recovered_result = benchmark_test.RunnerResult(
            status=benchmark_test.RunStatus.RECOVERED,
            answer=benchmark_test.AnswerPayload(
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
            recovery=benchmark_test.RecoveryInfo(
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
            def run(self, record: object, group: object) -> benchmark_test.RunnerResult:
                return recovered_result

        original_build_runner = getattr(benchmark_test, "build_runner", None)
        original_evaluate_answer = benchmark_test.evaluate_answer
        judge = object()
        try:
            benchmark_test.build_runner = lambda **kwargs: StubRunner()

            def fake_evaluate_answer(
                actual_record: object,
                *,
                short_answer_text: str,
                full_response_text: str,
                judge: object,
            ) -> benchmark_test.EvaluationResult:
                self.assertIs(record, actual_record)
                self.assertEqual("fallback-answer", short_answer_text)
                self.assertEqual("FINAL ANSWER: fallback-answer", full_response_text)
                self.assertIs(judge, judge_obj)
                return benchmark_test.EvaluationResult(
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
            benchmark_test.evaluate_answer = fake_evaluate_answer
            with tempfile.TemporaryDirectory() as tmpdir:
                results = benchmark_test.run_group(
                    group=benchmark_test.EXPERIMENT_GROUPS["chemqa_web_off"],
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
            if original_build_runner is None:
                delattr(benchmark_test, "build_runner")
            else:
                benchmark_test.build_runner = original_build_runner
            benchmark_test.evaluate_answer = original_evaluate_answer

    def test_run_group_accepts_structural_result_object_for_unscored_recovery(self) -> None:
        record = benchmark_test.BenchmarkRecord(
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
        original_build_runner = getattr(benchmark_test, "build_runner", None)
        original_evaluate_answer = benchmark_test.evaluate_answer
        try:
            benchmark_test.build_runner = lambda **kwargs: stub_runner

            def fail_evaluate_answer(*args, **kwargs):
                raise AssertionError("evaluate_answer should not be called for unscored recovery")

            benchmark_test.evaluate_answer = fail_evaluate_answer
            with tempfile.TemporaryDirectory() as tmpdir:
                results = benchmark_test.run_group(
                    group=benchmark_test.EXPERIMENT_GROUPS["chemqa_web_off"],
                    records=[record],
                    output_root=Path(tmpdir),
                    single_timeout=10,
                    chemqa_timeout=10,
                    judge=object(),
                    config_path=Path(tmpdir) / "cfg.json",
                    single_agent="benchmark-single-web-off",
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
            if original_build_runner is None:
                delattr(benchmark_test, "build_runner")
            else:
                benchmark_test.build_runner = original_build_runner
            benchmark_test.evaluate_answer = original_evaluate_answer

    def test_run_group_structural_unscored_recovery_without_failure_attr_uses_runner_meta_error(self) -> None:
        record = benchmark_test.BenchmarkRecord(
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
        original_build_runner = getattr(benchmark_test, "build_runner", None)
        original_evaluate_answer = benchmark_test.evaluate_answer
        try:
            benchmark_test.build_runner = lambda **kwargs: stub_runner

            def fail_evaluate_answer(*args, **kwargs):
                raise AssertionError("evaluate_answer should not be called for unscored recovery")

            benchmark_test.evaluate_answer = fail_evaluate_answer
            with tempfile.TemporaryDirectory() as tmpdir:
                results = benchmark_test.run_group(
                    group=benchmark_test.EXPERIMENT_GROUPS["chemqa_web_off"],
                    records=[record],
                    output_root=Path(tmpdir),
                    single_timeout=10,
                    chemqa_timeout=10,
                    judge=object(),
                    config_path=Path(tmpdir) / "cfg.json",
                    single_agent="benchmark-single-web-off",
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
            if original_build_runner is None:
                delattr(benchmark_test, "build_runner")
            else:
                benchmark_test.build_runner = original_build_runner
            benchmark_test.evaluate_answer = original_evaluate_answer

    def test_materialize_group_failure_results_writes_error_entries(self) -> None:
        records = [
            benchmark_test.BenchmarkRecord(
                record_id="r1",
                dataset="chembench",
                source_file="/tmp/demo.jsonl",
                eval_kind="chembench_open_ended",
                prompt="Q1",
                reference_answer="A",
                payload={},
            ),
            benchmark_test.BenchmarkRecord(
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
            results = benchmark_test.materialize_group_failure_results(
                group=benchmark_test.EXPERIMENT_GROUPS["chemqa_web_off"],
                records=records,
                output_root=Path(tmpdir),
                error_message="group crashed",
            )
            self.assertEqual(2, len(results))
            self.assertTrue(all(item.error == "group crashed" for item in results))
            self.assertTrue((Path(tmpdir) / "per-record" / "chemqa_web_off" / "r1.json").exists())
            self.assertTrue((Path(tmpdir) / "per-record" / "chemqa_web_off" / "r2.json").exists())

    def test_benchmark_test_build_error_group_record_result_preserves_legacy_compatibility(self) -> None:
        record = benchmark_test.BenchmarkRecord(
            record_id="demo-record",
            dataset="frontierscience",
            source_file="/tmp/frontier.jsonl",
            prompt="Question?",
            grading=benchmark_test.GradingSpec(
                kind="frontierscience_research",
                reference_answer="42",
                subset="frontierscience_Research",
                config={"track": "research"},
            ),
            raw_payload={"track": "research"},
        )
        entry = benchmark_test.build_error_group_record_result(
            group=benchmark_test.EXPERIMENT_GROUPS["single_llm_web_off"],
            record=record,
            error_message="boom",
            full_response_text="Reasoning\nFinal conclusion",
        )
        self.assertEqual("frontierscience_Research", entry.subset)
        self.assertEqual("Final conclusion", entry.short_answer_text)
        self.assertEqual("Reasoning\nFinal conclusion", entry.full_response_text)
        self.assertEqual("Reasoning\nFinal conclusion", entry.answer_text)

    def test_shared_reporting_build_error_group_record_result_requires_explicit_dependencies(self) -> None:
        record = benchmark_test.BenchmarkRecord(
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
                group=benchmark_test.EXPERIMENT_GROUPS["single_llm_web_off"],
                record=record,
                error_message="group crashed",
            )


if __name__ == "__main__":
    unittest.main()
