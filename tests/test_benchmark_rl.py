from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmark_rl.py"
SPEC = importlib.util.spec_from_file_location("benchmark_rl", MODULE_PATH)
benchmark_rl = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = benchmark_rl
SPEC.loader.exec_module(benchmark_rl)

MATERIALIZE_MODULE_PATH = Path(__file__).resolve().parents[2] / "skills" / "debateclaw-v1" / "scripts" / "materialize_runplan.py"
MATERIALIZE_SCRIPT_DIR = MATERIALIZE_MODULE_PATH.parent
if str(MATERIALIZE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(MATERIALIZE_SCRIPT_DIR))
MATERIALIZE_SPEC = importlib.util.spec_from_file_location("debateclaw_materialize_runplan", MATERIALIZE_MODULE_PATH)
materialize_runplan = importlib.util.module_from_spec(MATERIALIZE_SPEC)
assert MATERIALIZE_SPEC and MATERIALIZE_SPEC.loader
sys.modules[MATERIALIZE_SPEC.name] = materialize_runplan
MATERIALIZE_SPEC.loader.exec_module(materialize_runplan)


class BenchmarkRLModuleTests(unittest.TestCase):
    def make_record(self, record_id: str = "demo-record") -> benchmark_rl.BenchmarkRecord:
        return benchmark_rl.BenchmarkRecord(
            record_id=record_id,
            dataset="chembench",
            source_file="/tmp/demo.jsonl",
            eval_kind="chembench_open_ended",
            prompt="Question?",
            reference_answer="42",
            payload={"id": record_id},
        )

    def make_result(self, record: benchmark_rl.BenchmarkRecord, *, error: str | None = None) -> benchmark_rl.GroupRecordResult:
        return benchmark_rl.GroupRecordResult(
            group_id="review_loop_web_off",
            group_label="label",
            runner="review_loop",
            websearch=False,
            record_id=record.record_id,
            subset="chembench",
            dataset=record.dataset,
            source_file=record.source_file,
            eval_kind=record.eval_kind,
            prompt=record.prompt,
            reference_answer=record.reference_answer,
            answer_text="FINAL ANSWER: 42",
            evaluation={
                "eval_kind": record.eval_kind,
                "score": 1.0,
                "max_score": 1.0,
                "normalized_score": 1.0,
                "passed": error is None,
                "primary_metric": "answer_accuracy",
                "primary_metric_direction": "higher_is_better",
                "details": {},
            },
            runner_meta={},
            raw={},
            elapsed_seconds=1.0,
            error=error,
            short_answer_text="42",
            full_response_text="FINAL ANSWER: 42",
        )

    def make_runtime_context(self, root: Path, record: benchmark_rl.BenchmarkRecord) -> benchmark_rl.RuntimeContext:
        group = benchmark_rl.EXPERIMENT_GROUPS["review_loop_web_off"]
        return benchmark_rl.RuntimeContext(
            output_root=root,
            group=group,
            dataset_files=[root / "bench.jsonl"],
            records=[record],
            benchmark_root=root,
            config_path=root / "runtime-config" / "cfg.json",
            model_profile="review-loop-test",
            args_payload={
                "websearch": "off",
                "review_rounds": 3,
                "rebuttal_rounds": 3,
                "proposer_count": 3,
                "collector_agent": "benchmark-rl-collector",
                "collector_model": "collector-model",
                "judge_agent": "benchmark-judge",
                "judge_model": "judge-model",
            },
            status_path=root / "runtime-status.json",
            partial_results_path=root / "results.partial.json",
            runtime_manifest_path=root / "runtime-manifest.json",
        )

    def test_materialize_run_brief_escapes_literal_braces(self) -> None:
        run_plan = {
            "request_snapshot": {
                "goal": (
                    "QUESTION:\n"
                    "Combusting a \\pu{0.250 g} sample gives \\ce{CO2} and \\ce{H2O}. "
                    "What is x in \\ce{CH4N2Ox}?"
                ),
                "metadata": {"priority": "normal"},
            },
            "runtime_context": {
                "evidence_mode": "strict",
                "final_decider": "outer-entry-agent",
            },
        }
        rendered = materialize_runplan.render_run_brief(run_plan)
        self.assertIn("\\pu{{0.250 g}}", rendered)
        self.assertIn("\\ce{{CO2}}", rendered)
        self.assertIn("\\ce{{CH4N2Ox}}", rendered)

    def test_materialize_run_brief_escapes_math_braces(self) -> None:
        run_plan = {
            "request_snapshot": {
                "goal": (
                    "Given \\( v_0 = \\\\frac{k_{cat}[E]_0[S]}{K_M + [S]} \\) "
                    "and \\pu{40.8 cm^{3}}, determine {K_M}."
                ),
                "metadata": {"priority": "normal"},
            },
            "runtime_context": {
                "evidence_mode": "strict",
                "final_decider": "outer-entry-agent",
            },
        }
        rendered = materialize_runplan.render_run_brief(run_plan)
        self.assertIn("k_{{cat}}", rendered)
        self.assertIn("{{K_M}}", rendered)
        self.assertIn("\\pu{{40.8 cm^{{3}}}}", rendered)

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
            records = benchmark_rl.load_records([path])
            self.assertEqual(1, len(records))
            self.assertEqual("Solve me", records[0].prompt)
            self.assertEqual("42", records[0].reference_answer)

    def test_build_review_loop_goal_includes_superchem_bundle_instructions(self) -> None:
        record = benchmark_rl.BenchmarkRecord(
            record_id="superchem-demo",
            dataset="superchem",
            source_file="/tmp/demo.jsonl",
            eval_kind="superchem_multiple_choice_rpf",
            prompt="Question",
            reference_answer="A",
            payload={"options": {"A": "x"}},
        )
        bundle = benchmark_rl.benchmark_test.RuntimeBundle(
            bundle_dir=Path("/tmp/bundle"),
            question_markdown=Path("/tmp/bundle/question.md"),
            image_files=[Path("/tmp/bundle/images/img01.png")],
        )
        goal = benchmark_rl.build_review_loop_goal(record, websearch_enabled=False, input_bundle=bundle)
        self.assertIn("Do not use web search", goal)
        self.assertIn("Read `/tmp/bundle/question.md` first", goal)
        self.assertIn("FINAL ANSWER: <option letters>", goal)

    def test_fallback_collect_answer_prefers_latest_rebuttal_final_answer(self) -> None:
        summary_payload = {
            "final_candidates": ["proposer-1"],
            "proposals": [
                {
                    "proposer": "proposer-1",
                    "body": "Title: Demo\n\nFINAL ANSWER: B",
                }
            ],
            "reviews": [],
            "rebuttals": [
                {
                    "proposer": "proposer-1",
                    "body": "Rebuttal text\nFINAL ANSWER: C",
                }
            ],
        }
        collected = benchmark_rl.fallback_collect_answer(summary_payload)
        assert collected is not None
        self.assertEqual("C", collected["short_answer"])
        self.assertIn("FINAL ANSWER: C", collected["full_response_text"])

    def test_fallback_collect_answer_returns_none_without_candidates(self) -> None:
        self.assertIsNone(benchmark_rl.fallback_collect_answer({"final_candidates": []}))

    def test_proposal_payloads_for_candidates_filters_survivors(self) -> None:
        summary_payload = {
            "final_candidates": ["proposer-2"],
            "proposals": [
                {"proposer": "proposer-1", "body": "A"},
                {"proposer": "proposer-2", "body": "B"},
            ],
            "reviews": [{"target_proposer": "proposer-2", "body": "review"}],
            "rebuttals": [{"proposer": "proposer-2", "body": "rebuttal"}],
            "attack_registry": [{"target_proposer": "proposer-2", "attack_text": "x"}],
        }
        payloads = benchmark_rl.proposal_payloads_for_candidates(summary_payload)
        self.assertEqual(1, len(payloads))
        self.assertEqual("proposer-2", payloads[0]["candidate"])
        self.assertEqual(1, len(payloads[0]["review_history"]))
        self.assertEqual(1, len(payloads[0]["rebuttal_history"]))

    def test_export_csv_reports_writes_summary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = {
                "groups": {
                    "review_loop_web_off": {
                        "runner": "review_loop",
                        "websearch": False,
                        "count": 2,
                        "pass_count": 1,
                        "avg_normalized_score": 0.5,
                        "avg_answer_accuracy": 0.5,
                        "avg_rpf": 0.25,
                    }
                },
                "group_subset": {
                    "review_loop_web_off::chembench": {
                        "group_id": "review_loop_web_off",
                        "runner": "review_loop",
                        "websearch": False,
                        "subset": "chembench",
                        "count": 2,
                        "pass_count": 1,
                        "avg_normalized_score": 0.5,
                        "avg_answer_accuracy": 0.5,
                        "avg_rpf": 0.25,
                    }
                },
            }
            benchmark_rl.export_csv_reports(root, summary, ["review_loop_web_off"])
            self.assertTrue((root / "summary_by_group.csv").is_file())
            self.assertTrue((root / "summary_by_group_and_subset.csv").is_file())

    def test_benchmark_group_id_for_websearch(self) -> None:
        self.assertEqual("review_loop_web_on", benchmark_rl.benchmark_group_id_for_websearch("on"))
        self.assertEqual("review_loop_web_off", benchmark_rl.benchmark_group_id_for_websearch("off"))

    def test_safe_persist_group_record_result_degrades_on_write_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            record = self.make_record()
            entry = self.make_result(record)
            real_save_json = benchmark_rl.save_json

            def flaky_save(path: Path, payload: object) -> None:
                if "per-record" in str(path) and payload == asdict(entry):
                    raise OSError("disk full")
                real_save_json(path, payload)

            with mock.patch.object(benchmark_rl, "save_json", side_effect=flaky_save):
                persisted = benchmark_rl.safe_persist_group_record_result(
                    root,
                    benchmark_rl.EXPERIMENT_GROUPS["review_loop_web_off"],
                    record,
                    entry,
                )

            self.assertIsNotNone(persisted.error)
            self.assertIn("Failed to persist per-record result", persisted.error or "")
            saved = json.loads((root / "per-record" / "review_loop_web_off" / "demo-record.json").read_text(encoding="utf-8"))
            self.assertIn("persist_error", saved["runner_meta"])

    def test_finalize_outputs_reloads_existing_per_record_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            record = self.make_record()
            runtime = self.make_runtime_context(root, record)
            benchmark_rl.save_json(
                root / "per-record" / "review_loop_web_off" / "demo-record.json",
                asdict(self.make_result(record)),
            )

            payload = benchmark_rl.finalize_outputs(runtime, results=None, status="failed", fatal_error="boom")

            self.assertEqual(1, len(payload["results"]))
            self.assertEqual("failed", json.loads(runtime.status_path.read_text(encoding="utf-8"))["status"])
            self.assertTrue((root / "results.json").is_file())
            self.assertTrue((root / "results.partial.json").is_file())

    def test_main_keyboard_interrupt_still_writes_runtime_artifacts(self) -> None:
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

            class DummyConfigPool:
                def __init__(self, **_: object) -> None:
                    pass

                def config_for_group(self, group: object, *, model_profile: str) -> Path:
                    path = output_root / "runtime-config" / "review_loop_web_off-openclaw.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}", encoding="utf-8")
                    return path

                def judge_config_path(self) -> Path:
                    path = output_root / "runtime-config" / "benchmark-judge-openclaw.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}", encoding="utf-8")
                    return path

            argv = [
                "benchmark_rl.py",
                "--benchmark-root",
                str(root),
                "--openclaw-config",
                str(base_config),
                "--exact-output-dir",
                str(output_root),
                "--websearch",
                "off",
            ]

            with mock.patch.object(benchmark_rl, "ReviewLoopConfigPool", DummyConfigPool), \
                mock.patch.object(benchmark_rl, "JudgeClient", return_value=object()), \
                mock.patch.object(benchmark_rl, "run_group", side_effect=KeyboardInterrupt):
                with mock.patch.object(sys, "argv", argv):
                    exit_code = benchmark_rl.main()

            self.assertEqual(130, exit_code)
            self.assertTrue((output_root / "runtime-manifest.json").is_file())
            self.assertTrue((output_root / "runtime-status.json").is_file())
            self.assertTrue((output_root / "results.json").is_file())
            status = json.loads((output_root / "runtime-status.json").read_text(encoding="utf-8"))
            self.assertEqual("interrupted", status["status"])
            results = json.loads((output_root / "results.json").read_text(encoding="utf-8"))
            self.assertFalse(results["completed"])
            self.assertEqual("Interrupted by user", results["fatal_error"])


if __name__ == "__main__":
    unittest.main()
