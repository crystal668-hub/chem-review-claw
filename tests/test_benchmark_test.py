from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmark_test.py"
SPEC = importlib.util.spec_from_file_location("benchmark_test", MODULE_PATH)
benchmark_test = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = benchmark_test
SPEC.loader.exec_module(benchmark_test)


class JudgeStub:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def evaluate_json(self, prompt: str) -> dict[str, object]:
        self.prompts.append(prompt)
        return dict(self.payload)


class BenchmarkTestModuleTests(unittest.TestCase):
    def test_extract_final_answer_line_prefers_explicit_marker(self) -> None:
        text = "reasoning\nFINAL ANSWER: 42\n"
        self.assertEqual("42", benchmark_test.extract_final_answer_line(text))
        self.assertEqual("42", benchmark_test.extract_candidate_short_answer(text))

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

    def test_build_run_scoped_config_payload_uses_explicit_single_and_judge_models(self) -> None:
        base = {
            "agents": {"list": []},
            "tools": {"web": {"search": {"enabled": False}}},
            "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
        }
        group = benchmark_test.EXPERIMENT_GROUPS["single_llm_web_on"]
        payload = benchmark_test.build_run_scoped_config_payload(
            base,
            group=group,
            single_agent_model="qwen3.5-plus",
            judge_model="su8/gpt-5.4",
        )
        agents = {entry["id"]: entry for entry in payload["agents"]["list"]}
        self.assertEqual("qwen3.5-plus", agents["benchmark-single-web-on"]["model"])
        self.assertEqual("su8/gpt-5.4", agents["benchmark-judge"]["model"])

    def test_build_run_scoped_config_payload_chemqa_uses_judge_for_coordinator_only(self) -> None:
        base = {
            "agents": {"list": []},
            "tools": {"web": {"search": {"enabled": False}}},
            "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
        }
        group = benchmark_test.EXPERIMENT_GROUPS["chemqa_web_off"]
        payload = benchmark_test.build_run_scoped_config_payload(
            base,
            group=group,
            single_agent_model="qwen3.5-plus",
            judge_model="su8/gpt-5.4",
        )
        agents = {entry["id"]: entry for entry in payload["agents"]["list"]}
        self.assertEqual("su8/gpt-5.4", agents["debateB-coordinator"]["model"])
        for slot in ["debateB-1", "debateB-2", "debateB-3", "debateB-4", "debateB-5"]:
            self.assertEqual("qwen3.5-plus", agents[slot]["model"])

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
        result = benchmark_test.evaluate_chembench_open_ended(record, "FINAL ANSWER: 4")
        self.assertTrue(result.passed)
        self.assertEqual(0.0, result.score)
        self.assertEqual(1.0, result.normalized_score)
        self.assertEqual(0.0, result.details["mae"])

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
        text_record = benchmark_test.BenchmarkRecord(
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
        self.assertEqual("superchem_text_only", benchmark_test.classify_subset(text_record))
        self.assertEqual("superchem_multimodal", benchmark_test.classify_subset(multimodal_record))

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

    def test_sample_records_per_subset_pairs_superchem_by_source_uuid(self) -> None:
        records = [
            benchmark_test.BenchmarkRecord(
                record_id="s1-txt",
                dataset="superchem",
                source_file="/tmp/superchem.jsonl",
                eval_kind="superchem_multiple_choice_rpf",
                prompt="Q1",
                reference_answer="A",
                payload={"modality": "text_only", "source_uuid": "uuid-1"},
            ),
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
                record_id="s2-txt",
                dataset="superchem",
                source_file="/tmp/superchem.jsonl",
                eval_kind="superchem_multiple_choice_rpf",
                prompt="Q2",
                reference_answer="B",
                payload={"modality": "text_only", "source_uuid": "uuid-2"},
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
        self.assertEqual(2, len(sampled))
        self.assertEqual(
            {"superchem_text_only", "superchem_multimodal"},
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
        result = benchmark_test.evaluate_superchem_multiple_choice_rpf(record, "Reasoning\nFINAL ANSWER: B", judge=judge)
        self.assertTrue(result.passed)
        self.assertEqual(1.0, result.score)
        self.assertAlmostEqual(2.0 / 3.0, result.details["rpf"])
        self.assertEqual("B", result.details["parsed_prediction"])
        self.assertEqual(2, len(result.details["checkpoint_matches"]))
        self.assertEqual(1, len(judge.prompts))

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


if __name__ == "__main__":
    unittest.main()
