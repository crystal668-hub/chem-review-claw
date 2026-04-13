from __future__ import annotations

import importlib.util
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmark_test.py"
SPEC = importlib.util.spec_from_file_location("benchmark_test", MODULE_PATH)
benchmark_test = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = benchmark_test
SPEC.loader.exec_module(benchmark_test)


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


if __name__ == "__main__":
    unittest.main()
