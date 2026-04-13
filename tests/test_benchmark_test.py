from __future__ import annotations

import importlib.util
import sys
import unittest
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

    def test_aggregate_results_groups_by_experiment(self) -> None:
        sample = [
            benchmark_test.GroupRecordResult(
                group_id="g1",
                group_label="Group 1",
                runner="single_llm",
                websearch=False,
                record_id="r1",
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
