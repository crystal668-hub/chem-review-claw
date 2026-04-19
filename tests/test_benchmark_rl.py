from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmark_rl.py"
SPEC = importlib.util.spec_from_file_location("benchmark_rl", MODULE_PATH)
benchmark_rl = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = benchmark_rl
SPEC.loader.exec_module(benchmark_rl)


class BenchmarkRLModuleTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
