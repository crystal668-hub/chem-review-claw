from __future__ import annotations

import unittest

from benchmarking.datasets import BenchmarkRecord
from benchmarking.prompts import build_chemqa_goal, build_single_llm_prompt, resolve_chemqa_answer_kind


class BenchmarkPromptsTests(unittest.TestCase):
    def test_conformabench_prompts_require_smiles_final_answer(self) -> None:
        record = BenchmarkRecord(
            record_id="cb-1",
            dataset="conformabench",
            source_file="/tmp/conformabench.jsonl",
            eval_kind="conformabench_constructive",
            prompt="Design a molecule.",
            reference_answer="Points: 1.0, Item: ok",
            payload={},
        )

        self.assertEqual("structure_answer", resolve_chemqa_answer_kind(record))
        self.assertIn("FINAL ANSWER: <SMILES>", build_single_llm_prompt(record, websearch_enabled=False))
        self.assertIn("FINAL ANSWER: <SMILES>", build_chemqa_goal(record, websearch_enabled=True))

    def test_frontierscience_olympiad_uses_numeric_answer_kind(self) -> None:
        record = BenchmarkRecord(
            record_id="fs-1",
            dataset="frontierscience",
            source_file="/tmp/frontierscience.jsonl",
            eval_kind="frontierscience_olympiad",
            prompt="Calculate the pH.",
            reference_answer="4.7",
            payload={"track": "olympiad"},
        )

        self.assertEqual("numeric_short_answer", resolve_chemqa_answer_kind(record))
