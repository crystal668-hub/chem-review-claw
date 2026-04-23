from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from benchmarking.datasets import BenchmarkRecord, GradingSpec, load_records
from benchmarking.evaluation import EVALUATORS, EvaluationRegistryError, evaluate_record, register_evaluator


class BenchmarkDatasetsTests(unittest.TestCase):
    def test_load_records_builds_chembench_grading_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chembench" / "data" / "sample.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "id": "chem-1",
                        "prompt": "Q",
                        "target": "42",
                        "eval_kind": "chembench_open_ended",
                        "relative_tolerance": 0.1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            record = load_records([path])[0]

            self.assertEqual("chembench_open_ended", record.grading.kind)
            self.assertEqual("42", record.grading.reference_answer)
            self.assertEqual(0.1, record.grading.config["relative_tolerance"])

    def test_load_records_missing_prompt_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chembench" / "data" / "sample.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "id": "chem-1",
                        "target": "42",
                        "eval_kind": "chembench_open_ended",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Missing prompt/problem field"):
                load_records([path])

    def test_load_records_missing_answer_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chembench" / "data" / "sample.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "id": "chem-1",
                        "prompt": "Q",
                        "eval_kind": "chembench_open_ended",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Missing answer/target field"):
                load_records([path])

    def test_evaluate_record_uses_registry_dispatch(self) -> None:
        calls: list[tuple[str, str, object]] = []

        def evaluator(record: BenchmarkRecord, *, short_answer_text: str, full_response_text: str, judge: object) -> dict[str, object]:
            calls.append((record.record_id, short_answer_text, judge))
            return {"ok": True, "full_response_text": full_response_text}

        saved = dict(EVALUATORS)
        try:
            register_evaluator("unit_test_eval_kind", evaluator)
            record = BenchmarkRecord(
                record_id="chem-1",
                dataset="chembench",
                source_file="/tmp/sample.jsonl",
                prompt="Q",
                grading=GradingSpec(
                    kind="unit_test_eval_kind",
                    reference_answer="42",
                    subset="chembench",
                    config={},
                ),
                raw_payload={"id": "chem-1"},
            )

            result = evaluate_record(
                record,
                short_answer_text="42",
                full_response_text="FINAL ANSWER: 42",
                judge=object(),
            )

            self.assertEqual({"ok": True, "full_response_text": "FINAL ANSWER: 42"}, result)
            self.assertEqual(1, len(calls))
            self.assertEqual("chem-1", calls[0][0])
            self.assertEqual("42", calls[0][1])
            self.assertIn("unit_test_eval_kind", EVALUATORS)
        finally:
            EVALUATORS.clear()
            EVALUATORS.update(saved)

    def test_benchmark_record_keeps_compatibility_properties(self) -> None:
        payload = {"id": "chem-1", "target": "42"}
        record = BenchmarkRecord(
            record_id="chem-1",
            dataset="chembench",
            source_file="/tmp/sample.jsonl",
            prompt="Q",
            grading=GradingSpec(
                kind="chembench_open_ended",
                reference_answer="42",
                subset="chembench",
                config={"relative_tolerance": 0.1},
            ),
            raw_payload=payload,
        )

        self.assertEqual("chembench_open_ended", record.eval_kind)
        self.assertEqual("42", record.reference_answer)
        self.assertEqual(payload, record.payload)

    def test_benchmark_record_asdict_preserves_legacy_shape(self) -> None:
        payload = {"id": "chem-1", "target": "42", "options": {"A": "x"}}
        record = BenchmarkRecord(
            record_id="chem-1",
            dataset="chembench",
            source_file="/tmp/sample.jsonl",
            prompt="Q",
            eval_kind="chembench_open_ended",
            reference_answer="42",
            payload=payload,
        )

        serialized = asdict(record)

        self.assertEqual("chembench_open_ended", serialized["eval_kind"])
        self.assertEqual("42", serialized["reference_answer"])
        self.assertEqual(payload, serialized["payload"])
        self.assertEqual(
            {"record_id", "dataset", "source_file", "eval_kind", "prompt", "reference_answer", "payload"},
            set(serialized),
        )

    def test_benchmark_record_payload_and_grading_config_are_deep_copied(self) -> None:
        payload = {"id": "chem-1", "target": "42", "options": {"A": "x"}}
        record = BenchmarkRecord(
            record_id="chem-1",
            dataset="chembench",
            source_file="/tmp/sample.jsonl",
            prompt="Q",
            eval_kind="chembench_open_ended",
            reference_answer="42",
            payload=payload,
        )

        record.payload["options"]["A"] = "mutated payload"
        self.assertEqual("x", record.grading.config["options"]["A"])

        record.grading.config["options"]["A"] = "mutated config"
        self.assertEqual("mutated payload", record.payload["options"]["A"])
        self.assertEqual("x", payload["options"]["A"])

    def test_evaluate_record_unknown_kind_without_generic_fallback_raises_registry_error(self) -> None:
        record = BenchmarkRecord(
            record_id="chem-1",
            dataset="chembench",
            source_file="/tmp/sample.jsonl",
            prompt="Q",
            grading=GradingSpec(
                kind="missing_eval_kind",
                reference_answer="42",
                subset="chembench",
                config={},
            ),
            raw_payload={"id": "chem-1"},
        )
        saved = dict(EVALUATORS)
        try:
            EVALUATORS.clear()
            with self.assertRaises(EvaluationRegistryError):
                evaluate_record(
                    record,
                    short_answer_text="42",
                    full_response_text="FINAL ANSWER: 42",
                    judge=object(),
                )
        finally:
            EVALUATORS.clear()
            EVALUATORS.update(saved)


if __name__ == "__main__":
    unittest.main()
