from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SKILL_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import chemqa_artifact_flow as artifact_flow  # noqa: E402
import chemqa_review_artifacts as review_artifacts  # noqa: E402


class AnswerKindResolutionTest(unittest.TestCase):
    def test_resolves_known_benchmark_eval_kinds(self) -> None:
        self.assertEqual(
            "numeric_short_answer",
            artifact_flow.resolve_answer_kind({"eval_kind": "chembench_open_ended", "dataset": "chembench"}),
        )
        self.assertEqual(
            "numeric_short_answer",
            artifact_flow.resolve_answer_kind({"eval_kind": "frontierscience_olympiad", "dataset": "frontierscience"}),
        )
        self.assertEqual(
            "multi_part_research_answer",
            artifact_flow.resolve_answer_kind(
                {"eval_kind": "frontierscience_research", "dataset": "frontierscience", "track": "research"}
            ),
        )
        self.assertEqual(
            "multiple_choice",
            artifact_flow.resolve_answer_kind({"eval_kind": "superchem_multiple_choice_rpf", "dataset": "superchem"}),
        )
        self.assertEqual(
            "structure_answer",
            artifact_flow.resolve_answer_kind({"eval_kind": "conformabench_constructive", "dataset": "conformabench"}),
        )
        self.assertEqual(
            "generic_semantic_answer",
            artifact_flow.resolve_answer_kind({"eval_kind": "custom_eval", "dataset": "custom"}),
        )


class ArtifactFlowValidationTest(unittest.TestCase):
    def test_legacy_candidate_shape_allows_research_narrative_when_answer_kind_is_research(self) -> None:
        candidate_text = """
artifact_kind: candidate_submission
artifact_contract_version: react-reviewed-v2
phase: propose
owner: proposer-1
direct_answer: >-
  The reaction is most consistent with a ligand-assisted oxidative addition pathway.
  The rate trend, isotope control, and reported solvent dependence jointly support that assignment.
summary: Research-style answer with multiple linked claims.
submission_trace:
  - step: analysis
    status: success
    detail: Compared mechanistic evidence.
""".strip()

        checked = review_artifacts.check_candidate_submission(
            candidate_text,
            owner="proposer-1",
            answer_kind="multi_part_research_answer",
        )

        self.assertTrue(checked.ok, checked.errors)

    def test_research_answer_is_not_rejected_for_narrative_length(self) -> None:
        candidate = {
            "artifact_kind": "candidate_submission",
            "owner": "proposer-1",
            "direct_answer": (
                "The reaction is most consistent with a ligand-assisted oxidative addition pathway. "
                "The rate trend, isotope control, and reported solvent dependence jointly support that assignment."
            ),
            "summary": "Research-style answer with multiple linked claims.",
            "submission_trace": [{"step": "analysis", "status": "success", "detail": "Compared mechanistic evidence."}],
        }

        result = artifact_flow.validate_candidate_artifact(candidate, answer_kind="multi_part_research_answer")

        self.assertTrue(result.valid, result.errors)
        self.assertEqual(candidate["direct_answer"], result.artifact["payload"]["evaluator_answer"])

    def test_numeric_answer_requires_scalar_projection(self) -> None:
        candidate = {
            "artifact_kind": "candidate_submission",
            "owner": "proposer-1",
            "direct_answer": "The answer is not available from the prompt.",
            "summary": "No calculation.",
            "submission_trace": [{"step": "analysis", "status": "partial", "detail": "Could not calculate."}],
        }

        result = artifact_flow.validate_candidate_artifact(candidate, answer_kind="numeric_short_answer")

        self.assertFalse(result.valid)
        self.assertIn("numeric_short_answer requires a numeric evaluator_answer", result.errors)

    def test_answer_revision_rebuttal_updates_current_candidate_view(self) -> None:
        candidate = artifact_flow.validate_candidate_artifact(
            {
                "artifact_kind": "candidate_submission",
                "owner": "proposer-1",
                "direct_answer": "7.5",
                "summary": "Initial calculation.",
                "submission_trace": [{"step": "calc", "status": "success", "detail": "Initial rounded result."}],
            },
            answer_kind="numeric_short_answer",
        )
        review = artifact_flow.validate_review_artifact(
            {
                "artifact_kind": "formal_review",
                "phase": "review",
                "reviewer_lane": "proposer-2",
                "target_owner": "proposer-1",
                "target_kind": "candidate_submission",
                "verdict": "blocking",
                "counts_for_acceptance": True,
                "synthetic": False,
                "review_items": [
                    {
                        "item_id": "rounding",
                        "severity": "high",
                        "finding": "Use two decimals.",
                        "requested_change": "Report 7.59.",
                        "target_field": "evaluator_answer",
                    }
                ],
            }
        )
        rebuttal = artifact_flow.validate_rebuttal_artifact(
            {
                "artifact_kind": "rebuttal",
                "phase": "rebuttal",
                "owner": "proposer-1",
                "mode": "answer_revision",
                "response_summary": "Updated the rounded value.",
                "addressed_review_items": ["1:0:proposer-2:rounding"],
                "updated_answer": {"evaluator_answer": "7.59", "display_answer": "7.59 micrograms"},
            },
            answer_kind="numeric_short_answer",
        )

        state = artifact_flow.build_current_candidate_view(
            candidate_artifact=candidate.artifact,
            review_artifacts=[review.artifact],
            rebuttal_artifacts=[rebuttal.artifact],
        )

        self.assertEqual("7.59", state.candidate_view["payload"]["evaluator_answer"])
        self.assertEqual("7.59 micrograms", state.candidate_view["payload"]["display_answer"])
        self.assertEqual("addressed_by_revision", state.review_items["1:0:proposer-2:rounding"]["status"])

    def test_response_only_rebuttal_does_not_update_answer_fields(self) -> None:
        candidate = artifact_flow.validate_candidate_artifact(
            {
                "artifact_kind": "candidate_submission",
                "owner": "proposer-1",
                "direct_answer": "B",
                "summary": "Initial answer.",
                "submission_trace": [{"step": "choice", "status": "success", "detail": "Selected B."}],
            },
            answer_kind="multiple_choice",
        )
        rebuttal = artifact_flow.validate_rebuttal_artifact(
            {
                "artifact_kind": "rebuttal",
                "phase": "rebuttal",
                "owner": "proposer-1",
                "mode": "response_only",
                "response_summary": "The answer remains B.",
                "updated_answer": {"evaluator_answer": "C", "display_answer": "C"},
            },
            answer_kind="multiple_choice",
        )

        state = artifact_flow.build_current_candidate_view(
            candidate_artifact=candidate.artifact,
            review_artifacts=[],
            rebuttal_artifacts=[rebuttal.artifact],
        )

        self.assertEqual("B", state.candidate_view["payload"]["evaluator_answer"])


class ArtifactFinalizationTest(unittest.TestCase):
    def test_finalize_success_writes_final_artifact_manifest_and_qa_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            candidate = artifact_flow.validate_candidate_artifact(
                {
                    "artifact_kind": "candidate_submission",
                    "owner": "proposer-1",
                    "direct_answer": "7.59",
                    "summary": "Final calculation.",
                    "submission_trace": [{"step": "calc", "status": "success", "detail": "Computed mass."}],
                },
                answer_kind="numeric_short_answer",
                run_id="run-1",
            )
            state = artifact_flow.build_current_candidate_view(
                candidate_artifact=candidate.artifact,
                review_artifacts=[],
                rebuttal_artifacts=[],
            )

            result = artifact_flow.finalize_success(
                run_id="run-1",
                output_dir=output_dir,
                answer_kind="numeric_short_answer",
                candidate_state=state,
                acceptance_status="accepted",
                protocol_payload={"question": "How much product?"},
            )

            self.assertEqual("completed", result.terminal_state)
            for key in ("final_answer_artifact_path", "artifact_manifest_path", "qa_result_path", "candidate_view_path"):
                self.assertTrue(Path(result.status_overlay[key]).is_file(), key)
            qa_result = json.loads(Path(result.status_overlay["qa_result_path"]).read_text(encoding="utf-8"))
            self.assertEqual("7.59", qa_result["final_answer"]["direct_answer"])
            self.assertEqual("7.59", qa_result["final_answer"]["answer"])
            self.assertEqual("7.59", qa_result["final_answer"]["value"])
            self.assertEqual(result.status_overlay["final_answer_artifact_path"], qa_result["artifact_paths"]["final_answer_artifact"])

    def test_finalize_failure_with_projection_writes_recovery_eligibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = artifact_flow.finalize_failure(
                run_id="run-2",
                output_dir=output_dir,
                failure_code="protocol_stalled",
                failure_message="review phase stalled",
                answer_projection={
                    "answer_kind": "multiple_choice",
                    "evaluator_answer": "B",
                    "display_answer": "B",
                    "full_answer": "Recovered from current candidate view.",
                    "source_candidate_view_id": "candidate-view-run-2",
                },
                recovery_eligibility={
                    "evaluable": True,
                    "scored": True,
                    "reliability": "high_confidence_recovered",
                    "recovery_mode": "failure_artifact_answer_projection",
                    "reason": "last_valid_candidate_view",
                },
            )

            self.assertEqual("failed", result.terminal_state)
            qa_result = json.loads(Path(result.status_overlay["qa_result_path"]).read_text(encoding="utf-8"))
            self.assertNotIn("final_answer", qa_result)
            self.assertEqual("B", qa_result["answer_projection"]["evaluator_answer"])
            self.assertTrue(qa_result["recovery_eligibility"]["evaluable"])


if __name__ == "__main__":
    unittest.main()
