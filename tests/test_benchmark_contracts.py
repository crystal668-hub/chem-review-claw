import unittest

from benchmarking.contracts import (
    AnswerPayload,
    FailureInfo,
    RecoveryInfo,
    RunStatus,
    RunnerResult,
)
from benchmarking.experiments import ExperimentSpec


class BenchmarkContractsTests(unittest.TestCase):
    def test_answer_payload_defaults_to_empty_strings(self) -> None:
        payload = AnswerPayload()

        self.assertEqual("", payload.short_answer_text)
        self.assertEqual("", payload.full_response_text)

    def test_runner_result_only_scores_completed_or_scored_recovery(self) -> None:
        completed = RunnerResult(
            status=RunStatus.COMPLETED,
            answer=AnswerPayload(short_answer_text="42", full_response_text="FINAL ANSWER: 42"),
            raw={},
            runner_meta={},
        )
        recovered = RunnerResult(
            status=RunStatus.RECOVERED,
            answer=AnswerPayload(short_answer_text="42", full_response_text="FINAL ANSWER: 42"),
            raw={},
            runner_meta={},
            recovery=RecoveryInfo(source="proposer-1-proposal", scored=False, details={}),
        )
        scored_recovery = RunnerResult(
            status=RunStatus.RECOVERED,
            answer=AnswerPayload(short_answer_text="42", full_response_text="FINAL ANSWER: 42"),
            raw={},
            runner_meta={},
            recovery=RecoveryInfo(source="proposer-1-proposal", scored=True, details={}),
        )
        recovery_without_info = RunnerResult(
            status=RunStatus.RECOVERED,
            answer=AnswerPayload(short_answer_text="42", full_response_text="FINAL ANSWER: 42"),
            raw={},
            runner_meta={},
        )
        failed = RunnerResult(
            status=RunStatus.FAILED,
            answer=AnswerPayload(),
            raw={},
            runner_meta={},
            failure=FailureInfo(code="terminal_failure", message="review stalled", details={}),
        )

        self.assertTrue(completed.should_score())
        self.assertFalse(recovered.should_score())
        self.assertTrue(scored_recovery.should_score())
        self.assertFalse(recovery_without_info.should_score())
        self.assertFalse(failed.should_score())

    def test_experiment_spec_resolves_single_agent_override_explicitly(self) -> None:
        spec = ExperimentSpec(
            id="single_llm_web_off",
            label="Single LLM without web",
            runner_kind="single_llm",
            websearch_enabled=False,
            single_agent_id="benchmark-single-web-off",
        )

        self.assertEqual("benchmark-single-web-off", spec.resolve_single_agent_id(None))
        self.assertEqual("custom-single-agent", spec.resolve_single_agent_id("custom-single-agent"))
        self.assertEqual("custom-single-agent", spec.resolve_single_agent_id("  custom-single-agent  "))
        self.assertEqual("benchmark-single-web-off", spec.resolve_single_agent_id("   "))

    def test_public_package_exports_contract_types(self) -> None:
        import benchmarking

        self.assertIs(AnswerPayload, benchmarking.AnswerPayload)
        self.assertIs(ExperimentSpec, benchmarking.ExperimentSpec)
        self.assertIs(FailureInfo, benchmarking.FailureInfo)
        self.assertIs(RecoveryInfo, benchmarking.RecoveryInfo)
        self.assertIs(RunStatus, benchmarking.RunStatus)
        self.assertIs(RunnerResult, benchmarking.RunnerResult)


if __name__ == "__main__":
    unittest.main()
