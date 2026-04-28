from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SKILL_ROOT / "scripts"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEBATECLAW_SCRIPTS_DIR = PROJECT_ROOT / "skills" / "debateclaw-v1" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bundle_common  # noqa: E402
import chemqa_review_transport as transport  # noqa: E402
import collect_artifacts as collect_artifacts_module  # noqa: E402
import launch_from_preset as launch_from_preset_module  # noqa: E402
import materialize_runplan  # noqa: E402
import recover_run  # noqa: E402

if str(DEBATECLAW_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(DEBATECLAW_SCRIPTS_DIR))

DEBATE_STATE_PATH = Path.home() / ".clawteam" / "debateclaw" / "bin" / "debate_state.py"
SNAPSHOT_PATH = SCRIPTS_DIR / "chemqa_review_state_snapshot.py"
TEST_PYTHON = str((PROJECT_ROOT / ".venv" / "bin" / "python").expanduser())


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


driver_module = load_module(SCRIPTS_DIR / "chemqa_review_openclaw_driver.py", "chemqa_review_openclaw_driver_test")
debate_wrapper_module = load_module(DEBATECLAW_SCRIPTS_DIR / "openclaw_debate_agent.py", "openclaw_debate_agent_test")


class TransportHelpersTest(unittest.TestCase):
    def test_current_proposal_prefers_current_epoch_and_ignores_prior_epoch_rows(self) -> None:
        status_payload = {
            "epoch": 2,
            "proposals": [
                {"epoch": 1, "proposer": "proposer-1", "status": "active", "body": "old"},
                {"epoch": 1, "proposer": "proposer-2", "status": "active", "body": "other"},
            ],
        }
        self.assertIsNone(transport.current_proposal(status_payload, "proposer-1"))

        status_payload["proposals"].append({"epoch": 2, "proposer": "proposer-1", "status": "active", "body": "new"})
        current = transport.current_proposal(status_payload, "proposer-1")
        assert current is not None
        self.assertEqual(2, current["epoch"])
        self.assertEqual("new", current["body"])

    def test_placeholder_and_transport_review_shapes(self) -> None:
        placeholder = transport.render_placeholder_proposal("proposer-2")
        self.assertIn("artifact_kind: placeholder", placeholder)
        self.assertIn("reviewer_lane: proposer-2", placeholder)
        self.assertIn("target_owner: proposer-1", placeholder)

        review = transport.render_transport_review(reviewer="proposer-4", target="proposer-5")
        self.assertEqual(
            [],
            transport.validate_transport_review_shape(review, reviewer="proposer-4", target="proposer-5"),
        )

    def test_formal_review_validation_and_blocking(self) -> None:
        review = """
artifact_kind: formal_review
phase: review
reviewer_lane: proposer-3
target_owner: proposer-1
target_kind: candidate_submission
verdict: insufficient_evidence
review_items:
- severity: high
  finding: Missing anchor.
counts_for_acceptance: true
synthetic: false
""".strip()
        self.assertEqual(
            [],
            transport.validate_formal_review_shape(review, reviewer="proposer-3", target="proposer-1"),
        )
        self.assertTrue(transport.blocking_flag_for_review(review))

    def test_markdown_bold_metadata_is_accepted(self) -> None:
        review = """
# Formal Review

**artifact_kind:** formal_review
**phase:** review
**reviewer_lane:** proposer-2
**target_owner:** proposer-1
**target_kind:** candidate_submission
**verdict:** blocking

review_items:
- severity: medium
  finding: Needs evidence
counts_for_acceptance: true
synthetic: false
""".strip()
        self.assertEqual(
            [],
            transport.validate_formal_review_shape(review, reviewer="proposer-2", target="proposer-1"),
        )
        self.assertEqual("blocking", transport.parse_review_verdict(review))
        self.assertTrue(transport.blocking_flag_for_review(review))

    def test_candidate_markdown_inline_direct_answer_is_recovered(self) -> None:
        candidate = """
# Candidate Submission — proposer-1

**Direct answer: 6 peaks** in the **1H NMR** spectrum.

## Short justification
The answer follows from a stereogenic center making the adjacent CH2 hydrogens diastereotopic.

## Submission trace
- structure-parse: success via direct molecular reasoning
""".strip()
        checked = transport.check_candidate_submission(candidate, owner="proposer-1")
        self.assertTrue(checked.ok)
        self.assertEqual("6 peaks", checked.payload["direct_answer"])

    def test_candidate_prose_recovers_summary_and_default_trace(self) -> None:
        candidate = """
artifact_kind: candidate_submission
phase: propose
owner: proposer-1
direct_answer: C
reasoning: Product C best matches the mechanism and substituent pattern implied by the prompt.
""".strip()
        checked = transport.check_candidate_submission(candidate, owner="proposer-1")
        self.assertTrue(checked.ok)
        self.assertEqual("C", checked.payload["direct_answer"])
        self.assertIn("Product C best matches", checked.payload["summary"])
        self.assertEqual(1, len(checked.payload["submission_trace"]))

    def test_candidate_submission_rejects_narrative_direct_answer_for_scalar_answer_tasks(self) -> None:
        candidate = """
artifact_kind: candidate_submission
phase: propose
owner: proposer-1
direct_answer: Revised proposal for 2-fluoroethylamine (NCCF) as the target molecule. This epoch addresses the prior review items with a fuller narrative justification.
summary: Revised candidate after considering reviewer feedback.
submission_trace:
- step: structural_reasoning
  status: success
  detail: Re-ran the analysis and rewrote the candidate explanation.
""".strip()
        checked = transport.check_candidate_submission(candidate, owner="proposer-1")
        self.assertFalse(checked.ok)
        self.assertIn("direct_answer", " ".join(checked.errors))

    def test_candidate_submission_rejects_revised_design_direct_answer(self) -> None:
        candidate = """
artifact_kind: candidate_submission
phase: propose
owner: proposer-1
direct_answer: 'Revised design: 2-mercaptobenzonitrile addresses the epoch-1 acceptor-count violation.'
summary: Revised design after reviewer feedback.
submission_trace:
- step: structural_reasoning
  status: success
  detail: Re-ran the structural analysis and rewrote the candidate explanation.
""".strip()
        checked = transport.check_candidate_submission(candidate, owner="proposer-1")
        self.assertFalse(checked.ok)
        self.assertIn("direct_answer", " ".join(checked.errors))

    def test_formal_review_prose_recovers_summary_and_review_items(self) -> None:
        review = """
artifact_kind: formal_review
phase: review
reviewer_lane: proposer-2
target_owner: proposer-1
target_kind: candidate_submission
verdict: blocking

- The answer does not justify why the cited mechanism dominates.
- Add a concrete anchor for the rate-determining step.
counts_for_acceptance: true
synthetic: false
""".strip()
        checked = transport.check_formal_review(review, reviewer="proposer-2", target="proposer-1")
        self.assertTrue(checked.ok)
        self.assertGreaterEqual(len(checked.payload["review_items"]), 2)
        self.assertTrue(checked.payload["summary"])

    def test_formal_review_prose_without_items_recovers_single_item(self) -> None:
        review = """
artifact_kind: formal_review
phase: review
reviewer_lane: proposer-4
target_owner: proposer-1
target_kind: candidate_submission
verdict: insufficient_evidence
The answer overclaims certainty and does not reconcile the contradictory thermodynamic assumptions.
counts_for_acceptance: true
synthetic: false
""".strip()
        checked = transport.check_formal_review(review, reviewer="proposer-4", target="proposer-1")
        self.assertTrue(checked.ok)
        self.assertEqual(1, len(checked.payload["review_items"]))
        self.assertIn("overclaims certainty", checked.payload["summary"])

    def test_rebuttal_recovers_response_summary_from_updated_direct_answer(self) -> None:
        rebuttal = """
artifact_kind: rebuttal
phase: rebuttal
owner: proposer-1
updated_direct_answer: Revised answer after addressing the review comments.
""".strip()
        checked = transport.check_rebuttal(rebuttal, owner="proposer-1")
        self.assertTrue(checked.ok)
        self.assertEqual(
            "Revised answer after addressing the review comments.",
            checked.payload["response_summary"],
        )

    def test_empty_artifacts_still_fail(self) -> None:
        self.assertFalse(transport.check_candidate_submission("", owner="proposer-1").ok)
        self.assertFalse(transport.check_formal_review("", reviewer="proposer-2", target="proposer-1").ok)
        self.assertFalse(transport.check_rebuttal("", owner="proposer-1").ok)


class RecoverRunRespawnTest(unittest.TestCase):
    def test_respawn_actionable_roles_respawns_dead_coordinator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "clawteam-data"
            team = "chemqa-dead-coordinator"
            team_dir = data_dir / "teams" / team
            team_dir.mkdir(parents=True)
            (team_dir / "spawn_registry.json").write_text(
                json.dumps(
                    {
                        "debate-coordinator": {
                            "backend": "subprocess",
                            "pid": 999999,
                            "slot": "debateA-coordinator",
                            "command": ["/bin/echo", "coordinator"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                skill_root=str(SKILL_ROOT),
                team=team,
                runtime_dir=str(DEBATECLAW_SCRIPTS_DIR),
                workspace_root=str(root / "workspaces"),
                max_steps=1,
                max_respawns_per_role_phase_signature=1,
                json=True,
            )
            launched: list[dict[str, object]] = []

            class FakePopen:
                pid = 4242

                def __init__(self, command, **kwargs) -> None:
                    launched.append({"command": list(command), **kwargs})

            with mock.patch.dict(os.environ, {"CLAWTEAM_DATA_DIR": str(data_dir)}), \
                mock.patch.object(recover_run.subprocess, "Popen", FakePopen):
                recoverer = recover_run.RunRecoverer(args)
                recoverer.current_phase_signature = lambda: "phase=review;missing=4"  # type: ignore[method-assign]
                recoverer.next_action = lambda role: {  # type: ignore[method-assign]
                    "action": "wait" if role == "debate-coordinator" else "none",
                    "phase": "review",
                }

                self.assertTrue(recoverer.respawn_actionable_roles())

            self.assertEqual(1, len(launched))
            launch = launched[0]
            self.assertEqual(["/bin/echo", "coordinator"], launch["command"])
            self.assertEqual(
                (root / "workspaces" / "debateA-coordinator").resolve(),
                Path(str(launch["cwd"])).resolve(),
            )
            self.assertEqual(recover_run.subprocess.STDOUT, launch["stderr"])
            self.assertTrue(launch["text"])
            self.assertTrue(launch["start_new_session"])
            registry = json.loads((team_dir / "spawn_registry.json").read_text(encoding="utf-8"))
            self.assertEqual(4242, registry["debate-coordinator"]["pid"])
            self.assertEqual("recover_run_actionable_role", registry["debate-coordinator"]["last_respawn_reason"])


class ProtocolReconstructionTest(unittest.TestCase):
    def make_candidate(self, direct_answer: str = "6") -> str:
        return "\n".join(
            [
                "# Candidate",
                "",
                "## Direct answer",
                direct_answer,
                "",
                "## Justification",
                "Candidate says the answer is six peaks.",
                "",
                "## Submission trace",
                "- parse-smiles: Identified stereogenic center and diastereotopic CH2.",
                "- count-environments: Counted six distinct proton environments.",
            ]
        )

    def make_review(self, reviewer: str, verdict: str, *, summary: str, finding: str) -> str:
        return "\n".join(
            [
                "artifact_kind: formal_review",
                "phase: review",
                f"reviewer_lane: {reviewer}",
                "target_owner: proposer-1",
                "target_kind: candidate_submission",
                f"verdict: {verdict}",
                f"summary: {summary}",
                "review_items:",
                "- severity: medium",
                f"  finding: {finding}",
                "counts_for_acceptance: true",
                "synthetic: false",
            ]
        )

    def build_summary(self, *, proposer2_verdict: str = "non_blocking", include_second_round_for_p2: bool = False, exited_reviewers: list[str] | None = None) -> dict[str, Any]:
        proposals = [
            {
                "proposer": "proposer-1",
                "title": "ChemQA candidate submission",
                "body": self.make_candidate(),
                "artifact": {"archive_path": "/tmp/proposer-1.md"},
            },
            {"proposer": "proposer-2", "title": "Placeholder Proposal — proposer-2", "body": transport.render_placeholder_proposal("proposer-2")},
            {"proposer": "proposer-3", "title": "Placeholder Proposal — proposer-3", "body": transport.render_placeholder_proposal("proposer-3")},
            {"proposer": "proposer-4", "title": "Placeholder Proposal — proposer-4", "body": transport.render_placeholder_proposal("proposer-4")},
            {"proposer": "proposer-5", "title": "Placeholder Proposal — proposer-5", "body": transport.render_placeholder_proposal("proposer-5")},
        ]
        reviews = [
            {
                "reviewer": "proposer-2",
                "target_proposer": "proposer-1",
                "review_round": 1,
                "body": self.make_review(
                    "proposer-2",
                    proposer2_verdict,
                    summary="Search coverage assessment",
                    finding="Coverage is adequate." if proposer2_verdict == "non_blocking" else "Missing literature support.",
                ),
                "artifact": {"archive_path": "/tmp/review-p2-r1.md"},
            },
            {
                "reviewer": "proposer-3",
                "target_proposer": "proposer-1",
                "review_round": 1,
                "body": self.make_review("proposer-3", "non_blocking", summary="Evidence trace assessment", finding="Reasoning is anchored."),
                "artifact": {"archive_path": "/tmp/review-p3-r1.md"},
            },
            {
                "reviewer": "proposer-4",
                "target_proposer": "proposer-1",
                "review_round": 1,
                "body": self.make_review("proposer-4", "non_blocking", summary="Reasoning consistency assessment", finding="Internal logic is coherent."),
                "artifact": {"archive_path": "/tmp/review-p4-r1.md"},
            },
            {
                "reviewer": "proposer-5",
                "target_proposer": "proposer-1",
                "review_round": 1,
                "body": self.make_review("proposer-5", "non_blocking", summary="Counterevidence assessment", finding="No decisive counterexample found."),
                "artifact": {"archive_path": "/tmp/review-p5-r1.md"},
            },
            {
                "reviewer": "proposer-2",
                "target_proposer": "proposer-4",
                "review_round": 1,
                "body": transport.render_transport_review(reviewer="proposer-2", target="proposer-4"),
                "artifact": {"archive_path": "/tmp/transport-p2-p4.md"},
            },
        ]
        if include_second_round_for_p2:
            reviews.append(
                {
                    "reviewer": "proposer-2",
                    "target_proposer": "proposer-1",
                    "review_round": 2,
                    "body": self.make_review(
                        "proposer-2",
                        proposer2_verdict,
                        summary="Round 2 search coverage assessment",
                        finding="Still missing literature support." if proposer2_verdict != "non_blocking" else "Coverage remains adequate.",
                    ),
                    "artifact": {"archive_path": "/tmp/review-p2-r2.md"},
                }
            )
        exited_reviewers = list(exited_reviewers or [])
        return {
            "team_name": "chemqa-review-test-run",
            "workflow": "review-loop",
            "goal": "Question: How many peaks?",
            "status": "done",
            "phase": "done",
            "epoch": 1,
            "review_round": 2 if include_second_round_for_p2 else 1,
            "rebuttal_round": 0,
            "final_candidates": ["proposer-1", "proposer-2", "proposer-3", "proposer-4", "proposer-5"],
            "proposals": proposals,
            "reviews": reviews,
            "rebuttals": [],
            "exited_reviewer_lanes": exited_reviewers,
            "active_reviewer_lanes": [role for role in transport.REVIEWER_ROLES if role not in exited_reviewers],
            "reviewer_exit_reasons": {
                role: {"reason": f"{role} exited after repeated review stagnation"}
                for role in exited_reviewers
            },
        }

    def test_build_protocol_from_summary_is_strictly_valid_for_acceptance(self) -> None:
        protocol = transport.build_protocol_from_summary(self.build_summary())
        self.assertEqual("accepted", protocol["acceptance_status"])
        self.assertEqual(4, protocol["review_completion_status"]["required_candidate_reviews_submitted"])
        self.assertEqual([], collect_artifacts_module.validate_protocol(protocol)["errors"])

    def test_build_protocol_from_summary_uses_latest_blocking_review(self) -> None:
        protocol = transport.build_protocol_from_summary(
            self.build_summary(proposer2_verdict="blocking", include_second_round_for_p2=True)
        )
        self.assertEqual("rejected", protocol["acceptance_status"])
        self.assertEqual(["proposer-2"], protocol["acceptance_decision"]["blocking_reviewers"])
        self.assertEqual(2, protocol["reviewer_trajectories"]["proposer-2"]["latest_review_round"])
        self.assertEqual(4, protocol["review_completion_status"]["required_candidate_reviews_submitted"])
        self.assertEqual([], collect_artifacts_module.validate_protocol(protocol)["errors"])

    def test_build_protocol_from_summary_allows_completed_rejected_output_when_lane_missing(self) -> None:
        summary = self.build_summary()
        summary["reviews"] = [
            review for review in summary["reviews"]
            if not (review.get("reviewer") == "proposer-5" and review.get("target_proposer") == "proposer-1")
        ]
        protocol = transport.build_protocol_from_summary(summary)
        self.assertEqual("completed", protocol["terminal_state"])
        self.assertEqual("rejected", protocol["acceptance_status"])
        self.assertEqual("6", protocol["final_answer"]["direct_answer"])
        self.assertEqual(["proposer-5"], protocol["acceptance_decision"]["missing_required_reviewer_lanes"])
        self.assertEqual(False, protocol["review_completion_status"]["required_fixed_reviewer_lanes_complete"])
        self.assertEqual([], collect_artifacts_module.validate_protocol(protocol)["errors"])

    def test_apply_forced_missing_review_completion_marks_degraded_completion(self) -> None:
        summary = self.build_summary()
        summary["reviews"] = [
            review for review in summary["reviews"]
            if not (review.get("reviewer") == "proposer-5" and review.get("target_proposer") == "proposer-1")
        ]
        protocol = transport.build_protocol_from_summary(summary)
        forced = transport.apply_forced_missing_review_completion(
            protocol,
            reason="forced degraded completion after recovery attempts left missing reviewer lane proposer-5",
            missing_lanes=["proposer-5"],
            blockers=["missing formal review artifact for proposer-5->proposer-1"],
            recovery_cycles_without_progress=2,
        )
        self.assertTrue(forced["review_completion_status"]["forced_completion"])
        self.assertEqual(["proposer-5"], forced["review_completion_status"]["missing_required_reviewer_lanes"])
        self.assertEqual("low", forced["overall_confidence"]["level"])
        self.assertTrue(forced["final_answer"]["forced_completion"])
        self.assertIn("Recovery blocker: missing formal review artifact for proposer-5->proposer-1", forced["execution_warnings"])
        self.assertEqual([], collect_artifacts_module.validate_protocol(forced)["errors"])

    def test_build_protocol_from_summary_accepts_under_active_reviewer_quorum_after_exit(self) -> None:
        summary = self.build_summary(exited_reviewers=["proposer-5"])
        summary["reviews"] = [
            review for review in summary["reviews"]
            if not (review.get("reviewer") == "proposer-5" and review.get("target_proposer") == "proposer-1")
        ]
        protocol = transport.build_protocol_from_summary(summary)
        self.assertEqual("accepted", protocol["acceptance_status"])
        self.assertEqual("active_reviewer_quorum_after_lane_exit", protocol["acceptance_decision"]["acceptance_context"])
        self.assertTrue(protocol["acceptance_decision"]["accepted_under_degraded_quorum"])
        self.assertEqual(["proposer-5"], protocol["review_completion_status"]["exited_reviewer_lanes"])
        self.assertEqual(3, protocol["review_completion_status"]["required_candidate_reviews_expected_effective"])
        self.assertEqual(3, protocol["review_completion_status"]["required_candidate_reviews_submitted_effective"])
        self.assertTrue(protocol["review_completion_status"]["required_active_reviewer_lanes_complete"])
        self.assertFalse(protocol["review_completion_status"]["required_fixed_reviewer_lanes_complete"])


class MaterializeRunplanTest(unittest.TestCase):
    def test_build_command_map_uses_chemqa_driver(self) -> None:
        run_plan = {
            "run_id": "chemqa-review-test-run",
            "session_assignments": {
                "debate-coordinator": "sess-coord",
                "debate-1": "sess-1",
                "debate-2": "sess-2",
                "debate-3": "sess-3",
                "debate-4": "sess-4",
                "debate-5": "sess-5",
            },
            "slot_assignments": {
                "debate-coordinator": {"thinking": "high"},
                "debate-1": {"thinking": "high"},
                "debate-2": {"thinking": "medium"},
                "debate-3": {"thinking": "medium"},
                "debate-4": {"thinking": "medium"},
                "debate-5": {"thinking": "medium"},
            },
            "launch_spec": {
                "role_slots": {
                    "debate-coordinator": "debate-coordinator",
                    "proposer-1": "debate-1",
                    "proposer-2": "debate-2",
                    "proposer-3": "debate-3",
                    "proposer-4": "debate-4",
                    "proposer-5": "debate-5",
                }
            },
            "runtime_context": {
                "chemqa_review": {
                    "stop_loss": {
                        "stale_timeout_seconds": 300,
                        "respawn_cooldown_seconds": 120,
                        "max_model_attempts": 2,
                        "lane_retry_budget": 2,
                        "phase_repair_budget": 2,
                        "max_respawns_per_role_phase_signature": 2,
                    }
                }
            },
        }
        command_map = materialize_runplan.build_command_map(
            run_plan,
            wrapper_path=Path("/runtime/openclaw_debate_agent.py"),
            env_file="/tmp/openclaw.env",
            skill_root=SKILL_ROOT,
            clawteam_data_dir="/tmp/clawteam-data",
            openclaw_config=Path("/tmp/openclaw.json"),
        )
        command = command_map["proposer-2"]
        self.assertTrue(command[0].endswith("python") or command[0].endswith("python3") or "/python" in command[0])
        self.assertTrue(command[1].endswith("chemqa_review_openclaw_driver.py"))
        self.assertIn("--team", command)
        self.assertIn("chemqa-review-test-run", command)
        self.assertIn("--role", command)
        self.assertIn("proposer-2", command)
        self.assertIn("--runtime-dir", command)
        self.assertIn("/runtime", command)
        self.assertIn("--data-dir", command)
        self.assertIn("/tmp/clawteam-data", command)
        self.assertIn("--stale-timeout-seconds", command)
        self.assertIn("300", command)
        self.assertIn("--respawn-cooldown-seconds", command)
        self.assertIn("120", command)
        self.assertIn("--max-model-attempts", command)
        self.assertIn("2", command[command.index("--max-model-attempts") + 1])
        self.assertIn("--lane-retry-budget", command)
        self.assertIn("--phase-repair-budget", command)
        self.assertIn("2", command[command.index("--phase-repair-budget") + 1])
        self.assertIn("--max-respawns-per-role-phase-signature", command)
        self.assertIn("2", command[command.index("--max-respawns-per-role-phase-signature") + 1])


class NativeWorkflowStateTest(unittest.TestCase):
    def test_chemqa_native_progress_and_next_action(self) -> None:
        debate_state = load_module(DEBATE_STATE_PATH, "debate_state_for_native_workflow_tests")
        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env["CLAWTEAM_DATA_DIR"] = tmpdir
            previous_data_dir = os.environ.get("CLAWTEAM_DATA_DIR")
            os.environ["CLAWTEAM_DATA_DIR"] = tmpdir
            try:
                config = debate_state.DebateConfig(
                    team_name="chemqa-native-team",
                    workflow="chemqa-review",
                    goal="Question: test?",
                    evidence_policy="strict",
                    proposer_count=5,
                    max_review_rounds=3,
                    max_rebuttal_rounds=2,
                    max_epochs=3,
                )
                debate_state.init_debate_state(config, reset=True)
                with debate_state.connect("chemqa-native-team") as conn:
                    propose_progress = debate_state.propose_phase_progress(conn)
                    self.assertEqual(1, propose_progress["expected"])
                    self.assertEqual(0, propose_progress["actual"])
                    self.assertFalse(propose_progress["complete"])

                    reviewer_action = debate_state.next_action_payload(conn, agent="proposer-2")
                    self.assertEqual("wait", reviewer_action["action"])

                    fixture_dir = Path(tmpdir) / "fixtures"
                    fixture_dir.mkdir(parents=True, exist_ok=True)
                    proposal_path = fixture_dir / "proposal.yaml"
                    proposal_path.write_text(
                        "artifact_kind: candidate_submission\nphase: propose\nowner: proposer-1\ndirect_answer: '6'\nsummary: test\nsubmission_trace:\n- step: reasoning\n  status: success\n  detail: ok\n",
                        encoding="utf-8",
                    )
                    debate_state.submit_proposal(conn, agent="proposer-1", file_path=proposal_path)
                    advanced = debate_state.advance_state(conn, agent="debate-coordinator")
                    self.assertEqual("review", advanced["phase"])

                    review_progress = debate_state.review_phase_progress(conn)
                    self.assertEqual(4, review_progress["expected"])
                    self.assertEqual(0, review_progress["actual"])

                    reviewer_action = debate_state.next_action_payload(conn, agent="proposer-3")
                    self.assertEqual("review", reviewer_action["action"])
                    self.assertEqual(["proposer-1"], reviewer_action["targets"])
            finally:
                if previous_data_dir is None:
                    os.environ.pop("CLAWTEAM_DATA_DIR", None)
                else:
                    os.environ["CLAWTEAM_DATA_DIR"] = previous_data_dir


class DebateWrapperConfigTest(unittest.TestCase):
    def test_create_debate_runtime_config_whitelists_trusted_plugins(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config_path = Path(tmpdir) / "openclaw.json"
            base_config_path.write_text(
                json.dumps(
                    {
                        "plugins": {
                            "entries": {
                                "duckduckgo": {"kind": "builtin"},
                                "filesystem": {"kind": "builtin"},
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            temp_config_path = debate_wrapper_module.create_debate_runtime_config(
                base_config_path,
                env={"OPENCLAW_DEBATE_TRUSTED_PLUGINS": "duckduckgo"},
                slot="debate-2",
            )
            self.assertIsNotNone(temp_config_path)
            assert temp_config_path is not None
            payload = json.loads(temp_config_path.read_text(encoding="utf-8"))
            self.assertEqual(["duckduckgo"], payload["plugins"]["allow"])
            self.assertEqual({"duckduckgo": {"kind": "builtin"}}, payload["plugins"]["entries"])
            temp_config_path.unlink(missing_ok=True)

    def test_reset_slot_main_session_clears_lowercased_main_key_when_session_file_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config_path = Path(tmpdir) / "openclaw.json"
            base_config_path.write_text(
                json.dumps(
                    {
                        "agents": {
                            "list": [
                                {
                                    "id": "debateA-1",
                                    "workspace": "/tmp/debateA-1",
                                    "model": "dashscope-responses/qwen3.5-plus",
                                }
                            ]
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            store_path = Path(tmpdir) / "sessions.json"
            store_path.write_text(
                json.dumps(
                    {
                        "agent:debatea-1:main": {
                            "sessionId": "chemqa-review-test-session",
                            "sessionFile": "/tmp/smoke-chemqa-direct.jsonl",
                            "modelProvider": "dashscope-responses",
                            "model": "qwen3.5-plus",
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            original_session_store_path_for_slot = debate_wrapper_module.session_store_path_for_slot
            try:
                debate_wrapper_module.session_store_path_for_slot = lambda _slot: store_path
                debate_wrapper_module.reset_slot_main_session_if_session_id_changed(
                    "debateA-1",
                    "chemqa-review-test-session",
                    config_path=base_config_path,
                )
            finally:
                debate_wrapper_module.session_store_path_for_slot = original_session_store_path_for_slot

            payload = json.loads(store_path.read_text(encoding="utf-8"))
            self.assertEqual({}, payload)


class DriverInitConfigPropagationTest(unittest.TestCase):
    def test_driver_init_passes_explicit_config_to_main_session_reset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir = Path(tmpdir) / "runtime"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            (runtime_dir / "debate_state.py").write_text("# stub\n", encoding="utf-8")
            (runtime_dir / "openclaw_debate_agent.py").write_text("# stub\n", encoding="utf-8")
            config_path = Path(tmpdir) / "cleanroom-openclaw.json"
            config_path.write_text("{}", encoding="utf-8")

            original_load_module = driver_module.load_module_from_path
            original_resolve_skill_root = driver_module.resolve_skill_root
            original_default_runtime_dir = driver_module.default_runtime_dir
            original_load_cleanroom_runtime_lease_module = driver_module.load_cleanroom_runtime_lease_module
            original_resolve_clawteam_executable = driver_module.resolve_clawteam_executable

            class WrapperProbe:
                def __init__(self) -> None:
                    self.workspace_reset_config = None
                    self.main_session_reset_config = None

                def reset_slot_workspace_if_session_id_changed(self, _slot, _session_id, *, config_path=None):
                    self.workspace_reset_config = config_path

                def reset_slot_main_session_if_session_id_changed(self, _slot, _session_id, *, config_path=None):
                    self.main_session_reset_config = config_path

                def resolve_slot_workspace(self, _slot, *, config_path=None):
                    return Path(tmpdir) / "workspace"

            wrapper_probe = WrapperProbe()
            try:
                driver_module.load_module_from_path = lambda *_args, **_kwargs: wrapper_probe
                driver_module.resolve_skill_root = lambda value: Path(value).resolve()
                driver_module.default_runtime_dir = lambda: runtime_dir
                driver_module.load_cleanroom_runtime_lease_module = lambda _skill_root: None
                driver_module.resolve_clawteam_executable = lambda **_kwargs: "/tmp/fake-clawteam"

                driver_module.ChemQAReviewDriver(
                    argparse.Namespace(
                        skill_root=str(SKILL_ROOT),
                        runtime_dir=str(runtime_dir),
                        config_file=str(config_path),
                        team="chemqa-review-init-config",
                        role="debate-coordinator",
                        slot="debateA-coordinator",
                        env_file="",
                        data_dir="",
                        lease_dir="",
                        session_id="chemqa-review-init-config-session",
                        prompt=None,
                        message=None,
                        thinking=None,
                        poll_seconds=20,
                        stale_timeout_seconds=300,
                        max_model_attempts=1,
                        model_timeout_seconds=None,
                        candidate_timeout_seconds=None,
                        review_timeout_seconds=None,
                        rebuttal_timeout_seconds=None,
                        coordinator_timeout_seconds=None,
                        subprocess_timeout_grace_seconds=30,
                        respawn_cooldown_seconds=120,
                        lane_retry_budget=2,
                        phase_repair_budget=1,
                        max_respawns_per_role_phase_signature=1,
                    )
                )
            finally:
                driver_module.load_module_from_path = original_load_module
                driver_module.resolve_skill_root = original_resolve_skill_root
                driver_module.default_runtime_dir = original_default_runtime_dir
                driver_module.load_cleanroom_runtime_lease_module = original_load_cleanroom_runtime_lease_module
                driver_module.resolve_clawteam_executable = original_resolve_clawteam_executable

            self.assertEqual(config_path.resolve(), wrapper_probe.workspace_reset_config)
            self.assertEqual(config_path.resolve(), wrapper_probe.main_session_reset_config)


class ClawteamResolutionTest(unittest.TestCase):
    def test_resolve_clawteam_executable_falls_back_when_path_is_stripped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fallback = Path(tmpdir) / "clawteam"
            fallback.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fallback.chmod(0o755)

            resolved = bundle_common.resolve_clawteam_executable(
                env={"PATH": "/usr/bin:/bin"},
                fallback_paths=[fallback],
            )

            self.assertEqual(str(fallback.resolve()), resolved)

    def test_resolve_clawteam_executable_uses_real_home_when_home_is_cleanroom(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            real_home = Path(tmpdir) / "real-home"
            cleanroom_home = Path(tmpdir) / "cleanroom-home"
            fallback = real_home / ".local" / "share" / "uv" / "tools" / "clawteam" / "bin" / "clawteam"
            fallback.parent.mkdir(parents=True, exist_ok=True)
            fallback.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fallback.chmod(0o755)
            cleanroom_home.mkdir(parents=True, exist_ok=True)

            original_home = os.environ.get("HOME")
            original_real_home = os.environ.get("OPENCLAW_REAL_HOME")
            try:
                os.environ["HOME"] = str(cleanroom_home)
                os.environ["OPENCLAW_REAL_HOME"] = str(real_home)
                resolved = bundle_common.resolve_clawteam_executable(
                    env={"PATH": "/usr/bin:/bin"},
                )
            finally:
                if original_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = original_home
                if original_real_home is None:
                    os.environ.pop("OPENCLAW_REAL_HOME", None)
                else:
                    os.environ["OPENCLAW_REAL_HOME"] = original_real_home

            self.assertEqual(str(fallback.resolve()), resolved)

    def test_current_task_uses_resolved_clawteam_executable(self) -> None:
        driver = driver_module.ChemQAReviewDriver.__new__(driver_module.ChemQAReviewDriver)
        driver.args = argparse.Namespace(team="chemqa-review-clawteam", role="proposer-2")
        driver.data_dir = "/tmp/clawteam-data"
        driver.clawteam_executable = "/tmp/resolved-clawteam"

        original_run = driver_module.subprocess.run
        captured: dict[str, object] = {}
        try:
            def fake_run(command, env=None, check=False, capture_output=False, text=False):
                captured["command"] = list(command)
                return subprocess.CompletedProcess(command, 0, stdout="[]", stderr="")

            driver_module.subprocess.run = fake_run
            payload = driver_module.ChemQAReviewDriver.current_task(driver)
        finally:
            driver_module.subprocess.run = original_run

        self.assertIsNone(payload)
        command = captured["command"]
        assert isinstance(command, list)
        self.assertEqual("/tmp/resolved-clawteam", command[0])
        self.assertEqual(
            [
                "/tmp/resolved-clawteam",
                "--data-dir",
                "/tmp/clawteam-data",
                "--json",
                "task",
                "list",
                "chemqa-review-clawteam",
                "--owner",
                "proposer-2",
            ],
            command,
        )

    def test_launch_from_preset_resolves_bare_clawteam_command_before_run(self) -> None:
        original_run_json = launch_from_preset_module.run_json
        original_resolve_clawteam_executable = launch_from_preset_module.resolve_clawteam_executable
        original_subprocess_run = launch_from_preset_module.subprocess.run
        original_argv = sys.argv
        captured: dict[str, object] = {}
        try:
            launch_from_preset_module.run_json = lambda command, cwd: (
                {"run_id": "demo-run"}
                if "compile_runplan.py" in str(command[1])
                else {
                    "launch_command": ["clawteam", "launch", "--template", "demo"],
                    "clawteam_data_dir": "/tmp/demo-clawteam-data",
                    "openclaw_config_path": "/tmp/demo-openclaw.json",
                }
            )
            launch_from_preset_module.resolve_clawteam_executable = lambda **_kwargs: "/tmp/resolved-clawteam"

            def fake_run(command, cwd=None, env=None, check=False, capture_output=False, text=False):
                captured["command"] = list(command)
                captured["cwd"] = cwd
                captured["env"] = dict(env or {})
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            launch_from_preset_module.subprocess.run = fake_run
            sys.argv = [
                "launch_from_preset.py",
                "--root",
                str(SKILL_ROOT),
                "--preset",
                "chemqa-review@1",
                "--goal",
                "demo goal",
                "--launch-mode",
                "run",
            ]

            exit_code = launch_from_preset_module.main()
        finally:
            launch_from_preset_module.run_json = original_run_json
            launch_from_preset_module.resolve_clawteam_executable = original_resolve_clawteam_executable
            launch_from_preset_module.subprocess.run = original_subprocess_run
            sys.argv = original_argv

        self.assertEqual(0, exit_code)
        command = captured["command"]
        assert isinstance(command, list)
        self.assertEqual("/tmp/resolved-clawteam", command[0])
        self.assertEqual(["launch", "--template", "demo"], command[1:])
        env = captured["env"]
        assert isinstance(env, dict)
        self.assertEqual("/tmp/demo-clawteam-data", env["CLAWTEAM_DATA_DIR"])
        self.assertEqual("/tmp/demo-openclaw.json", env["OPENCLAW_CONFIG_PATH"])


class ChemProviderIntegrationTest(unittest.TestCase):
    def test_required_skills_include_chem_provider_bundles(self) -> None:
        expected = {
            "debateclaw-v1",
            "paper-retrieval",
            "paper-access",
            "paper-parse",
            "paper-rerank",
            "rdkit",
            "pubchem",
            "opsin",
            "chem-calculator",
        }
        self.assertEqual(expected, set(bundle_common.REQUIRED_SKILLS))

        report = bundle_common.dependency_report(SKILL_ROOT)
        self.assertEqual(expected, set(report))
        self.assertEqual([], bundle_common.missing_skills_from_report(report))

    def test_prompt_contracts_include_chem_provider_routing_rules(self) -> None:
        proposer = (SKILL_ROOT / "prompts" / "contracts" / "proposer-main.md").read_text(encoding="utf-8")
        reasoning = (SKILL_ROOT / "prompts" / "contracts" / "reviewer-reasoning-consistency.md").read_text(
            encoding="utf-8"
        )
        evidence = (SKILL_ROOT / "prompts" / "contracts" / "reviewer-evidence-trace.md").read_text(
            encoding="utf-8"
        )
        counter = (SKILL_ROOT / "prompts" / "contracts" / "reviewer-counterevidence.md").read_text(
            encoding="utf-8"
        )
        required_skills = (SKILL_ROOT / "prompts" / "modules" / "context" / "required-skills.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("chem-calculator", proposer)
        self.assertIn("FrontierScience", proposer)
        self.assertIn("SuperChem", proposer)
        self.assertIn("extract available SMILES/name text first", proposer)
        self.assertIn("rdkit", proposer)
        self.assertIn("opsin", proposer)
        self.assertIn("pubchem", proposer)

        self.assertIn("chem-calculator", reasoning)
        self.assertIn("result.json", reasoning)
        self.assertIn("tool_trace", reasoning)

        self.assertIn("result.json", evidence)
        self.assertIn("tool_trace", evidence)
        self.assertIn("result.json", counter)
        self.assertIn("tool_trace", counter)

        self.assertIn("rdkit", required_skills)
        self.assertIn("pubchem", required_skills)
        self.assertIn("opsin", required_skills)
        self.assertIn("chem-calculator", required_skills)


class OpenClawResolutionTest(unittest.TestCase):
    def test_resolve_openclaw_executable_falls_back_when_path_is_stripped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fallback = Path(tmpdir) / "openclaw"
            fallback.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fallback.chmod(0o755)

            resolved = debate_wrapper_module.resolve_openclaw_executable(
                env={"PATH": "/usr/bin:/bin"},
                fallback_paths=[fallback],
            )

            self.assertEqual(str(fallback.resolve()), resolved)

    def test_wrapper_main_uses_resolved_openclaw_executable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = Path(tmpdir) / "openclaw.json"
            base_config.write_text(json.dumps({"agents": {"list": []}}, ensure_ascii=False), encoding="utf-8")
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("", encoding="utf-8")

            original_parse_args = debate_wrapper_module.parse_args
            original_parse_env_entries = debate_wrapper_module.parse_env_entries
            original_reset_workspace = debate_wrapper_module.reset_slot_workspace_if_session_id_changed
            original_reset_session = debate_wrapper_module.reset_slot_main_session_if_session_id_changed
            original_create_runtime_config = debate_wrapper_module.create_debate_runtime_config
            original_resolve_slot = debate_wrapper_module.resolve_effective_slot_id
            original_resolve_openclaw = getattr(debate_wrapper_module, "resolve_openclaw_executable", None)
            original_write_lease = debate_wrapper_module.write_cleanroom_lease
            original_run = debate_wrapper_module.subprocess.run

            captured: dict[str, object] = {}
            try:
                debate_wrapper_module.parse_args = lambda: argparse.Namespace(
                    slot="debateA-1",
                    session_id="chemqa-review-test-session",
                    config_file=str(base_config),
                    env_file=str(env_file),
                    prompt=None,
                    message="probe",
                    thinking="high",
                    timeout=None,
                    json=False,
                )
                debate_wrapper_module.parse_env_entries = lambda _path: {}
                debate_wrapper_module.reset_slot_workspace_if_session_id_changed = lambda *args, **kwargs: None
                debate_wrapper_module.reset_slot_main_session_if_session_id_changed = lambda *args, **kwargs: None
                debate_wrapper_module.create_debate_runtime_config = lambda *_args, **_kwargs: None
                debate_wrapper_module.resolve_effective_slot_id = lambda *_args, **_kwargs: "debateA-1"
                debate_wrapper_module.resolve_openclaw_executable = lambda **_kwargs: "/tmp/resolved-openclaw"
                debate_wrapper_module.write_cleanroom_lease = lambda **_kwargs: (None, None)

                def fake_run(command, env=None, check=False):
                    captured["command"] = list(command)
                    return subprocess.CompletedProcess(command, 0)

                debate_wrapper_module.subprocess.run = fake_run

                exit_code = debate_wrapper_module.main()
            finally:
                debate_wrapper_module.parse_args = original_parse_args
                debate_wrapper_module.parse_env_entries = original_parse_env_entries
                debate_wrapper_module.reset_slot_workspace_if_session_id_changed = original_reset_workspace
                debate_wrapper_module.reset_slot_main_session_if_session_id_changed = original_reset_session
                debate_wrapper_module.create_debate_runtime_config = original_create_runtime_config
                debate_wrapper_module.resolve_effective_slot_id = original_resolve_slot
                if original_resolve_openclaw is None:
                    delattr(debate_wrapper_module, "resolve_openclaw_executable")
                else:
                    debate_wrapper_module.resolve_openclaw_executable = original_resolve_openclaw
                debate_wrapper_module.write_cleanroom_lease = original_write_lease
                debate_wrapper_module.subprocess.run = original_run

        self.assertEqual(0, exit_code)
        command = captured["command"]
        assert isinstance(command, list)
        self.assertEqual("/tmp/resolved-openclaw", command[0])
        self.assertEqual(
            [
                "/tmp/resolved-openclaw",
                "agent",
                "--local",
                "--agent",
                "debateA-1",
                "--session-id",
                "chemqa-review-test-session",
                "--message",
                "probe",
                "--thinking",
                "high",
            ],
            command,
        )

    def test_wrapper_main_prepends_resolved_node_directory_for_openclaw_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = Path(tmpdir) / "openclaw.json"
            base_config.write_text(json.dumps({"agents": {"list": []}}, ensure_ascii=False), encoding="utf-8")
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("", encoding="utf-8")

            original_path = os.environ.get("PATH")
            original_parse_args = debate_wrapper_module.parse_args
            original_parse_env_entries = debate_wrapper_module.parse_env_entries
            original_reset_workspace = debate_wrapper_module.reset_slot_workspace_if_session_id_changed
            original_reset_session = debate_wrapper_module.reset_slot_main_session_if_session_id_changed
            original_create_runtime_config = debate_wrapper_module.create_debate_runtime_config
            original_resolve_slot = debate_wrapper_module.resolve_effective_slot_id
            original_resolve_openclaw = getattr(debate_wrapper_module, "resolve_openclaw_executable", None)
            original_resolve_node = getattr(debate_wrapper_module, "resolve_node_executable", None)
            original_write_lease = debate_wrapper_module.write_cleanroom_lease
            original_run = debate_wrapper_module.subprocess.run

            captured: dict[str, object] = {}
            try:
                os.environ["PATH"] = "/usr/bin:/bin"
                debate_wrapper_module.parse_args = lambda: argparse.Namespace(
                    slot="debateA-1",
                    session_id="chemqa-review-test-session",
                    config_file=str(base_config),
                    env_file=str(env_file),
                    prompt=None,
                    message="probe",
                    thinking="high",
                    timeout=None,
                    json=False,
                )
                debate_wrapper_module.parse_env_entries = lambda _path: {}
                debate_wrapper_module.reset_slot_workspace_if_session_id_changed = lambda *args, **kwargs: None
                debate_wrapper_module.reset_slot_main_session_if_session_id_changed = lambda *args, **kwargs: None
                debate_wrapper_module.create_debate_runtime_config = lambda *_args, **_kwargs: None
                debate_wrapper_module.resolve_effective_slot_id = lambda *_args, **_kwargs: "debateA-1"
                debate_wrapper_module.resolve_openclaw_executable = lambda **_kwargs: "/tmp/tool-bin/openclaw"
                debate_wrapper_module.resolve_node_executable = lambda **_kwargs: "/tmp/node-bin/node"
                debate_wrapper_module.write_cleanroom_lease = lambda **_kwargs: (None, None)

                def fake_run(command, env=None, check=False):
                    captured["command"] = list(command)
                    captured["path"] = str((env or {}).get("PATH") or "")
                    return subprocess.CompletedProcess(command, 0)

                debate_wrapper_module.subprocess.run = fake_run

                exit_code = debate_wrapper_module.main()
            finally:
                if original_path is None:
                    os.environ.pop("PATH", None)
                else:
                    os.environ["PATH"] = original_path
                debate_wrapper_module.parse_args = original_parse_args
                debate_wrapper_module.parse_env_entries = original_parse_env_entries
                debate_wrapper_module.reset_slot_workspace_if_session_id_changed = original_reset_workspace
                debate_wrapper_module.reset_slot_main_session_if_session_id_changed = original_reset_session
                debate_wrapper_module.create_debate_runtime_config = original_create_runtime_config
                debate_wrapper_module.resolve_effective_slot_id = original_resolve_slot
                if original_resolve_openclaw is None:
                    delattr(debate_wrapper_module, "resolve_openclaw_executable")
                else:
                    debate_wrapper_module.resolve_openclaw_executable = original_resolve_openclaw
                if original_resolve_node is None:
                    delattr(debate_wrapper_module, "resolve_node_executable")
                else:
                    debate_wrapper_module.resolve_node_executable = original_resolve_node
                debate_wrapper_module.write_cleanroom_lease = original_write_lease
                debate_wrapper_module.subprocess.run = original_run

        self.assertEqual(0, exit_code)
        self.assertEqual("/tmp/tool-bin/openclaw", captured["command"][0])
        expected_prefix = str((Path("/tmp/node-bin")).resolve()) + os.pathsep
        self.assertTrue(str(captured["path"]).startswith(expected_prefix))


class WorkerTimeoutSalvageTest(unittest.TestCase):
    def test_call_model_captures_valid_candidate_before_workspace_file_is_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace = root / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            data_dir = root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            wrapper_path = root / "wrapper.py"
            wrapper_path.write_text(
                "\n".join(
                    [
                        "import pathlib",
                        "import sys",
                        "import time",
                        "",
                        "workspace = pathlib.Path.cwd()",
                        "proposal = workspace / 'proposal.yaml'",
                        "proposal.write_text(",
                        "    \"artifact_kind: candidate_submission\\n\"",
                        "    \"artifact_contract_version: react-reviewed-v2\\n\"",
                        "    \"phase: propose\\n\"",
                        "    \"owner: proposer-1\\n\"",
                        "    \"direct_answer: CCO\\n\"",
                        "    \"summary: test candidate\\n\"",
                        "    \"submission_trace:\\n\"",
                        "    \"- step: reasoning\\n\"",
                        "    \"  status: success\\n\"",
                        "    \"  detail: wrote a valid candidate before deleting it.\\n\",",
                        "    encoding='utf-8',",
                        ")",
                        "time.sleep(1.0)",
                        "proposal.unlink()",
                        "sys.exit(0)",
                    ]
                ),
                encoding="utf-8",
            )

            driver = driver_module.ChemQAReviewDriver.__new__(driver_module.ChemQAReviewDriver)
            driver.args = argparse.Namespace(
                slot="debate-1",
                session_id="sess-1",
                env_file="/tmp/unused.env",
                config_file=None,
                thinking=None,
                role="proposer-1",
                team="capture-team",
                model_timeout_seconds=None,
                candidate_timeout_seconds=5,
                review_timeout_seconds=None,
                rebuttal_timeout_seconds=None,
                coordinator_timeout_seconds=None,
                subprocess_timeout_grace_seconds=5,
                data_dir=str(data_dir),
            )
            driver.workspace = workspace
            driver.base_wrapper_path = wrapper_path
            driver.data_dir = str(data_dir)
            driver.initial_prompt = ""
            driver.initial_prompt_used = False

            driver_module.ChemQAReviewDriver.call_model(driver, ["Write proposal.yaml."], artifact_kind="candidate_submission")

            capture_path = driver_module.ChemQAReviewDriver.candidate_capture_path(driver)
            self.assertTrue(capture_path.is_file())
            self.assertFalse((workspace / transport.proposal_filename()).exists())
            captured = capture_path.read_text(encoding="utf-8")
            self.assertIn("artifact_kind: candidate_submission", captured)
            self.assertIn("direct_answer: CCO", captured)

    def test_attempt_model_artifact_salvages_valid_review_after_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = driver_module.ChemQAReviewDriver.__new__(driver_module.ChemQAReviewDriver)
            driver.args = argparse.Namespace(
                max_model_attempts=1,
                lane_retry_budget=2,
                role="proposer-2",
                model_timeout_seconds=None,
                candidate_timeout_seconds=None,
                review_timeout_seconds=None,
                rebuttal_timeout_seconds=None,
                coordinator_timeout_seconds=None,
                subprocess_timeout_grace_seconds=30,
            )
            driver.workspace = Path(tmpdir)

            review_path = driver.workspace / transport.review_filename("proposer-1")

            def fake_call_model(_prompt_parts: list[str], *, artifact_kind: str) -> None:
                self.assertEqual("formal_review", artifact_kind)
                review_path.write_text(
                    "\n".join(
                        [
                            "artifact_kind: formal_review",
                            "artifact_contract_version: react-reviewed-v2",
                            "phase: review",
                            "reviewer_lane: proposer-2",
                            "target_owner: proposer-1",
                            "target_kind: candidate_submission",
                            "verdict: non_blocking",
                            "summary: Timed-out turn still wrote a valid artifact.",
                            "review_items: []",
                            "counts_for_acceptance: true",
                            "synthetic: false",
                        ]
                    ),
                    encoding="utf-8",
                )
                raise driver_module.DriverError("OpenClaw model turn timed out after 420s for proposer-2.")

            driver.call_model = fake_call_model

            recovered = driver_module.ChemQAReviewDriver.attempt_model_artifact(
                driver,
                filename=transport.review_filename("proposer-1"),
                instructions=["Write the review file."],
                checker=lambda text: transport.check_formal_review(text, reviewer="proposer-2", target="proposer-1"),
                artifact_kind="formal_review",
                failure_key="review:proposer-2:proposer-1",
                failure_reason="proposer-2 failed to produce a valid formal review for proposer-1",
                status_payload={"phase": "review"},
                next_action_payload={"phase": "review", "action": "review"},
            )

            self.assertEqual(review_path, recovered)
            self.assertTrue(review_path.is_file())
            self.assertIn("artifact_kind: formal_review", review_path.read_text(encoding="utf-8"))


class DriverRespawnTest(unittest.TestCase):
    def test_ensure_required_lanes_running_respawns_dead_actionable_reviewers(self) -> None:
        driver = driver_module.ChemQAReviewDriver.__new__(driver_module.ChemQAReviewDriver)
        driver.args = argparse.Namespace(team="chemqa-review-test-run", max_respawns_per_role_phase_signature=1)
        driver.last_respawn_events = []
        driver.last_respawn_attempt_at = {}
        driver.all_tasks = lambda: [
            {"owner": "proposer-2", "status": "pending"},
            {"owner": "proposer-3", "status": "completed"},
        ]
        registry = {
            "debate-coordinator": {"pid": 123},
            "proposer-2": {"pid": 0, "command": ["python3", "worker.py"]},
            "proposer-3": {"pid": 0, "command": ["python3", "worker.py"]},
        }
        driver.load_spawn_registry = lambda: registry
        saved_payloads: list[dict[str, object]] = []
        driver.save_spawn_registry = lambda payload: saved_payloads.append(payload)
        driver.next_action_for_agent = lambda role: (
            {"action": "review", "phase": "review"} if role == "proposer-2" else {"action": "stop", "phase": "done"}
        )
        driver.role_process_is_running = lambda role, entry: False
        driver.current_phase_signature = lambda: "review-round-1"
        respawned: list[tuple[str, str]] = []
        driver.respawn_role_from_registry = lambda role, entry, reason: respawned.append((role, reason)) or True

        driver_module.ChemQAReviewDriver.ensure_required_lanes_running(driver)

        self.assertEqual([("proposer-2", "missing_or_dead_role_process")], respawned)
        self.assertTrue(saved_payloads)

    def test_ensure_required_lanes_running_does_not_respawn_same_role_after_phase_budget_exhausted(self) -> None:
        driver = driver_module.ChemQAReviewDriver.__new__(driver_module.ChemQAReviewDriver)
        driver.args = argparse.Namespace(team="chemqa-review-test-run", max_respawns_per_role_phase_signature=1)
        driver.last_respawn_events = []
        driver.last_respawn_attempt_at = {}
        driver.all_tasks = lambda: [{"owner": "proposer-2", "status": "pending"}]
        registry = {
            "_budget_state": {
                "phase_signature": "review-round-1",
                "respawns_by_role": {"proposer-2": 1},
            },
            "proposer-2": {"pid": 0, "command": ["python3", "worker.py"]},
        }
        driver.load_spawn_registry = lambda: registry
        saved_payloads: list[dict[str, object]] = []
        driver.save_spawn_registry = lambda payload: saved_payloads.append(payload)
        driver.next_action_for_agent = lambda _role: {"action": "review", "phase": "review"}
        driver.role_process_is_running = lambda role, entry: False
        driver.current_phase_signature = lambda: "review-round-1"
        respawned: list[tuple[str, str]] = []
        driver.respawn_role_from_registry = lambda role, entry, reason: respawned.append((role, reason)) or True

        driver_module.ChemQAReviewDriver.ensure_required_lanes_running(driver)

        self.assertEqual([], respawned)
        self.assertEqual([], saved_payloads)

    def test_slot_from_registry_entry_prefers_explicit_slot_or_command_flag(self) -> None:
        self.assertEqual(
            "debateB-3",
            driver_module.ChemQAReviewDriver._slot_from_registry_entry({"slot": "debateB-3"}),
        )
        self.assertEqual(
            "debateA-4",
            driver_module.ChemQAReviewDriver._slot_from_registry_entry(
                {"command": ["python3", "driver.py", "--slot", "debateA-4", "--team", "demo"]}
            ),
        )


class CoordinatorStagnationReviewerExitTest(unittest.TestCase):
    def test_maybe_handle_stagnation_does_not_mark_progress_on_probe_completion_alone(self) -> None:
        helper = ProtocolReconstructionTest()
        stalled_status = helper.build_summary()
        stalled_status["status"] = "running"
        stalled_status["phase"] = "review"
        stalled_status["review_round"] = 1

        driver = driver_module.ChemQAReviewDriver.__new__(driver_module.ChemQAReviewDriver)
        driver.args = argparse.Namespace(stale_timeout_seconds=600, phase_repair_budget=2)
        driver.current_task = lambda: {"status": "in_progress"}
        driver.stale_for_seconds = lambda: 600
        driver.last_repair_signature = ""
        driver.repair_cycles_without_progress = 0
        driver.last_recovery_payload = {}
        driver.reviewer_exits = {}
        driver.run_recovery_cycle = lambda: {"status": "done", "blockers": ["probe exited without state change"]}
        statuses = [stalled_status, stalled_status]
        driver.status = lambda: statuses.pop(0)
        next_actions = [{"phase": "review", "action": "review"}, {"phase": "review", "action": "review"}]
        driver.next_action = lambda: next_actions.pop(0)
        driver.candidate_submission_text = lambda _status: helper.make_candidate()
        marked: list[tuple[str, str]] = []
        driver.mark_reviewer_exited = lambda reviewer, **kwargs: marked.append((reviewer, kwargs["reason"])) or True
        advanced: list[str] = []
        driver.advance = lambda: advanced.append("advanced")
        progress_marks: list[str] = []
        driver.mark_progress = lambda: progress_marks.append("progress")
        terminal_failures: list[tuple[str, list[str]]] = []
        driver.force_complete_with_missing_reviews = lambda **_kwargs: self.fail("should not force degraded completion without missing lanes")
        driver.emit_terminal_failure = lambda **kwargs: terminal_failures.append(
            (kwargs["reason"], list(kwargs.get("blockers") or []))
        )

        result = driver_module.ChemQAReviewDriver.maybe_handle_stagnation(
            driver,
            stalled_status,
            {"phase": "review", "action": "review"},
        )

        self.assertIsNone(result)
        self.assertEqual([], marked)
        self.assertEqual([], advanced)
        self.assertEqual([], progress_marks)
        self.assertEqual([], terminal_failures)
        self.assertEqual(1, driver.repair_cycles_without_progress)

    def test_maybe_handle_stagnation_marks_missing_reviewer_exited_and_continues(self) -> None:
        helper = ProtocolReconstructionTest()
        stalled_status = helper.build_summary()
        stalled_status["status"] = "running"
        stalled_status["phase"] = "review"
        stalled_status["review_round"] = 1
        stalled_status["reviews"] = [
            review for review in stalled_status["reviews"]
            if not (review.get("reviewer") == "proposer-5" and review.get("target_proposer") == "proposer-1")
        ]
        stalled_status["phase_progress"] = {
            "actual": 3,
            "complete": False,
            "counts_by_target": [{"blocking": 0, "submitted": 3, "target": "proposer-1"}],
            "expected": 4,
            "kind": "review",
            "missing_reviewer_lanes": ["proposer-5"],
            "round": 1,
            "targets": ["proposer-1"],
            "active_reviewer_lanes": ["proposer-2", "proposer-3", "proposer-4", "proposer-5"],
            "exited_reviewer_lanes": [],
        }
        status_after_exit = dict(stalled_status)
        status_after_exit["exited_reviewer_lanes"] = ["proposer-5"]
        status_after_exit["active_reviewer_lanes"] = ["proposer-2", "proposer-3", "proposer-4"]
        status_after_exit["reviewer_exit_reasons"] = {
            "proposer-5": {"reason": "missing formal review artifact for proposer-5->proposer-1"}
        }
        status_after_exit["phase_progress"] = {
            "actual": 3,
            "complete": True,
            "counts_by_target": [{"blocking": 0, "submitted": 3, "target": "proposer-1"}],
            "expected": 3,
            "kind": "review",
            "missing_reviewer_lanes": [],
            "round": 1,
            "targets": ["proposer-1"],
            "active_reviewer_lanes": ["proposer-2", "proposer-3", "proposer-4"],
            "exited_reviewer_lanes": ["proposer-5"],
        }

        driver = driver_module.ChemQAReviewDriver.__new__(driver_module.ChemQAReviewDriver)
        driver.args = argparse.Namespace(stale_timeout_seconds=600, phase_repair_budget=2)
        driver.current_task = lambda: {"status": "in_progress"}
        driver.stale_for_seconds = lambda: 600
        driver.last_repair_signature = ""
        driver.repair_cycles_without_progress = 0
        driver.last_recovery_payload = {}
        driver.reviewer_exits = {}
        driver.run_recovery_cycle = lambda: {"status": "running", "blockers": ["missing formal review artifact for proposer-5->proposer-1"]}
        statuses = [stalled_status, status_after_exit]
        driver.status = lambda: statuses.pop(0)
        next_actions = [{"phase": "review", "action": "review"}, {"phase": "review", "action": "advance"}]
        driver.next_action = lambda: next_actions.pop(0)
        driver.candidate_submission_text = lambda _status: helper.make_candidate()
        marked: list[tuple[str, str]] = []
        driver.mark_reviewer_exited = lambda reviewer, **kwargs: marked.append((reviewer, kwargs["reason"])) or True
        advanced: list[str] = []
        driver.advance = lambda: advanced.append("advanced")
        progress_marks: list[str] = []
        driver.mark_progress = lambda: progress_marks.append("progress")
        driver.force_complete_with_missing_reviews = lambda **_kwargs: self.fail("should continue debate instead of forcing degraded completion")
        driver.emit_terminal_failure = lambda **_kwargs: self.fail("should not terminal-fail when reviewer exit unlocks active quorum")

        result = driver_module.ChemQAReviewDriver.maybe_handle_stagnation(
            driver,
            stalled_status,
            {"phase": "review", "action": "review"},
        )

        self.assertIsNone(result)
        self.assertEqual([("proposer-5", "reviewer exited after repeated review stagnation: missing formal review artifact for proposer-5->proposer-1")], marked)
        self.assertEqual(["advanced"], advanced)
        self.assertEqual(["progress"], progress_marks)


class CoordinatorTimeoutSalvageTest(unittest.TestCase):
    def test_attempt_model_artifact_rejects_unchanged_preexisting_candidate_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = driver_module.ChemQAReviewDriver.__new__(driver_module.ChemQAReviewDriver)
            driver.args = argparse.Namespace(
                max_model_attempts=1,
                lane_retry_budget=2,
                role="proposer-1",
                model_timeout_seconds=None,
                candidate_timeout_seconds=None,
                review_timeout_seconds=None,
                rebuttal_timeout_seconds=None,
                coordinator_timeout_seconds=None,
                subprocess_timeout_grace_seconds=30,
            )
            driver.workspace = Path(tmpdir)
            proposal_path = driver.workspace / transport.proposal_filename()
            proposal_path.write_text(
                "\n".join(
                    [
                        "artifact_kind: candidate_submission",
                        "artifact_contract_version: react-reviewed-v2",
                        "phase: propose",
                        "owner: proposer-1",
                        "direct_answer: CN(C)CCF",
                        "summary: stale candidate left by an earlier epoch.",
                        "submission_trace:",
                        "- step: structural_reasoning",
                        "  status: success",
                        "  detail: stale candidate.",
                    ]
                ),
                encoding="utf-8",
            )

            driver.call_model = lambda _prompt_parts, *, artifact_kind: self.assertEqual("candidate_submission", artifact_kind)
            recorded_failures: list[tuple[str, str, list[str] | None]] = []
            driver.record_lane_failure = (
                lambda failure_key, *, reason, problems=None: recorded_failures.append((failure_key, reason, problems)) or 1
            )
            driver.emit_terminal_failure = lambda **_kwargs: self.fail("unchanged stale artifact should not terminal-fail on first detection")

            with self.assertRaises(driver_module.DriverError):
                driver_module.ChemQAReviewDriver.attempt_model_artifact(
                    driver,
                    filename=transport.proposal_filename(),
                    instructions=["Write the revised proposal file."],
                    checker=lambda text: transport.check_candidate_submission(text, owner="proposer-1"),
                artifact_kind="candidate_submission",
                failure_key="propose:candidate",
                failure_reason="proposer-1 failed to produce a valid candidate submission",
                status_payload={"phase": "propose", "epoch": 2},
                next_action_payload={"phase": "propose", "action": "propose"},
                require_file_change=True,
            )

            self.assertEqual(1, len(recorded_failures))
            self.assertEqual("propose:candidate", recorded_failures[0][0])
            self.assertIn("was not updated by model turn", " ".join(recorded_failures[0][2] or []))

    def test_generate_protocol_with_model_salvages_valid_artifact_after_timeout(self) -> None:
        helper = ProtocolReconstructionTest()
        summary_payload = helper.build_summary()
        deterministic_protocol = transport.build_protocol_from_summary(summary_payload)

        with tempfile.TemporaryDirectory() as tmpdir:
            driver = driver_module.ChemQAReviewDriver.__new__(driver_module.ChemQAReviewDriver)
            driver.args = argparse.Namespace(max_model_attempts=1, role="debate-coordinator")
            driver.workspace = Path(tmpdir)

            protocol_path = driver.workspace / transport.coordinator_protocol_filename()

            def fake_call_model(_prompt_parts: list[str], *, artifact_kind: str) -> None:
                self.assertEqual("coordinator_protocol", artifact_kind)
                payload = json.loads(json.dumps(deterministic_protocol, ensure_ascii=False))
                payload["overall_confidence"] = {
                    "level": "high",
                    "rationale": "Coordinator finished writing before the wrapper timeout fired.",
                }
                protocol_path.write_text(
                    transport.check_protocol(json.dumps(payload, ensure_ascii=False)).normalized_text,
                    encoding="utf-8",
                )
                raise driver_module.DriverError("OpenClaw model turn timed out after 300s for debate-coordinator.")

            driver.call_model = fake_call_model

            protocol_payload, generation_mode = driver_module.ChemQAReviewDriver.generate_protocol_with_model(
                driver,
                summary_payload=summary_payload,
                deterministic_protocol=deterministic_protocol,
            )

            self.assertEqual("model_timeout_salvaged", generation_mode)
            self.assertEqual("accepted", protocol_payload["acceptance_status"])
            self.assertEqual("high", protocol_payload["overall_confidence"]["level"])
            self.assertTrue(protocol_path.is_file())


class RunStatusShapeTest(unittest.TestCase):
    def test_ensure_candidate_submission_retries_after_duplicate_epoch_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = driver_module.ChemQAReviewDriver.__new__(driver_module.ChemQAReviewDriver)
            driver.args = argparse.Namespace(role="proposer-1", lane_retry_budget=2, max_model_attempts=2)
            driver.workspace = Path(tmpdir)
            proposal_path = driver.workspace / transport.proposal_filename()

            attempt_prompts: list[list[str]] = []

            def fake_attempt_model_artifact(*, instructions, **_kwargs):
                attempt_prompts.append(list(instructions))
                if len(attempt_prompts) == 1:
                    proposal_path.write_text(
                        "\n".join(
                            [
                                "artifact_kind: candidate_submission",
                                "artifact_contract_version: react-reviewed-v2",
                                "phase: propose",
                                "owner: proposer-1",
                                "direct_answer: CN(C)CCF",
                                "summary: repeated stale candidate.",
                                "submission_trace:",
                                "- step: structural_reasoning",
                                "  status: success",
                                "  detail: repeated stale candidate.",
                            ]
                        ),
                        encoding="utf-8",
                    )
                else:
                    proposal_path.write_text(
                        "\n".join(
                            [
                                "artifact_kind: candidate_submission",
                                "artifact_contract_version: react-reviewed-v2",
                                "phase: propose",
                                "owner: proposer-1",
                                "direct_answer: NCCF",
                                "summary: revised candidate after duplicate rejection.",
                                "submission_trace:",
                                "- step: literature_alignment",
                                "  status: success",
                                "  detail: revised after duplicate rejection.",
                            ]
                        ),
                        encoding="utf-8",
                    )
                return proposal_path

            driver.attempt_model_artifact = fake_attempt_model_artifact
            submit_calls: list[str] = []

            def fake_submit_proposal(path: Path):
                submit_calls.append(path.read_text(encoding="utf-8"))
                if len(submit_calls) == 1:
                    raise driver_module.DriverError(
                        "Command failed (1): submit-proposal\nSTDOUT:\n\nSTDERR:\n"
                        "Proposal matches a prior submission from epoch 1: artifact_kind: candidate_submission"
                    )
                return {"ok": True}

            driver.submit_proposal = fake_submit_proposal
            driver.status = lambda: {
                "epoch": 2,
                "proposals": [
                    {
                        "epoch": 2,
                        "proposer": "proposer-1",
                        "status": "active",
                        "body": proposal_path.read_text(encoding="utf-8"),
                    }
                ],
            }
            progress_marks: list[str] = []
            driver.mark_progress = lambda: progress_marks.append("progress")

            driver_module.ChemQAReviewDriver.ensure_candidate_submission(
                driver,
                {"phase": "propose", "epoch": 2},
                {"phase": "propose", "action": "propose"},
            )

            self.assertEqual(2, len(attempt_prompts))
            self.assertEqual(2, len(submit_calls))
            self.assertIn("duplicate", "\n".join(attempt_prompts[1]).lower())
            self.assertEqual(["progress"], progress_marks)

    def test_sync_run_status_keeps_protocol_done_in_artifact_finalizing_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = driver_module.ChemQAReviewDriver.__new__(driver_module.ChemQAReviewDriver)
            driver.args = argparse.Namespace(team="chemqa-review-run-status", role="debate-coordinator")
            driver.lane_failures = {}
            driver.reviewer_exit_state = lambda: {}
            driver.repair_cycles_without_progress = 0
            driver.last_recovery_payload = {}
            driver.last_respawn_events = []
            driver.all_tasks = lambda: []
            driver.last_progress_change = 0.0
            driver.last_progress_key = ""
            store = driver_module.FileControlStore(Path(tmpdir))
            driver.store = store

            driver_module.ChemQAReviewDriver.sync_run_status(
                driver,
                {"status": "done", "terminal_state": "failed", "failure_reason": "engine stopped", "phase": "done"},
                {"action": "stop", "advance_ready": False, "message": "done"},
            )

            payload = json.loads((store.control / "run-status" / "chemqa-review-run-status.json").read_text(encoding="utf-8"))
            self.assertEqual("running", payload["status"])
            self.assertEqual("failed", payload["protocol_terminal_state"])
            self.assertEqual("finalizing", payload["artifact_flow_state"])
            self.assertEqual("running", payload["benchmark_terminal_state"])
            self.assertEqual("running", payload["terminal_state"])
            self.assertEqual("engine_terminal_failure", payload["terminal_reason_code"])
            self.assertEqual("engine stopped", payload["terminal_reason"])

    def test_emit_terminal_failure_writes_done_failed_reason_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = driver_module.ChemQAReviewDriver.__new__(driver_module.ChemQAReviewDriver)
            driver.args = argparse.Namespace(team="chemqa-review-terminal-failure", role="debate-coordinator")
            driver.terminal_failure_emitted = False
            driver.workspace = Path(tmpdir)
            driver.lane_failures = {"proposer-2": {"reason": "bad output"}}
            driver.repair_cycles_without_progress = 1
            driver.current_task = lambda: {"status": "in_progress"}
            driver.reviewer_exit_state = lambda: {"proposer-5": {"reason": "timeout"}}
            store = driver_module.FileControlStore(Path(tmpdir))
            driver.store = store

            with self.assertRaises(driver_module.TerminalFailure):
                driver_module.ChemQAReviewDriver.emit_terminal_failure(
                    driver,
                    reason="phase stagnation",
                    status_payload={"phase": "review", "status": "running"},
                    next_action_payload={"phase": "review", "action": "review"},
                    blockers=["missing review"],
                )

            payload = json.loads((store.control / "run-status" / "chemqa-review-terminal-failure.json").read_text(encoding="utf-8"))
            self.assertEqual("done", payload["status"])
            self.assertEqual("failed", payload["terminal_state"])
            self.assertEqual("terminal_failure", payload["terminal_reason_code"])
            self.assertEqual("phase stagnation", payload["terminal_reason"])


class RecoveryRespawnBudgetTest(unittest.TestCase):
    def test_respawn_actionable_roles_can_initialize_budget_state_while_iterating_registry(self) -> None:
        recoverer = recover_run.RunRecoverer.__new__(recover_run.RunRecoverer)
        recoverer.args = argparse.Namespace(team="chemqa-review-test-run", max_respawns_per_role_phase_signature=1)
        recoverer.actions = []
        recoverer.blockers = []
        recoverer.data_dir = ""
        recoverer.team_dir = lambda: Path("/tmp/team-dir")
        recoverer.workspace_for = lambda _role: Path("/tmp/workspace")
        recoverer.current_phase_signature = lambda: "propose-epoch-1"
        registry = {
            "proposer-1": {"pid": 0, "command": ["python3", "worker.py"]},
        }
        recoverer.load_spawn_registry = lambda: registry
        saved_payloads: list[dict[str, object]] = []
        recoverer.save_spawn_registry = lambda payload: saved_payloads.append(dict(payload))
        recoverer.next_action = lambda _role: {"action": "propose", "phase": "propose"}
        recoverer.role_process_is_running = lambda role, entry: False

        original_popen = recover_run.subprocess.Popen
        try:
            class FakeProc:
                pid = 12345

            recover_run.subprocess.Popen = lambda *args, **kwargs: FakeProc()
            changed = recover_run.RunRecoverer.respawn_actionable_roles(recoverer)
        finally:
            recover_run.subprocess.Popen = original_popen

        self.assertTrue(changed)
        self.assertEqual(["respawn-role proposer-1 pid=12345"], recoverer.actions)
        self.assertEqual(1, saved_payloads[-1]["_budget_state"]["respawns_by_role"]["proposer-1"])

    def test_respawn_actionable_roles_does_not_respawn_same_role_after_phase_budget_exhausted(self) -> None:
        recoverer = recover_run.RunRecoverer.__new__(recover_run.RunRecoverer)
        recoverer.args = argparse.Namespace(team="chemqa-review-test-run", max_respawns_per_role_phase_signature=1)
        recoverer.actions = []
        recoverer.blockers = []
        recoverer.data_dir = ""
        recoverer.team_dir = lambda: Path("/tmp/team-dir")
        recoverer.workspace_for = lambda _role: Path("/tmp/workspace")
        recoverer.current_phase_signature = lambda: "review-round-1"
        registry = {
            "_budget_state": {
                "phase_signature": "review-round-1",
                "respawns_by_role": {"proposer-2": 1},
            },
            "proposer-2": {"pid": 0, "command": ["python3", "worker.py"]},
        }
        recoverer.load_spawn_registry = lambda: registry
        saved_payloads: list[dict[str, object]] = []
        recoverer.save_spawn_registry = lambda payload: saved_payloads.append(payload)
        recoverer.next_action = lambda _role: {"action": "review", "phase": "review"}
        recoverer.role_process_is_running = lambda role, entry: False

        changed = recover_run.RunRecoverer.respawn_actionable_roles(recoverer)

        self.assertFalse(changed)
        self.assertEqual([], saved_payloads)


class RecoveryInvocationTest(unittest.TestCase):
    def test_run_recovery_cycle_uses_single_step_and_passes_respawn_budget(self) -> None:
        driver = driver_module.ChemQAReviewDriver.__new__(driver_module.ChemQAReviewDriver)
        driver.args = argparse.Namespace(team="chemqa-review-test-run", max_respawns_per_role_phase_signature=1)
        driver.skill_root = SKILL_ROOT
        driver.runtime_root = Path("/tmp/runtime")
        driver.workspace_root = Path("/tmp/workspaces")
        driver.data_dir = ""
        driver.last_recovery_payload = {}

        original_subprocess_run = driver_module.subprocess.run
        captured: dict[str, object] = {}
        try:
            def fake_run(command, env=None, check=False, capture_output=False, text=False):
                captured["command"] = list(command)
                return subprocess.CompletedProcess(command, 0, stdout='{"status":"running","blockers":[]}', stderr="")

            driver_module.subprocess.run = fake_run
            payload = driver_module.ChemQAReviewDriver.run_recovery_cycle(driver)
        finally:
            driver_module.subprocess.run = original_subprocess_run

        self.assertEqual({"status": "running", "blockers": []}, payload)
        command = captured["command"]
        assert isinstance(command, list)
        self.assertIn("--max-steps", command)
        self.assertEqual("1", command[command.index("--max-steps") + 1])
        self.assertIn("--max-respawns-per-role-phase-signature", command)
        self.assertEqual("1", command[command.index("--max-respawns-per-role-phase-signature") + 1])

    def test_finalize_protocol_payload_writes_completed_with_artifact_collection_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            driver = driver_module.ChemQAReviewDriver.__new__(driver_module.ChemQAReviewDriver)
            driver.args = argparse.Namespace(team="chemqa-review-finalize", role="debate-coordinator")
            driver.skill_root = root
            driver.workspace = root / "workspace"
            driver.workspace.mkdir(parents=True, exist_ok=True)
            driver.data_dir = ""
            driver.store = driver_module.FileControlStore(root)
            driver.team_dir = lambda: None
            driver.workspace_path = lambda name: driver.workspace / name

            protocol_payload = {
                "artifact_kind": "coordinator_protocol",
                "artifact_contract_version": "react-reviewed-v2",
                "terminal_state": "completed",
                "question": "Question: test?",
                "final_answer": {"direct_answer": "A"},
                "acceptance_status": "accepted",
                "review_completion_status": {"status": "complete"},
                "candidate_submission": {},
                "acceptance_decision": {"status": "accepted"},
                "submission_trace": [],
                "submission_cycles": [],
                "proposer_trajectory": {},
                "reviewer_trajectories": {},
                "review_statuses": {},
                "final_review_items": {},
                "overall_confidence": {"level": "high", "rationale": "ok"},
            }

            original_subprocess_run = driver_module.subprocess.run

            def fake_subprocess_run(*_args, **_kwargs):
                return subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps({"artifact_paths": {"qa_result": "/tmp/qa.json"}}), stderr="")

            driver_module.subprocess.run = fake_subprocess_run
            try:
                driver_module.ChemQAReviewDriver.finalize_protocol_payload(
                    driver,
                    protocol_payload=protocol_payload,
                    protocol_generation_mode="model",
                )
            finally:
                driver_module.subprocess.run = original_subprocess_run

            payload = json.loads((root / "control" / "run-status" / "chemqa-review-finalize.json").read_text(encoding="utf-8"))
            self.assertEqual("done", payload["status"])
            self.assertEqual("completed", payload["terminal_state"])
            self.assertEqual("ok", payload["artifact_collection"]["status"])


class RecoverySlotMappingTest(unittest.TestCase):
    def test_workspace_for_prefers_spawn_registry_slot_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_root:
            recoverer = recover_run.RunRecoverer.__new__(recover_run.RunRecoverer)
            recoverer.workspace_root = Path(workspace_root)
            recoverer.load_spawn_registry = lambda: {
                "proposer-2": {
                    "command": ["python3", "driver.py", "--slot", "debateB-2", "--team", "demo"],
                }
            }
            path = recover_run.RunRecoverer.workspace_for(recoverer, "proposer-2")
            self.assertEqual(Path(workspace_root) / "debateB-2", path)

    def test_debate_state_commands_prefer_virtualenv_python(self) -> None:
        original_virtual_env = os.environ.get("VIRTUAL_ENV")
        original_run = recover_run.subprocess.run
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_root = Path(tmpdir) / ".venv"
            python_path = venv_root / "bin" / "python"
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.write_text("", encoding="utf-8")
            recoverer = recover_run.RunRecoverer.__new__(recover_run.RunRecoverer)
            recoverer.data_dir = ""
            recoverer.debate_state_path = Path("/tmp/debate_state.py")
            os.environ["VIRTUAL_ENV"] = str(venv_root)

            calls: list[list[str]] = []

            def fake_run(command, **kwargs):
                calls.append(list(command))
                return subprocess.CompletedProcess(command, 0, stdout="{}\n", stderr="")

            recover_run.subprocess.run = fake_run
            try:
                payload = recover_run.RunRecoverer.debate_state_json(recoverer, "status", "--team", "demo", "--json")
            finally:
                recover_run.subprocess.run = original_run
                if original_virtual_env is None:
                    os.environ.pop("VIRTUAL_ENV", None)
                else:
                    os.environ["VIRTUAL_ENV"] = original_virtual_env

            self.assertEqual({}, payload)
            self.assertEqual([[str(python_path), "/tmp/debate_state.py", "status", "--team", "demo", "--json"]], calls)


class RecoveryScriptTest(unittest.TestCase):
    def run_cmd(self, env: dict[str, str], *command: str) -> str:
        result = subprocess.run(list(command), env=env, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise AssertionError(
                f"Command failed ({result.returncode}): {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        return result.stdout

    def test_recover_review_round_submits_existing_formal_reviews(self) -> None:
        debate_state = load_module(DEBATE_STATE_PATH, "debate_state_for_recovery_tests")
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as workspace_root:
            env = os.environ.copy()
            env["CLAWTEAM_DATA_DIR"] = tmpdir
            previous_data_dir = os.environ.get("CLAWTEAM_DATA_DIR")
            os.environ["CLAWTEAM_DATA_DIR"] = tmpdir
            try:
                config = debate_state.DebateConfig(
                    team_name="chemqa-review-recovery-team",
                    workflow="review-loop",
                    goal="Question: test?",
                    evidence_policy="strict",
                    proposer_count=5,
                    max_review_rounds=2,
                    max_rebuttal_rounds=1,
                    max_epochs=3,
                )
                debate_state.init_debate_state(config, reset=True)

                fixtures = Path(tmpdir) / "fixtures"
                fixtures.mkdir(parents=True, exist_ok=True)

                def write_fixture(name: str, text: str) -> Path:
                    path = fixtures / name
                    path.write_text(text, encoding="utf-8")
                    return path

                candidate_path = write_fixture(
                    "proposal-1.md",
                    "# Candidate\n\n## Direct answer\n6\n\n## Submission trace\n- retrieval: skipped\n",
                )
                self.run_cmd(env, TEST_PYTHON, str(DEBATE_STATE_PATH), "submit-proposal", "--team", config.team_name, "--agent", "proposer-1", "--file", str(candidate_path))
                for role in ("proposer-2", "proposer-3", "proposer-4", "proposer-5"):
                    path = write_fixture(f"{role}.md", transport.render_placeholder_proposal(role))
                    self.run_cmd(env, TEST_PYTHON, str(DEBATE_STATE_PATH), "submit-proposal", "--team", config.team_name, "--agent", role, "--file", str(path))
                self.run_cmd(env, TEST_PYTHON, str(DEBATE_STATE_PATH), "advance", "--team", config.team_name, "--agent", "debate-coordinator")

                workspaces = Path(workspace_root)
                for slot in ("debate-1", "debate-2", "debate-3", "debate-4", "debate-5"):
                    (workspaces / slot).mkdir(parents=True, exist_ok=True)

                (workspaces / "debate-2" / transport.review_filename("proposer-1")).write_text(
                    """# Review\n\n**artifact_kind:** formal_review\n**phase:** review\n**reviewer_lane:** proposer-2\n**target_owner:** proposer-1\n**target_kind:** candidate_submission\n**verdict:** blocking\n\nreview_items:\n- severity: medium\n  finding: Missing literature support\ncounts_for_acceptance: true\nsynthetic: false\n""".strip(),
                    encoding="utf-8",
                )
                for role in ("proposer-3", "proposer-4", "proposer-5"):
                    (workspaces / f"debate-{role.split('-')[-1]}" / transport.review_filename("proposer-1")).write_text(
                        "\n".join(
                            [
                                "artifact_kind: formal_review",
                                "phase: review",
                                f"reviewer_lane: {role}",
                                "target_owner: proposer-1",
                                "target_kind: candidate_submission",
                                "verdict: non_blocking",
                                "review_items:",
                                "- severity: none",
                                "  finding: acceptable",
                                "counts_for_acceptance: true",
                                "synthetic: false",
                            ]
                        ),
                        encoding="utf-8",
                    )

                args = argparse.Namespace(
                    skill_root=str(SKILL_ROOT),
                    team=config.team_name,
                    runtime_dir=str(DEBATE_STATE_PATH.parent),
                    workspace_root=str(workspaces),
                    max_steps=1,
                    json=False,
                )
                recoverer = recover_run.RunRecoverer(args)
                status_before = recoverer.status()
                self.assertEqual("review", status_before["phase"])
                changed = recoverer.recover_review(status_before)
                self.assertTrue(changed)
                status_after = recoverer.status()
                self.assertGreaterEqual(int(status_after["phase_progress"]["actual"] or 0), 16)
                self.assertTrue(status_after["phase_progress"]["complete"])
            finally:
                if previous_data_dir is None:
                    os.environ.pop("CLAWTEAM_DATA_DIR", None)
                else:
                    os.environ["CLAWTEAM_DATA_DIR"] = previous_data_dir

    def test_recover_propose_repairs_existing_candidate_file(self) -> None:
        debate_state = load_module(DEBATE_STATE_PATH, "debate_state_for_recovery_propose_tests")
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as workspace_root:
            env = os.environ.copy()
            env["CLAWTEAM_DATA_DIR"] = tmpdir
            previous_data_dir = os.environ.get("CLAWTEAM_DATA_DIR")
            os.environ["CLAWTEAM_DATA_DIR"] = tmpdir
            try:
                config = debate_state.DebateConfig(
                    team_name="chemqa-review-recover-propose",
                    workflow="chemqa-review",
                    goal="Question: test?",
                    evidence_policy="strict",
                    proposer_count=5,
                    max_review_rounds=2,
                    max_rebuttal_rounds=1,
                    max_epochs=2,
                )
                debate_state.init_debate_state(config, reset=True)

                workspaces = Path(workspace_root)
                (workspaces / "debate-1").mkdir(parents=True, exist_ok=True)
                proposal_path = workspaces / "debate-1" / transport.proposal_filename()
                proposal_path.write_text(
                    "\n".join(
                        [
                            "artifact_kind: candidate_submission",
                            "phase: propose",
                            "owner: proposer-1",
                            "direct_answer: B",
                            "reasoning: B is consistent with the stoichiometry and limiting reagent calculation.",
                        ]
                    ),
                    encoding="utf-8",
                )

                args = argparse.Namespace(
                    skill_root=str(SKILL_ROOT),
                    team=config.team_name,
                    runtime_dir=str(DEBATE_STATE_PATH.parent),
                    workspace_root=str(workspaces),
                    max_steps=1,
                    json=False,
                )
                recoverer = recover_run.RunRecoverer(args)
                status_before = recoverer.status()
                self.assertEqual("propose", status_before["phase"])
                changed = recoverer.recover_propose(status_before)
                self.assertTrue(changed)
                status_after = recoverer.status()
                self.assertEqual(1, len(status_after["proposals"]))
                repaired_text = proposal_path.read_text(encoding="utf-8")
                self.assertIn("summary:", repaired_text)
                self.assertIn("submission_trace:", repaired_text)
            finally:
                if previous_data_dir is None:
                    os.environ.pop("CLAWTEAM_DATA_DIR", None)
                else:
                    os.environ["CLAWTEAM_DATA_DIR"] = previous_data_dir

    def test_recover_propose_uses_captured_candidate_when_workspace_file_is_missing(self) -> None:
        debate_state = load_module(DEBATE_STATE_PATH, "debate_state_for_recovery_propose_capture_tests")
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as workspace_root:
            env = os.environ.copy()
            env["CLAWTEAM_DATA_DIR"] = tmpdir
            previous_data_dir = os.environ.get("CLAWTEAM_DATA_DIR")
            os.environ["CLAWTEAM_DATA_DIR"] = tmpdir
            try:
                config = debate_state.DebateConfig(
                    team_name="chemqa-review-recover-propose-capture",
                    workflow="chemqa-review",
                    goal="Question: test?",
                    evidence_policy="strict",
                    proposer_count=5,
                    max_review_rounds=2,
                    max_rebuttal_rounds=1,
                    max_epochs=2,
                )
                debate_state.init_debate_state(config, reset=True)

                workspaces = Path(workspace_root)
                (workspaces / "debate-1").mkdir(parents=True, exist_ok=True)
                capture_dir = Path(tmpdir) / "teams" / config.team_name / "artifacts" / "captures" / "proposer-1"
                capture_dir.mkdir(parents=True, exist_ok=True)
                capture_path = capture_dir / "proposal.captured.yaml"
                capture_path.write_text(
                    "\n".join(
                        [
                            "artifact_kind: candidate_submission",
                            "artifact_contract_version: react-reviewed-v2",
                            "phase: propose",
                            "owner: proposer-1",
                            "direct_answer: N",
                            "summary: recovered from stable capture",
                            "submission_trace:",
                            "- step: reasoning",
                            "  status: success",
                            "  detail: captured before workspace artifact disappeared.",
                        ]
                    ),
                    encoding="utf-8",
                )

                args = argparse.Namespace(
                    skill_root=str(SKILL_ROOT),
                    team=config.team_name,
                    runtime_dir=str(DEBATE_STATE_PATH.parent),
                    workspace_root=str(workspaces),
                    max_steps=1,
                    json=False,
                )
                recoverer = recover_run.RunRecoverer(args)
                status_before = recoverer.status()
                self.assertEqual("propose", status_before["phase"])
                changed = recoverer.recover_propose(status_before)
                self.assertTrue(changed)
                status_after = recoverer.status()
                self.assertEqual(1, len(status_after["proposals"]))
                proposal_entry = dict(status_after["proposals"][0])
                proposal_body = str(proposal_entry.get("body") or "")
                if not proposal_body.strip():
                    artifact = proposal_entry.get("artifact") or {}
                    for key in ("source_path", "archive_path"):
                        candidate = str(artifact.get(key) or "").strip()
                        if candidate:
                            proposal_body = Path(candidate).read_text(encoding="utf-8")
                            break
                self.assertIn("direct_answer: N", proposal_body)
                self.assertIn("summary: recovered from stable capture", proposal_body)
            finally:
                if previous_data_dir is None:
                    os.environ.pop("CLAWTEAM_DATA_DIR", None)
                else:
                    os.environ["CLAWTEAM_DATA_DIR"] = previous_data_dir

    def test_recover_propose_uses_archived_candidate_when_workspace_and_capture_are_missing(self) -> None:
        debate_state = load_module(DEBATE_STATE_PATH, "debate_state_for_recovery_propose_archive_tests")
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as workspace_root:
            env = os.environ.copy()
            env["CLAWTEAM_DATA_DIR"] = tmpdir
            previous_data_dir = os.environ.get("CLAWTEAM_DATA_DIR")
            os.environ["CLAWTEAM_DATA_DIR"] = tmpdir
            try:
                config = debate_state.DebateConfig(
                    team_name="chemqa-review-recover-propose-archive",
                    workflow="chemqa-review",
                    goal="Question: test?",
                    evidence_policy="strict",
                    proposer_count=5,
                    max_review_rounds=2,
                    max_rebuttal_rounds=1,
                    max_epochs=2,
                )
                debate_state.init_debate_state(config, reset=True)

                workspaces = Path(workspace_root)
                (workspaces / "debate-1").mkdir(parents=True, exist_ok=True)
                archive_dir = Path(tmpdir) / "teams" / config.team_name / "debate" / "artifacts" / "proposals" / "epoch-002"
                archive_dir.mkdir(parents=True, exist_ok=True)
                archive_path = archive_dir / "proposer-1.md"
                archive_path.write_text(
                    "\n".join(
                        [
                            "artifact_kind: candidate_submission",
                            "artifact_contract_version: react-reviewed-v2",
                            "phase: propose",
                            "owner: proposer-1",
                            "direct_answer: NCCF",
                            "summary: recovered from archived epoch candidate",
                            "submission_trace:",
                            "- step: structural_reasoning",
                            "  status: success",
                            "  detail: archived candidate remained valid after rollback.",
                        ]
                    ),
                    encoding="utf-8",
                )

                args = argparse.Namespace(
                    skill_root=str(SKILL_ROOT),
                    team=config.team_name,
                    runtime_dir=str(DEBATE_STATE_PATH.parent),
                    workspace_root=str(workspaces),
                    max_steps=1,
                    json=False,
                )
                recoverer = recover_run.RunRecoverer(args)
                status_before = recoverer.status()
                self.assertEqual("propose", status_before["phase"])
                changed = recoverer.recover_propose(status_before)
                self.assertTrue(changed)
                status_after = recoverer.status()
                self.assertEqual(1, len(status_after["proposals"]))
                proposal_entry = dict(status_after["proposals"][0])
                proposal_body = str(proposal_entry.get("body") or "")
                if not proposal_body.strip():
                    artifact = proposal_entry.get("artifact") or {}
                    for key in ("source_path", "archive_path"):
                        candidate = str(artifact.get(key) or "").strip()
                        if candidate:
                            proposal_body = Path(candidate).read_text(encoding="utf-8")
                            break
                self.assertIn("direct_answer: NCCF", proposal_body)
                self.assertIn("summary: recovered from archived epoch candidate", proposal_body)
            finally:
                if previous_data_dir is None:
                    os.environ.pop("CLAWTEAM_DATA_DIR", None)
                else:
                    os.environ["CLAWTEAM_DATA_DIR"] = previous_data_dir

    def test_recover_propose_duplicate_candidate_adds_blocker_instead_of_raising(self) -> None:
        debate_state = load_module(DEBATE_STATE_PATH, "debate_state_for_recovery_propose_duplicate_tests")
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as workspace_root:
            env = os.environ.copy()
            env["CLAWTEAM_DATA_DIR"] = tmpdir
            previous_data_dir = os.environ.get("CLAWTEAM_DATA_DIR")
            os.environ["CLAWTEAM_DATA_DIR"] = tmpdir
            try:
                config = debate_state.DebateConfig(
                    team_name="chemqa-review-recover-propose-duplicate",
                    workflow="chemqa-review",
                    goal="Question: test?",
                    evidence_policy="strict",
                    proposer_count=5,
                    max_review_rounds=2,
                    max_rebuttal_rounds=1,
                    max_epochs=2,
                )
                debate_state.init_debate_state(config, reset=True)

                fixtures = Path(tmpdir) / "fixtures"
                fixtures.mkdir(parents=True, exist_ok=True)
                prior_candidate_path = fixtures / "proposal.yaml"
                prior_candidate_text = transport.check_candidate_submission(
                    "\n".join(
                        [
                            "artifact_kind: candidate_submission",
                            "artifact_contract_version: react-reviewed-v2",
                            "phase: propose",
                            "owner: proposer-1",
                            "direct_answer: NCCF",
                            "summary: prior epoch candidate",
                            "submission_trace:",
                            "- step: structural_reasoning",
                            "  status: success",
                            "  detail: prior epoch reasoning.",
                        ]
                    ),
                    owner="proposer-1",
                ).normalized_text
                prior_candidate_path.write_text(
                    prior_candidate_text,
                    encoding="utf-8",
                )
                self.run_cmd(
                    env,
                    TEST_PYTHON,
                    str(DEBATE_STATE_PATH),
                    "submit-proposal",
                    "--team",
                    config.team_name,
                    "--agent",
                    "proposer-1",
                    "--file",
                    str(prior_candidate_path),
                )

                state_db = Path(tmpdir) / "teams" / config.team_name / "debate" / "state.db"
                with sqlite3.connect(state_db) as conn:
                    conn.execute("UPDATE meta SET value = '2' WHERE key = 'epoch'")
                    conn.execute("UPDATE meta SET value = 'propose' WHERE key = 'phase'")
                    conn.execute("UPDATE meta SET value = '0' WHERE key = 'review_round'")
                    conn.execute("UPDATE meta SET value = '0' WHERE key = 'rebuttal_round'")
                    conn.execute("UPDATE meta SET value = '[\"proposer-1\"]' WHERE key = 'phase_targets_json'")
                    conn.commit()

                workspaces = Path(workspace_root)
                (workspaces / "debate-1").mkdir(parents=True, exist_ok=True)
                capture_dir = Path(tmpdir) / "teams" / config.team_name / "artifacts" / "captures" / "proposer-1"
                capture_dir.mkdir(parents=True, exist_ok=True)
                capture_path = capture_dir / "proposal.captured.yaml"
                capture_path.write_text(prior_candidate_text, encoding="utf-8")

                args = argparse.Namespace(
                    skill_root=str(SKILL_ROOT),
                    team=config.team_name,
                    runtime_dir=str(DEBATE_STATE_PATH.parent),
                    workspace_root=str(workspaces),
                    max_steps=1,
                    json=False,
                )
                recoverer = recover_run.RunRecoverer(args)
                status_before = recoverer.status()
                self.assertEqual("propose", status_before["phase"])

                changed = recoverer.recover_propose(status_before)

                self.assertFalse(changed)
                self.assertEqual([], recoverer.actions)
                self.assertTrue(recoverer.blockers)
                self.assertIn("stale candidate proposal source", recoverer.blockers[0])
            finally:
                if previous_data_dir is None:
                    os.environ.pop("CLAWTEAM_DATA_DIR", None)
                else:
                    os.environ["CLAWTEAM_DATA_DIR"] = previous_data_dir

    def test_recover_propose_does_not_resubmit_stale_prior_epoch_candidate_after_epoch_increment(self) -> None:
        debate_state = load_module(DEBATE_STATE_PATH, "debate_state_for_recovery_propose_stale_epoch_tests")
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as workspace_root:
            env = os.environ.copy()
            env["CLAWTEAM_DATA_DIR"] = tmpdir
            previous_data_dir = os.environ.get("CLAWTEAM_DATA_DIR")
            os.environ["CLAWTEAM_DATA_DIR"] = tmpdir
            try:
                config = debate_state.DebateConfig(
                    team_name="chemqa-review-recover-propose-stale-epoch",
                    workflow="chemqa-review",
                    goal="Question: test?",
                    evidence_policy="strict",
                    proposer_count=5,
                    max_review_rounds=2,
                    max_rebuttal_rounds=1,
                    max_epochs=3,
                )
                debate_state.init_debate_state(config, reset=True)

                state_db = Path(tmpdir) / "teams" / config.team_name / "debate" / "state.db"
                with sqlite3.connect(state_db) as conn:
                    conn.execute("UPDATE meta SET value = '2' WHERE key = 'epoch'")
                    conn.execute("UPDATE meta SET value = 'propose' WHERE key = 'phase'")
                    conn.execute("UPDATE meta SET value = '0' WHERE key = 'review_round'")
                    conn.execute("UPDATE meta SET value = '0' WHERE key = 'rebuttal_round'")
                    conn.execute("UPDATE meta SET value = '[\"proposer-1\"]' WHERE key = 'phase_targets_json'")
                    conn.commit()

                fixtures = Path(tmpdir) / "fixtures"
                fixtures.mkdir(parents=True, exist_ok=True)
                prior_candidate_path = fixtures / "proposal.yaml"
                prior_candidate_text = transport.check_candidate_submission(
                    "\n".join(
                        [
                            "artifact_kind: candidate_submission",
                            "artifact_contract_version: react-reviewed-v2",
                            "phase: propose",
                            "owner: proposer-1",
                            "direct_answer: CN(C)CCF",
                            "summary: epoch-2 revised candidate",
                            "submission_trace:",
                            "- step: structural_reasoning",
                            "  status: success",
                            "  detail: epoch-2 candidate reasoning.",
                        ]
                    ),
                    owner="proposer-1",
                ).normalized_text
                prior_candidate_path.write_text(prior_candidate_text, encoding="utf-8")
                self.run_cmd(
                    env,
                    TEST_PYTHON,
                    str(DEBATE_STATE_PATH),
                    "submit-proposal",
                    "--team",
                    config.team_name,
                    "--agent",
                    "proposer-1",
                    "--file",
                    str(prior_candidate_path),
                )

                with sqlite3.connect(state_db) as conn:
                    conn.execute("UPDATE meta SET value = '3' WHERE key = 'epoch'")
                    conn.execute("UPDATE meta SET value = 'propose' WHERE key = 'phase'")
                    conn.execute("UPDATE meta SET value = '0' WHERE key = 'review_round'")
                    conn.execute("UPDATE meta SET value = '0' WHERE key = 'rebuttal_round'")
                    conn.execute("UPDATE meta SET value = '[\"proposer-1\"]' WHERE key = 'phase_targets_json'")
                    conn.commit()

                workspaces = Path(workspace_root)
                (workspaces / "debate-1").mkdir(parents=True, exist_ok=True)
                (workspaces / "debate-1" / transport.proposal_filename()).write_text(
                    prior_candidate_text,
                    encoding="utf-8",
                )
                capture_dir = Path(tmpdir) / "teams" / config.team_name / "artifacts" / "captures" / "proposer-1"
                capture_dir.mkdir(parents=True, exist_ok=True)
                (capture_dir / "proposal.captured.yaml").write_text(prior_candidate_text, encoding="utf-8")

                args = argparse.Namespace(
                    skill_root=str(SKILL_ROOT),
                    team=config.team_name,
                    runtime_dir=str(DEBATE_STATE_PATH.parent),
                    workspace_root=str(workspaces),
                    max_steps=1,
                    json=False,
                )
                recoverer = recover_run.RunRecoverer(args)
                recoverer.submit_proposal = lambda **_kwargs: self.fail(
                    "recover_propose should not resubmit a stale prior-epoch candidate after the run has advanced epochs"
                )
                status_before = recoverer.status()
                self.assertEqual("propose", status_before["phase"])
                self.assertEqual(3, int(status_before["epoch"]))

                changed = recoverer.recover_propose(status_before)

                self.assertFalse(changed)
                self.assertEqual([], recoverer.actions)
                self.assertTrue(recoverer.blockers)
            finally:
                if previous_data_dir is None:
                    os.environ.pop("CLAWTEAM_DATA_DIR", None)
                else:
                    os.environ["CLAWTEAM_DATA_DIR"] = previous_data_dir

    def test_recover_rebuttal_repairs_existing_rebuttal_file(self) -> None:
        debate_state = load_module(DEBATE_STATE_PATH, "debate_state_for_recovery_rebuttal_tests")
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as workspace_root:
            env = os.environ.copy()
            env["CLAWTEAM_DATA_DIR"] = tmpdir
            previous_data_dir = os.environ.get("CLAWTEAM_DATA_DIR")
            os.environ["CLAWTEAM_DATA_DIR"] = tmpdir
            try:
                config = debate_state.DebateConfig(
                    team_name="chemqa-review-recover-rebuttal",
                    workflow="chemqa-review",
                    goal="Question: test?",
                    evidence_policy="strict",
                    proposer_count=5,
                    max_review_rounds=2,
                    max_rebuttal_rounds=1,
                    max_epochs=2,
                )
                debate_state.init_debate_state(config, reset=True)

                fixtures = Path(tmpdir) / "fixtures"
                fixtures.mkdir(parents=True, exist_ok=True)
                proposal_path = fixtures / "proposal.yaml"
                proposal_path.write_text(
                    "\n".join(
                        [
                            "artifact_kind: candidate_submission",
                            "phase: propose",
                            "owner: proposer-1",
                            "direct_answer: A",
                            "summary: initial answer",
                            "submission_trace:",
                            "- step: reasoning",
                            "  status: success",
                            "  detail: initial trace",
                        ]
                    ),
                    encoding="utf-8",
                )
                self.run_cmd(env, TEST_PYTHON, str(DEBATE_STATE_PATH), "submit-proposal", "--team", config.team_name, "--agent", "proposer-1", "--file", str(proposal_path))
                self.run_cmd(env, TEST_PYTHON, str(DEBATE_STATE_PATH), "advance", "--team", config.team_name, "--agent", "debate-coordinator")

                for role in ("proposer-2", "proposer-3", "proposer-4", "proposer-5"):
                    review_path = fixtures / f"review-{role}.yaml"
                    review_path.write_text(
                        "\n".join(
                            [
                                "artifact_kind: formal_review",
                                "phase: review",
                                f"reviewer_lane: {role}",
                                "target_owner: proposer-1",
                                "target_kind: candidate_submission",
                                "verdict: blocking",
                                "summary: review summary",
                                "review_items:",
                                "- severity: high",
                                "  finding: needs revision",
                                "counts_for_acceptance: true",
                                "synthetic: false",
                            ]
                        ),
                        encoding="utf-8",
                    )
                    self.run_cmd(
                        env,
                        TEST_PYTHON,
                        str(DEBATE_STATE_PATH),
                        "submit-review",
                        "--team",
                        config.team_name,
                        "--agent",
                        role,
                        "--target",
                        "proposer-1",
                        "--blocking",
                        "yes",
                        "--file",
                        str(review_path),
                    )
                self.run_cmd(env, TEST_PYTHON, str(DEBATE_STATE_PATH), "advance", "--team", config.team_name, "--agent", "debate-coordinator")

                workspaces = Path(workspace_root)
                (workspaces / "debate-1").mkdir(parents=True, exist_ok=True)
                rebuttal_path = workspaces / "debate-1" / transport.rebuttal_filename()
                rebuttal_path.write_text(
                    "\n".join(
                        [
                            "artifact_kind: rebuttal",
                            "phase: rebuttal",
                            "owner: proposer-1",
                            "updated_direct_answer: Revised answer after corrections.",
                        ]
                    ),
                    encoding="utf-8",
                )

                args = argparse.Namespace(
                    skill_root=str(SKILL_ROOT),
                    team=config.team_name,
                    runtime_dir=str(DEBATE_STATE_PATH.parent),
                    workspace_root=str(workspaces),
                    max_steps=1,
                    json=False,
                )
                recoverer = recover_run.RunRecoverer(args)
                status_before = recoverer.status()
                self.assertEqual("rebuttal", status_before["phase"])
                changed = recoverer.recover_rebuttal(status_before)
                self.assertTrue(changed)
                status_after = recoverer.status()
                self.assertEqual(1, len(status_after["rebuttals"]))
                repaired_text = rebuttal_path.read_text(encoding="utf-8")
                self.assertIn("response_summary:", repaired_text)
            finally:
                if previous_data_dir is None:
                    os.environ.pop("CLAWTEAM_DATA_DIR", None)
                else:
                    os.environ["CLAWTEAM_DATA_DIR"] = previous_data_dir

    def test_write_run_status_done_failed_can_set_stalled_reason_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recoverer = recover_run.RunRecoverer.__new__(recover_run.RunRecoverer)
            recoverer.args = argparse.Namespace(team="chemqa-review-recover-status")
            recoverer.actions = []
            recoverer.blockers = []
            recoverer.store = recover_run.FileControlStore(Path(tmpdir))

            recover_run.RunRecoverer.write_run_status(
                recoverer,
                state={"phase": "review"},
                status="done",
                recovery_cycles_without_progress=1,
                progress_made=False,
                terminal_state="failed",
                terminal_reason_code="stalled",
                terminal_reason="Recovery stopped without reaching done.",
            )

            payload = json.loads((Path(tmpdir) / "control" / "run-status" / "chemqa-review-recover-status.json").read_text(encoding="utf-8"))
            self.assertEqual("done", payload["status"])
            self.assertEqual("failed", payload["terminal_state"])
            self.assertEqual("stalled", payload["terminal_reason_code"])

    def test_recover_run_stalled_single_step_does_not_publish_done_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recoverer = recover_run.RunRecoverer.__new__(recover_run.RunRecoverer)
            recoverer.args = argparse.Namespace(team="chemqa-review-recover-status", max_steps=1, json=False)
            recoverer.actions = []
            recoverer.blockers = []
            recoverer.store = recover_run.FileControlStore(Path(tmpdir))
            recoverer._phase_signature = lambda _state: "review:1"
            recoverer.status = lambda: {"status": "running", "phase": "review"}
            recoverer.repair_invalid_review_state = lambda _state: False
            recoverer.recover_propose = lambda _state: False
            recoverer.recover_review = lambda _state: False
            recoverer.recover_rebuttal = lambda _state: False
            recoverer.next_action = lambda _agent: {"action": "wait"}
            recoverer.respawn_actionable_roles = lambda: False

            exit_code = recover_run.RunRecoverer.run(recoverer)

            self.assertEqual(1, exit_code)
            payload = json.loads((Path(tmpdir) / "control" / "run-status" / "chemqa-review-recover-status.json").read_text(encoding="utf-8"))
            self.assertEqual("running", payload["status"])
            self.assertNotIn("terminal_state", payload)
            self.assertNotIn("terminal_reason_code", payload)


class SnapshotScriptTest(unittest.TestCase):
    def run_cmd(self, env: dict[str, str], *command: str) -> str:
        result = subprocess.run(list(command), env=env, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise AssertionError(
                f"Command failed ({result.returncode}): {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        return result.stdout

    def test_snapshot_reports_missing_submissions_and_reviews(self) -> None:
        debate_state = load_module(DEBATE_STATE_PATH, "debate_state_for_tests")
        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env["CLAWTEAM_DATA_DIR"] = tmpdir
            previous_data_dir = os.environ.get("CLAWTEAM_DATA_DIR")
            os.environ["CLAWTEAM_DATA_DIR"] = tmpdir
            try:
                config = debate_state.DebateConfig(
                    team_name="chemqa-review-test-team",
                    workflow="review-loop",
                    goal="Question: test?",
                    evidence_policy="strict",
                    proposer_count=5,
                    max_review_rounds=2,
                    max_rebuttal_rounds=1,
                    max_epochs=3,
                )
                debate_state.init_debate_state(config, reset=True)

                proposal_dir = Path(tmpdir) / "fixtures"
                proposal_dir.mkdir(parents=True, exist_ok=True)

                def write_file(name: str, text: str) -> Path:
                    path = proposal_dir / name
                    path.write_text(text, encoding="utf-8")
                    return path

                candidate_path = write_file(
                    "proposal-1.md",
                    "# Candidate\n\n## Direct answer\nA\n\n## Submission trace\n- retrieval: skipped\n",
                )
                self.run_cmd(env, TEST_PYTHON, str(DEBATE_STATE_PATH), "submit-proposal", "--team", config.team_name, "--agent", "proposer-1", "--file", str(candidate_path))

                snapshot = json.loads(
                    self.run_cmd(
                        env,
                        TEST_PYTHON,
                        str(SNAPSHOT_PATH),
                        "--skill-root",
                        str(SKILL_ROOT),
                        "--team",
                        config.team_name,
                        "--agent",
                        "proposer-1",
                        "--force",
                    )
                )
                self.assertEqual(["proposer-2", "proposer-3", "proposer-4", "proposer-5"], snapshot["missing_proposer_submissions"])
                self.assertEqual(0, snapshot["qualifying_candidate_reviews_count"])

                for role in ("proposer-2", "proposer-3", "proposer-4", "proposer-5"):
                    path = write_file(f"{role}.md", transport.render_placeholder_proposal(role))
                    self.run_cmd(env, TEST_PYTHON, str(DEBATE_STATE_PATH), "submit-proposal", "--team", config.team_name, "--agent", role, "--file", str(path))

                self.run_cmd(env, TEST_PYTHON, str(DEBATE_STATE_PATH), "advance", "--team", config.team_name, "--agent", "debate-coordinator")

                formal_review = write_file(
                    "review-proposer-1.md",
                    "\n".join(
                        [
                            "artifact_kind: formal_review",
                            "phase: review",
                            "reviewer_lane: proposer-2",
                            "target_owner: proposer-1",
                            "target_kind: candidate_submission",
                            "verdict: non_blocking",
                            "review_items:",
                            "- severity: none",
                            "  finding: acceptable",
                            "counts_for_acceptance: true",
                            "synthetic: false",
                        ]
                    ),
                )
                self.run_cmd(
                    env,
                    TEST_PYTHON,
                    str(DEBATE_STATE_PATH),
                    "submit-review",
                    "--team",
                    config.team_name,
                    "--agent",
                    "proposer-2",
                    "--target",
                    "proposer-1",
                    "--blocking",
                    "no",
                    "--file",
                    str(formal_review),
                )

                snapshot = json.loads(
                    self.run_cmd(
                        env,
                        TEST_PYTHON,
                        str(SNAPSHOT_PATH),
                        "--skill-root",
                        str(SKILL_ROOT),
                        "--team",
                        config.team_name,
                        "--agent",
                        "proposer-2",
                        "--force",
                    )
                )
                self.assertEqual(1, snapshot["qualifying_candidate_reviews_count"])
                self.assertEqual(["proposer-3", "proposer-4", "proposer-5"], snapshot["missing_required_reviewer_lanes"])
            finally:
                if previous_data_dir is None:
                    os.environ.pop("CLAWTEAM_DATA_DIR", None)
                else:
                    os.environ["CLAWTEAM_DATA_DIR"] = previous_data_dir

    def test_load_terminal_status_accepts_new_done_failed_and_legacy_terminal_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = driver_module.FileControlStore(Path(tmpdir))
            store.update_run_status(
                "run-new",
                {"run_id": "run-new", "status": "done", "terminal_state": "failed", "terminal_reason_code": "stalled"},
            )
            store.update_run_status(
                "run-legacy",
                {"run_id": "run-legacy", "status": "terminal_failure"},
            )

            payload_new = json.loads((Path(tmpdir) / "control" / "run-status" / "run-new.json").read_text(encoding="utf-8"))
            payload_legacy = json.loads((Path(tmpdir) / "control" / "run-status" / "run-legacy.json").read_text(encoding="utf-8"))
            snapshot_module_new = load_module(SNAPSHOT_PATH, "chemqa_review_state_snapshot_runtime_test")
            snapshot_module_legacy = load_module(SNAPSHOT_PATH, "chemqa_review_state_snapshot_runtime_test_legacy")
            self.assertEqual(
                payload_new,
                snapshot_module_new.load_terminal_status(store, "run-new"),
            )
            self.assertEqual(
                payload_legacy,
                snapshot_module_legacy.load_terminal_status(store, "run-legacy"),
            )

    def test_write_run_status_running_does_not_emit_terminal_failure_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recoverer = recover_run.RunRecoverer.__new__(recover_run.RunRecoverer)
            recoverer.args = argparse.Namespace(team="chemqa-review-recover-running")
            recoverer.actions = ["respawn-role proposer-4 pid=123"]
            recoverer.blockers = ["missing formal review artifact"]
            recoverer.store = recover_run.FileControlStore(Path(tmpdir))

            recover_run.RunRecoverer.write_run_status(
                recoverer,
                state={"phase": "review", "review_round": 1},
                status="running",
                recovery_cycles_without_progress=1,
                progress_made=False,
                terminal_state="failed",
                terminal_reason_code="stalled",
                terminal_reason="should not be persisted for running status",
            )

            payload = json.loads((Path(tmpdir) / "control" / "run-status" / "chemqa-review-recover-running.json").read_text(encoding="utf-8"))
            self.assertEqual("running", payload["status"])
            self.assertEqual("review", payload["phase"])
            self.assertNotIn("terminal_state", payload)
            self.assertNotIn("terminal_reason_code", payload)
            self.assertNotIn("terminal_reason", payload)

    def test_run_leaves_run_status_running_when_single_step_recovery_stalls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recoverer = recover_run.RunRecoverer.__new__(recover_run.RunRecoverer)
            recoverer.args = argparse.Namespace(team="chemqa-review-recover-loop", max_steps=1, json=False)
            recoverer.actions = []
            recoverer.blockers = ["missing formal review artifact for proposer-4->proposer-1"]
            recoverer.store = recover_run.FileControlStore(Path(tmpdir))
            recoverer._phase_signature = lambda state: "review-1"
            recoverer.status = lambda: {
                "status": "running",
                "phase": "review",
                "review_round": 1,
                "rebuttal_round": 0,
                "phase_progress": {"kind": "review", "complete": False, "actual": 3, "expected": 4},
            }
            recoverer.repair_invalid_review_state = lambda _state: False
            recoverer.recover_propose = lambda _state: False
            recoverer.recover_review = lambda _state: False
            recoverer.recover_rebuttal = lambda _state: False
            recoverer.next_action = lambda _agent: {"action": "wait"}
            recoverer.advance = lambda: None
            recoverer.respawn_actionable_roles = lambda: False

            exit_code = recover_run.RunRecoverer.run(recoverer)

            self.assertEqual(1, exit_code)
            payload = json.loads((Path(tmpdir) / "control" / "run-status" / "chemqa-review-recover-loop.json").read_text(encoding="utf-8"))
            self.assertEqual("running", payload["status"])
            self.assertEqual("review", payload["phase"])
            self.assertEqual(0, payload["recovery_cycles_without_progress"])
            self.assertNotIn("terminal_state", payload)
            self.assertNotIn("terminal_reason_code", payload)


if __name__ == "__main__":
    unittest.main()
