#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import tempfile
from pathlib import Path

try:
    from workspace import runtime_paths
except ModuleNotFoundError:  # pragma: no cover - direct test execution fallback
    import runtime_paths

DEBATE_STATE = str(runtime_paths.clawteam_home / "debateclaw" / "bin" / "debate_state.py")
RECOVER_RUN = str(runtime_paths.skills_root / "chemqa-review" / "scripts" / "recover_run.py")
SKILL_ROOT = str(runtime_paths.skills_root / "chemqa-review")

PROPOSAL_BODY = """artifact_kind: candidate_submission
artifact_contract_version: react-reviewed-v2
phase: propose
owner: proposer-1
direct_answer: initial candidate
summary: initial candidate summary
submission_trace:
  - step: local-calc
    status: success
    detail: placeholder
"""

REVIEW_BODIES = {
    "proposer-2": """artifact_kind: formal_review
artifact_contract_version: react-reviewed-v2
phase: review
reviewer_lane: proposer-2
target_owner: proposer-1
target_kind: candidate_submission
verdict: non_blocking
summary: search coverage is not the issue
review_items:
  - item_id: search-coverage-1
    severity: low
    finding: no retrieval gap
    requested_change: none
counts_for_acceptance: true
synthetic: false
""",
    "proposer-3": """artifact_kind: formal_review
artifact_contract_version: react-reviewed-v2
phase: review
reviewer_lane: proposer-3
target_owner: proposer-1
target_kind: candidate_submission
verdict: blocking
summary: evidence trace issue
review_items:
  - item_id: trace-1
    severity: high
    finding: missing anchor for the governing formula
    requested_change: justify or remove the unsupported formula
counts_for_acceptance: true
synthetic: false
""",
    "proposer-4": """artifact_kind: formal_review
artifact_contract_version: react-reviewed-v2
phase: review
reviewer_lane: proposer-4
target_owner: proposer-1
target_kind: candidate_submission
verdict: blocking
summary: reasoning consistency issue
review_items:
  - item_id: reasoning-1
    severity: high
    finding: the derivation is dimensionally inconsistent
    requested_change: replace it with a dimensionally consistent derivation
counts_for_acceptance: true
synthetic: false
""",
    "proposer-5": """artifact_kind: formal_review
artifact_contract_version: react-reviewed-v2
phase: review
reviewer_lane: proposer-5
target_owner: proposer-1
target_kind: candidate_submission
verdict: blocking
summary: counterevidence issue
review_items:
  - item_id: counterevidence-1
    severity: high
    finding: the direct rates may already absorb wavelength dependence
    requested_change: explain why the direct rates are not being double counted
counts_for_acceptance: true
synthetic: false
""",
}

REBUTTAL_BODY = """artifact_kind: rebuttal
artifact_contract_version: react-reviewed-v2
phase: rebuttal
owner: proposer-1
concede: true
response_summary: conceding the failed candidate after blocking review
response_items:
  - item_id: reasoning-1
    severity: high
    finding: accepted the dimensional inconsistency
updated_direct_answer: revised answer needed
"""


def run_cmd(args: list[str], *, env: dict[str, str], expect_json: bool = False, check: bool = True) -> dict | str:
    result = subprocess.run(args, env=env, check=False, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, args, output=result.stdout, stderr=result.stderr)
    stdout = result.stdout.strip()
    if expect_json:
        return json.loads(stdout)
    return stdout


class TeamHarness:
    def __init__(self, *, data_dir: str, team: str) -> None:
        self.data_dir = data_dir
        self.team = team
        self.env = os.environ.copy()
        self.env["CLAWTEAM_DATA_DIR"] = data_dir
        self.tmp = Path(data_dir) / "fixtures"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.proposal_file = self.tmp / f"{team}-proposal.yaml"
        self.rebuttal_file = self.tmp / f"{team}-rebuttal.yaml"
        self.review_files: dict[str, Path] = {}
        self.cycle_index = 0

    def init(self, *, max_epochs: int) -> None:
        run_cmd(
            [
                "python3",
                DEBATE_STATE,
                "init",
                "--team",
                self.team,
                "--workflow",
                "chemqa-review",
                "--goal",
                "epoch loop validation",
                "--evidence-policy",
                "test policy",
                "--proposer-count",
                "5",
                "--max-review-rounds",
                "3",
                "--max-rebuttal-rounds",
                "2",
                "--max-epochs",
                str(max_epochs),
                "--reset",
            ],
            env=self.env,
        )
        self.proposal_file.write_text(PROPOSAL_BODY, encoding="utf-8")
        self.rebuttal_file.write_text(REBUTTAL_BODY, encoding="utf-8")
        for reviewer, body in REVIEW_BODIES.items():
            path = self.tmp / f"{self.team}-{reviewer}.yaml"
            path.write_text(body, encoding="utf-8")
            self.review_files[reviewer] = path

    def status(self) -> dict:
        return run_cmd(["python3", DEBATE_STATE, "status", "--team", self.team, "--json"], env=self.env, expect_json=True)

    def next_action(self, agent: str) -> dict:
        return run_cmd(["python3", DEBATE_STATE, "next-action", "--team", self.team, "--agent", agent, "--json"], env=self.env, expect_json=True)

    def advance(self) -> dict:
        return run_cmd(["python3", DEBATE_STATE, "advance", "--team", self.team, "--agent", "debate-coordinator", "--json"], env=self.env, expect_json=True)

    def submit_epoch_failure_cycle(self) -> None:
        self.cycle_index += 1
        proposal_body = PROPOSAL_BODY.replace("initial candidate", f"candidate epoch {self.cycle_index}")
        self.proposal_file.write_text(proposal_body, encoding="utf-8")
        run_cmd(["python3", DEBATE_STATE, "submit-proposal", "--team", self.team, "--agent", "proposer-1", "--file", str(self.proposal_file)], env=self.env)
        adv = self.advance()
        assert adv["phase"] == "review", adv
        for reviewer, blocking in (("proposer-2", "no"), ("proposer-3", "yes"), ("proposer-4", "yes"), ("proposer-5", "yes")):
            run_cmd(
                [
                    "python3",
                    DEBATE_STATE,
                    "submit-review",
                    "--team",
                    self.team,
                    "--agent",
                    reviewer,
                    "--target",
                    "proposer-1",
                    "--blocking",
                    blocking,
                    "--file",
                    str(self.review_files[reviewer]),
                ],
                env=self.env,
            )
        adv = self.advance()
        assert adv["phase"] == "rebuttal", adv
        run_cmd(
            [
                "python3",
                DEBATE_STATE,
                "submit-rebuttal",
                "--team",
                self.team,
                "--agent",
                "proposer-1",
                "--file",
                str(self.rebuttal_file),
                "--concede",
            ],
            env=self.env,
        )


def assert_epoch_restarts_and_limits(tmpdir: str) -> None:
    team = TeamHarness(data_dir=tmpdir, team="chemqa-max-epochs")
    team.init(max_epochs=3)
    status = team.status()
    assert status["max_epochs"] == 3, status

    team.submit_epoch_failure_cycle()
    adv = team.advance()
    assert adv["phase"] == "propose", adv
    assert adv["epoch"] == 2, adv

    proposer_next = team.next_action("proposer-1")
    assert proposer_next["action"] == "propose", proposer_next
    revision_context = proposer_next.get("revision_context")
    assert revision_context, proposer_next
    assert revision_context["source_epoch"] == 1, revision_context
    required_ids = {item.get("item_id") for item in revision_context["required_revision_items"]}
    assert {"trace-1", "reasoning-1", "counterevidence-1"}.issubset(required_ids), revision_context

    team.submit_epoch_failure_cycle()
    adv = team.advance()
    assert adv["phase"] == "propose", adv
    assert adv["epoch"] == 3, adv

    team.submit_epoch_failure_cycle()
    adv = team.advance()
    assert adv["phase"] == "done", adv
    assert adv["terminal_state"] == "failed", adv
    assert "max_epochs_exhausted_after_candidate_failures" in adv["failure_reason"], adv
    status = team.status()
    assert status["terminal_state"] == "failed", status
    assert status["epoch"] == 3, status


def assert_invalid_review_guard_and_recover(tmpdir: str) -> None:
    team = TeamHarness(data_dir=tmpdir, team="chemqa-invalid-review")
    team.init(max_epochs=2)

    db_path = Path(tmpdir) / "teams" / team.team / "debate" / "state.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE meta SET value='review' WHERE key='phase'")
        conn.execute("UPDATE meta SET value='2' WHERE key='review_round'")
        conn.execute("UPDATE meta SET value='[\"proposer-1\"]' WHERE key='phase_targets_json'")
        conn.commit()

    reviewer_next = team.next_action("proposer-2")
    assert reviewer_next["action"] == "wait", reviewer_next
    assert reviewer_next.get("state_issue") == "no_active_candidate_review_target", reviewer_next

    recover_payload = run_cmd(
        [
            "python3",
            RECOVER_RUN,
            "--skill-root",
            SKILL_ROOT,
            "--team",
            team.team,
            "--max-steps",
            "1",
            "--json",
        ],
        env=team.env,
        expect_json=True,
        check=False,
    )
    assert any("repair-invalid-review-state -> epoch 2 propose" in action for action in recover_payload.get("actions", [])), recover_payload
    status = team.status()
    assert status["phase"] == "propose", status
    assert status["epoch"] == 2, status


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="chemqa-epoch-tests-") as tmpdir:
        assert_epoch_restarts_and_limits(tmpdir)
        assert_invalid_review_guard_and_recover(tmpdir)
        print(json.dumps({"ok": True, "tempdir": tmpdir}, ensure_ascii=False))
