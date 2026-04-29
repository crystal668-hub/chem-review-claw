#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from bundle_common import default_runtime_dir, dump_json, load_module_from_path, resolve_clawteam_executable, resolve_skill_root
from bundle_common import resolve_python_interpreter
from chemqa_artifact_flow import finalize_failure, resolve_answer_kind
from chemqa_review_artifacts import (
    CANDIDATE_OWNER,
    ROLE_TO_SEMANTIC_ROLE,
    apply_forced_missing_review_completion,
    blocking_flag_for_review,
    build_protocol_from_summary,
    check_candidate_submission,
    check_formal_review,
    check_protocol,
    check_rebuttal,
    check_transport_review,
    coordinator_protocol_filename,
    current_proposal,
    is_reviewer_role,
    liveness_summary,
    missing_required_reviewer_lanes,
    pretty_json,
    proposal_filename,
    proposal_is_transport_placeholder,
    qualifying_candidate_reviews,
    render_placeholder_proposal,
    render_terminal_failure,
    render_transport_review,
    review_exists,
    review_filename,
    rebuttal_exists,
    rebuttal_filename,
    terminal_failure_filename,
)
from control_store import FileControlStore

POLL_SECONDS_DEFAULT = 20
STALE_TIMEOUT_SECONDS_DEFAULT = 300
MODEL_ATTEMPTS_DEFAULT = 1
CANDIDATE_MODEL_TIMEOUT_SECONDS_DEFAULT = 240
REVIEW_MODEL_TIMEOUT_SECONDS_DEFAULT = 420
REBUTTAL_MODEL_TIMEOUT_SECONDS_DEFAULT = 300
COORDINATOR_MODEL_TIMEOUT_SECONDS_DEFAULT = 300
SUBPROCESS_TIMEOUT_GRACE_SECONDS_DEFAULT = 30
RESPAWN_COOLDOWN_SECONDS_DEFAULT = 120
LANE_RETRY_BUDGET_DEFAULT = 2
PHASE_REPAIR_BUDGET_DEFAULT = 1
MAX_RESPAWNS_PER_ROLE_PHASE_SIGNATURE_DEFAULT = 1
BLOCKER_FILENAME = "chemqa_review_blocker.json"
CANDIDATE_CAPTURE_FILENAME = "proposal.captured.yaml"
CANDIDATE_CAPTURE_POLL_SECONDS = 0.5


def current_python() -> str:
    return resolve_python_interpreter()


def load_cleanroom_runtime_lease_module(skill_root: Path):
    module_path = skill_root.parent / "benchmark-cleanroom" / "scripts" / "runtime_lease.py"
    if not module_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("benchmark_cleanroom_runtime_lease", module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


class DriverError(RuntimeError):
    pass


class TerminalFailure(RuntimeError):
    pass


@dataclass
class TurnOutcome:
    returncode: int | None
    stop_reason: str = ""
    timed_out: bool = False
    aborted: bool = False
    hard_error: str = ""
    transcript_path: str = ""
    tool_call_count: int = 0
    assistant_text_tail: str = ""
    stdout_preview: str = ""
    stderr_preview: str = ""

    def as_payload(self) -> dict[str, Any]:
        return {
            "returncode": self.returncode,
            "stop_reason": self.stop_reason,
            "timed_out": self.timed_out,
            "aborted": self.aborted,
            "hard_error": self.hard_error,
            "transcript_path": self.transcript_path,
            "tool_call_count": self.tool_call_count,
            "assistant_text_tail": self.assistant_text_tail,
            "stdout_preview": self.stdout_preview,
            "stderr_preview": self.stderr_preview,
        }


@dataclass
class ArtifactOutcome:
    state: str
    filename: str
    path: str
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    normalized_text: str = ""
    changed_since_turn: bool = False
    classification: str = ""

    def as_payload(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "filename": self.filename,
            "path": self.path,
            "validation_errors": list(self.validation_errors),
            "validation_warnings": list(self.validation_warnings),
            "normalized_text": self.normalized_text,
            "changed_since_turn": self.changed_since_turn,
            "classification": self.classification,
        }


@dataclass
class PhaseAttemptState:
    role: str
    phase: str
    artifact_kind: str
    turn_index: int
    max_phase_turns: int
    classification: str
    last_turn_outcome: TurnOutcome | None = None
    last_artifact_outcome: ArtifactOutcome | None = None
    last_feedback: str = ""

    def as_payload(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "phase": self.phase,
            "artifact_kind": self.artifact_kind,
            "turn_index": self.turn_index,
            "max_phase_turns": self.max_phase_turns,
            "classification": self.classification,
            "last_turn": self.last_turn_outcome.as_payload() if self.last_turn_outcome is not None else None,
            "last_artifact": self.last_artifact_outcome.as_payload() if self.last_artifact_outcome is not None else None,
            "last_feedback": self.last_feedback,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChemQA-specific OpenClaw debate driver.")
    parser.add_argument("--skill-root", default=str(Path(__file__).resolve().parents[1]), help="chemqa-review skill root")
    parser.add_argument("--team", required=True, help="Debate team / run id")
    parser.add_argument("--role", required=True, choices=sorted(ROLE_TO_SEMANTIC_ROLE), help="ChemQA role name")
    parser.add_argument("--slot", required=True, help="OpenClaw slot id")
    parser.add_argument("--session-id", required=True, help="Explicit OpenClaw session id")
    parser.add_argument("--env-file", default=str(Path.home() / ".openclaw" / ".env"))
    parser.add_argument("--config-file", help="Explicit OpenClaw config path for this run")
    parser.add_argument("--runtime-dir", help="Path to deployed DebateClaw runtime helpers")
    parser.add_argument("--data-dir", help="Explicit CLAWTEAM_DATA_DIR override")
    parser.add_argument("--lease-dir", help="Optional benchmark cleanroom lease directory")
    parser.add_argument("-p", "--prompt", help="Initial ClawTeam task prompt")
    parser.add_argument("-m", "--message", help="Initial OpenClaw-compatible message")
    parser.add_argument("--thinking", choices=("off", "minimal", "low", "medium", "high", "xhigh"))
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS_DEFAULT)
    parser.add_argument("--stale-timeout-seconds", type=int, default=STALE_TIMEOUT_SECONDS_DEFAULT)
    parser.add_argument("--max-model-attempts", type=int, default=MODEL_ATTEMPTS_DEFAULT)
    parser.add_argument("--model-timeout-seconds", type=int, help="Override all model-turn timeouts")
    parser.add_argument("--candidate-timeout-seconds", type=int, help="Override candidate-submission model timeout")
    parser.add_argument("--review-timeout-seconds", type=int, help="Override formal-review model timeout")
    parser.add_argument("--rebuttal-timeout-seconds", type=int, help="Override rebuttal model timeout")
    parser.add_argument("--coordinator-timeout-seconds", type=int, help="Override terminal coordinator model timeout")
    parser.add_argument("--subprocess-timeout-grace-seconds", type=int, default=SUBPROCESS_TIMEOUT_GRACE_SECONDS_DEFAULT)
    parser.add_argument("--respawn-cooldown-seconds", type=int, default=RESPAWN_COOLDOWN_SECONDS_DEFAULT)
    parser.add_argument("--lane-retry-budget", type=int, default=LANE_RETRY_BUDGET_DEFAULT)
    parser.add_argument("--phase-repair-budget", type=int, default=PHASE_REPAIR_BUDGET_DEFAULT)
    parser.add_argument(
        "--max-respawns-per-role-phase-signature",
        type=int,
        default=MAX_RESPAWNS_PER_ROLE_PHASE_SIGNATURE_DEFAULT,
    )
    return parser.parse_args()


class ChemQAReviewDriver:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.skill_root = resolve_skill_root(args.skill_root)
        self.cleanroom_runtime_lease = load_cleanroom_runtime_lease_module(self.skill_root)
        self.store = FileControlStore(self.skill_root)
        self.runtime_root = Path(args.runtime_dir).expanduser().resolve() if args.runtime_dir else default_runtime_dir()
        self.debate_state_path = self.runtime_root / "debate_state.py"
        self.base_wrapper_path = self.runtime_root / "openclaw_debate_agent.py"
        if not self.debate_state_path.is_file():
            raise SystemExit(f"Missing DebateClaw runtime helper: {self.debate_state_path}")
        if not self.base_wrapper_path.is_file():
            raise SystemExit(f"Missing DebateClaw OpenClaw wrapper: {self.base_wrapper_path}")

        runtime_root_str = str(self.runtime_root)
        if runtime_root_str not in sys.path:
            sys.path.insert(0, runtime_root_str)
        wrapper_module = load_module_from_path("chemqa_driver_openclaw_wrapper", self.base_wrapper_path)
        explicit_config = Path(args.config_file).expanduser().resolve() if args.config_file else None
        wrapper_module.reset_slot_workspace_if_session_id_changed(args.slot, args.session_id, config_path=explicit_config)
        wrapper_module.reset_slot_main_session_if_session_id_changed(
            args.slot,
            args.session_id,
            config_path=explicit_config,
        )
        self.workspace = wrapper_module.resolve_slot_workspace(args.slot, config_path=explicit_config)
        self.workspace_root = self.workspace.parent

        self.initial_prompt = (args.message or args.prompt or "").strip()
        self.initial_prompt_used = False
        self.last_progress_key = ""
        self.last_progress_change = time.time()
        self.last_blocker_report = 0.0
        arg_data_dir = (args.data_dir or "").strip()
        env_data_dir = os.environ.get("CLAWTEAM_DATA_DIR", "").strip()
        fallback_data_dir = str((self.skill_root / "generated" / "clawteam-data").resolve())
        self.data_dir = arg_data_dir or env_data_dir or fallback_data_dir
        try:
            self.clawteam_executable = resolve_clawteam_executable(env=self.env)
        except FileNotFoundError as exc:
            raise SystemExit("Missing clawteam executable in PATH or fallback locations.") from exc
        self.lane_failures: dict[str, dict[str, Any]] = {}
        self.reviewer_exits: dict[str, dict[str, Any]] = {}
        self.repair_cycles_without_progress = 0
        self.last_repair_signature = ""
        self.terminal_failure_emitted = False
        self.last_recovery_payload: dict[str, Any] = {}
        self.last_respawn_events: list[dict[str, Any]] = []
        self.last_respawn_attempt_at: dict[str, float] = {}
        self.last_turn_outcome: TurnOutcome | None = None
        self.current_role_phase_state: PhaseAttemptState | None = None
        self.cleanroom_lease_handle = self._open_cleanroom_lease()

    def _lease_dir(self) -> str:
        return (self.args.lease_dir or os.environ.get("BENCHMARK_CLEANROOM_LEASE_DIR", "")).strip()

    def _run_id_for_lease(self) -> str:
        return os.environ.get("BENCHMARK_CLEANROOM_RUN_ID", "").strip() or self.args.team

    def _open_cleanroom_lease(self):
        if self.cleanroom_runtime_lease is None:
            return None
        lease_dir = self._lease_dir()
        if not lease_dir:
            return None
        session_id = str(self.args.session_id or "").strip()
        if not session_id:
            return None
        handle = self.cleanroom_runtime_lease.open_lease(
            lease_dir,
            run_id=self._run_id_for_lease(),
            role=self.args.role,
            slot=self.args.slot,
            session_id=session_id,
        )
        handle.write(
            run_id=self._run_id_for_lease(),
            role=self.args.role,
            slot=self.args.slot,
            session_id=session_id,
            status="starting",
            cwd=self.workspace,
            home=os.environ.get("HOME", ""),
            extra={"component": "chemqa_review_openclaw_driver"},
        )
        return handle

    def update_cleanroom_lease(self, status: str, extra: dict[str, Any] | None = None) -> None:
        if self.cleanroom_lease_handle is None:
            return
        self.cleanroom_lease_handle.write(
            run_id=self._run_id_for_lease(),
            role=self.args.role,
            slot=self.args.slot,
            session_id=str(self.args.session_id or ""),
            status=status,
            cwd=self.workspace,
            home=os.environ.get("HOME", ""),
            extra={"component": "chemqa_review_openclaw_driver", **dict(extra or {})},
        )

    def remove_cleanroom_lease(self) -> None:
        if self.cleanroom_lease_handle is None:
            return
        self.cleanroom_lease_handle.remove()

    @property
    def env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.data_dir:
            env["CLAWTEAM_DATA_DIR"] = self.data_dir
        return env

    def run(self) -> int:
        self.ensure_task_status("in_progress")
        self.update_cleanroom_lease("running")
        try:
            if self.args.role == "debate-coordinator":
                exit_code = self.run_coordinator_loop()
            else:
                exit_code = self.run_worker_loop()
            self.update_cleanroom_lease("completed", {"exit_code": exit_code})
            return exit_code
        except TerminalFailure:
            self.update_cleanroom_lease("terminal_failure")
            self.save_session()
            self.ensure_task_status("completed")
            return 2
        except DriverError as exc:
            self.update_cleanroom_lease("driver_error", {"error": str(exc)})
            self.emit_driver_error_marker(exc)
            raise
        finally:
            self.remove_cleanroom_lease()

    def current_task(self) -> dict[str, Any] | None:
        command = [self.clawteam_executable]
        if self.data_dir:
            command.extend(["--data-dir", self.data_dir])
        command.extend(["--json", "task", "list", self.args.team, "--owner", self.args.role])
        result = subprocess.run(command, env=self.env, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise DriverError(f"Failed to query task list for {self.args.role}: {result.stdout}\n{result.stderr}")
        tasks = json.loads(result.stdout or "[]")
        if not tasks:
            return None
        return dict(tasks[0])

    def ensure_task_status(self, status: str) -> None:
        task = self.current_task()
        if not task:
            return
        if str(task.get("status")) == status:
            return
        command = [self.clawteam_executable]
        if self.data_dir:
            command.extend(["--data-dir", self.data_dir])
        command.extend(["task", "update", self.args.team, str(task["id"]), "--status", status])
        result = subprocess.run(command, env=self.env, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise DriverError(f"Failed to update task {task['id']} to {status}: {result.stdout}\n{result.stderr}")

    def save_session(self) -> None:
        command = [self.clawteam_executable]
        if self.data_dir:
            command.extend(["--data-dir", self.data_dir])
        command.extend(["session", "save", self.args.team, "--session-id", self.args.session_id])
        subprocess.run(command, env=self.env, check=False, capture_output=True, text=True)

    def status(self) -> dict[str, Any]:
        return self._debate_state_json("status", "--team", self.args.team, "--agent", self.args.role, "--json")

    def next_action(self) -> dict[str, Any]:
        return self._debate_state_json("next-action", "--team", self.args.team, "--agent", self.args.role, "--json")

    def summary(self) -> dict[str, Any]:
        return self._debate_state_json("summary", "--team", self.args.team, "--json", "--include-bodies")

    def advance(self) -> str:
        return self._debate_state_text("advance", "--team", self.args.team, "--agent", self.args.role)

    def submit_proposal(self, file_path: Path) -> dict[str, Any]:
        return self._debate_state_json(
            "submit-proposal",
            "--team",
            self.args.team,
            "--agent",
            self.args.role,
            "--file",
            str(file_path),
        )

    def submit_review(self, *, file_path: Path, target: str, blocking: bool) -> dict[str, Any]:
        return self._debate_state_json(
            "submit-review",
            "--team",
            self.args.team,
            "--agent",
            self.args.role,
            "--target",
            target,
            "--blocking",
            "yes" if blocking else "no",
            "--file",
            str(file_path),
        )

    def submit_rebuttal(self, *, file_path: Path, concede: bool) -> dict[str, Any]:
        argv = [
            "submit-rebuttal",
            "--team",
            self.args.team,
            "--agent",
            self.args.role,
            "--file",
            str(file_path),
        ]
        if concede:
            argv.append("--concede")
        return self._debate_state_json(*argv)

    def _debate_state_json(self, *argv: str) -> dict[str, Any]:
        command = [current_python(), str(self.debate_state_path), *argv]
        result = subprocess.run(command, env=self.env, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise DriverError(f"Command failed ({result.returncode}): {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise DriverError(f"Command did not return JSON: {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}") from exc

    def _debate_state_text(self, *argv: str) -> str:
        command = [current_python(), str(self.debate_state_path), *argv]
        result = subprocess.run(command, env=self.env, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise DriverError(f"Command failed ({result.returncode}): {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        return (result.stdout or "").strip()

    def state_progress_key(self, status_payload: dict[str, Any], next_action_payload: dict[str, Any]) -> str:
        payload = {
            "phase": status_payload.get("phase"),
            "status": status_payload.get("status"),
            "action": next_action_payload.get("action"),
            "advance_ready": next_action_payload.get("advance_ready"),
            "phase_progress": status_payload.get("phase_progress"),
            "review_round": status_payload.get("review_round"),
            "rebuttal_round": status_payload.get("rebuttal_round"),
            "proposals": len(status_payload.get("proposals") or []),
            "reviews": len(status_payload.get("reviews") or []),
            "rebuttals": len(status_payload.get("rebuttals") or []),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    def refresh_progress_clock(self, status_payload: dict[str, Any], next_action_payload: dict[str, Any]) -> None:
        key = self.state_progress_key(status_payload, next_action_payload)
        if key != self.last_progress_key:
            self.last_progress_key = key
            self.last_progress_change = time.time()
            self.repair_cycles_without_progress = 0
            self.last_repair_signature = ""

    def stale_for_seconds(self) -> float:
        return max(0.0, time.time() - self.last_progress_change)

    def sleep(self) -> None:
        time.sleep(max(1, self.args.poll_seconds))

    def workspace_path(self, filename: str) -> Path:
        return self.workspace / filename

    def candidate_capture_path(self) -> Path:
        team_dir = self.team_dir()
        if team_dir is not None:
            path = team_dir / "artifacts" / "captures" / self.args.role / CANDIDATE_CAPTURE_FILENAME
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        return self.workspace_path(CANDIDATE_CAPTURE_FILENAME)

    def capture_valid_candidate_submission(self, source_path: Path) -> Path | None:
        if self.args.role != CANDIDATE_OWNER or not source_path.is_file():
            return None
        try:
            checked = check_candidate_submission(
                source_path.read_text(encoding="utf-8"),
                owner=self.args.role,
                answer_kind=self.answer_kind(),
            )
        except OSError:
            return None
        if not checked.ok:
            return None
        capture_path = self.candidate_capture_path()
        capture_path.write_text(checked.normalized_text, encoding="utf-8")
        return capture_path

    def best_available_candidate_submission_path(self) -> Path | None:
        if self.args.role != CANDIDATE_OWNER:
            return None
        candidates = [self.workspace_path(proposal_filename()), self.candidate_capture_path()]
        for path in candidates:
            if not path.is_file():
                continue
            checked = check_candidate_submission(path.read_text(encoding="utf-8"), owner=self.args.role, answer_kind=self.answer_kind())
            if not checked.ok:
                continue
            if checked.normalized_text and path == self.workspace_path(proposal_filename()):
                path.write_text(checked.normalized_text, encoding="utf-8")
            return path
        return None

    def emit_driver_error_marker(self, exc: Exception) -> None:
        payload: dict[str, Any] = {
            "team": self.args.team,
            "role": self.args.role,
            "slot": self.args.slot,
            "session_id": self.args.session_id,
            "status": "driver_error",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "error": str(exc),
            "lane_failures": self.lane_failures,
            "repair_cycles_without_progress": self.repair_cycles_without_progress,
        }
        current_role_phase_state = getattr(self, "current_role_phase_state", None)
        if current_role_phase_state is not None:
            payload["role_phase"] = current_role_phase_state.as_payload()
        for label, loader in (("status", self.status), ("next_action", self.next_action)):
            try:
                payload[label] = loader()
            except Exception as inner_exc:  # best-effort diagnostics only
                payload[f"{label}_error"] = str(inner_exc)
        dump_json(self.workspace_path(BLOCKER_FILENAME), payload)

    def maybe_include_initial_prompt(self, parts: list[str]) -> list[str]:
        if self.initial_prompt and not self.initial_prompt_used:
            self.initial_prompt_used = True
            return [self.initial_prompt, *parts]
        return parts

    def resolve_model_timeout_seconds(self, artifact_kind: str) -> int:
        explicit = self.args.model_timeout_seconds
        if explicit is not None:
            return max(1, int(explicit))
        if artifact_kind == "candidate_submission":
            if self.args.candidate_timeout_seconds is not None:
                return max(1, int(self.args.candidate_timeout_seconds))
            return CANDIDATE_MODEL_TIMEOUT_SECONDS_DEFAULT
        if artifact_kind == "formal_review":
            if self.args.review_timeout_seconds is not None:
                return max(1, int(self.args.review_timeout_seconds))
            return REVIEW_MODEL_TIMEOUT_SECONDS_DEFAULT
        if artifact_kind == "rebuttal":
            if self.args.rebuttal_timeout_seconds is not None:
                return max(1, int(self.args.rebuttal_timeout_seconds))
            return REBUTTAL_MODEL_TIMEOUT_SECONDS_DEFAULT
        if artifact_kind == "coordinator_protocol":
            if self.args.coordinator_timeout_seconds is not None:
                return max(1, int(self.args.coordinator_timeout_seconds))
            return COORDINATOR_MODEL_TIMEOUT_SECONDS_DEFAULT
        return CANDIDATE_MODEL_TIMEOUT_SECONDS_DEFAULT

    def resolve_subprocess_timeout_seconds(self, artifact_kind: str) -> int:
        return max(
            1,
            self.resolve_model_timeout_seconds(artifact_kind) + max(0, int(self.args.subprocess_timeout_grace_seconds)),
        )

    @staticmethod
    def _trim_preview(text: str, limit: int = 240) -> str:
        stripped = " ".join((text or "").split())
        if len(stripped) <= limit:
            return stripped
        return stripped[: limit - 3] + "..."

    def turn_result_path(self) -> Path:
        return self.workspace_path(".chemqa-turn-result.json")

    def _load_turn_result(self) -> dict[str, Any]:
        path = self.turn_result_path()
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _turn_outcome_from_sidecar(self, *, fallback_returncode: int, stdout: str, stderr: str) -> TurnOutcome:
        payload = self._load_turn_result()
        return TurnOutcome(
            returncode=int(payload.get("returncode")) if payload.get("returncode") is not None else fallback_returncode,
            stop_reason=str(payload.get("stop_reason") or ""),
            timed_out=bool(payload.get("timed_out")),
            aborted=bool(payload.get("aborted")),
            hard_error=str(payload.get("hard_error") or ""),
            transcript_path=str(payload.get("transcript_path") or ""),
            tool_call_count=int(payload.get("tool_call_count") or 0),
            assistant_text_tail=str(payload.get("assistant_text_tail") or ""),
            stdout_preview=str(payload.get("stdout_preview") or self._trim_preview(stdout)),
            stderr_preview=str(payload.get("stderr_preview") or self._trim_preview(stderr)),
        )

    def call_model(self, prompt_parts: list[str], *, artifact_kind: str) -> TurnOutcome:
        message = "\n\n---\n\n".join(part.strip() for part in self.maybe_include_initial_prompt(prompt_parts) if part.strip())
        model_timeout = self.resolve_model_timeout_seconds(artifact_kind)
        turn_result_path = self.turn_result_path()
        turn_result_path.unlink(missing_ok=True)
        command = [
            current_python(),
            str(self.base_wrapper_path),
            "--slot",
            self.args.slot,
            "--session-id",
            self.args.session_id,
            "--env-file",
            self.args.env_file,
            "--timeout",
            str(model_timeout),
            "--message",
            message,
            "--turn-result-file",
            str(turn_result_path),
        ]
        if self.args.config_file:
            command.extend(["--config-file", str(self.args.config_file)])
        if self.args.thinking:
            command.extend(["--thinking", self.args.thinking])
        timeout_seconds = self.resolve_subprocess_timeout_seconds(artifact_kind)
        deadline = time.monotonic() + timeout_seconds
        stdout = ""
        stderr = ""
        try:
            proc = subprocess.Popen(
                command,
                cwd=str(self.workspace),
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            while True:
                if artifact_kind == "candidate_submission":
                    self.capture_valid_candidate_submission(self.workspace_path(proposal_filename()))
                returncode = proc.poll()
                if returncode is not None:
                    out, err = proc.communicate()
                    stdout = out or ""
                    stderr = err or ""
                    result = subprocess.CompletedProcess(command, returncode, stdout, stderr)
                    break
                if time.monotonic() >= deadline:
                    proc.kill()
                    out, err = proc.communicate()
                    stdout = out or ""
                    stderr = err or ""
                    raise DriverError(
                        f"OpenClaw model turn timed out after {model_timeout}s for {self.args.role}.\n"
                        f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                    )
                time.sleep(CANDIDATE_CAPTURE_POLL_SECONDS if artifact_kind == "candidate_submission" else 0.1)
        except OSError as exc:
            raise DriverError(f"Failed to launch OpenClaw model turn for {self.args.role}: {exc}") from exc
        if artifact_kind == "candidate_submission":
            self.capture_valid_candidate_submission(self.workspace_path(proposal_filename()))
        self.last_turn_outcome = self._turn_outcome_from_sidecar(
            fallback_returncode=int(result.returncode),
            stdout=result.stdout,
            stderr=result.stderr,
        )
        if result.returncode != 0:
            self.last_turn_outcome.hard_error = self.last_turn_outcome.hard_error or "wrapper_nonzero_exit"
            raise DriverError(
                f"OpenClaw model turn failed ({result.returncode}) for {self.args.role}.\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        return self.last_turn_outcome

    @staticmethod
    def _is_timeout_error(exc: DriverError) -> bool:
        return "timed out after" in str(exc)

    def record_lane_failure(self, key: str, *, reason: str, problems: list[str] | None = None) -> int:
        payload = self.lane_failures.setdefault(key, {"count": 0, "reason": reason, "problems": []})
        payload["count"] = int(payload.get("count") or 0) + 1
        payload["reason"] = reason
        if problems:
            payload["problems"] = list(problems)
        return int(payload["count"])

    def mark_progress(self) -> None:
        self.last_progress_change = time.time()
        self.repair_cycles_without_progress = 0
        self.last_repair_signature = ""

    def maybe_emit_blocker_report(self, status_payload: dict[str, Any]) -> None:
        stale_for = self.stale_for_seconds()
        if stale_for < self.args.stale_timeout_seconds:
            return
        now = time.time()
        if now - self.last_blocker_report < self.args.stale_timeout_seconds:
            return
        task_status = ""
        task = self.current_task()
        if task:
            task_status = str(task.get("status") or "")
        payload = liveness_summary(status_payload, coordinator_task_status=task_status)
        payload.update(
            {
                "team": self.args.team,
                "role": self.args.role,
                "stale_for_seconds": round(stale_for, 1),
                "lane_failures": self.lane_failures,
                "phase_repair_budget": self.args.phase_repair_budget,
                "repair_cycles_without_progress": self.repair_cycles_without_progress,
                "qualifying_candidate_reviews_count": len(qualifying_candidate_reviews(status_payload)),
            }
        )
        current_role_phase_state = getattr(self, "current_role_phase_state", None)
        if current_role_phase_state is not None:
            payload["role_phase"] = current_role_phase_state.as_payload()
        blocker_path = self.workspace_path(BLOCKER_FILENAME)
        dump_json(blocker_path, payload)
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)
        self.last_blocker_report = now

    def emit_terminal_failure(self, *, reason: str, status_payload: dict[str, Any], next_action_payload: dict[str, Any], blockers: list[str] | None = None) -> None:
        if self.terminal_failure_emitted:
            raise TerminalFailure(reason)
        phase = str(next_action_payload.get("phase") or status_payload.get("phase") or "unknown")
        liveness = liveness_summary(status_payload, coordinator_task_status=str((self.current_task() or {}).get("status") or ""))
        failure_path = self.workspace_path(terminal_failure_filename())
        failure_text = render_terminal_failure(
            team=self.args.team,
            role=self.args.role,
            reason=reason,
            phase=phase,
            phase_signature=liveness["phase_signature"],
            state_excerpt={"status": status_payload, "next_action": next_action_payload},
            lane_failures=self.lane_failures,
            repair_cycles_without_progress=self.repair_cycles_without_progress,
            blockers=blockers or [],
        )
        failure_path.write_text(failure_text, encoding="utf-8")
        if self.args.role == "debate-coordinator":
            protocol_path = self.workspace_path(coordinator_protocol_filename())
            protocol_payload = {
                "artifact_kind": "coordinator_protocol",
                "artifact_contract_version": "react-reviewed-v2",
                "terminal_state": "failed",
                "question": "",
                "final_answer": {},
                "acceptance_status": "failed",
                "review_completion_status": {"status": "failed", "phase": phase},
                "candidate_submission": {},
                "acceptance_decision": {"status": "failed", "reason": reason},
                "submission_trace": [],
                "submission_cycles": [],
                "proposer_trajectory": {},
                "reviewer_trajectories": {},
                "review_statuses": {},
                "final_review_items": {},
                "overall_confidence": {"level": "low", "rationale": reason},
                "failure_reason": reason,
                "terminal_failure_artifact": str(failure_path),
                "execution_warnings": blockers or [],
            }
            protocol_text = check_protocol(json.dumps(protocol_payload, ensure_ascii=False)).normalized_text
            protocol_path.write_text(protocol_text, encoding="utf-8")
        skill_root = getattr(self, "skill_root", None)
        output_root = Path(skill_root) if skill_root is not None else self.store.root
        output_dir = output_root / "generated" / "artifacts" / self.args.team
        failure = finalize_failure(
            run_id=self.args.team,
            output_dir=output_dir,
            failure_code="terminal_failure",
            failure_message=reason,
            missing_artifacts=[],
            validation_errors=[],
            open_review_items=[],
            diagnostic_paths=[str(failure_path)],
        )
        self.store.update_run_status(
            self.args.team,
            {
                "run_id": self.args.team,
                "status": "done",
                "protocol_terminal_state": "failed",
                "artifact_flow_state": "finalization_failed",
                "benchmark_terminal_state": "failed",
                "terminal_state": "failed",
                "terminal_reason_code": "terminal_failure",
                "terminal_reason": reason,
                "role": self.args.role,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "reason": reason,
                "transport_failure_artifact_path": str(failure_path),
                "failure_artifact_path": failure.status_overlay["failure_artifact_path"],
                "artifact_manifest_path": failure.status_overlay["artifact_manifest_path"],
                "qa_result_path": failure.status_overlay["qa_result_path"],
                "artifact_paths": failure.artifact_paths,
                "lane_failures": self.lane_failures,
                "reviewer_exit_reasons": self.reviewer_exit_state(),
                "repair_cycles_without_progress": self.repair_cycles_without_progress,
            },
        )
        self.terminal_failure_emitted = True
        raise TerminalFailure(reason)

    def all_tasks(self) -> list[dict[str, Any]]:
        command = [self.clawteam_executable]
        if self.data_dir:
            command.extend(["--data-dir", self.data_dir])
        command.extend(["--json", "task", "list", self.args.team])
        result = subprocess.run(command, env=self.env, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise DriverError(f"Failed to query task list for {self.args.team}: {result.stdout}\n{result.stderr}")
        return [dict(item) for item in json.loads(result.stdout or "[]") if isinstance(item, dict)]

    def sync_run_status(self, status_payload: dict[str, Any], next_action_payload: dict[str, Any]) -> None:
        engine_done = str(status_payload.get("status") or "") == "done"
        payload: dict[str, Any] = {
            "run_id": self.args.team,
            "status": "running",
            "role": self.args.role,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phase": status_payload.get("phase"),
            "review_round": status_payload.get("review_round"),
            "rebuttal_round": status_payload.get("rebuttal_round"),
            "phase_progress": status_payload.get("phase_progress"),
            "next_action": {
                "action": next_action_payload.get("action"),
                "advance_ready": next_action_payload.get("advance_ready"),
                "message": next_action_payload.get("message"),
            },
            "lane_failures": self.lane_failures,
            "reviewer_exit_reasons": self.reviewer_exit_state(),
            "exited_reviewer_lanes": list(self.reviewer_exit_state().keys()),
            "active_reviewer_lanes": [role for role in ROLE_TO_SEMANTIC_ROLE if is_reviewer_role(role) and role not in self.reviewer_exit_state()],
            "repair_cycles_without_progress": self.repair_cycles_without_progress,
            "qualifying_candidate_reviews_count": len(qualifying_candidate_reviews(status_payload)),
        }
        current_role_phase_state = getattr(self, "current_role_phase_state", None)
        if current_role_phase_state is not None:
            payload["role_phase"] = current_role_phase_state.as_payload()
        if engine_done:
            payload["protocol_terminal_state"] = str(status_payload.get("terminal_state") or "completed")
            payload["artifact_flow_state"] = "finalizing"
            payload["benchmark_terminal_state"] = "running"
            payload["terminal_state"] = "running"
            failure_reason = str(status_payload.get("failure_reason") or "").strip()
            if failure_reason:
                payload["terminal_reason"] = failure_reason
                if payload["protocol_terminal_state"] == "failed":
                    payload["terminal_reason_code"] = "engine_terminal_failure"
        if self.last_recovery_payload:
            payload["last_recovery"] = self.last_recovery_payload
        if self.last_respawn_events:
            payload["last_respawn_events"] = self.last_respawn_events[-8:]
        try:
            payload["tasks"] = {
                str(item.get("owner") or "unknown"): {
                    "id": item.get("id"),
                    "status": item.get("status"),
                    "lockedBy": item.get("lockedBy"),
                    "updatedAt": item.get("updatedAt"),
                }
                for item in self.all_tasks()
            }
        except DriverError as exc:
            payload["task_query_error"] = str(exc)
        self.store.update_run_status(self.args.team, payload)

    def spawn_registry_path(self) -> Path | None:
        team_dir = self.team_dir()
        if team_dir is None:
            return None
        return team_dir / "spawn_registry.json"

    def load_spawn_registry(self) -> dict[str, Any]:
        path = self.spawn_registry_path()
        if path is None or not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DriverError(f"Spawn registry is not valid JSON: {path} ({exc})") from exc
        return data if isinstance(data, dict) else {}

    def save_spawn_registry(self, payload: dict[str, Any]) -> None:
        path = self.spawn_registry_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @staticmethod
    def _budget_state_from_registry(registry: dict[str, Any]) -> dict[str, Any]:
        payload = registry.get("_budget_state") or {}
        if not isinstance(payload, dict):
            payload = {}
        role_counts = payload.get("respawns_by_role") or {}
        if not isinstance(role_counts, dict):
            role_counts = {}
        return {
            "phase_signature": str(payload.get("phase_signature") or ""),
            "respawns_by_role": {
                str(role): int(count or 0)
                for role, count in role_counts.items()
            },
        }

    def current_phase_signature(self) -> str:
        try:
            status_payload = self.status()
        except Exception:
            return ""
        summary = liveness_summary(
            status_payload,
            coordinator_task_status=str((self.current_task() or {}).get("status") or ""),
        )
        return str(summary.get("phase_signature") or "")

    def _prepare_respawn_budget_state(self, registry: dict[str, Any], *, phase_signature: str) -> tuple[dict[str, Any], bool]:
        budget_state = self._budget_state_from_registry(registry)
        changed = False
        if budget_state["phase_signature"] != phase_signature:
            budget_state = {
                "phase_signature": phase_signature,
                "respawns_by_role": {},
            }
            changed = True
        return budget_state, changed

    @staticmethod
    def _slot_from_registry_entry(entry: dict[str, Any] | None) -> str:
        if not isinstance(entry, dict):
            return ""
        explicit = str(entry.get("slot") or "").strip()
        if explicit:
            return explicit
        command = list(entry.get("command") or [])
        for index, token in enumerate(command[:-1]):
            if str(token) != "--slot":
                continue
            candidate = str(command[index + 1] or "").strip()
            if candidate:
                return candidate
        for key in ("cwd", "workspace"):
            candidate = str(entry.get(key) or "").strip()
            if candidate:
                return Path(candidate).name
        return ""

    @staticmethod
    def slot_for_role(role: str) -> str:
        return "debate-coordinator" if role == "debate-coordinator" else f"debate-{role.split('-')[-1]}"

    def next_action_for_agent(self, agent: str) -> dict[str, Any]:
        return self._debate_state_json("next-action", "--team", self.args.team, "--agent", agent, "--json")

    def role_process_is_running(self, role: str, entry: dict[str, Any]) -> bool:
        pid = int(entry.get("pid") or 0)
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        proc_cmdline = Path("/proc") / str(pid) / "cmdline"
        try:
            raw = proc_cmdline.read_text(encoding="utf-8")
        except OSError:
            return True
        joined = raw.replace("\x00", " ")
        return (
            "chemqa_review_openclaw_driver.py" in joined
            and self.args.team in joined
            and role in joined
        )

    def role_should_be_running(self, role: str, next_action_payload: dict[str, Any]) -> bool:
        action = str(next_action_payload.get("action") or "")
        phase = str(next_action_payload.get("phase") or "")
        if action == "stop" or phase == "done":
            return False
        if role == CANDIDATE_OWNER:
            return action in {"propose", "rebuttal"}
        if is_reviewer_role(role):
            return action in {"propose", "review"}
        return False

    def respawn_role_from_registry(self, role: str, entry: dict[str, Any], *, reason: str) -> bool:
        now = time.time()
        cooldown = max(0, int(self.args.respawn_cooldown_seconds))
        if now - self.last_respawn_attempt_at.get(role, 0.0) < cooldown:
            return False
        command = list(entry.get("command") or [])
        if not command:
            return False
        slot = self._slot_from_registry_entry(entry) or self.slot_for_role(role)
        cwd = self.workspace_root / slot
        cwd.mkdir(parents=True, exist_ok=True)
        team_dir = self.team_dir()
        if team_dir is None:
            return False
        log_path = team_dir / "spawn-logs" / f"{role}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = self.env.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        with log_path.open("a", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        registry = self.load_spawn_registry()
        updated = dict(entry)
        updated["pid"] = proc.pid
        updated["slot"] = slot
        updated["cwd"] = str(cwd)
        updated["workspace"] = str(cwd)
        updated["respawn_count"] = int(updated.get("respawn_count") or 0) + 1
        updated["last_respawn_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        updated["last_respawn_reason"] = reason
        registry[role] = updated
        self.save_spawn_registry(registry)
        self.last_respawn_attempt_at[role] = now
        self.last_respawn_events.append({
            "role": role,
            "pid": proc.pid,
            "reason": reason,
            "at": updated["last_respawn_at"],
        })
        return True

    def ensure_required_lanes_running(self) -> None:
        try:
            tasks_by_owner = {
                str(item.get("owner") or "unknown"): item
                for item in self.all_tasks()
            }
            registry = self.load_spawn_registry()
        except DriverError:
            return
        phase_signature = self.current_phase_signature()
        budget_state, budget_changed = self._prepare_respawn_budget_state(registry, phase_signature=phase_signature)
        max_respawns = max(0, int(getattr(self.args, "max_respawns_per_role_phase_signature", 0)))
        exited_reviewers = set(self.reviewer_exit_state())
        for role, entry in list(registry.items()):
            if role == "_budget_state":
                continue
            if role == "debate-coordinator" or role in exited_reviewers:
                continue
            task = tasks_by_owner.get(role) or {}
            if str(task.get("status") or "") == "completed":
                continue
            try:
                next_action_payload = self.next_action_for_agent(role)
            except DriverError:
                continue
            if not self.role_should_be_running(role, next_action_payload):
                continue
            if self.role_process_is_running(role, entry):
                continue
            if max_respawns >= 0 and int(budget_state["respawns_by_role"].get(role) or 0) >= max_respawns:
                continue
            budget_state["respawns_by_role"][role] = int(budget_state["respawns_by_role"].get(role) or 0) + 1
            registry["_budget_state"] = budget_state
            self.save_spawn_registry(registry)
            self.respawn_role_from_registry(role, entry, reason="missing_or_dead_role_process")
            budget_changed = False
        if budget_changed:
            registry["_budget_state"] = budget_state
            self.save_spawn_registry(registry)

    def maybe_salvage_timed_out_artifact(self, *, file_path: Path, checker: Callable[[str], Any], artifact_kind: str) -> tuple[bool, list[str]]:
        if not file_path.is_file():
            print(
                f"{self.args.role}: timeout while producing {artifact_kind}; no `{file_path.name}` artifact was left.",
                file=sys.stderr,
                flush=True,
            )
            return False, [f"model turn timed out and left no `{file_path.name}` artifact"]
        raw_text = file_path.read_text(encoding="utf-8")
        checked = checker(raw_text)
        if checked.normalized_text:
            file_path.write_text(checked.normalized_text, encoding="utf-8")
        if checked.ok:
            print(
                f"{self.args.role}: salvaged valid `{file_path.name}` after {artifact_kind} timeout.",
                file=sys.stderr,
                flush=True,
            )
            return True, []
        return False, ["model turn timed out before wrapper completion", *list(checked.errors)]

    @staticmethod
    def _file_snapshot(path: Path) -> tuple[int, int, str] | None:
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8")
        stat = path.stat()
        return (stat.st_mtime_ns, stat.st_size, hashlib.sha256(text.encode("utf-8")).hexdigest())

    @staticmethod
    def _normalized_candidate_text(checker: Callable[[str], Any], path: Path) -> str | None:
        if not path.is_file():
            return None
        raw_text = path.read_text(encoding="utf-8")
        checked = checker(raw_text)
        normalized = str(getattr(checked, "normalized_text", "") or "").strip()
        if normalized:
            return normalized
        return raw_text.strip() or None

    @staticmethod
    def _duplicate_proposal_reason(exc: DriverError) -> str:
        message = str(exc)
        if "Proposal matches a prior submission from epoch" not in message:
            return ""
        return message.split("STDERR:\n", 1)[-1].strip() or message.strip()

    def _build_artifact_feedback(
        self,
        *,
        filename: str,
        artifact_state: str,
        last_errors: list[str],
        minimal_template_lines: list[str] | None,
    ) -> tuple[list[str], str, str]:
        if artifact_state == "missing":
            lines = [
                "Previous turn ended normally, but the required artifact was not found.",
                f"Required artifact: `{filename}`.",
                "This phase is still in progress. Continue from your prior context.",
                f"When ready, write `{filename}` in the current workspace.",
            ]
            return lines, "waiting_for_artifact", "The previous turn did not complete the phase because the artifact is still missing."
        if artifact_state == "present_stale":
            lines = [
                f"The existing `{filename}` appears unchanged from before this turn.",
                "This phase is still in progress. Continue from your prior context.",
                f"Rewrite `{filename}` for the current phase instead of reusing the old file unchanged.",
            ]
            return lines, "repairing_stale_artifact", f"The existing `{filename}` must be rewritten for the current phase."
        lines = [
            f"The previous turn did not complete the phase because `{filename}` is present but still invalid.",
            "Rewrite the file as pure YAML only.",
            *(f"- {item}" for item in last_errors),
        ]
        if minimal_template_lines:
            lines.extend([
                "Use this minimal valid template shape:",
                *minimal_template_lines,
            ])
        return lines, "repairing_invalid_artifact", f"The previous turn left an invalid `{filename}` artifact that still needs repair."

    def _observe_artifact_outcome(
        self,
        *,
        file_path: Path,
        filename: str,
        checker: Callable[[str], Any],
        pre_call_snapshot: tuple[int, int, str] | None,
        pre_call_normalized: str | None,
        require_file_change: bool,
    ) -> ArtifactOutcome:
        if not file_path.is_file():
            return ArtifactOutcome(
                state="missing",
                filename=filename,
                path=str(file_path),
                validation_errors=[],
                validation_warnings=[],
                normalized_text="",
                changed_since_turn=False,
                classification="incomplete_turn_no_artifact",
            )
        raw_text = file_path.read_text(encoding="utf-8")
        checked = checker(raw_text)
        normalized_text = str(checked.normalized_text or raw_text)
        if checked.normalized_text:
            file_path.write_text(checked.normalized_text, encoding="utf-8")
        post_snapshot = self._file_snapshot(file_path)
        changed_since_turn = post_snapshot != pre_call_snapshot
        if checked.ok:
            post_normalized = normalized_text.strip()
            if require_file_change and pre_call_snapshot is not None:
                if post_snapshot == pre_call_snapshot or (
                    pre_call_normalized is not None and post_normalized == pre_call_normalized
                ):
                    return ArtifactOutcome(
                        state="present_stale",
                        filename=filename,
                        path=str(file_path),
                        validation_errors=[f"`{filename}` was not updated by model turn"],
                        validation_warnings=list(getattr(checked, "warnings", []) or []),
                        normalized_text=post_normalized,
                        changed_since_turn=changed_since_turn,
                        classification="artifact_stale",
                    )
            return ArtifactOutcome(
                state="present_valid",
                filename=filename,
                path=str(file_path),
                validation_errors=[],
                validation_warnings=list(getattr(checked, "warnings", []) or []),
                normalized_text=post_normalized,
                changed_since_turn=changed_since_turn,
                classification="submitted",
            )
        return ArtifactOutcome(
            state="present_invalid",
            filename=filename,
            path=str(file_path),
            validation_errors=list(checked.errors),
            validation_warnings=list(getattr(checked, "warnings", []) or []),
            normalized_text=normalized_text.strip(),
            changed_since_turn=changed_since_turn,
            classification="artifact_invalid",
        )

    def attempt_model_artifact(
        self,
        *,
        filename: str,
        instructions: list[str],
        checker: Callable[[str], Any],
        artifact_kind: str,
        failure_key: str,
        failure_reason: str,
        status_payload: dict[str, Any],
        next_action_payload: dict[str, Any],
        minimal_template_lines: list[str] | None = None,
        max_attempts_override: int | None = None,
        require_file_change: bool = False,
    ) -> Path:
        file_path = self.workspace_path(filename)
        last_errors: list[str] = []
        last_turn_outcome = getattr(self, "last_turn_outcome", None)
        max_attempts = max(1, int(max_attempts_override or self.args.max_model_attempts))
        for attempt in range(1, max_attempts + 1):
            pre_call_snapshot = self._file_snapshot(file_path)
            pre_call_normalized = self._normalized_candidate_text(checker, file_path) if require_file_change else None
            print(
                f"{self.args.role}: attempt {attempt}/{max_attempts} for {artifact_kind} -> {file_path.name}",
                file=sys.stderr,
                flush=True,
            )
            corrective = []
            classification = "running"
            feedback = ""
            if attempt > 1:
                corrective, classification, feedback = self._build_artifact_feedback(
                    filename=filename,
                    artifact_state="missing" if not last_errors else (
                        "present_stale"
                        if any("not updated by model turn" in item for item in last_errors)
                        else "present_invalid"
                    ),
                    last_errors=last_errors,
                    minimal_template_lines=minimal_template_lines,
                )
            try:
                turn_outcome = self.call_model([
                    *instructions,
                    *corrective,
                    "Behavior constraints for this turn:",
                    "- Do not explore the workspace unless the runtime explicitly asked for it.",
                    "- Do not run directory listings, broad find/grep scans, or other orientation commands.",
                    "- Write the requested artifact file directly in the current slot workspace, then stop.",
                    "Current state excerpt:",
                    pretty_json(next_action_payload),
                ], artifact_kind=artifact_kind)
                last_turn_outcome = turn_outcome
            except DriverError as exc:
                turn_outcome = getattr(self, "last_turn_outcome", None) or last_turn_outcome or TurnOutcome(returncode=None, hard_error=str(exc))
                last_turn_outcome = turn_outcome
                if not self._is_timeout_error(exc):
                    self.current_role_phase_state = PhaseAttemptState(
                        role=self.args.role,
                        phase=str(next_action_payload.get("phase") or status_payload.get("phase") or ""),
                        artifact_kind=artifact_kind,
                        turn_index=attempt,
                        max_phase_turns=max_attempts,
                        classification="failed_hard_error",
                        last_turn_outcome=turn_outcome,
                        last_artifact_outcome=ArtifactOutcome(
                            state="missing",
                            filename=filename,
                            path=str(file_path),
                            validation_errors=[],
                            normalized_text="",
                            changed_since_turn=False,
                            classification="wrapper_hard_error",
                        ),
                        last_feedback=str(exc),
                    )
                    raise
                salvaged, timeout_errors = self.maybe_salvage_timed_out_artifact(
                    file_path=file_path,
                    checker=checker,
                    artifact_kind=artifact_kind,
                )
                if salvaged:
                    self.current_role_phase_state = PhaseAttemptState(
                        role=self.args.role,
                        phase=str(next_action_payload.get("phase") or status_payload.get("phase") or ""),
                        artifact_kind=artifact_kind,
                        turn_index=attempt,
                        max_phase_turns=max_attempts,
                        classification="submitted",
                        last_turn_outcome=turn_outcome,
                        last_artifact_outcome=ArtifactOutcome(
                            state="present_valid",
                            filename=filename,
                            path=str(file_path),
                            validation_errors=[],
                            normalized_text=file_path.read_text(encoding="utf-8").strip(),
                            changed_since_turn=True,
                            classification="turn_timeout_artifact_salvaged",
                        ),
                        last_feedback="",
                    )
                    return file_path
                last_errors = timeout_errors
                self.current_role_phase_state = PhaseAttemptState(
                    role=self.args.role,
                    phase=str(next_action_payload.get("phase") or status_payload.get("phase") or ""),
                    artifact_kind=artifact_kind,
                    turn_index=attempt,
                    max_phase_turns=max_attempts,
                    classification="waiting_for_artifact",
                    last_turn_outcome=turn_outcome,
                    last_artifact_outcome=ArtifactOutcome(
                        state="missing",
                        filename=filename,
                        path=str(file_path),
                        validation_errors=list(last_errors),
                        normalized_text="",
                        changed_since_turn=False,
                        classification="turn_timeout_no_artifact",
                    ),
                    last_feedback="The previous turn timed out before producing a valid artifact.",
                )
                print(
                    f"{self.args.role}: timeout while producing {artifact_kind}: {'; '.join(last_errors)}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            artifact_outcome = self._observe_artifact_outcome(
                file_path=file_path,
                filename=filename,
                checker=checker,
                pre_call_snapshot=pre_call_snapshot,
                pre_call_normalized=pre_call_normalized,
                require_file_change=require_file_change,
            )
            if artifact_outcome.state == "present_valid":
                self.current_role_phase_state = PhaseAttemptState(
                    role=self.args.role,
                    phase=str(next_action_payload.get("phase") or status_payload.get("phase") or ""),
                    artifact_kind=artifact_kind,
                    turn_index=attempt,
                    max_phase_turns=max_attempts,
                    classification="submitted",
                    last_turn_outcome=turn_outcome,
                    last_artifact_outcome=artifact_outcome,
                    last_feedback="",
                )
                return file_path
            last_errors = list(artifact_outcome.validation_errors)
            corrective, classification, feedback = self._build_artifact_feedback(
                filename=filename,
                artifact_state=artifact_outcome.state,
                last_errors=last_errors,
                minimal_template_lines=minimal_template_lines,
            )
            self.current_role_phase_state = PhaseAttemptState(
                role=self.args.role,
                phase=str(next_action_payload.get("phase") or status_payload.get("phase") or ""),
                artifact_kind=artifact_kind,
                turn_index=attempt,
                max_phase_turns=max_attempts,
                classification=classification,
                last_turn_outcome=turn_outcome,
                last_artifact_outcome=artifact_outcome,
                last_feedback=feedback,
            )
            if artifact_outcome.state == "missing":
                print(
                    f"{self.args.role}: missing `{filename}` after model turn for {artifact_kind}.",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if artifact_outcome.state == "present_stale":
                print(
                    f"{self.args.role}: stale `{filename}` was reused without changes for {artifact_kind}.",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            print(
                f"{self.args.role}: invalid {artifact_kind} in `{filename}`: {'; '.join(last_errors)}",
                file=sys.stderr,
                flush=True,
            )
        count = self.record_lane_failure(failure_key, reason=failure_reason, problems=last_errors)
        self.current_role_phase_state = PhaseAttemptState(
            role=self.args.role,
            phase=str(next_action_payload.get("phase") or status_payload.get("phase") or ""),
            artifact_kind=artifact_kind,
            turn_index=max_attempts,
            max_phase_turns=max_attempts,
            classification="failed_budget_exhausted",
            last_turn_outcome=last_turn_outcome,
            last_artifact_outcome=getattr(getattr(self, "current_role_phase_state", None), "last_artifact_outcome", None),
            last_feedback=f"{failure_reason}; phase budget exhausted.",
        )
        if count >= self.args.lane_retry_budget:
            self.emit_terminal_failure(
                reason=f"{failure_reason}; lane retry budget exhausted for {failure_key}",
                status_payload=status_payload,
                next_action_payload=next_action_payload,
                blockers=last_errors,
            )
        raise DriverError(f"{failure_reason}: {'; '.join(last_errors) if last_errors else 'unknown validation failure'}")

    def run_worker_loop(self) -> int:
        while True:
            status_payload = self.status()
            next_action_payload = self.next_action()
            self.refresh_progress_clock(status_payload, next_action_payload)
            action = str(next_action_payload.get("action") or "")
            phase = str(next_action_payload.get("phase") or status_payload.get("phase") or "")

            if action == "stop" or str(status_payload.get("status")) == "done":
                self.save_session()
                self.ensure_task_status("completed")
                return 0

            if phase == "propose":
                if self.args.role == CANDIDATE_OWNER and action == "propose":
                    self.ensure_candidate_submission(status_payload, next_action_payload)
                elif is_reviewer_role(self.args.role) and action == "propose":
                    self.ensure_placeholder_submission(status_payload)
                self.sleep()
                continue

            if phase == "review":
                if action == "review":
                    self.ensure_pending_reviews(status_payload, next_action_payload)
                self.sleep()
                continue

            if phase == "rebuttal":
                if self.args.role == CANDIDATE_OWNER and action == "rebuttal":
                    self.ensure_rebuttal(status_payload, next_action_payload)
                self.sleep()
                continue

            self.sleep()

    def run_coordinator_loop(self) -> int:
        while True:
            status_payload = self.status()
            next_action_payload = self.next_action()
            self.refresh_progress_clock(status_payload, next_action_payload)
            self.ensure_required_lanes_running()
            self.sync_run_status(status_payload, next_action_payload)
            action = str(next_action_payload.get("action") or "")
            if action == "stop" or str(status_payload.get("status")) == "done":
                self.ensure_protocol_artifact(status_payload)
                self.save_session()
                self.ensure_task_status("completed")
                return 0
            if action == "advance":
                self.advance()
                self.mark_progress()
                continue
            self.maybe_emit_blocker_report(status_payload)
            stagnation_result = self.maybe_handle_stagnation(status_payload, next_action_payload)
            if stagnation_result is not None:
                return stagnation_result
            self.sleep()

    def maybe_handle_stagnation(self, status_payload: dict[str, Any], next_action_payload: dict[str, Any]) -> int | None:
        if self.stale_for_seconds() < self.args.stale_timeout_seconds:
            return None
        pre = liveness_summary(status_payload, coordinator_task_status=str((self.current_task() or {}).get("status") or ""))
        if pre["phase_signature"] == self.last_repair_signature:
            self.repair_cycles_without_progress += 1
        else:
            self.last_repair_signature = pre["phase_signature"]
            self.repair_cycles_without_progress = 1
        recovery = self.run_recovery_cycle()
        refreshed_status = self.status()
        refreshed_next = self.next_action()
        post = liveness_summary(refreshed_status, coordinator_task_status=str((self.current_task() or {}).get("status") or ""))
        if post["phase_signature"] != pre["phase_signature"]:
            self.mark_progress()
            return None
        if self.recovery_payload_indicates_progress(recovery):
            self.mark_progress()
            return None

        blockers = list(recovery.get("blockers") or [])
        missing_lanes = missing_required_reviewer_lanes(refreshed_status)
        candidate_checked = check_candidate_submission(
            self.candidate_submission_text(refreshed_status),
            owner=CANDIDATE_OWNER,
            answer_kind=self.answer_kind(),
        )
        phase = str(refreshed_next.get("phase") or refreshed_status.get("phase") or "")
        if phase == "review" and missing_lanes and candidate_checked.ok:
            blocker_text = " ".join(blockers)
            exitable_lanes = [lane for lane in missing_lanes if lane in blocker_text]
            if not exitable_lanes and self.repair_cycles_without_progress >= self.args.phase_repair_budget:
                exitable_lanes = list(missing_lanes)
            exited_any = False
            for lane in exitable_lanes:
                exited_any = self.mark_reviewer_exited(
                    lane,
                    reason=f"reviewer exited after repeated review stagnation: {'; '.join(blockers) if blockers else 'missing formal review artifact'}",
                    phase=phase,
                    review_round=int(refreshed_status.get('review_round') or 0),
                    blockers=blockers,
                ) or exited_any
            if exited_any:
                refreshed_status = self.status()
                refreshed_next = self.next_action()
                if str(refreshed_next.get("action") or "") == "advance":
                    self.advance()
                self.mark_progress()
                return None

        if self.repair_cycles_without_progress >= self.args.phase_repair_budget:
            if phase == "review" and candidate_checked.ok and missing_lanes:
                reason = (
                    "forced degraded completion after recovery attempts left missing required reviewer lanes: "
                    + ", ".join(missing_lanes)
                )
                return self.force_complete_with_missing_reviews(
                    reason=reason,
                    missing_lanes=missing_lanes,
                    blockers=blockers,
                )
            self.emit_terminal_failure(
                reason="phase stagnation detected after recovery cycles without progress",
                status_payload=refreshed_status,
                next_action_payload=refreshed_next,
                blockers=blockers,
            )
        return None

    def run_recovery_cycle(self) -> dict[str, Any]:
        recover_script = self.skill_root / "scripts" / "recover_run.py"
        command = [
            current_python(),
            str(recover_script),
            "--skill-root",
            str(self.skill_root),
            "--team",
            self.args.team,
            "--runtime-dir",
            str(self.runtime_root),
            "--workspace-root",
            str(self.workspace_root),
            "--max-steps",
            "1",
            "--max-respawns-per-role-phase-signature",
            str(getattr(self.args, "max_respawns_per_role_phase_signature", MAX_RESPAWNS_PER_ROLE_PHASE_SIGNATURE_DEFAULT)),
            "--json",
        ]
        result = subprocess.run(command, env=self.env, check=False, capture_output=True, text=True)
        if result.returncode not in {0, 1}:
            raise DriverError(f"Recovery command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}\n{result.stderr}")
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise DriverError(f"Recovery command did not return JSON: {result.stdout}\n{result.stderr}") from exc
        self.last_recovery_payload = payload if isinstance(payload, dict) else {}
        return payload

    @staticmethod
    def recovery_payload_indicates_progress(payload: dict[str, Any]) -> bool:
        if bool(payload.get("progress_made")):
            return True
        actions = [str(item or "").strip() for item in (payload.get("actions") or [])]
        progress_prefixes = (
            "respawn-role ",
            "submit-proposal ",
            "submit-review ",
            "submit-rebuttal ",
            "advance ",
            "repair-invalid-review-state ",
        )
        return any(action.startswith(progress_prefixes) for action in actions)

    def ensure_placeholder_submission(self, status_payload: dict[str, Any]) -> None:
        if current_proposal(status_payload, self.args.role):
            return
        target = self.workspace_path(proposal_filename())
        target.write_text(render_placeholder_proposal(self.args.role), encoding="utf-8")
        self.submit_proposal(target)
        refreshed = self.status()
        if not current_proposal(refreshed, self.args.role):
            raise DriverError(f"Placeholder proposal for {self.args.role} was not registered in debate state.")
        self.mark_progress()

    def ensure_candidate_submission(self, status_payload: dict[str, Any], next_action_payload: dict[str, Any]) -> None:
        base_instructions = [
            "Runtime request for this turn:",
            f"- You are in `propose` for the main candidate owner. Write exactly one pure-YAML file named `{proposal_filename()}`.",
            "- No markdown headings, bullets outside YAML, or code fences.",
            "- Do not submit transport commands yourself. The runtime wrapper will do that after this turn.",
            "- Do not wait or poll. Produce the file and stop.",
        ]
        duplicate_feedback: list[str] = []
        duplicate_reason = ""
        max_attempts = max(1, int(getattr(self.args, "max_model_attempts", MODEL_ATTEMPTS_DEFAULT)))

        for submit_attempt in range(1, max_attempts + 1):
            instructions = list(base_instructions)
            if duplicate_feedback:
                instructions.extend(duplicate_feedback)
            self.attempt_model_artifact(
                filename=proposal_filename(),
                instructions=instructions,
                checker=lambda text: check_candidate_submission(text, owner=self.args.role, answer_kind=self.answer_kind()),
                artifact_kind="candidate_submission",
                failure_key="propose:candidate",
                failure_reason=f"{self.args.role} failed to produce a valid candidate submission",
                status_payload=status_payload,
                next_action_payload=next_action_payload,
                minimal_template_lines=[
                    "artifact_kind: candidate_submission",
                    "artifact_contract_version: react-reviewed-v2",
                    "phase: propose",
                    f"owner: {self.args.role}",
                    'direct_answer: "6"',
                    "summary: Short explanation of why the answer has that many distinct proton environments.",
                    "submission_trace:",
                    "  - step: structural_reasoning",
                    "    status: success",
                    "    detail: Counted distinct proton environments from the provided SMILES.",
                    "evidence_limits:",
                    "  - Based on first-principles NMR equivalence reasoning from the provided structure.",
                    "claim_anchors: []",
                ],
                max_attempts_override=max_attempts,
                require_file_change=True,
            )
            target = self.best_available_candidate_submission_path()
            if target is None:
                raise DriverError(f"{self.args.role} did not leave a valid candidate submission in workspace or capture.")
            try:
                self.submit_proposal(target)
            except DriverError as exc:
                duplicate_reason = self._duplicate_proposal_reason(exc)
                if not duplicate_reason:
                    raise
                duplicate_feedback = [
                    "Previous turn produced a candidate that was rejected because it duplicated a prior epoch submission.",
                    f"- {duplicate_reason}",
                    "- Do not repeat the conceded prior answer unchanged.",
                    "- Revise the direct answer and supporting trace so the new candidate explicitly addresses the prior review items.",
                ]
                continue
            refreshed = self.status()
            if not current_proposal(refreshed, self.args.role):
                raise DriverError(f"Candidate submission for {self.args.role} was not registered.")
            self.mark_progress()
            return

        count = self.record_lane_failure(
            "propose:candidate:duplicate_epoch_submission",
            reason=f"{self.args.role} kept resubmitting a duplicate candidate across epochs",
            problems=[duplicate_reason] if duplicate_reason else None,
        )
        if count >= self.args.lane_retry_budget:
            self.emit_terminal_failure(
                reason=f"{self.args.role} kept resubmitting a duplicate candidate across epochs; lane retry budget exhausted",
                status_payload=status_payload,
                next_action_payload=next_action_payload,
                blockers=[duplicate_reason] if duplicate_reason else [],
            )
        raise DriverError(
            f"{self.args.role} kept resubmitting a duplicate candidate across epochs"
            + (f": {duplicate_reason}" if duplicate_reason else "")
        )

    def ensure_pending_reviews(self, status_payload: dict[str, Any], next_action_payload: dict[str, Any]) -> None:
        targets = [str(item) for item in (next_action_payload.get("targets") or [])]
        target_map = {
            str(item.get("proposer")): dict(item)
            for item in (next_action_payload.get("target_proposals") or [])
            if isinstance(item, dict)
        }
        review_round = int(next_action_payload.get("review_round") or 0)
        for target in targets:
            current_status = self.status()
            if review_exists(current_status, reviewer=self.args.role, target=target, review_round=review_round):
                continue
            proposal_payload = target_map.get(target) or {}
            if target == CANDIDATE_OWNER and is_reviewer_role(self.args.role) and not proposal_is_transport_placeholder(proposal_payload):
                self.ensure_formal_review(current_status, next_action_payload, target)
            else:
                self.ensure_transport_review(target)

    def ensure_transport_review(self, target: str) -> None:
        file_path = self.workspace_path(review_filename(target))
        file_path.write_text(render_transport_review(reviewer=self.args.role, target=target), encoding="utf-8")
        body = file_path.read_text(encoding="utf-8")
        checked = check_transport_review(body, reviewer=self.args.role, target=target)
        if not checked.ok:
            raise DriverError(f"Generated transport review is invalid for {self.args.role}->{target}: {'; '.join(checked.errors)}")
        file_path.write_text(checked.normalized_text, encoding="utf-8")
        submitted = self.submit_review(file_path=file_path, target=target, blocking=False)
        refreshed = self.status()
        submitted_round = int(submitted.get("review_round") or 0)
        if review_exists(refreshed, reviewer=self.args.role, target=target, review_round=submitted_round) or review_exists(refreshed, reviewer=self.args.role, target=target):
            self.mark_progress()
            return
        raise DriverError(f"Transport review for {self.args.role}->{target} was not registered.")

    def ensure_formal_review(self, status_payload: dict[str, Any], next_action_payload: dict[str, Any], target: str) -> None:
        file_path = self.attempt_model_artifact(
            filename=review_filename(target),
            instructions=[
                "Runtime request for this turn:",
                f"- You are in `review` and must write exactly one pure-YAML formal review file named `{review_filename(target)}`.",
                f"- The target is `{target}` and this review counts toward ChemQA acceptance.",
                "- No markdown headings, narrative body outside YAML, or code fences.",
                "- For self-contained numeric, stoichiometric, symmetry, or equilibrium questions whose givens are already in the prompt, default to reviewing only the question, candidate artifact, and prior rebuttal/review history; do not launch literature retrieval unless you can name a concrete missing external fact that could change the answer.",
                "- If the target artifact already contains the full calculation chain, keep the review tight and local instead of scanning sibling skills or external sources.",
                "- Do not submit transport commands yourself. The runtime wrapper will do that after this turn.",
                "- Do not wait or poll. Produce the file and stop.",
            ],
            checker=lambda text: check_formal_review(text, reviewer=self.args.role, target=target),
            artifact_kind="formal_review",
            failure_key=f"review:{self.args.role}:{target}",
            failure_reason=f"{self.args.role} failed to produce a valid formal review for {target}",
            status_payload=status_payload,
            next_action_payload=next_action_payload,
            minimal_template_lines=[
                "artifact_kind: formal_review",
                "artifact_contract_version: react-reviewed-v2",
                "phase: review",
                f"reviewer_lane: {self.args.role}",
                f"target_owner: {target}",
                "target_kind: candidate_submission",
                "verdict: non_blocking",
                "summary: Brief review summary.",
                "review_items: []",
                "counts_for_acceptance: true",
                "synthetic: false",
            ],
        )
        body = file_path.read_text(encoding="utf-8")
        submitted = self.submit_review(file_path=file_path, target=target, blocking=blocking_flag_for_review(body))
        refreshed = self.status()
        submitted_round = int(submitted.get("review_round") or 0)
        if review_exists(refreshed, reviewer=self.args.role, target=target, review_round=submitted_round) or review_exists(refreshed, reviewer=self.args.role, target=target):
            self.mark_progress()
            return
        raise DriverError(f"Formal review for {self.args.role}->{target} was not registered.")

    def ensure_rebuttal(self, status_payload: dict[str, Any], next_action_payload: dict[str, Any]) -> None:
        file_path = self.attempt_model_artifact(
            filename=rebuttal_filename(),
            instructions=[
                "Runtime request for this turn:",
                f"- You are in `rebuttal` and must write exactly one pure-YAML file named `{rebuttal_filename()}`.",
                "- If you intend to concede, include `concede: true` as a YAML field.",
                "- No markdown headings, narrative body outside YAML, or code fences.",
                "- Do not submit transport commands yourself. The runtime wrapper will do that after this turn.",
                "- Do not wait or poll. Produce the file and stop.",
            ],
            checker=lambda text: check_rebuttal(text, owner=self.args.role),
            artifact_kind="rebuttal",
            failure_key=f"rebuttal:{self.args.role}",
            failure_reason=f"{self.args.role} failed to produce a valid rebuttal",
            status_payload=status_payload,
            next_action_payload=next_action_payload,
            minimal_template_lines=[
                "artifact_kind: rebuttal",
                "artifact_contract_version: react-reviewed-v2",
                "phase: rebuttal",
                f"owner: {self.args.role}",
                "concede: false",
                "response_summary: Brief rebuttal summary.",
                "response_items: []",
            ],
        )
        body = file_path.read_text(encoding="utf-8")
        checked = check_rebuttal(body, owner=self.args.role)
        submitted = self.submit_rebuttal(file_path=file_path, concede=bool(checked.payload.get("concede")))
        refreshed = self.status()
        submitted_round = int(submitted.get("rebuttal_round") or 0)
        if rebuttal_exists(refreshed, proposer=self.args.role, rebuttal_round=submitted_round) or rebuttal_exists(refreshed, proposer=self.args.role):
            self.mark_progress()
            return
        raise DriverError(f"Rebuttal for {self.args.role} was not registered.")

    def team_dir(self) -> Path | None:
        data_dir = getattr(self, "data_dir", "")
        if not data_dir:
            return None
        path = Path(data_dir).expanduser().resolve() / "teams" / self.args.team
        path.mkdir(parents=True, exist_ok=True)
        return path

    def state_db_path(self) -> Path | None:
        team_dir = self.team_dir()
        if team_dir is None:
            return None
        path = team_dir / "debate" / "state.db"
        return path if path.exists() else None

    def reviewer_exit_state(self) -> dict[str, dict[str, Any]]:
        cached = dict(getattr(self, "reviewer_exits", {}) or {})
        db_path = self.state_db_path()
        if db_path is None:
            return cached
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", ("chemqa_exited_reviewers_json",)).fetchone()
        if not row or row[0] in (None, ""):
            return dict(self.reviewer_exits)
        try:
            payload = json.loads(str(row[0] or "{}"))
        except json.JSONDecodeError:
            return cached
        if not isinstance(payload, dict):
            return cached
        state = {role: dict(item) for role, item in payload.items() if isinstance(item, dict)}
        self.reviewer_exits = state
        return dict(state)

    def candidate_submission_text(self, status_payload: dict[str, Any]) -> str:
        proposal = current_proposal(status_payload, CANDIDATE_OWNER) or {}
        body = str(proposal.get("body") or "")
        if body.strip():
            return body
        artifact = proposal.get("artifact") or {}
        for key in ("archive_path", "source_path"):
            candidate = str(artifact.get(key) or "").strip()
            if not candidate:
                continue
            path = Path(candidate).expanduser().resolve()
            if path.is_file():
                return path.read_text(encoding="utf-8")
        team_dir = self.team_dir()
        if team_dir is not None:
            candidates = sorted(
                (team_dir / "debate" / "artifacts" / "proposals").glob("epoch-*/proposer-1.md"),
                reverse=True,
            )
            for path in candidates:
                if path.is_file():
                    return path.read_text(encoding="utf-8")
            capture_path = self.candidate_capture_path()
            if capture_path.is_file():
                return capture_path.read_text(encoding="utf-8")
        proposal_path = self.workspace_path(proposal_filename())
        if proposal_path.is_file():
            return proposal_path.read_text(encoding="utf-8")
        return ""

    def mark_reviewer_exited(self, reviewer: str, *, reason: str, phase: str, review_round: int, blockers: list[str]) -> bool:
        if reviewer not in ROLE_TO_SEMANTIC_ROLE or not is_reviewer_role(reviewer):
            return False
        state = self.reviewer_exit_state()
        if reviewer in state:
            return False
        payload = {
            "reason": reason,
            "phase": phase,
            "review_round": review_round,
            "blockers": list(blockers),
            "exited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "recovery_cycles_without_progress": self.repair_cycles_without_progress,
        }
        state[reviewer] = payload
        self.reviewer_exits = state
        db_path = self.state_db_path()
        if db_path is not None:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                    ("chemqa_exited_reviewers_json", json.dumps(state, ensure_ascii=False)),
                )
                conn.commit()
        return True

    def mark_engine_done_for_forced_completion(self, *, reason: str, final_candidates: list[str]) -> None:
        db_path = self.state_db_path()
        if db_path is None:
            return
        with sqlite3.connect(db_path) as conn:
            rows = {
                "phase": "done",
                "status": "done",
                "terminal_state": "completed",
                "failure_reason": reason,
                "final_candidates_json": json.dumps(final_candidates, ensure_ascii=False),
                "chemqa_exited_reviewers_json": json.dumps(self.reviewer_exit_state(), ensure_ascii=False),
            }
            for key, value in rows.items():
                conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, value))
            conn.commit()

    def generate_protocol_with_model(self, *, summary_payload: dict[str, Any], deterministic_protocol: dict[str, Any]) -> tuple[dict[str, Any], str]:
        scaffold_checked = check_protocol(json.dumps(deterministic_protocol, ensure_ascii=False))
        if not scaffold_checked.ok:
            raise DriverError(f"Deterministic protocol reconstruction failed: {'; '.join(scaffold_checked.errors)}")

        file_path = self.workspace_path(coordinator_protocol_filename())
        file_path.write_text(scaffold_checked.normalized_text, encoding="utf-8")
        last_errors: list[str] = []

        for attempt in range(1, self.args.max_model_attempts + 1):
            corrective: list[str] = []
            if attempt > 1:
                corrective = [
                    f"Previous turn did not leave a valid pure-YAML `{coordinator_protocol_filename()}` file.",
                    "Rewrite the file as pure YAML only and preserve the required coordinator protocol schema.",
                    *(f"- {item}" for item in last_errors),
                ]
                file_path.write_text(scaffold_checked.normalized_text, encoding="utf-8")

            timeout_salvaged = False
            try:
                self.call_model([
                    "Runtime request for this turn:",
                    f"- You are the terminal coordinator. Write exactly one pure-YAML file named `{coordinator_protocol_filename()}`.",
                    "- Start from the deterministic scaffold already placed in the workspace and refine it into the final protocol.",
                    "- Preserve the schema required by the collector.",
                    "- Use the completed debate evidence to make `final_answer`, `acceptance_decision`, `overall_confidence`, and related summary fields clean and explicit.",
                    "- Do not run transport commands, waiting loops, or extra polling.",
                    *corrective,
                    "Behavior constraints for this turn:",
                    "- Do not explore the workspace unless the runtime explicitly asked for it.",
                    "- Rewrite the requested protocol file directly, then stop.",
                    "Deterministic protocol scaffold:",
                    scaffold_checked.normalized_text,
                    "Completed debate summary (JSON):",
                    pretty_json(summary_payload),
                ], artifact_kind="coordinator_protocol")
            except DriverError as exc:
                if not self._is_timeout_error(exc):
                    last_turn = getattr(self, "last_turn_outcome", None)
                    if last_turn is not None and last_turn.aborted:
                        last_errors = [
                            "coordinator model turn aborted before producing a refined protocol",
                        ]
                        break
                    raise
                timeout_salvaged = True
                print(
                    f"Coordinator model turn timed out for {self.args.role}; checking for a salvaged `{coordinator_protocol_filename()}` artifact.",
                    file=sys.stderr,
                    flush=True,
                )
            if not file_path.is_file():
                if timeout_salvaged:
                    last_errors = [
                        f"coordinator model turn timed out and left no `{coordinator_protocol_filename()}` artifact"
                    ]
                    continue
                last_errors = [f"missing `{coordinator_protocol_filename()}` after coordinator model turn"]
                continue
            raw_text = file_path.read_text(encoding="utf-8")
            checked = check_protocol(raw_text)
            if checked.normalized_text:
                file_path.write_text(checked.normalized_text, encoding="utf-8")
            if checked.ok:
                return checked.payload, "model_timeout_salvaged" if timeout_salvaged else "model"
            last_errors = list(checked.errors)
            if timeout_salvaged:
                last_errors.insert(0, "coordinator model turn timed out before wrapper completion")

        print(
            "Coordinator model protocol generation failed validation; falling back to deterministic protocol: "
            + ("; ".join(last_errors) if last_errors else "unknown validation failure"),
            file=sys.stderr,
            flush=True,
        )
        file_path.write_text(scaffold_checked.normalized_text, encoding="utf-8")
        return scaffold_checked.payload, "deterministic_fallback"

    def finalize_protocol_payload(self, *, protocol_payload: dict[str, Any], protocol_generation_mode: str) -> None:
        checked = check_protocol(json.dumps(protocol_payload, ensure_ascii=False))
        if not checked.ok:
            raise DriverError(f"Final protocol validation failed: {'; '.join(checked.errors)}")

        workspace_protocol_path = self.workspace_path(coordinator_protocol_filename())
        workspace_protocol_path.write_text(checked.normalized_text, encoding="utf-8")

        team_dir = self.team_dir()
        protocol_path = workspace_protocol_path
        if team_dir is not None:
            protocol_path = team_dir / coordinator_protocol_filename()
            protocol_path.write_text(checked.normalized_text, encoding="utf-8")

        output_dir = self.skill_root / "generated" / "artifacts" / self.args.team
        answer_kind = resolve_answer_kind(self.runtime_answer_metadata())
        self.store.update_run_status(
            self.args.team,
            {
                "run_id": self.args.team,
                "status": "running",
                "role": self.args.role,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "protocol_terminal_state": str(protocol_payload.get("terminal_state") or "completed"),
                "artifact_flow_state": "finalizing",
                "benchmark_terminal_state": "running",
                "terminal_state": "running",
                "protocol_generation_mode": protocol_generation_mode,
                "protocol_path": str(protocol_path),
                "workspace_protocol_path": str(workspace_protocol_path),
                "artifacts_output_dir": str(output_dir),
                "answer_kind": answer_kind,
            },
        )
        collect_payload: dict[str, Any] = {}
        collect_error = ""
        collect_script = self.skill_root / "scripts" / "collect_artifacts.py"
        source_dir = team_dir or self.workspace
        command = [
            current_python(),
            str(collect_script),
            "--skill-root",
            str(self.skill_root),
            "--source-dir",
            str(source_dir),
            "--protocol-file",
            str(protocol_path),
            "--output-dir",
            str(output_dir),
            "--answer-kind",
            answer_kind,
            "--json",
        ]
        result = subprocess.run(command, env=self.env, check=False, capture_output=True, text=True)
        if result.returncode == 0:
            try:
                collect_payload = json.loads(result.stdout or "{}")
            except json.JSONDecodeError:
                collect_error = f"Artifact collection returned non-JSON output:\n{result.stdout}\n{result.stderr}"
        else:
            collect_error = (
                f"Artifact collection failed ({result.returncode}): {' '.join(command)}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        final_answer = protocol_payload.get("final_answer")
        final_answer_preview = ""
        if isinstance(final_answer, dict):
            final_answer_preview = str(
                final_answer.get("direct_answer")
                or final_answer.get("answer")
                or final_answer.get("value")
                or ""
            ).strip()
        elif final_answer not in (None, ""):
            final_answer_preview = str(final_answer).strip()

        run_status_payload = {
            "run_id": self.args.team,
            "status": "done",
            "protocol_terminal_state": str(protocol_payload.get("terminal_state") or "completed"),
            "artifact_flow_state": "finalized",
            "benchmark_terminal_state": "completed",
            "terminal_state": "completed",
            "role": self.args.role,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "acceptance_status": protocol_payload.get("acceptance_status"),
            "review_completion_status": protocol_payload.get("review_completion_status"),
            "protocol_generation_mode": protocol_generation_mode,
            "protocol_path": str(protocol_path),
            "workspace_protocol_path": str(workspace_protocol_path),
            "artifacts_output_dir": str(output_dir),
            "final_answer_preview": final_answer_preview,
            "answer_kind": answer_kind,
            "artifact_collection": {
                "status": "error" if collect_error else "ok",
            },
        }
        if collect_payload:
            run_status_payload["artifact_paths"] = collect_payload.get("artifact_paths") or {}
            status_overlay = collect_payload.get("status_overlay") if isinstance(collect_payload.get("status_overlay"), dict) else {}
            for key, value in status_overlay.items():
                if key in {"status", "protocol_terminal_state", "artifact_flow_state", "benchmark_terminal_state", "terminal_state", "qa_result_path", "final_answer_artifact_path", "failure_artifact_path", "artifact_manifest_path", "candidate_view_path", "artifacts_output_dir", "artifact_paths"}:
                    run_status_payload[key] = value
        if collect_error:
            failure = finalize_failure(
                run_id=self.args.team,
                output_dir=output_dir,
                failure_code="artifact_collection_error",
                failure_message=collect_error,
                diagnostic_paths=[str(protocol_path)],
            )
            run_status_payload.update(failure.status_overlay)
            run_status_payload["artifact_collection"] = {"status": "error"}
            run_status_payload["artifact_collection_error"] = collect_error
            print(collect_error, file=sys.stderr, flush=True)
        self.store.update_run_status(self.args.team, run_status_payload)

    def runtime_answer_metadata(self) -> dict[str, Any]:
        skill_root = getattr(self, "skill_root", None)
        team = str(getattr(getattr(self, "args", None), "team", "") or "").strip()
        if skill_root is None or not team:
            return {}
        runplan_path = Path(skill_root) / "control" / "runplans" / f"{team}.json"
        if not runplan_path.is_file():
            return {}
        try:
            runplan = json.loads(runplan_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        request_metadata = ((runplan.get("request_snapshot") or {}).get("metadata") or {})
        runtime_context = runplan.get("runtime_context") or {}
        return {
            "answer_kind": runtime_context.get("answer_kind") or request_metadata.get("answer_kind"),
            "eval_kind": request_metadata.get("eval_kind"),
            "dataset": request_metadata.get("dataset"),
            "track": request_metadata.get("track"),
        }

    def answer_kind(self) -> str:
        return resolve_answer_kind(self.runtime_answer_metadata())

    def ensure_protocol_artifact(self, status_payload: dict[str, Any]) -> None:
        summary_payload = self.summary()
        deterministic_protocol = build_protocol_from_summary(summary_payload)
        protocol_payload, protocol_generation_mode = self.generate_protocol_with_model(
            summary_payload=summary_payload,
            deterministic_protocol=deterministic_protocol,
        )
        self.finalize_protocol_payload(protocol_payload=protocol_payload, protocol_generation_mode=protocol_generation_mode)

    def force_complete_with_missing_reviews(self, *, reason: str, missing_lanes: list[str], blockers: list[str]) -> int:
        summary_payload = self.summary()
        deterministic_protocol = build_protocol_from_summary(summary_payload)
        forced_protocol = apply_forced_missing_review_completion(
            deterministic_protocol,
            reason=reason,
            missing_lanes=missing_lanes,
            blockers=blockers,
            recovery_cycles_without_progress=self.repair_cycles_without_progress,
        )
        final_candidates = [CANDIDATE_OWNER] if current_proposal(summary_payload, CANDIDATE_OWNER) else []
        self.mark_engine_done_for_forced_completion(reason=reason, final_candidates=final_candidates)
        self.finalize_protocol_payload(
            protocol_payload=forced_protocol,
            protocol_generation_mode="forced_missing_review_completion",
        )
        self.save_session()
        self.ensure_task_status("completed")
        return 0


def main() -> int:
    args = parse_args()
    try:
        return ChemQAReviewDriver(args).run()
    except DriverError as exc:
        print(f"chemqa-review driver error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
