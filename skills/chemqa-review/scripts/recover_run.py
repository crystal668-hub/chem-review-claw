#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

from bundle_common import default_runtime_dir, resolve_python_interpreter, resolve_skill_root
from control_store import FileControlStore
from chemqa_review_artifacts import (
    CANDIDATE_OWNER,
    REVIEWER_ROLES,
    blocking_flag_for_review,
    check_candidate_submission,
    check_formal_review,
    check_rebuttal,
    check_transport_review,
    current_proposal,
    pretty_json,
    proposal_filename,
    repair_candidate_submission_text,
    repair_formal_review_text,
    repair_rebuttal_text,
    render_placeholder_proposal,
    render_transport_review,
    review_filename,
    rebuttal_filename,
)

CANDIDATE_CAPTURE_FILENAME = "proposal.captured.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover and auto-progress a stalled ChemQA review run.")
    parser.add_argument("--skill-root", default=str(Path(__file__).resolve().parents[1]), help="chemqa-review skill root")
    parser.add_argument("--team", required=True, help="Debate team / run id")
    parser.add_argument("--runtime-dir", help="Path to deployed DebateClaw runtime helpers")
    parser.add_argument("--workspace-root", default=str(Path.home() / ".openclaw" / "debateclaw" / "workspaces"))
    parser.add_argument("--max-steps", type=int, default=40, help="Maximum reconcile/advance iterations")
    parser.add_argument("--max-respawns-per-role-phase-signature", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="Print final JSON summary")
    return parser.parse_args()


class RecoverError(RuntimeError):
    pass


class RunRecoverer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.skill_root = resolve_skill_root(args.skill_root)
        self.store = FileControlStore(self.skill_root)
        self.runtime_root = Path(args.runtime_dir).expanduser().resolve() if args.runtime_dir else default_runtime_dir()
        self.debate_state_path = self.runtime_root / "debate_state.py"
        if not self.debate_state_path.is_file():
            raise SystemExit(f"Missing DebateClaw runtime helper: {self.debate_state_path}")
        self.workspace_root = Path(args.workspace_root).expanduser().resolve()
        self.data_dir = os.environ.get("CLAWTEAM_DATA_DIR", "").strip()
        self.actions: list[str] = []
        self.blockers: list[str] = []

    @property
    def env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.data_dir:
            env["CLAWTEAM_DATA_DIR"] = self.data_dir
        return env

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
    def _fallback_slot_for_role(role: str) -> str:
        return "debate-coordinator" if role == "debate-coordinator" else f"debate-{role.split('-')[-1]}"

    def workspace_for(self, role: str) -> Path:
        registry = self.load_spawn_registry()
        slot = self._slot_from_registry_entry(registry.get(role)) or self._fallback_slot_for_role(role)
        path = self.workspace_root / slot
        path.mkdir(parents=True, exist_ok=True)
        return path

    def candidate_capture_path_for(self, role: str) -> Path:
        team_dir = self.team_dir()
        if team_dir is not None:
            path = team_dir / "artifacts" / "captures" / role / CANDIDATE_CAPTURE_FILENAME
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        return self.workspace_for(role) / CANDIDATE_CAPTURE_FILENAME

    def team_dir(self) -> Path | None:
        if not self.data_dir:
            return None
        path = Path(self.data_dir).expanduser().resolve() / "teams" / self.args.team
        path.mkdir(parents=True, exist_ok=True)
        return path

    def spawn_registry_path(self) -> Path | None:
        team_dir = self.team_dir()
        if team_dir is None:
            return None
        return team_dir / "spawn_registry.json"

    def state_db_path(self) -> Path | None:
        team_dir = self.team_dir()
        if team_dir is None:
            return None
        return team_dir / "debate" / "state.db"

    def load_spawn_registry(self) -> dict[str, Any]:
        path = self.spawn_registry_path()
        if path is None or not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def save_spawn_registry(self, payload: dict[str, Any]) -> None:
        path = self.spawn_registry_path()
        if path is None:
            return
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
            return self._phase_signature(self.status())
        except Exception:
            return ""

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
        return "chemqa_review_openclaw_driver.py" in joined and self.args.team in joined and role in joined

    def respawn_actionable_roles(self) -> bool:
        changed = False
        registry = self.load_spawn_registry()
        team_dir = self.team_dir()
        if not registry or team_dir is None:
            return False
        phase_signature = self.current_phase_signature()
        budget_state, budget_changed = self._prepare_respawn_budget_state(registry, phase_signature=phase_signature)
        max_respawns = max(0, int(getattr(self.args, "max_respawns_per_role_phase_signature", 0)))
        for role, entry in registry.items():
            if role == "_budget_state":
                continue
            if role == "debate-coordinator":
                continue
            payload = self.next_action(role)
            action = str(payload.get("action") or "")
            if action not in {"propose", "review", "rebuttal"}:
                continue
            if self.role_process_is_running(role, entry):
                continue
            command = list(entry.get("command") or [])
            if not command:
                continue
            if max_respawns >= 0 and int(budget_state["respawns_by_role"].get(role) or 0) >= max_respawns:
                continue
            log_path = team_dir / "spawn-logs" / f"{role}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                proc = subprocess.Popen(
                    command,
                    cwd=str(self.workspace_for(role)),
                    env=self.env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            updated = dict(entry)
            updated["pid"] = proc.pid
            updated["slot"] = self._slot_from_registry_entry(entry) or self._fallback_slot_for_role(role)
            updated["cwd"] = str(self.workspace_for(role))
            updated["workspace"] = str(self.workspace_for(role))
            updated["respawn_count"] = int(updated.get("respawn_count") or 0) + 1
            updated["last_respawn_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            updated["last_respawn_reason"] = "recover_run_actionable_role"
            registry[role] = updated
            budget_state["respawns_by_role"][role] = int(budget_state["respawns_by_role"].get(role) or 0) + 1
            registry["_budget_state"] = budget_state
            self.actions.append(f"respawn-role {role} pid={proc.pid}")
            changed = True
        if changed:
            self.save_spawn_registry(registry)
        elif budget_changed:
            registry["_budget_state"] = budget_state
            self.save_spawn_registry(registry)
        return changed

    def write_run_status(
        self,
        *,
        state: dict[str, Any],
        status: str,
        recovery_cycles_without_progress: int,
        progress_made: bool,
        terminal_state: str = "",
        terminal_reason_code: str = "",
        terminal_reason: str = "",
    ) -> None:
        payload = {
            "run_id": self.args.team,
            "status": status,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phase": state.get("phase"),
            "review_round": state.get("review_round"),
            "rebuttal_round": state.get("rebuttal_round"),
            "phase_progress": state.get("phase_progress"),
            "actions": self.actions,
            "blockers": self.blockers,
            "progress_made": progress_made,
            "recovery_cycles_without_progress": recovery_cycles_without_progress,
        }
        if status == "done":
            effective_terminal_state = terminal_state or str(state.get("terminal_state") or "completed")
            payload["terminal_state"] = effective_terminal_state
            failure_reason = terminal_reason or str(state.get("failure_reason") or "").strip()
            if terminal_reason_code:
                payload["terminal_reason_code"] = terminal_reason_code
            elif effective_terminal_state == "failed":
                payload["terminal_reason_code"] = "engine_terminal_failure"
            if failure_reason:
                payload["terminal_reason"] = failure_reason
        self.store.update_run_status(self.args.team, payload)

    def debate_state_json(self, *argv: str) -> dict[str, Any]:
        command = [resolve_python_interpreter(), str(self.debate_state_path), *argv]
        result = subprocess.run(command, env=self.env, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RecoverError(
                f"Command failed ({result.returncode}): {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        return json.loads(result.stdout)

    def debate_state_text(self, *argv: str) -> str:
        command = [resolve_python_interpreter(), str(self.debate_state_path), *argv]
        result = subprocess.run(command, env=self.env, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RecoverError(
                f"Command failed ({result.returncode}): {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        return (result.stdout or "").strip()

    def status(self) -> dict[str, Any]:
        return self.debate_state_json("status", "--team", self.args.team, "--json")

    def next_action(self, agent: str) -> dict[str, Any]:
        return self.debate_state_json("next-action", "--team", self.args.team, "--agent", agent, "--json")

    def submit_proposal(self, *, agent: str, file_path: Path) -> None:
        self.debate_state_json("submit-proposal", "--team", self.args.team, "--agent", agent, "--file", str(file_path))
        self.actions.append(f"submit-proposal {agent} <- {file_path.name}")

    def submit_review(self, *, agent: str, target: str, file_path: Path, blocking: bool) -> None:
        self.debate_state_json(
            "submit-review",
            "--team",
            self.args.team,
            "--agent",
            agent,
            "--target",
            target,
            "--blocking",
            "yes" if blocking else "no",
            "--file",
            str(file_path),
        )
        self.actions.append(f"submit-review {agent}->{target} blocking={'yes' if blocking else 'no'}")

    def submit_rebuttal(self, *, agent: str, file_path: Path, concede: bool) -> None:
        argv = ["submit-rebuttal", "--team", self.args.team, "--agent", agent, "--file", str(file_path)]
        if concede:
            argv.append("--concede")
        self.debate_state_json(*argv)
        self.actions.append(f"submit-rebuttal {agent} concede={'yes' if concede else 'no'}")

    def advance(self) -> None:
        self.debate_state_text("advance", "--team", self.args.team, "--agent", "debate-coordinator", "--json")
        self.actions.append("advance debate-coordinator")

    def _latest_prior_candidate_review_body(self, status_payload: dict[str, Any], *, reviewer: str, target: str) -> str:
        matches = [
            review for review in (status_payload.get("reviews") or [])
            if str(review.get("reviewer")) == reviewer and str(review.get("target_proposer")) == target
        ]
        matches.sort(key=lambda item: int(item.get("review_round") or 0), reverse=True)
        for review in matches:
            body = str(review.get("body") or "").strip()
            if body:
                return body
        return ""

    def _latest_prior_rebuttal_body(self, status_payload: dict[str, Any], *, proposer: str) -> str:
        matches = [
            rebuttal for rebuttal in (status_payload.get("rebuttals") or [])
            if str(rebuttal.get("proposer")) == proposer
        ]
        matches.sort(key=lambda item: int(item.get("rebuttal_round") or 0), reverse=True)
        for rebuttal in matches:
            body = str(rebuttal.get("body") or "").strip()
            if body:
                return body
        return ""

    def _ensure_text_file(self, path: Path, text: str) -> Path:
        path.write_text(text, encoding="utf-8")
        return path

    def _phase_signature(self, status_payload: dict[str, Any]) -> str:
        payload = {
            "phase": status_payload.get("phase"),
            "status": status_payload.get("status"),
            "review_round": status_payload.get("review_round"),
            "rebuttal_round": status_payload.get("rebuttal_round"),
            "phase_progress": status_payload.get("phase_progress"),
            "reviews": len(status_payload.get("reviews") or []),
            "rebuttals": len(status_payload.get("rebuttals") or []),
            "proposals": len(status_payload.get("proposals") or []),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    def recover_propose(self, status_payload: dict[str, Any]) -> bool:
        changed = False
        workflow = str(status_payload.get("workflow") or "")
        if workflow == "chemqa-review":
            expected = [CANDIDATE_OWNER]
        else:
            expected = [f"proposer-{i}" for i in range(1, int(status_payload.get("proposer_count") or 0) + 1)]
        for agent in expected:
            if current_proposal(status_payload, agent):
                continue
            workspace = self.workspace_for(agent)
            file_path = workspace / proposal_filename()
            if agent == CANDIDATE_OWNER:
                capture_path = self.candidate_capture_path_for(agent)
                checked = None
                source_path = None
                invalid_reasons: list[str] = []
                for candidate_path in (file_path, capture_path):
                    if not candidate_path.is_file():
                        continue
                    repaired = repair_candidate_submission_text(candidate_path.read_text(encoding="utf-8"), owner=agent)
                    checked = check_candidate_submission(repaired, owner=agent)
                    if checked.ok:
                        candidate_path.write_text(checked.normalized_text, encoding="utf-8")
                        source_path = candidate_path
                        break
                    invalid_reasons.append(f"{candidate_path}: {'; '.join(checked.errors)}")
                if source_path is not None and checked is not None:
                    self.submit_proposal(agent=agent, file_path=source_path)
                    changed = True
                elif invalid_reasons:
                    self.blockers.append(f"candidate proposal sources exist but are invalid: {' | '.join(invalid_reasons)}")
                else:
                    self.blockers.append(
                        f"missing candidate proposal sources for {agent}: workspace={file_path}; capture={capture_path}"
                    )
                continue
            checked_text = render_placeholder_proposal(agent)
            self._ensure_text_file(file_path, checked_text)
            self.submit_proposal(agent=agent, file_path=file_path)
            changed = True
        return changed

    def recover_review(self, status_payload: dict[str, Any]) -> bool:
        changed = False
        for agent in [CANDIDATE_OWNER, *REVIEWER_ROLES]:
            payload = self.next_action(agent)
            if str(payload.get("action") or "") != "review":
                continue
            for target in [str(item) for item in (payload.get("targets") or [])]:
                workspace = self.workspace_for(agent)
                file_path = workspace / review_filename(target)
                if target == CANDIDATE_OWNER and agent in REVIEWER_ROLES:
                    body = ""
                    if file_path.is_file():
                        body = file_path.read_text(encoding="utf-8")
                    if not body.strip():
                        body = self._latest_prior_candidate_review_body(status_payload, reviewer=agent, target=target)
                    if not body.strip():
                        self.blockers.append(f"missing formal review artifact for {agent}->{target}")
                        continue
                    repaired = repair_formal_review_text(body, reviewer=agent, target=target)
                    checked = check_formal_review(repaired, reviewer=agent, target=target)
                    if not checked.ok:
                        self.blockers.append(f"invalid formal review artifact for {agent}->{target}: {'; '.join(checked.errors)}")
                        continue
                    self._ensure_text_file(file_path, checked.normalized_text)
                    self.submit_review(agent=agent, target=target, file_path=file_path, blocking=blocking_flag_for_review(checked.normalized_text))
                    changed = True
                    continue

                body = render_transport_review(reviewer=agent, target=target)
                if file_path.is_file():
                    existing = file_path.read_text(encoding="utf-8")
                    checked_existing = check_transport_review(existing, reviewer=agent, target=target)
                    if checked_existing.ok:
                        body = checked_existing.normalized_text
                self._ensure_text_file(file_path, body)
                self.submit_review(agent=agent, target=target, file_path=file_path, blocking=False)
                changed = True
        return changed

    def recover_rebuttal(self, status_payload: dict[str, Any]) -> bool:
        changed = False
        payload = self.next_action(CANDIDATE_OWNER)
        if str(payload.get("action") or "") != "rebuttal":
            return False
        workspace = self.workspace_for(CANDIDATE_OWNER)
        file_path = workspace / rebuttal_filename()
        body = ""
        if file_path.is_file():
            body = file_path.read_text(encoding="utf-8")
        if not body.strip():
            body = self._latest_prior_rebuttal_body(status_payload, proposer=CANDIDATE_OWNER)
        if not body.strip():
            self.blockers.append(f"missing rebuttal artifact for {CANDIDATE_OWNER}")
            return False
        repaired = repair_rebuttal_text(body, owner=CANDIDATE_OWNER)
        checked = check_rebuttal(repaired, owner=CANDIDATE_OWNER)
        if not checked.ok:
            self.blockers.append(f"invalid rebuttal artifact for {CANDIDATE_OWNER}: {'; '.join(checked.errors)}")
            return False
        self._ensure_text_file(file_path, checked.normalized_text)
        self.submit_rebuttal(agent=CANDIDATE_OWNER, file_path=file_path, concede=bool(checked.payload.get("concede")))
        changed = True
        return changed

    def repair_invalid_review_state(self, status_payload: dict[str, Any]) -> bool:
        if str(status_payload.get("workflow") or "") != "chemqa-review":
            return False
        if str(status_payload.get("phase") or "") != "review":
            return False
        proposals = [
            proposal for proposal in (status_payload.get("proposals") or [])
            if int(proposal.get("epoch") or 0) == int(status_payload.get("epoch") or 0)
            and str(proposal.get("proposer") or "") == CANDIDATE_OWNER
            and str(proposal.get("status") or "") == "active"
        ]
        if proposals:
            return False
        db_path = self.state_db_path()
        if db_path is None or not db_path.is_file():
            self.blockers.append("cannot repair invalid review state without state.db")
            return False
        epoch = int(status_payload.get("epoch") or 1)
        max_epochs = int(status_payload.get("max_epochs") or 1)
        with sqlite3.connect(db_path) as conn:
            if epoch < max_epochs:
                conn.execute("UPDATE meta SET value = ? WHERE key = 'epoch'", (str(epoch + 1),))
                conn.execute("UPDATE meta SET value = 'propose' WHERE key = 'phase'")
                conn.execute("UPDATE meta SET value = '0' WHERE key = 'review_round'")
                conn.execute("UPDATE meta SET value = '0' WHERE key = 'rebuttal_round'")
                conn.execute("UPDATE meta SET value = ? WHERE key = 'phase_targets_json'", ('["proposer-1"]',))
                conn.commit()
                self.actions.append(f"repair-invalid-review-state -> epoch {epoch + 1} propose")
                return True
            conn.execute("UPDATE meta SET value = 'done' WHERE key = 'phase'")
            conn.execute("UPDATE meta SET value = 'done' WHERE key = 'status'")
            conn.execute("UPDATE meta SET value = 'failed' WHERE key = 'terminal_state'")
            conn.execute("UPDATE meta SET value = ? WHERE key = 'failure_reason'", (f'max_epochs_exhausted_after_candidate_failures (epoch={epoch}, max_epochs={max_epochs})',))
            conn.execute("UPDATE meta SET value = '[]' WHERE key = 'final_candidates_json'")
            conn.commit()
        self.actions.append("repair-invalid-review-state -> terminal failed")
        return True

    def run(self) -> int:
        last_signature = ""
        recovery_cycles_without_progress = 0
        progress_made = False
        for _ in range(self.args.max_steps):
            status_payload = self.status()
            signature = self._phase_signature(status_payload)
            if str(status_payload.get("status") or "") == "done" or str(status_payload.get("phase") or "") == "done":
                payload = {"status": "done", "actions": self.actions, "blockers": self.blockers, "state": status_payload, "progress_made": progress_made, "recovery_cycles_without_progress": recovery_cycles_without_progress}
                self.write_run_status(
                    state=status_payload,
                    status="done",
                    recovery_cycles_without_progress=recovery_cycles_without_progress,
                    progress_made=progress_made,
                )
                if self.args.json:
                    print(json.dumps(payload, ensure_ascii=False, indent=2))
                else:
                    print("Run is done.")
                return 0

            changed = self.repair_invalid_review_state(status_payload)
            phase = str(status_payload.get("phase") or "")
            if not changed:
                if phase == "propose":
                    changed = self.recover_propose(status_payload)
                elif phase == "review":
                    changed = self.recover_review(status_payload)
                elif phase == "rebuttal":
                    changed = self.recover_rebuttal(status_payload)

            coordinator_payload = self.next_action("debate-coordinator")
            if str(coordinator_payload.get("action") or "") == "advance":
                self.advance()
                changed = True

            self.respawn_actionable_roles()

            if changed:
                progress_made = True
                recovery_cycles_without_progress = 0
                last_signature = ""
                continue

            if signature == last_signature:
                recovery_cycles_without_progress += 1
            else:
                recovery_cycles_without_progress = 0
                last_signature = signature

            if recovery_cycles_without_progress >= 1:
                break

        final_state = self.status()
        stalled = str(final_state.get("status") or "") != "done"
        payload = {
            "status": "done",
            "actions": self.actions,
            "blockers": self.blockers,
            "state": final_state,
            "progress_made": progress_made,
            "recovery_cycles_without_progress": recovery_cycles_without_progress,
        }
        if stalled:
            payload["terminal_state"] = "failed"
            payload["terminal_reason_code"] = "stalled"
        self.write_run_status(
            state=final_state,
            status=payload["status"],
            recovery_cycles_without_progress=recovery_cycles_without_progress,
            progress_made=progress_made,
            terminal_state="failed" if stalled else "",
            terminal_reason_code="stalled" if stalled else "",
            terminal_reason=(
                f"Recovery stopped without reaching done after {recovery_cycles_without_progress} stagnant cycle(s)."
                if stalled
                else ""
            ),
        )
        if self.args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print("Recovery stopped without reaching done.")
            print("Final state:")
            print(pretty_json(final_state))
        return 1 if stalled else 0


def main() -> int:
    args = parse_args()
    recoverer = RunRecoverer(args)
    try:
        return recoverer.run()
    except RecoverError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
