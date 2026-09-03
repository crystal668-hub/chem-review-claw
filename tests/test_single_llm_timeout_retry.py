from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

from benchmarking.core.datasets import BenchmarkRecord
from benchmarking.runtime.agent_workspace import (
    AttemptWorkspaceManager,
    WorkspaceIsolationError,
    WorkspaceTemplate,
)
from benchmarking.runtime.workspace_policy import (
    ContaminationAudit,
    ProtectedRoot,
    WorkspaceAudit,
    adjudicate_workspace_findings,
)
from benchmarking.workflow.prompts import build_single_llm_prompt
from benchmarking.workflow.runners.single_llm import SingleLLMRunner


@dataclass(frozen=True)
class Group:
    id: str
    skills_enabled: bool
    websearch: bool = True


@dataclass
class CompletedProcess:
    returncode: int = 0
    stdout: str = "{}"
    stderr: str = ""


class SingleLLMTimeoutRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        templates = {}
        for template_id in ("single-llm-skills-on-v1", "single-llm-skills-off-v1"):
            template_root = root / "templates" / template_id
            template_root.mkdir(parents=True)
            (template_root / "AGENTS.md").write_text("# test\n", encoding="utf-8")
            templates[template_id] = WorkspaceTemplate(template_id=template_id, source_dir=template_root)
        self.workspace_manager = AttemptWorkspaceManager(
            runtime_root=root / "runtime" / "runs",
            output_root=root / "output",
            run_id="run-1",
            invocation_id="invocation-1",
            templates=templates,
            protected_roots=(
                ProtectedRoot("benchmark_runtime_root", root / "runtime" / "runs", "test.runtime_root"),
                ProtectedRoot("current_output_root", root / "output", "test.output_root"),
            ),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _record(self) -> BenchmarkRecord:
        return BenchmarkRecord(
            record_id="record-1",
            dataset="frontierscience",
            source_file="/tmp/frontierscience.jsonl",
            eval_kind="frontierscience_olympiad",
            prompt="Identify X.",
            reference_answer="X",
            payload={"track": "olympiad"},
        )

    def _vgb_record(self) -> BenchmarkRecord:
        return BenchmarkRecord(
            record_id="vgb-record-1",
            dataset="verifier_grounded_rdkit",
            source_file="/tmp/verifier_grounded_rdkit.jsonl",
            eval_kind="verifier_grounded",
            prompt="Return a molecule.",
            reference_answer="hidden",
            payload={
                "verifier_grounded": {
                    "answer_schema": {"final_answer_prefix": "FINAL ANSWER:"},
                }
            },
        )

    def _runner(
        self,
        *,
        captured_commands: list[list[str]],
        captured_envs: list[dict[str, str]] | None = None,
        captured_timeouts: list[int | None] | None = None,
        timeout_once: bool = True,
        no_timeout: bool = False,
        build_prompt: Any | None = None,
        config_path: Path = Path("/tmp/openclaw.json"),
        write_attempt_marker: bool = False,
        observed_old_marker: list[bool] | None = None,
        contamination_auditor: Any | None = None,
        failure_stderr: str | None = None,
        pypi_cutoff: str | None = None,
    ) -> SingleLLMRunner:
        calls = {"count": 0}
        prompt_builder = build_prompt or (lambda *args, **kwargs: "BASE PROMPT")

        def run_subprocess(command: list[str], *, env: dict[str, str], timeout: int) -> CompletedProcess:
            calls["count"] += 1
            captured_commands.append(command)
            if captured_envs is not None:
                captured_envs.append(dict(env))
            if captured_timeouts is not None:
                captured_timeouts.append(timeout)
            if write_attempt_marker:
                workspace = Path(env["BENCHMARK_WORKSPACE_DIR"])
                if observed_old_marker is not None:
                    observed_old_marker.append((workspace / "attempt-marker.txt").exists())
                (workspace / "attempt-marker.txt").write_text(str(calls["count"]), encoding="utf-8")
            if timeout_once and calls["count"] == 1:
                raise subprocess.TimeoutExpired(command, timeout)
            if failure_stderr is not None:
                return CompletedProcess(returncode=1, stderr=failure_stderr)
            return CompletedProcess()

        return SingleLLMRunner(
            agent_id="benchmark-single-skills-on",
            timeout_seconds=10,
            config_path=config_path,
            runtime_bundle_root=Path("/tmp/bundles"),
            run_subprocess=run_subprocess,
            parse_json_stdout=lambda result, command: {
                "result": {"payloads": [{"text": "Reasoning\nFINAL ANSWER: X"}], "meta": {}}
            },
            unwrap_agent_payload=lambda payload: payload["result"],
            summarize_payloads=lambda payloads: "\n\n".join(str(item.get("text") or "") for item in payloads),
            normalize_answer_tracks=lambda *, full_response_text: ("X", full_response_text),
            ensure_runtime_bundle=lambda record, *, bundle_root: None,
            build_single_llm_prompt=prompt_builder,
            slugify=lambda value, **kwargs: str(value),
            benchmark_agent_thinking="high",
            workspace_manager=self.workspace_manager,
            contamination_auditor=contamination_auditor or (lambda **_kwargs: ContaminationAudit(status="clean")),
            timeout_retries=1,
            timeout_retry_backoff_seconds=(0,),
            sleep_fn=lambda seconds: None,
            no_timeout=no_timeout,
            pypi_cutoff=pypi_cutoff,
        )

    def test_vgb_attempt_uses_fresh_python_environment_and_cleans_it_before_archive(self) -> None:
        captured_commands: list[list[str]] = []
        captured_envs: list[dict[str, str]] = []
        runner = self._runner(
            captured_commands=captured_commands,
            captured_envs=captured_envs,
            timeout_once=False,
            pypi_cutoff="2026-09-03T00:00:00Z",
        )

        result = runner.run(self._vgb_record(), Group(id="single_llm_skills_on", skills_enabled=True))

        environment = captured_envs[0]
        attempt_python = Path(environment["BENCHMARK_ATTEMPT_PYTHON"])
        self.assertEqual(str(attempt_python.parent.parent), environment["VIRTUAL_ENV"])
        self.assertEqual(str(attempt_python), environment["UV_PYTHON"])
        self.assertEqual("2026-09-03T00:00:00Z", environment["UV_EXCLUDE_NEWER"])
        self.assertEqual(str(attempt_python.parent), environment["PATH"].split(":")[0])
        self.assertFalse(attempt_python.exists())
        self.assertEqual("clear", result.runner_meta["dependency_audit"]["status"])
        self.assertTrue(result.runner_meta["attempt_environment_cleanup"]["venv_removed"])
        manifest = result.runner_meta["attempt_environment"]
        self.assertEqual("vgb-record-1", manifest["identity"]["record_id"])
        self.assertEqual("2026-09-03T00:00:00Z", manifest["pypi"]["cutoff"])
        self.assertEqual("generated", manifest["pypi"]["replay_lock"]["status"])
        self.assertIn("uv", manifest["native_tools"])
        archive_workspace = Path(result.runner_meta["workspace_isolation"]["archive_workspace"])
        self.assertTrue((archive_workspace / "scratch" / "notes" / "dependency-manifest.json").is_file())

    def _message_from_command(self, command: list[str]) -> str:
        return command[command.index("--message") + 1]

    def test_skills_on_timeout_retry_keeps_original_prompt(self) -> None:
        captured_commands: list[list[str]] = []
        runner = self._runner(captured_commands=captured_commands)

        result = runner.run(self._record(), Group(id="single_llm_skills_on", skills_enabled=True))

        self.assertEqual(2, len(captured_commands))
        first_prompt = self._message_from_command(captured_commands[0])
        self.assertIn("BASE PROMPT", first_prompt)
        self.assertIn("BENCHMARK WORKSPACE FILE CONTRACT", first_prompt)
        self.assertIn("omit workdir", first_prompt)
        self.assertIn('cd "$BENCHMARK_SKILL_SCRATCH_DIR" &&', first_prompt)
        retry_prompt = self._message_from_command(captured_commands[1])
        self.assertIn("BASE PROMPT", retry_prompt)
        self.assertNotIn("RETRY FOCUS GUIDANCE", retry_prompt)
        self.assertNotIn("reduce tool use", retry_prompt)
        self.assertNotIn("retry_focus_prompt_applied", result.runner_meta["timeout_retry"]["attempt_history"][0])
        self.assertNotIn("retry_focus_prompt_applied", result.runner_meta["timeout_retry"]["attempt_history"][1])

    def test_skills_off_timeout_retry_keeps_original_prompt(self) -> None:
        captured_commands: list[list[str]] = []
        captured_envs: list[dict[str, str]] = []
        runner = self._runner(captured_commands=captured_commands, captured_envs=captured_envs)

        result = runner.run(self._record(), Group(id="single_llm_skills_off", skills_enabled=False))

        self.assertEqual(2, len(captured_commands))
        self.assertIn("BASE PROMPT", self._message_from_command(captured_commands[0]))
        self.assertIn("BENCHMARK WORKSPACE FILE CONTRACT", self._message_from_command(captured_commands[0]))
        self.assertIn("omit workdir", self._message_from_command(captured_commands[0]))
        self.assertIn('cd "$BENCHMARK_SKILL_SCRATCH_DIR" &&', self._message_from_command(captured_commands[0]))
        self.assertIn("BASE PROMPT", self._message_from_command(captured_commands[1]))
        self.assertIn("BENCHMARK WORKSPACE FILE CONTRACT", self._message_from_command(captured_commands[1]))
        for environment in captured_envs:
            self.assertIn("BENCHMARK_WORKSPACE_DIR", environment)
            self.assertIn("BENCHMARK_SKILL_SCRATCH_DIR", environment)
            self.assertIn("BENCHMARK_SKILL_REQUEST_DIR", environment)
            self.assertIn("BENCHMARK_SKILL_OUTPUT_DIR", environment)
            self.assertIn("BENCHMARK_SKILL_NOTES_DIR", environment)
        self.assertNotIn("retry_focus_prompt_applied", result.runner_meta["timeout_retry"]["attempt_history"][0])
        self.assertNotIn("retry_focus_prompt_applied", result.runner_meta["timeout_retry"]["attempt_history"][1])

    def test_no_timeout_mode_omits_prompt_budget_and_wrapper_timeout(self) -> None:
        captured_commands: list[list[str]] = []
        captured_timeouts: list[int | None] = []
        captured_prompt_kwargs: dict[str, object] = {}

        def prompt_builder(*args, **kwargs):
            captured_prompt_kwargs.update(kwargs)
            return build_single_llm_prompt(*args, **kwargs)

        runner = self._runner(
            captured_commands=captured_commands,
            captured_timeouts=captured_timeouts,
            timeout_once=False,
            no_timeout=True,
            build_prompt=prompt_builder,
        )

        result = runner.run(self._record(), Group(id="single_llm_skills_on", skills_enabled=True))

        self.assertEqual(1, len(captured_commands))
        self.assertEqual([24 * 60 * 60], captured_timeouts)
        self.assertNotIn("--timeout", captured_commands[0])
        prompt = self._message_from_command(captured_commands[0])
        self.assertNotIn("Time budget:", prompt)
        self.assertIn("Chemistry skill catalog:", prompt)
        self.assertNotIn("act-like-a-chemist", prompt)
        self.assertNotIn("Read act-like-a-chemist first", prompt)
        self.assertNotIn("Atomic Coverage Checklist", prompt)
        self.assertNotIn("Do not skip task-relevant derivation steps", prompt)
        self.assertIsNone(captured_prompt_kwargs.get("time_budget_seconds"))
        self.assertEqual("no_timeout", result.runner_meta["timeout_mode"])
        self.assertFalse(result.runner_meta["timeout_retry"]["triggered"])

    def test_skills_on_attempt_uses_stable_workspace_scratch_directory(self) -> None:
        captured_commands: list[list[str]] = []
        captured_envs: list[dict[str, str]] = []
        runner = self._runner(captured_commands=captured_commands, captured_envs=captured_envs, timeout_once=False)

        result = runner.run(self._record(), Group(id="single_llm_skills_on", skills_enabled=True))

        self.assertEqual(1, len(captured_commands))
        session_id = captured_commands[0][captured_commands[0].index("--session-id") + 1]
        scratch_dir = Path(captured_envs[0]["BENCHMARK_SKILL_SCRATCH_DIR"])
        self.assertFalse(scratch_dir.exists())
        self.assertTrue(str(scratch_dir).startswith(str(self.workspace_manager.runtime_root)))
        self.assertEqual(str(scratch_dir / "requests"), captured_envs[0]["BENCHMARK_SKILL_REQUEST_DIR"])
        self.assertEqual(str(scratch_dir / "outputs"), captured_envs[0]["BENCHMARK_SKILL_OUTPUT_DIR"])
        self.assertNotIn(str(scratch_dir), self._message_from_command(captured_commands[0]))
        self.assertEqual("scratch", scratch_dir.name)
        scratch_meta = result.runner_meta["skill_scratch"]
        self.assertEqual(str(scratch_dir), scratch_meta["scratch_dir"])
        self.assertEqual("record-1", scratch_meta["record_id"])
        self.assertEqual(2, scratch_meta["scratch_contract_version"])
        self.assertEqual(session_id, scratch_meta["session_id"])
        self.assertEqual("single_llm_skills_on", scratch_meta["group_id"])

    def test_skills_on_scratch_directory_uses_configured_agent_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace = root / "benchmark" / "workspaces" / "benchmark-single-skills-on"
            config_path = root / "runtime-config" / "single-openclaw.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                json.dumps(
                    {
                        "agents": {
                            "list": [
                                {
                                    "id": "benchmark-single-skills-on",
                                    "workspace": str(workspace),
                                    "agentDir": str(root / "agents" / "benchmark-single-skills-on" / "agent"),
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            captured_commands: list[list[str]] = []
            captured_envs: list[dict[str, str]] = []
            runner = self._runner(
                captured_commands=captured_commands,
                captured_envs=captured_envs,
                timeout_once=False,
                config_path=config_path,
            )

            runner.run(self._record(), Group(id="single_llm_skills_on", skills_enabled=True))

            session_id = captured_commands[0][captured_commands[0].index("--session-id") + 1]
            scratch_dir = Path(captured_envs[0]["BENCHMARK_SKILL_SCRATCH_DIR"])
            self.assertNotEqual((workspace / ".benchmark-scratch" / "record-1" / session_id).resolve(), scratch_dir)
            self.assertTrue(str(scratch_dir).startswith(str(self.workspace_manager.runtime_root)))

    def test_timeout_retry_archives_each_attempt_and_rebuilds_clean_workspace(self) -> None:
        captured_commands: list[list[str]] = []
        observed_old_marker: list[bool] = []
        runner = self._runner(
            captured_commands=captured_commands,
            write_attempt_marker=True,
            observed_old_marker=observed_old_marker,
        )

        result = runner.run(self._record(), Group(id="single_llm_skills_on", skills_enabled=True))

        self.assertEqual([False, False], observed_old_marker)
        history = result.runner_meta["timeout_retry"]["attempt_history"]
        self.assertEqual(2, len(history))
        first_workspace = Path(history[0]["workspace_isolation"]["archive_workspace"])
        second_workspace = Path(history[1]["workspace_isolation"]["archive_workspace"])
        self.assertNotEqual(first_workspace, second_workspace)
        self.assertEqual("1", (first_workspace / "attempt-marker.txt").read_text(encoding="utf-8"))
        self.assertEqual("2", (second_workspace / "attempt-marker.txt").read_text(encoding="utf-8"))
        self.assertFalse(Path(history[1]["workspace_isolation"]["active_workspace"]).exists())

    def test_contamination_finding_discards_scoreable_answer(self) -> None:
        runner = self._runner(
            captured_commands=[],
            timeout_once=False,
            contamination_auditor=lambda **_kwargs: ContaminationAudit(
                status="contaminated",
                findings=(
                    {
                        "rule_id": "forbidden_path",
                        "policy_id": "benchmark_runtime_root",
                        "tool_name": "read",
                        "candidate_source": "read.path",
                        "resolved_path": "/tmp/benchmark-runtime/old/answer.xyz",
                        "matched_root": "/tmp/benchmark-runtime",
                        "command_excerpt": "../old/answer.xyz",
                    },
                ),
            ),
        )

        result = runner.run(self._record(), Group(id="single_llm_skills_on", skills_enabled=True))

        self.assertFalse(result.should_score())
        self.assertEqual("benchmark_workspace_contamination", result.failure.code)
        isolation = result.runner_meta["workspace_isolation"]
        self.assertEqual("confirmed", isolation["contamination_status"])
        self.assertEqual("non_evaluable", isolation["adjudication"])
        self.assertTrue(Path(result.runner_meta["workspace_isolation"]["archive_manifest"]).is_file())

    def test_write_only_boundary_finding_preserves_scoreable_answer(self) -> None:
        finding = {
            "rule_id": "protected_path_access",
            "tool_call_id": "call-write",
            "policy_id": "benchmark_runtime_root",
            "tool_name": "write",
            "candidate_source": "write.path",
            "access_mode": "write",
            "operation_outcome": "succeeded",
            "resolved_path": "/tmp/benchmark-runtime/sibling/generated.py",
            "matched_root": "/tmp/benchmark-runtime",
            "resource_provenance": "unknown",
            "information_exposure": "none",
            "boundary_effect": "violated",
            "evidence": {},
        }
        runner = self._runner(
            captured_commands=[],
            timeout_once=False,
            contamination_auditor=lambda **_kwargs: adjudicate_workspace_findings((finding,)),
        )

        result = runner.run(self._record(), Group(id="single_llm_skills_on", skills_enabled=True))

        self.assertTrue(result.should_score())
        self.assertEqual("X", result.answer.short_answer_text)
        isolation = result.runner_meta["workspace_isolation"]
        self.assertEqual("scoreable_degraded", isolation["adjudication"])
        self.assertEqual("clear", isolation["contamination_status"])

    def test_seal_failure_discards_scoreable_answer(self) -> None:
        runner = self._runner(captured_commands=[], timeout_once=False)
        failure = WorkspaceIsolationError(
            "workspace_archive_failed",
            "archive failed",
            details={"forced": True},
        )

        with mock.patch.object(self.workspace_manager, "seal", side_effect=failure):
            result = runner.run(self._record(), Group(id="single_llm_skills_on", skills_enabled=True))

        self.assertFalse(result.should_score())
        self.assertEqual("workspace_archive_failed", result.failure.code)
        self.assertFalse(result.runner_meta["workspace_isolation"]["archive_ok"])

    def test_provider_failure_keeps_transcript_available_for_workspace_audit(self) -> None:
        root = Path(self.temporary.name)
        agent_root = root / "agents" / "benchmark-single-skills-on"
        sessions_root = agent_root / "sessions"
        sessions_root.mkdir(parents=True)
        config_path = root / "openclaw.json"
        config_path.write_text(
            json.dumps(
                {
                    "agents": {
                        "list": [
                            {
                                "id": "benchmark-single-skills-on",
                                "agentDir": str(agent_root / "agent"),
                                "model": "openai/gpt-5.5",
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )

        def run_subprocess(command: list[str], *, env: dict[str, str], timeout: int) -> CompletedProcess:
            session_id = command[command.index("--session-id") + 1]
            transcript = sessions_root / f"{session_id}.jsonl"
            transcript.write_text(
                json.dumps({"type": "session", "id": session_id, "cwd": env["BENCHMARK_WORKSPACE_DIR"]})
                + "\n",
                encoding="utf-8",
            )
            (sessions_root / "sessions.json").write_text(
                json.dumps(
                    {
                        f"agent:benchmark-single-skills-on:explicit:{session_id}": {
                            "sessionId": session_id,
                            "sessionFile": str(transcript),
                            "modelProvider": "openai",
                            "model": "gpt-5.5",
                        }
                    }
                ),
                encoding="utf-8",
            )
            return CompletedProcess(
                returncode=1,
                stderr="rate limit exceeded: Concurrency limit exceeded for account",
            )

        runner = SingleLLMRunner(
            agent_id="benchmark-single-skills-on",
            timeout_seconds=10,
            config_path=config_path,
            runtime_bundle_root=root / "bundles",
            run_subprocess=run_subprocess,
            parse_json_stdout=lambda result, command: {},
            unwrap_agent_payload=lambda payload: payload,
            summarize_payloads=lambda payloads: "",
            normalize_answer_tracks=lambda *, full_response_text: ("", full_response_text),
            ensure_runtime_bundle=lambda record, *, bundle_root: None,
            build_single_llm_prompt=lambda *args, **kwargs: "BASE PROMPT",
            slugify=lambda value, **kwargs: str(value),
            benchmark_agent_thinking="high",
            workspace_manager=self.workspace_manager,
            timeout_retries=0,
        )

        result = runner.run(self._record(), Group(id="single_llm_skills_on", skills_enabled=True))

        self.assertEqual("provider_rate_limit_error", result.failure.code)
        isolation = result.runner_meta["workspace_isolation"]
        self.assertEqual("complete", isolation["audit_execution_status"])
        self.assertEqual("clear", isolation["contamination_status"])
        transcript_path = result.runner_meta["session_isolation"]["postflight_entry_session_file"]
        self.assertTrue(Path(transcript_path).is_file())
        self.assertEqual("provider_rate_limit_error", result.raw["execution_error"]["code"])

    def test_unavailable_audit_does_not_mask_preexisting_subprocess_failure(self) -> None:
        stderr = 'Error: Thinking level "high" is not supported for minimax/MiniMax-M3. Use one of: off, adaptive.\n'
        runner = self._runner(
            captured_commands=[],
            timeout_once=False,
            failure_stderr=stderr,
            contamination_auditor=lambda **_kwargs: WorkspaceAudit(
                audit_execution_status="unavailable",
                boundary_status="unknown",
                contamination_status="indeterminate",
                adjudication="non_evaluable",
                findings=(
                    {
                        "rule_id": "transcript_unavailable",
                        "candidate_source": "session_isolation.postflight_entry_session_file",
                    },
                ),
            ),
        )

        result = runner.run(self._record(), Group(id="single_llm_skills_on", skills_enabled=True))

        self.assertIsNotNone(result.failure)
        self.assertEqual("openclaw_thinking_level_unsupported", result.failure.code)
        self.assertEqual(stderr.strip(), result.failure.message)
        self.assertEqual(stderr, result.raw["execution_error"]["stderr_excerpt"])
        self.assertNotIn("discarded_runner_raw", result.raw)
        isolation = result.runner_meta["workspace_isolation"]
        self.assertEqual("unavailable", isolation["audit_execution_status"])
        self.assertEqual("indeterminate", isolation["contamination_status"])
        self.assertEqual("non_evaluable", isolation["adjudication"])

    def test_confirmed_contamination_still_overrides_preexisting_subprocess_failure(self) -> None:
        runner = self._runner(
            captured_commands=[],
            timeout_once=False,
            failure_stderr="OpenClaw startup failed\n",
            contamination_auditor=lambda **_kwargs: WorkspaceAudit(
                audit_execution_status="complete",
                boundary_status="violated",
                contamination_status="confirmed",
                adjudication="non_evaluable",
                findings=(
                    {
                        "rule_id": "protected_path_access",
                        "information_exposure": "possible",
                        "boundary_effect": "violated",
                    },
                ),
            ),
        )

        result = runner.run(self._record(), Group(id="single_llm_skills_on", skills_enabled=True))

        self.assertIsNotNone(result.failure)
        self.assertEqual("benchmark_workspace_contamination", result.failure.code)
        self.assertEqual("confirmed", result.runner_meta["workspace_isolation"]["contamination_status"])
        self.assertEqual("openclaw_subprocess_failed", result.raw["discarded_runner_raw"]["execution_error"]["code"])

    def test_provider_access_denied_preserves_primary_error_and_attempt_history(self) -> None:
        raw_error = "403 Access to model denied. Please make sure you are eligible for using the model."
        stderr = (
            "[agent/embedded] embedded run agent end: error=LLM request failed. "
            f"rawError={raw_error}"
        )
        runner = self._runner(
            captured_commands=[],
            timeout_once=False,
            failure_stderr=stderr,
        )

        result = runner.run(self._record(), Group(id="single_llm_skills_on", skills_enabled=True))

        assert result.failure is not None
        self.assertEqual("provider_access_denied", result.failure.code)
        self.assertEqual(
            "Access to model denied. Please make sure you are eligible for using the model.",
            result.failure.message,
        )
        execution_error = result.raw["execution_error"]
        self.assertEqual(403, execution_error["primary_error"]["status_code"])
        self.assertEqual(raw_error, execution_error["primary_error"]["raw"])
        attempt_error = result.runner_meta["timeout_retry"]["attempt_history"][0]["execution_error"]
        self.assertEqual(execution_error, attempt_error)
        self.assertFalse(result.runner_meta["timeout_retry"]["triggered"])

    def test_stream_read_error_retries_with_fresh_attempt(self) -> None:
        captured_commands: list[list[str]] = []
        stderr = "\n".join(
            (
                "[openai-transport] [responses] error provider=openai message=stream_read_error",
                "[agent/embedded] embedded run agent end: isError=true "
                "error=LLM request timed out. rawError=stream_read_error",
                "[agent/embedded] embedded run failover decision: decision=surface_error "
                "reason=timeout rawError=stream_read_error",
                '[diagnostic] lane task error: error="FailoverError: LLM request timed out."',
            )
        )
        runner = self._runner(
            captured_commands=captured_commands,
            timeout_once=False,
            failure_stderr=stderr,
        )

        result = runner.run(self._record(), Group(id="single_llm_skills_on", skills_enabled=True))

        self.assertEqual(2, len(captured_commands))
        session_ids = [command[command.index("--session-id") + 1] for command in captured_commands]
        self.assertEqual(2, len(set(session_ids)))
        retry = result.runner_meta["timeout_retry"]
        self.assertTrue(retry["triggered"])
        self.assertTrue(retry["exhausted"])
        self.assertEqual(1, retry["retries_used"])
        self.assertEqual("provider_transport_error", retry["retry_reason"])
        self.assertEqual(
            ["provider_transport_error", "provider_transport_error"],
            [item["failure_code"] for item in retry["attempt_history"]],
        )
        archived_workspaces = [
            Path(item["workspace_isolation"]["archive_workspace"])
            for item in retry["attempt_history"]
        ]
        self.assertNotEqual(archived_workspaces[0], archived_workspaces[1])
        self.assertTrue(all(path.is_dir() for path in archived_workspaces))


if __name__ == "__main__":
    unittest.main()
