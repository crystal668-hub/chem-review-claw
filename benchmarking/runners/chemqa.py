from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..contracts import AnswerPayload, FailureInfo, RecoveryInfo, RunnerResult, RunStatus


class ChemQARunner:
    def __init__(
        self,
        *,
        chemqa_root: Path,
        timeout_seconds: int,
        config_path: Path,
        slot_set: str,
        review_rounds: int | None,
        rebuttal_rounds: int | None,
        model_profile: str,
        runtime_bundle_root: Path,
        launch_workspace_root: Path,
        launch_script: Path,
        collect_script: Path,
        runtime_dir: Path,
        current_python,
        run_subprocess,
        parse_json_stdout,
        deep_copy_jsonish,
        ensure_runtime_bundle,
        build_chemqa_goal,
        cleanup_manifest_path,
        build_cleanup_manifest_payload,
        write_cleanup_manifest,
        register_pending_cleanup_manifest,
        update_cleanup_manifest,
        invoke_cleanroom_cleanup,
        unregister_pending_cleanup_manifest,
        now_stamp,
        slugify,
        default_chemqa_preset: str,
        default_openclaw_env_file: Path,
        actual_slot_ids,
        chemqa_workspace_roots,
        normalize_chemqa_run_status,
        is_chemqa_terminal_status,
        is_chemqa_success_status,
        build_chemqa_full_response,
        build_chemqa_response_from_submission,
        load_yaml_mapping,
        normalize_space,
        benchmark_error_factory=None,
        cleanup_error_factory=None,
        benchmark_agent_thinking: str | None = None,
    ) -> None:
        self.chemqa_root = chemqa_root
        self.timeout_seconds = timeout_seconds
        self.config_path = config_path
        self.slot_set = slot_set
        self.review_rounds = review_rounds
        self.rebuttal_rounds = rebuttal_rounds
        self.model_profile = model_profile
        self.runtime_bundle_root = runtime_bundle_root
        self.launch_workspace_root = launch_workspace_root
        self.launch_script = launch_script
        self.collect_script = collect_script
        self.runtime_dir = runtime_dir
        self._current_python = current_python
        self._run_subprocess = run_subprocess
        self._parse_json_stdout = parse_json_stdout
        self._deep_copy_jsonish = deep_copy_jsonish
        self._ensure_runtime_bundle = ensure_runtime_bundle
        self._build_chemqa_goal = build_chemqa_goal
        self._cleanup_manifest_path = cleanup_manifest_path
        self._build_cleanup_manifest_payload = build_cleanup_manifest_payload
        self._write_cleanup_manifest = write_cleanup_manifest
        self._register_pending_cleanup_manifest = register_pending_cleanup_manifest
        self._update_cleanup_manifest = update_cleanup_manifest
        self._invoke_cleanroom_cleanup = invoke_cleanroom_cleanup
        self._unregister_pending_cleanup_manifest = unregister_pending_cleanup_manifest
        self._now_stamp = now_stamp
        self._slugify = slugify
        self._default_chemqa_preset = default_chemqa_preset
        self._default_openclaw_env_file = default_openclaw_env_file
        self._actual_slot_ids = actual_slot_ids
        self._chemqa_workspace_roots = chemqa_workspace_roots
        self._normalize_chemqa_run_status = normalize_chemqa_run_status
        self._is_chemqa_terminal_status = is_chemqa_terminal_status
        self._is_chemqa_success_status = is_chemqa_success_status
        self._build_chemqa_full_response = build_chemqa_full_response
        self._build_chemqa_response_from_submission = build_chemqa_response_from_submission
        self._load_yaml_mapping = load_yaml_mapping
        self._normalize_space = normalize_space
        self._benchmark_error_factory = benchmark_error_factory
        self._cleanup_error_factory = cleanup_error_factory
        self._benchmark_agent_thinking = benchmark_agent_thinking

    def _status_path(self, run_id: str) -> Path:
        return self.chemqa_root / "control" / "run-status" / f"{run_id}.json"

    def _read_run_status(self, run_id: str) -> dict[str, Any]:
        status_path = self._status_path(run_id)
        if not status_path.is_file():
            return {}
        return self._normalize_chemqa_run_status(json.loads(status_path.read_text(encoding="utf-8")))

    def _wait_for_terminal_status(self, run_id: str, *, timeout_seconds: int) -> dict[str, Any]:
        import time

        deadline = time.time() + timeout_seconds
        last_status: dict[str, Any] = {}
        while time.time() < deadline:
            last_status = self._read_run_status(run_id)
            if self._is_chemqa_terminal_status(last_status):
                return last_status
            time.sleep(30)
        error_message = (
            f"ChemQA run `{run_id}` did not reach a terminal state within {timeout_seconds}s. Last status: {last_status}"
        )
        if self._benchmark_error_factory is not None:
            raise self._benchmark_error_factory(error_message)
        raise RuntimeError(error_message)

    def _candidate_protocol_dirs(self, run_id: str, run_status: dict[str, Any]) -> list[Path]:
        candidates: list[Path] = []
        explicit_protocol = str(run_status.get("protocol_path") or "").strip()
        explicit_workspace_protocol = str(run_status.get("workspace_protocol_path") or "").strip()
        if explicit_protocol:
            candidates.append(Path(explicit_protocol).expanduser().resolve().parent)
        if explicit_workspace_protocol:
            candidates.append(Path(explicit_workspace_protocol).expanduser().resolve().parent)

        protocol_dir = self.chemqa_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id
        candidates.append(protocol_dir)
        coordinator_slot = self._actual_slot_ids(self.slot_set)["debate-coordinator"]
        coordinator_workspace = self._chemqa_workspace_roots[self.slot_set] / coordinator_slot
        candidates.append(coordinator_workspace)

        deduped: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate.resolve()) if candidate.exists() else str(candidate)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def _resolve_existing_qa_result(self, run_id: str, run_status: dict[str, Any]) -> Path | None:
        explicit_output_dir = str(run_status.get("artifacts_output_dir") or "").strip()
        candidate_dirs = []
        if explicit_output_dir:
            candidate_dirs.append(Path(explicit_output_dir).expanduser().resolve())
        candidate_dirs.append(self.chemqa_root / "generated" / "artifacts" / run_id)
        for directory in candidate_dirs:
            path = directory / "qa_result.json"
            if path.is_file():
                return path
        return None

    def _candidate_submission_paths(self, run_id: str, run_status: dict[str, Any]) -> list[Path]:
        import re

        candidates: list[Path] = []
        for root in self._candidate_protocol_dirs(run_id, run_status):
            if not root.exists():
                continue
            for path in root.rglob("proposer-1.md"):
                if "proposals" not in path.parts:
                    continue
                candidates.append(path.resolve())

        deduped: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)

        def sort_key(path: Path) -> tuple[int, float, str]:
            match = re.search(r"epoch-(\d+)", str(path))
            epoch = int(match.group(1)) if match else -1
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            return (epoch, mtime, str(path))

        return sorted(deduped, key=sort_key, reverse=True)

    def _build_candidate_submission_fallback(self, run_id: str, run_status: dict[str, Any]) -> tuple[str, str, dict[str, Any]] | None:
        for proposal_path in self._candidate_submission_paths(run_id, run_status):
            proposal_payload = self._load_yaml_mapping(proposal_path)
            if not proposal_payload:
                continue
            short_answer_text, full_response_text = self._build_chemqa_response_from_submission(final_submission=proposal_payload)
            if short_answer_text:
                return short_answer_text, full_response_text, {
                    "fallback_source": "proposer-1-proposal",
                    "proposal_path": str(proposal_path),
                    "proposal_payload": proposal_payload,
                }

        preview = self._normalize_space(str(run_status.get("final_answer_preview") or ""))
        if preview:
            return preview, f"FINAL ANSWER: {preview}", {
                "fallback_source": "run-status-final-answer-preview",
            }
        return None

    def _collect_artifacts_from_source(self, *, source_dir: Path, output_dir: Path, env: dict[str, str]) -> None:
        command = [
            self._current_python(),
            str(self.collect_script),
            "--skill-root",
            str(self.chemqa_root),
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
        ]
        result = self._run_subprocess(command, env=env, cwd=self.chemqa_root, timeout=120)
        self._parse_json_stdout(result, command)

    def _ensure_artifacts(
        self,
        run_id: str,
        *,
        env: dict[str, str],
        run_status: dict[str, Any],
        wait_seconds: int = 120,
        poll_seconds: int = 5,
    ) -> Path:
        import time

        deadline = time.time() + wait_seconds
        last_seen_status = run_status
        checked_sources: list[str] = []
        while time.time() < deadline:
            last_seen_status = self._read_run_status(run_id) or last_seen_status
            qa_result_path = self._resolve_existing_qa_result(run_id, last_seen_status)
            if qa_result_path is not None:
                return qa_result_path

            output_dir = Path(
                str(last_seen_status.get("artifacts_output_dir") or (self.chemqa_root / "generated" / "artifacts" / run_id))
            ).expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)

            for source_dir in self._candidate_protocol_dirs(run_id, last_seen_status):
                checked_sources.append(str(source_dir))
                if (source_dir / "chemqa_review_protocol.yaml").is_file() or (source_dir / "chemqa_review_protocol.yml").is_file():
                    self._collect_artifacts_from_source(source_dir=source_dir, output_dir=output_dir, env=env)
                    qa_result_path = output_dir / "qa_result.json"
                    if qa_result_path.is_file():
                        return qa_result_path
            time.sleep(poll_seconds)

        error_message = (
            f"ChemQA run `{run_id}` reached terminal state but artifacts were not resolved within {wait_seconds}s. "
            f"Last run status: {last_seen_status}. Checked sources: {checked_sources}"
        )
        if self._benchmark_error_factory is not None:
            raise self._benchmark_error_factory(error_message)
        raise RuntimeError(error_message)

    def run(self, record: Any, group: Any) -> RunnerResult:
        payload: dict[str, Any] = {}
        run_id = f"benchmark-{group.id}-{self._slugify(record.record_id, limit=40)}-{self._now_stamp()}"
        input_bundle = self._ensure_runtime_bundle(record, bundle_root=self.runtime_bundle_root)
        goal = self._build_chemqa_goal(record, websearch_enabled=group.websearch, input_bundle=input_bundle)
        launch_root = self.launch_workspace_root / group.id / self._slugify(record.record_id, limit=80)
        launch_home = launch_root / "home"
        template_dir = launch_home / ".clawteam" / "templates"
        command_map_dir = launch_root / "command-maps"
        template_dir.mkdir(parents=True, exist_ok=True)
        command_map_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self._cleanup_manifest_path(self.launch_workspace_root.parent, run_id)
        initial_manifest = self._build_cleanup_manifest_payload(
            run_id=run_id,
            benchmark_kind="chemqa",
            group_id=group.id,
            output_root=self.launch_workspace_root.parent,
            launch_home=launch_home,
            control_roots=[
                self.chemqa_root / "control" / "runplans" / f"{run_id}.json",
                self.chemqa_root / "control" / "run-status" / f"{run_id}.json",
            ],
            generated_roots=[
                self.chemqa_root / "generated" / "command-maps" / f"{run_id}-command-map.json",
                self.chemqa_root / "generated" / "prompt-bundles" / f"{run_id}-prompts.json",
                self.chemqa_root / "generated" / "runtime-context" / f"{run_id}-context.json",
            ],
            artifact_roots=[
                self.chemqa_root / "generated" / "artifacts" / run_id,
                self.chemqa_root / "generated" / "clawteam-data" / "runs" / run_id,
                launch_root,
            ],
            extra={
                "record_id": record.record_id,
                "template_dir": str(template_dir),
                "command_map_dir": str(command_map_dir),
            },
        )
        self._write_cleanup_manifest(manifest_path, initial_manifest)
        self._register_pending_cleanup_manifest(manifest_path)
        command = [
            self._current_python(),
            str(self.launch_script),
            "--root",
            str(self.chemqa_root),
            "--preset",
            self._default_chemqa_preset,
            "--goal",
            goal,
            "--run-id",
            run_id,
            "--model-profile",
            self.model_profile,
            "--slot-set",
            self.slot_set,
            "--openclaw-config",
            str(self.config_path),
            "--template-dir",
            str(template_dir),
            "--command-map-dir",
            str(command_map_dir),
            "--runtime-dir",
            str(self.runtime_dir),
            "--launch-mode",
            "run",
        ]
        if input_bundle is not None:
            command.extend(["--additional-file-workspace", str(input_bundle.bundle_dir)])
        if self.review_rounds is not None:
            command.extend(["--review-rounds", str(self.review_rounds)])
        if self.rebuttal_rounds is not None:
            command.extend(["--rebuttal-rounds", str(self.rebuttal_rounds)])

        env = os.environ.copy()
        env["HOME"] = str(launch_home)
        env["OPENCLAW_CONFIG_PATH"] = str(self.config_path)
        env["OPENCLAW_ENV_FILE"] = str(self._default_openclaw_env_file)
        env["OPENCLAW_DEBATE_TRUSTED_PLUGINS"] = "duckduckgo" if group.websearch else "__none__"
        env["BENCHMARK_CLEANROOM_RUN_ID"] = run_id
        env["BENCHMARK_CLEANROOM_LEASE_DIR"] = str((self.launch_workspace_root.parent / "cleanroom" / "leases").resolve())

        try:
            result = self._run_subprocess(command, env=env, cwd=self.chemqa_root, timeout=self.timeout_seconds)
            payload = self._parse_json_stdout(result, command)
            materialize = self._deep_copy_jsonish((payload.get("materialize") or {}))
            self._update_cleanup_manifest(
                manifest_path,
                {
                    "launch_home": str(launch_home.resolve()),
                    "clawteam_data_dir": str(
                        Path(str(materialize.get("clawteam_data_dir") or (self.chemqa_root / "generated" / "clawteam-data" / "runs" / run_id)))
                        .expanduser()
                        .resolve()
                    ),
                    "session_assignments": self._deep_copy_jsonish(((payload.get("compile") or {}).get("session_assignments") or {})),
                    "control_roots": [
                        str(self.chemqa_root / "control" / "runplans" / f"{run_id}.json"),
                        str(self.chemqa_root / "control" / "run-status" / f"{run_id}.json"),
                    ],
                    "generated_roots": [
                        str((command_map_dir / f"{run_id}-command-map.json").resolve()),
                        str(self.chemqa_root / "generated" / "prompt-bundles" / f"{run_id}-prompts.json"),
                        str(self.chemqa_root / "generated" / "runtime-context" / f"{run_id}-context.json"),
                        str(template_dir),
                    ],
                    "artifact_roots": [
                        str(self.chemqa_root / "generated" / "artifacts" / run_id),
                        str(self.chemqa_root / "generated" / "clawteam-data" / "runs" / run_id),
                        str(launch_root.resolve()),
                    ],
                    "launch_payload": self._deep_copy_jsonish(payload),
                },
            )
            run_status = self._wait_for_terminal_status(run_id, timeout_seconds=self.timeout_seconds)
            terminal_state = str(run_status.get("terminal_state") or "")
            terminal_reason_code = str(run_status.get("terminal_reason_code") or "")
            legacy_status = str(run_status.get("legacy_status") or "")
            artifact_collection = self._deep_copy_jsonish(run_status.get("artifact_collection") or {})

            if not self._is_chemqa_success_status(run_status):
                message = (
                    f"ChemQA run ended with non-success status: "
                    f"{terminal_state or legacy_status or 'unknown'}"
                )
                runner_meta = {
                    "run_id": run_id,
                    "launch": payload,
                    "acceptance_status": None,
                    "terminal_state": terminal_state or "unknown",
                    "terminal_reason_code": terminal_reason_code or "",
                    "artifact_collection": artifact_collection,
                    "run_status": run_status,
                    "non_success_terminal_status": legacy_status or terminal_state or "unknown",
                    "missing_reviewer_lanes": list(((run_status.get("phase_progress") or {}).get("missing_reviewer_lanes") or [])),
                    "error": message,
                }
                if legacy_status:
                    runner_meta["legacy_status"] = legacy_status
                if input_bundle is not None:
                    runner_meta["runtime_bundle"] = input_bundle.to_meta()
                fallback_payload = self._build_candidate_submission_fallback(run_id, run_status)
                if fallback_payload is not None:
                    short_answer_text, full_response_text, fallback_meta = fallback_payload
                    runner_meta.update(
                        {
                            "fallback_used": True,
                            **fallback_meta,
                        }
                    )
                    return RunnerResult(
                        status=RunStatus.RECOVERED,
                        answer=AnswerPayload(
                            short_answer_text=short_answer_text,
                            full_response_text=full_response_text,
                        ),
                        raw={"run_status": run_status, "fallback": fallback_meta},
                        runner_meta=runner_meta,
                        recovery=RecoveryInfo(
                            source=str(fallback_meta["fallback_source"]),
                            scored=False,
                            details=fallback_meta,
                        ),
                    )
                return RunnerResult(
                    status=RunStatus.FAILED,
                    answer=AnswerPayload(),
                    raw={"run_status": run_status},
                    runner_meta=runner_meta,
                    failure=FailureInfo(
                        code=terminal_reason_code or "chemqa_non_success_terminal_status",
                        message=message,
                        details={"run_status": run_status},
                    ),
                )

            qa_result_path = self._ensure_artifacts(run_id, env=env, run_status=run_status)
            qa_result = json.loads(qa_result_path.read_text(encoding="utf-8"))
            short_answer_text, full_response_text = self._build_chemqa_full_response(qa_result=qa_result)
            runner_meta = {
                "run_id": run_id,
                "launch": payload,
                "qa_result_path": str(qa_result_path),
                "acceptance_status": qa_result.get("acceptance_status"),
                "terminal_state": terminal_state or qa_result.get("terminal_state"),
                "terminal_reason_code": terminal_reason_code or "",
                "artifact_collection": artifact_collection,
                "run_status": run_status,
            }
            if legacy_status:
                runner_meta["legacy_status"] = legacy_status
            if input_bundle is not None:
                runner_meta["runtime_bundle"] = input_bundle.to_meta()
            return RunnerResult(
                status=RunStatus.COMPLETED,
                answer=AnswerPayload(
                    short_answer_text=short_answer_text,
                    full_response_text=full_response_text,
                ),
                raw=qa_result,
                runner_meta=runner_meta,
            )
        finally:
            try:
                cleanup_report = self._invoke_cleanroom_cleanup(manifest_path=manifest_path)
            except Exception as exc:
                self._unregister_pending_cleanup_manifest(manifest_path)
                if self._cleanup_error_factory is not None:
                    raise self._cleanup_error_factory(f"ChemQA cleanup failed for run `{run_id}`: {exc}") from exc
                raise
            else:
                self._unregister_pending_cleanup_manifest(manifest_path)
                if payload:
                    payload.setdefault("cleanup_report", cleanup_report)
