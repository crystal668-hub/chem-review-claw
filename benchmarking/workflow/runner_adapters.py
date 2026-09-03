from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from benchmarking.core.answer_processing import normalize_answer_tracks, normalize_space
from benchmarking.core.convergence import ConvergencePolicy
from benchmarking.core.status import (
    is_chemqa_success_status,
    is_chemqa_terminal_status,
    normalize_chemqa_run_status,
)
from benchmarking.runtime import bundles as runtime_bundles
from benchmarking.runtime import paths as runtime_paths
from benchmarking.runtime import subprocess_utils
from benchmarking.runtime.agent_workspace import (
    AttemptWorkspaceManager,
    default_workspace_templates,
)
from benchmarking.runtime.cancellation import (
    CancellationOutcome,
    CancellationReason,
    CancellationToken,
    OwnedProcessRegistry,
)
from benchmarking.runtime.cleanroom import (
    CleanroomError,
    CleanroomRuntime,
    iter_pending_cleanup_manifests,
)
from benchmarking.runtime.config_pool import actual_slot_ids
from benchmarking.runtime.session_isolation import inspect_postflight_session
from benchmarking.runtime.workspace_policy import ContaminationAudit, ProtectedRoot
from benchmarking.workflow.chemqa_response import (
    build_chemqa_full_response,
    build_chemqa_response_from_submission,
    load_yaml_mapping,
)
from benchmarking.workflow.errors import BenchmarkError, CleanupFatalError
from benchmarking.workflow.experiments import (
    DEFAULT_CHEMQA_PRESET,
    DEFAULT_SINGLE_AGENT_THINKING,
)
from benchmarking.workflow.prompts import (
    build_chemqa_goal,
    build_single_llm_prompt,
    resolve_chemqa_answer_kind,
)
from benchmarking.workflow.run_state import now_stamp, slugify
from benchmarking.workflow.runners import ChemQARunner as BaseChemQARunner
from benchmarking.workflow.runners import SingleLLMRunner as BaseSingleLLMRunner
from benchmarking.workflow.runners import build_runner as build_base_runner

DEFAULT_OPENCLAW_ENV_FILE = runtime_paths.openclaw_env
DEFAULT_CLEANROOM_ROOT = runtime_paths.skills_root / "benchmark-cleanroom"


def _compatibility_protected_roots(*, runtime_root: Path, output_root: Path) -> tuple[ProtectedRoot, ...]:
    return (
        ProtectedRoot("benchmark_runtime_root", runtime_root, "compatibility.runtime_root"),
        ProtectedRoot("current_output_root", output_root, "compatibility.output_root"),
    )


def _load_cleanroom_runtime(*, process_registry: OwnedProcessRegistry | None = None) -> CleanroomRuntime:
    try:
        return CleanroomRuntime.load(
            cleanroom_root=DEFAULT_CLEANROOM_ROOT,
            current_python=lambda: subprocess_utils.current_python(),
            run_subprocess=lambda *args, **kwargs: subprocess_utils.run_owned_subprocess(
                *args,
                process_registry=process_registry,
                **kwargs,
            ),
        )
    except CleanroomError as exc:
        raise BenchmarkError(str(exc)) from exc


def run_pending_cleanroom_cleanup() -> list[dict[str, Any]]:
    if not iter_pending_cleanup_manifests():
        return []
    return _load_cleanroom_runtime().run_pending_cleanroom_cleanup()


class _CancellationRunnerMixin:
    _cancellation_token: CancellationToken
    _process_registry: OwnedProcessRegistry

    def cancel(self, reason: CancellationReason) -> None:
        self._cancellation_enabled = True
        self._cancellation_token.cancel(reason)

    def wait_cancelled(self, deadline: float) -> CancellationOutcome:
        return self._process_registry.wait_cancelled(deadline)


class SingleLLMRunner(_CancellationRunnerMixin, BaseSingleLLMRunner):
    def __init__(
        self,
        *,
        agent_id: str,
        timeout_seconds: int,
        config_path: Path,
        runtime_bundle_root: Path,
        configured_skills: tuple[str, ...] | list[str] = (),
        vgb_configured_skills: tuple[str, ...] | list[str] = (),
        skill_health_summary: dict[str, Any] | None = None,
        convergence_policy: ConvergencePolicy | None = None,
        timeout_retries: int = 3,
        timeout_retry_backoff_seconds: tuple[int | float, ...] | list[int | float] = (5, 15, 45),
        sleep_fn=time.sleep,
        benchmark_agent_thinking: str = DEFAULT_SINGLE_AGENT_THINKING,
        no_timeout: bool = False,
        pypi_cutoff: str | None = None,
        workspace_manager: AttemptWorkspaceManager | None = None,
        contamination_auditor=None,
        cancellation_token: CancellationToken | None = None,
        process_registry: OwnedProcessRegistry | None = None,
    ) -> None:
        self._cancellation_enabled = cancellation_token is not None
        self._cancellation_token = cancellation_token or CancellationToken()
        self._process_registry = process_registry or OwnedProcessRegistry(
            cancellation_token=self._cancellation_token
        )
        compatibility_manager = workspace_manager is None
        if workspace_manager is None:
            compatibility_root = runtime_bundle_root.expanduser().resolve()
            runtime_root = compatibility_root / ".benchmark-test-workspaces" / "runs"
            output_root = compatibility_root / ".benchmark-test-output"
            workspace_manager = AttemptWorkspaceManager(
                runtime_root=runtime_root,
                output_root=output_root,
                run_id="runner-test",
                invocation_id=uuid.uuid4().hex,
                templates=default_workspace_templates(runtime_paths.project_root),
                protected_roots=_compatibility_protected_roots(
                    runtime_root=runtime_root,
                    output_root=output_root,
                ),
            )
        if compatibility_manager and contamination_auditor is None:
            contamination_auditor = lambda **_kwargs: ContaminationAudit(status="clean")
        super().__init__(
            agent_id=agent_id,
            timeout_seconds=timeout_seconds,
            config_path=config_path,
            runtime_bundle_root=runtime_bundle_root,
            configured_skills=configured_skills,
            vgb_configured_skills=vgb_configured_skills,
            skill_health_summary=skill_health_summary,
            convergence_policy=convergence_policy,
            timeout_retries=timeout_retries,
            timeout_retry_backoff_seconds=timeout_retry_backoff_seconds,
            sleep_fn=sleep_fn,
            no_timeout=no_timeout,
            pypi_cutoff=pypi_cutoff,
            run_subprocess=lambda *args, **kwargs: subprocess_utils.run_owned_subprocess(
                *args,
                cancellation_token=self._cancellation_token,
                process_registry=self._process_registry,
                **kwargs,
            ),
            parse_json_stdout=subprocess_utils.parse_json_stdout,
            unwrap_agent_payload=subprocess_utils.unwrap_agent_payload,
            summarize_payloads=subprocess_utils.summarize_payloads,
            normalize_answer_tracks=normalize_answer_tracks,
            ensure_runtime_bundle=lambda record, *, bundle_root: runtime_bundles.ensure_runtime_bundle(
                record,
                bundle_root=bundle_root,
            ),
            build_single_llm_prompt=build_single_llm_prompt,
            slugify=slugify,
            benchmark_agent_thinking=benchmark_agent_thinking,
            workspace_manager=workspace_manager,
            allowed_workspace_roots=(
                runtime_paths.skills_root,
                runtime_paths.project_root / "scripts" / "run_skill.py",
            ),
            contamination_auditor=contamination_auditor,
        )


class ChemQARunner(_CancellationRunnerMixin, BaseChemQARunner):
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
        convergence_policy: ConvergencePolicy | None = None,
        workspace_manager: AttemptWorkspaceManager | None = None,
        contamination_auditor=None,
        cancellation_token: CancellationToken | None = None,
        process_registry: OwnedProcessRegistry | None = None,
    ) -> None:
        self._cancellation_enabled = cancellation_token is not None
        self._cancellation_token = cancellation_token or CancellationToken()
        self._process_registry = process_registry or OwnedProcessRegistry(
            cancellation_token=self._cancellation_token
        )
        compatibility_manager = workspace_manager is None
        if workspace_manager is None:
            compatibility_root = runtime_bundle_root.expanduser().resolve()
            runtime_root = compatibility_root / ".benchmark-test-workspaces" / "runs"
            output_root = compatibility_root / ".benchmark-test-output"
            workspace_manager = AttemptWorkspaceManager(
                runtime_root=runtime_root,
                output_root=output_root,
                run_id="chemqa-runner-test",
                invocation_id=uuid.uuid4().hex,
                templates=default_workspace_templates(runtime_paths.project_root),
                protected_roots=_compatibility_protected_roots(
                    runtime_root=runtime_root,
                    output_root=output_root,
                ),
            )
        if compatibility_manager and contamination_auditor is None:
            contamination_auditor = lambda **_kwargs: ContaminationAudit(status="clean")
        cleanroom = _load_cleanroom_runtime(process_registry=self._process_registry)
        super().__init__(
            chemqa_root=chemqa_root,
            timeout_seconds=timeout_seconds,
            config_path=config_path,
            slot_set=slot_set,
            review_rounds=review_rounds,
            rebuttal_rounds=rebuttal_rounds,
            model_profile=model_profile,
            runtime_bundle_root=runtime_bundle_root,
            launch_workspace_root=launch_workspace_root,
            launch_script=chemqa_root / "scripts" / "launch_from_preset.py",
            collect_script=chemqa_root / "scripts" / "collect_artifacts.py",
            runtime_dir=chemqa_root.parent / "debateclaw-v1" / "scripts",
            current_python=lambda: subprocess_utils.current_python(),
            run_subprocess=lambda *args, **kwargs: subprocess_utils.run_owned_subprocess(
                *args,
                cancellation_token=self._cancellation_token,
                process_registry=self._process_registry,
                **kwargs,
            ),
            parse_json_stdout=subprocess_utils.parse_json_stdout,
            deep_copy_jsonish=subprocess_utils.deep_copy_jsonish,
            ensure_runtime_bundle=lambda record, *, bundle_root: runtime_bundles.ensure_runtime_bundle(
                record,
                bundle_root=bundle_root,
            ),
            build_chemqa_goal=build_chemqa_goal,
            resolve_chemqa_answer_kind=resolve_chemqa_answer_kind,
            cleanup_manifest_path=cleanroom.cleanup_manifest_path,
            build_cleanup_manifest_payload=cleanroom.build_cleanup_manifest_payload,
            write_cleanup_manifest=cleanroom.write_cleanup_manifest,
            register_pending_cleanup_manifest=cleanroom.register_pending_cleanup_manifest,
            update_cleanup_manifest=cleanroom.update_cleanup_manifest,
            invoke_cleanroom_cleanup=cleanroom.invoke_cleanroom_cleanup,
            unregister_pending_cleanup_manifest=cleanroom.unregister_pending_cleanup_manifest,
            now_stamp=now_stamp,
            slugify=slugify,
            default_chemqa_preset=DEFAULT_CHEMQA_PRESET,
            default_openclaw_env_file=DEFAULT_OPENCLAW_ENV_FILE,
            actual_slot_ids=actual_slot_ids,
            workspace_manager=workspace_manager,
            session_audit_resolver=lambda agent_id, session_id, *, session_store_path=None: inspect_postflight_session(
                agent_id,
                session_id,
                config_path=config_path,
                session_store_path=session_store_path,
            ),
            contamination_auditor=contamination_auditor,
            allowed_workspace_roots=(runtime_paths.skills_root,),
            unique_run_suffix=not compatibility_manager,
            normalize_chemqa_run_status=normalize_chemqa_run_status,
            is_chemqa_terminal_status=is_chemqa_terminal_status,
            is_chemqa_success_status=is_chemqa_success_status,
            build_chemqa_full_response=build_chemqa_full_response,
            build_chemqa_response_from_submission=build_chemqa_response_from_submission,
            load_yaml_mapping=load_yaml_mapping,
            normalize_space=normalize_space,
            benchmark_error_factory=BenchmarkError,
            cleanup_error_factory=CleanupFatalError,
            convergence_policy=convergence_policy,
        )

    def _wait_for_terminal_status(self, run_id: str, *, timeout_seconds: int) -> dict[str, Any]:
        if not hasattr(self, "_is_chemqa_terminal_status"):
            self._is_chemqa_terminal_status = is_chemqa_terminal_status
        if not hasattr(self, "_normalize_chemqa_run_status"):
            self._normalize_chemqa_run_status = normalize_chemqa_run_status
        if not hasattr(self, "_benchmark_error_factory"):
            self._benchmark_error_factory = BenchmarkError
        if not hasattr(self, "convergence_policy"):
            self.convergence_policy = ConvergencePolicy(timeout_seconds=timeout_seconds)
        return super()._wait_for_terminal_status(run_id, timeout_seconds=timeout_seconds)

    def _candidate_protocol_dirs(self, run_id: str, run_status: dict[str, Any]) -> list[Path]:
        if not hasattr(self, "_actual_slot_ids"):
            self._actual_slot_ids = actual_slot_ids
        return super()._candidate_protocol_dirs(run_id, run_status)

    def _build_candidate_submission_fallback(
        self,
        run_id: str,
        run_status: dict[str, Any],
    ) -> tuple[str, str, dict[str, Any]] | None:
        if not hasattr(self, "_load_yaml_mapping"):
            self._load_yaml_mapping = load_yaml_mapping
        if not hasattr(self, "_build_chemqa_response_from_submission"):
            self._build_chemqa_response_from_submission = build_chemqa_response_from_submission
        if not hasattr(self, "_normalize_space"):
            self._normalize_space = normalize_space
        return super()._build_candidate_submission_fallback(run_id, run_status)


def build_runner(*, runner_kind: str, **kwargs: Any) -> Any:
    return build_base_runner(
        runner_kind=runner_kind,
        chemqa_runner_cls=ChemQARunner,
        single_llm_runner_cls=SingleLLMRunner,
        **kwargs,
    )
