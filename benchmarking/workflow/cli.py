#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from benchmarking.analysis.launcher import launch_automated_evaluation
from benchmarking.core.answer_processing import normalize_answer_tracks
from benchmarking.core.convergence import ConvergencePolicy
from benchmarking.core.datasets import BenchmarkRecord as _BenchmarkRecord
from benchmarking.core.datasets import classify_subset
from benchmarking.core.reporting import (
    GroupRecordResult as _GroupRecordResult,
)
from benchmarking.core.reporting import (
    aggregate_results,
)
from benchmarking.core.reporting import (
    build_error_group_record_result as _build_error_group_record_result,
)
from benchmarking.core.reporting import (
    materialize_group_failure_results as _materialize_group_failure_results,
)
from benchmarking.dashboard.progress import ProgressWriter
from benchmarking.runtime import config_pool as runtime_config_pool
from benchmarking.runtime import judge as judge_runtime
from benchmarking.runtime import paths as runtime_paths
from benchmarking.runtime import subprocess_utils
from benchmarking.runtime.agent_workspace import (
    AttemptIdentity,
    AttemptOutcome,
    AttemptWorkspaceManager,
    WorkspaceIsolationError,
    default_workspace_templates,
)
from benchmarking.runtime.cancellation import (
    CancellationReason,
    CancellationToken,
    OwnedProcessRegistry,
)
from benchmarking.runtime.openclaw_env import (
    build_openclaw_subprocess_env,
    proxy_environment_report,
)
from benchmarking.runtime.vgb_bridge import load_release_config
from benchmarking.runtime.web_search_preflight import run_web_search_preflight
from benchmarking.scoring.evaluators.verifier_grounded import (
    evaluate_verifier_grounded,
    run_verifier_grounded_evaluation,
    validate_verifier_grounded_release,
)
from benchmarking.scoring.registry import evaluate_record, register_default_evaluators
from benchmarking.scoring.results import build_execution_error_evaluation
from benchmarking.skills.health import check_all_skill_health, summarize_skill_health
from benchmarking.workflow import (
    dataset_selection,
    experiments,
    run_state,
    runner_adapters,
    runtime_config,
)
from benchmarking.workflow import orchestration as _orchestration
from benchmarking.workflow.errors import BenchmarkError as _BenchmarkError

DEFAULT_BENCHMARK_ROOT = runtime_paths.benchmarks_root
DEFAULT_CHEMQA_ROOT = runtime_paths.skills_root / "chemqa-review"
DEFAULT_OPENCLAW_CONFIG = runtime_paths.openclaw_config
DEFAULT_OUTPUT_DIR = runtime_paths.project_state_root / "benchmark-runs"


register_default_evaluators()


def install_cancellation_signal_handlers(
    cancellation_token: CancellationToken,
) -> dict[signal.Signals, Any]:
    previous_handlers: dict[signal.Signals, Any] = {}

    def request_cancellation(signum: int, _frame: Any) -> None:
        signal_value = signal.Signals(signum)
        cancellation_token.cancel(
            CancellationReason(
                source="signal",
                signal_name=signal_value.name,
                message=f"Benchmark run cancelled by {signal_value.name}",
            )
        )

    for signal_value in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signal_value] = signal.getsignal(signal_value)
        signal.signal(signal_value, request_cancellation)
    return previous_handlers


def restore_signal_handlers(previous_handlers: dict[signal.Signals, Any]) -> None:
    for signal_value, previous_handler in previous_handlers.items():
        signal.signal(signal_value, previous_handler)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run three-group skills benchmark experiments.")
    parser.add_argument("--benchmark-root", default=str(DEFAULT_BENCHMARK_ROOT), help="formal-benchmarks/ 根目录")
    parser.add_argument("--chemqa-root", default=str(DEFAULT_CHEMQA_ROOT), help="chemqa-review skill 根目录")
    parser.add_argument("--openclaw-config", default=str(DEFAULT_OPENCLAW_CONFIG), help="基础 OpenClaw 配置文件")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="分类结果根目录；默认按用途、benchmark、模型和 run ID 分层",
    )
    parser.add_argument(
        "--exact-output-dir",
        help="若提供，则直接把该目录作为本次输出根目录，绕过默认分类层级",
    )
    parser.add_argument(
        "--merge-existing-per-record",
        action="store_true",
        help="聚合结果时合并输出目录中已存在的 per-record 结果，适合断点续跑/部分重跑",
    )
    parser.add_argument(
        "--groups",
        default=",".join(experiments.EXPERIMENT_GROUPS.keys()),
        help="要运行的实验组，逗号分隔。默认三组全跑",
    )
    parser.add_argument(
        "--datasets",
        help="仅运行指定数据集，逗号分隔；默认扫描 formal-benchmarks/*/data/*.jsonl",
    )
    parser.add_argument(
        "--subsets",
        help=(
            "仅运行指定子集，逗号分隔；例如 "
            "frontierscience_Research,superchem_multimodal"
        ),
    )
    parser.add_argument(
        "--random-count-per-subset",
        type=int,
        help=(
            "按子集随机抽样时，每个子集抽取多少题；当前支持 chembench / "
            "frontierscience_Olympiad / frontierscience_Research / "
            "superchem_multimodal"
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="随机抽样的 seed，默认 0，便于复现",
    )
    parser.add_argument(
        "--files",
        help="仅运行指定 jsonl 文件，逗号分隔，优先级高于 --datasets",
    )
    parser.add_argument("--limit", type=int, help="最多运行多少条题目")
    parser.add_argument("--offset", type=int, default=0, help="跳过前多少条题目")
    parser.add_argument(
        "--record-ids",
        help="仅运行指定 record/task id，逗号分隔；按给定顺序运行",
    )
    parser.add_argument(
        "--single-agent-id-override",
        help="覆盖 single_llm 组的 agent id；未提供时按实验组规范使用默认 baseline agent",
    )
    parser.add_argument(
        "--single-agent-model",
        default=experiments.DEFAULT_SINGLE_AGENT_MODEL,
        help="单一 LLM baseline runtime model，默认锁定为 qwen3.5-plus",
    )
    parser.add_argument(
        "--single-agent-thinking",
        default=experiments.DEFAULT_SINGLE_AGENT_THINKING,
        choices=experiments.THINKING_LEVEL_CHOICES,
        help="单一 LLM baseline OpenClaw thinking level，默认 high",
    )
    parser.add_argument(
        "--chemqa-model-profile",
        default=experiments.DEFAULT_CHEMQA_MODEL_PROFILE,
        help="ChemQA fixed-lane review 所用 model profile，默认使用当前 benchmark 固定 profile",
    )
    parser.add_argument("--judge-agent", default=experiments.DEFAULT_JUDGE_AGENT, help="rubric / 语义评测所用 judge agent id")
    parser.add_argument(
        "--judge-model",
        default=experiments.DEFAULT_JUDGE_MODEL,
        help="judge runtime model，默认锁定为 openai/gpt-5.5",
    )
    parser.add_argument(
        "--judge-agent-thinking",
        default=experiments.DEFAULT_JUDGE_AGENT_THINKING,
        choices=experiments.THINKING_LEVEL_CHOICES,
        help="judge OpenClaw thinking level，默认 high",
    )
    parser.add_argument("--single-timeout", type=int, default=7200, help="单一 LLM 每题超时秒数，默认 7200 秒（2 小时）")
    parser.add_argument(
        "--single-timeout-retries",
        type=int,
        default=3,
        help="单一 LLM timeout-family 失败后的 fresh-session 最大重试次数，默认 3",
    )
    parser.add_argument(
        "--no-timeout",
        action="store_true",
        help="取消单题作答时间上限，让模型自由探索，但保留进程级兜底安全阀",
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="本轮 benchmark run 结束后不启动自动化评估流程，适合临时测试型 run",
    )
    parser.add_argument(
        "--single-timeout-retry-backoff-seconds",
        default="5,15,45",
        help="单一 LLM timeout 重试前等待秒数，逗号分隔，默认 5,15,45",
    )
    parser.add_argument("--chemqa-timeout", type=int, default=1800, help="ChemQA fixed-lane review 每题超时秒数")
    parser.add_argument("--judge-timeout", type=int, default=300, help="Judge 每次评测超时秒数")
    parser.add_argument(
        "--max-unchanged-status-polls",
        type=int,
        default=2,
        help="ChemQA convergence limit for unchanged status polls",
    )
    parser.add_argument(
        "--max-recovery-attempts",
        type=int,
        default=2,
        help="ChemQA convergence limit for recovery attempts",
    )
    parser.add_argument(
        "--max-concurrent-groups",
        type=int,
        default=2,
        help="最多同时运行多少个实验组；默认 2，以降低 WSL 峰值资源占用",
    )
    parser.add_argument(
        "--inter-wave-delay-seconds",
        type=int,
        default=10,
        help="相邻波次之间的等待秒数，默认 10，用于给系统释放资源的窗口",
    )
    parser.add_argument("--review-rounds", type=int, help="ChemQA review rounds 覆盖值")
    parser.add_argument("--rebuttal-rounds", type=int, help="ChemQA rebuttal rounds 覆盖值")
    parser.add_argument("--list-datasets", action="store_true", help="列出可发现的数据集文件后退出")
    parser.add_argument(
        "--print-selected-records",
        action="store_true",
        help="打印本次实际选中的题目清单后退出",
    )
    return parser.parse_args()







def parse_retry_backoff_seconds(raw: str, *, max_retries: int) -> tuple[float, ...]:
    if max_retries <= 0:
        return ()
    values: list[float] = []
    for item in str(raw or "").split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            value = float(stripped)
        except ValueError as exc:
            raise _BenchmarkError(f"Invalid --single-timeout-retry-backoff-seconds value: {stripped}") from exc
        if value < 0:
            raise _BenchmarkError("--single-timeout-retry-backoff-seconds values must be non-negative")
        values.append(value)
    if not values:
        raise _BenchmarkError("--single-timeout-retry-backoff-seconds must include at least one value")
    while len(values) < max_retries:
        values.append(values[-1])
    return tuple(values[:max_retries])



def build_group_waves(group_ids: list[str], *, max_concurrent_groups: int) -> list[list[str]]:
    if max_concurrent_groups <= 0:
        raise _BenchmarkError("--max-concurrent-groups 必须是正整数")
    waves: list[list[str]] = []
    for index in range(0, len(group_ids), max_concurrent_groups):
        waves.append(group_ids[index : index + max_concurrent_groups])
    return waves



def record_group_progress_failure(
    progress_writer: ProgressWriter,
    *,
    group_id: str,
    records: list[_BenchmarkRecord],
    error_message: str,
) -> None:
    progress_writer.group_started(group_id)
    progress_writer.error(group_id=group_id, message=error_message)
    for index, record in enumerate(records, start=1):
        progress_writer.record_started(group_id, record.record_id, index=index)
        progress_writer.record_completed(group_id, record.record_id, status="failed", score=0.0)
    progress_writer.group_completed(group_id, status="failed")


def run_benchmark_web_search_preflight(
    *,
    group_ids: list[str],
    config_pool: runtime_config_pool.ConfigPool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for group_id in group_ids:
        group = experiments.EXPERIMENT_GROUPS[group_id]
        if not group.websearch:
            continue
        spec = experiments.EXPERIMENT_SPECS.get(group_id)
        agent_id = (
            spec.resolve_single_agent_id(args.single_agent_id_override)
            if spec is not None and group.runner == "single_llm"
            else experiments.JUDGE_AGENT_ID
        )
        config_path = config_pool.config_for_group(group)
        effective_agent_id = agent_id or experiments.JUDGE_AGENT_ID
        identity = AttemptIdentity(
            run_id=config_pool.workspace_manager.run_id,
            invocation_id=config_pool.workspace_manager.invocation_id,
            group_id=group.id,
            runner_kind="web_search_preflight",
            agent_id=effective_agent_id,
            record_id="web-search-preflight",
            attempt_index=0,
            session_id=f"web-search-preflight-{uuid.uuid4().hex[:12]}",
            template_id="single-llm-skills-on-v1"
            if bool(getattr(group, "skills_enabled", False))
            else "single-llm-skills-off-v1",
        )
        try:
            lease = config_pool.workspace_manager.prepare(identity)
        except WorkspaceIsolationError as exc:
            reports[group_id] = {
                "available": False,
                "error": exc.message,
                "workspace_isolation": {
                    "preflight_ok": False,
                    "archive_ok": False,
                    "execution_error": dict(exc.details),
                },
            }
            continue
        environment = os.environ.copy()
        environment.update(
            {
                "BENCHMARK_WORKSPACE_DIR": str(lease.active_workspace),
                "BENCHMARK_SKILL_SCRATCH_DIR": str(lease.scratch_dir),
                "BENCHMARK_SKILL_REQUEST_DIR": str(lease.request_dir),
                "BENCHMARK_SKILL_OUTPUT_DIR": str(lease.output_dir),
                "BENCHMARK_SKILL_NOTES_DIR": str(lease.notes_dir),
                "BENCHMARK_PROJECT_ROOT": str(runtime_paths.project_root),
                "BENCHMARK_SKILL_RUNNER": str(runtime_paths.project_root / "scripts" / "run_skill.py"),
            }
        )
        report = run_web_search_preflight(
            agent_id=effective_agent_id,
            config_path=config_path,
            current_python_path=subprocess_utils.current_python(),
            run_subprocess=subprocess_utils.run_subprocess,
            base_env=environment,
        )
        transcript_path = str(report.get("transcript_path") or "").strip()
        policy = config_pool.workspace_manager.policy_for_lease(
            lease,
            role="single_llm",
            skills_enabled=bool(getattr(group, "skills_enabled", False)),
            read_scopes=(runtime_paths.skills_root, runtime_paths.project_root / "scripts" / "run_skill.py"),
        )
        contamination_audit = config_pool.workspace_manager.audit_attempt(
            lease,
            {"session_isolation": {"postflight_entry_session_file": transcript_path}},
            environment=environment,
            policy=policy,
        )
        isolation_meta = lease.to_meta()
        isolation_meta.update(contamination_audit.to_payload())
        isolation_meta.update({"policy_digest": policy.digest, "policy": policy.to_payload()})
        if contamination_audit.adjudication == "non_evaluable":
            report["available"] = False
            report["error"] = "web_search preflight workspace contamination audit failed"
        try:
            archive = config_pool.workspace_manager.seal(
                lease,
                AttemptOutcome(
                    runner_status="completed" if report.get("available") is True else "failed",
                    archive_reason="attempt_terminal",
                    contamination_audit=contamination_audit,
                ),
            )
            isolation_meta.update(archive.to_meta())
        except WorkspaceIsolationError as exc:
            isolation_meta["archive_ok"] = False
            isolation_meta["archive_error"] = dict(exc.details)
            report["available"] = False
            report["error"] = exc.message
        report["workspace_isolation"] = isolation_meta
        reports[group_id] = report
    return {
        "enabled": True,
        "provider": "duckduckgo",
        "reports": reports,
        "available": all(bool(report.get("available")) for report in reports.values()) if reports else True,
        "proxy_env": proxy_environment_report(build_openclaw_subprocess_env(base_env=os.environ.copy())),
    }



def main() -> int:
    args = parse_args()
    single_timeout_retries = max(0, int(getattr(args, "single_timeout_retries", 3)))
    single_timeout_retry_backoff_seconds = parse_retry_backoff_seconds(
        str(getattr(args, "single_timeout_retry_backoff_seconds", "5,15,45")),
        max_retries=single_timeout_retries,
    )
    timeout_mode = "no_timeout" if bool(getattr(args, "no_timeout", False)) else "bounded"
    single_convergence_policy = ConvergencePolicy(
        timeout_seconds=args.single_timeout,
        max_unchanged_status_polls=args.max_unchanged_status_polls,
        max_recovery_attempts=args.max_recovery_attempts,
    )
    chemqa_convergence_policy = ConvergencePolicy(
        timeout_seconds=args.chemqa_timeout,
        max_unchanged_status_polls=args.max_unchanged_status_polls,
        max_recovery_attempts=args.max_recovery_attempts,
    )
    convergence_policy_meta = {
        "single_llm": single_convergence_policy.to_meta(),
        "chemqa": chemqa_convergence_policy.to_meta(),
    }
    group_ids = experiments.select_group_ids(args.groups)
    dataset_files = dataset_selection.select_dataset_files(args)
    if args.list_datasets:
        dataset_selection.print_dataset_listing(dataset_files)
        return 0
    if not dataset_files:
        raise _BenchmarkError("No benchmark files discovered.")

    all_records = dataset_selection.filter_records_by_ids(
        dataset_selection.filter_records_by_subsets(dataset_selection.load_records(dataset_files), args.subsets),
        getattr(args, "record_ids", None),
    )
    if args.random_count_per_subset is not None:
        selected_pool = dataset_selection.sample_records_per_subset(
            all_records,
            per_subset_count=args.random_count_per_subset,
            seed=args.random_seed,
        )
    else:
        selected_pool = all_records

    records = dataset_selection.apply_offset_limit(selected_pool, offset=args.offset, limit=args.limit)
    if not records:
        raise _BenchmarkError("No benchmark records selected.")
    if args.print_selected_records:
        dataset_selection.print_selected_records(records)
        return 0

    verifier_release_config = (
        load_release_config()
        if any(record.grading.kind == "verifier_grounded" for record in records)
        else None
    )
    evaluator_overrides = None
    if verifier_release_config is not None:
        for record in records:
            if record.grading.kind == "verifier_grounded":
                validate_verifier_grounded_release(
                    record,
                    release_config=verifier_release_config,
                )
        verifier_runner = partial(
            run_verifier_grounded_evaluation,
            release_config=verifier_release_config,
        )
        evaluator_overrides = {
            "verifier_grounded": partial(
                evaluate_verifier_grounded,
                verifier_runner=verifier_runner,
            )
        }
    evaluate_invocation_record = partial(
        evaluate_record,
        evaluator_overrides=evaluator_overrides,
    )

    if args.exact_output_dir:
        output_root = Path(args.exact_output_dir).expanduser().resolve()
    else:
        output_root = dataset_selection.default_run_output_root(
            output_dir=args.output_dir,
            dataset_files=dataset_files,
            records=records,
            single_agent_model=args.single_agent_model,
            timestamp=run_state.now_stamp(),
        )
    run_state.ensure_dir(output_root)
    run_id = output_root.name
    invocation_id = str(uuid.uuid4())
    workspace_manager = AttemptWorkspaceManager(
        runtime_root=runtime_paths.benchmark_runtime_root / "runs",
        output_root=output_root,
        run_id=run_id,
        invocation_id=invocation_id,
        templates=default_workspace_templates(runtime_paths.project_root),
        protected_roots=runtime_config.build_production_protected_roots(
            runtime_root=runtime_paths.benchmark_runtime_root / "runs",
            output_root=output_root,
        ),
    )
    try:
        workspace_startup_recovery = workspace_manager.recover_all_incomplete()
    except WorkspaceIsolationError as exc:
        raise _BenchmarkError(f"Benchmark workspace startup recovery failed: {exc.message}") from exc
    cancellation_token = CancellationToken()
    process_registry = OwnedProcessRegistry(cancellation_token=cancellation_token)
    pending_records_by_group = {
        group_id: run_state.pending_records_for_group(
            records,
            output_root=output_root,
            group_id=group_id,
            merge_existing_per_record=bool(args.merge_existing_per_record),
        )
        for group_id in group_ids
    }

    skill_health_reports = check_all_skill_health(experiments.BENCHMARK_SKILLS_ALLOWLIST, workspace_root=runtime_paths.project_root)
    skill_health_summary = summarize_skill_health(skill_health_reports)
    run_state.save_json(output_root / "skill-health.json", {"summary": skill_health_summary, "skills": skill_health_reports})
    effective_experiment_specs = experiments.build_effective_experiment_specs(
        experiments.EXPERIMENT_SPECS,
        skill_health_reports=skill_health_reports,
    )

    config_pool = runtime_config_pool.ConfigPool(
        base_config_path=Path(args.openclaw_config).expanduser().resolve(),
        output_root=output_root,
        context=runtime_config.runtime_config_context(experiment_specs=effective_experiment_specs),
        run_id=run_id,
        invocation_id=invocation_id,
        workspace_manager=workspace_manager,
        single_agent_model=args.single_agent_model,
        judge_model=args.judge_model,
        single_agent_id_override=args.single_agent_id_override,
    )
    web_search_preflight = run_benchmark_web_search_preflight(
        group_ids=[group_id for group_id in group_ids if pending_records_by_group[group_id]],
        config_pool=config_pool,
        args=args,
    )
    run_state.save_json(output_root / "web-search-preflight.json", web_search_preflight)
    judge = judge_runtime.JudgeClient(
        judge_agent=args.judge_agent,
        timeout_seconds=args.judge_timeout,
        config_path=config_pool.judge_config_path(),
        thinking=args.judge_agent_thinking,
        workspace_manager=workspace_manager,
        cancellation_token=cancellation_token,
        process_registry=process_registry,
    )
    group_waves = build_group_waves(group_ids, max_concurrent_groups=args.max_concurrent_groups)
    progress_writer = ProgressWriter(
        output_root,
        total_records=sum(len(group_records) for group_records in pending_records_by_group.values()),
        groups=group_ids,
    )
    progress_writer.run_started()
    previous_signal_handlers = install_cancellation_signal_handlers(cancellation_token)

    build_error_result = partial(
        _build_error_group_record_result,
        classify_subset_fn=classify_subset,
        normalize_answer_tracks_fn=normalize_answer_tracks,
        build_execution_error_evaluation_fn=build_execution_error_evaluation,
        deep_copy_jsonish_fn=subprocess_utils.deep_copy_jsonish,
    )
    materialize_failure_results = partial(
        _materialize_group_failure_results,
        save_json_fn=run_state.save_json,
        slugify_fn=run_state.slugify,
        classify_subset_fn=classify_subset,
        normalize_answer_tracks_fn=normalize_answer_tracks,
        build_execution_error_evaluation_fn=build_execution_error_evaluation,
        deep_copy_jsonish_fn=subprocess_utils.deep_copy_jsonish,
    )
    execute_group = partial(
        _orchestration.run_group,
        chemqa_slot_sets=experiments.CHEMQA_SLOT_SETS,
        build_runner_fn=runner_adapters.build_runner,
        evaluate_answer_fn=evaluate_invocation_record,
        build_error_group_record_result_fn=build_error_result,
        classify_subset_fn=classify_subset,
        save_json_fn=run_state.save_json,
        slugify_fn=run_state.slugify,
    )

    group_results: dict[str, list[_GroupRecordResult]] = {}
    cancellation_errors: list[dict[str, Any]] = []
    try:
        for wave_index, wave_group_ids in enumerate(group_waves, start=1):
            if cancellation_token.is_cancelled:
                break
            started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            run_state.write_wave_status(
                output_root,
                wave_index=wave_index,
                wave_group_ids=wave_group_ids,
                status="running",
                started_at=started_at,
                inter_wave_delay_seconds=args.inter_wave_delay_seconds,
            )
            with ThreadPoolExecutor(max_workers=max(1, len(wave_group_ids))) as executor:
                future_map = {}
                for group_id in wave_group_ids:
                    if cancellation_token.is_cancelled:
                        break
                    group = experiments.EXPERIMENT_GROUPS[group_id]
                    group_records = pending_records_by_group[group_id]
                    if not group_records:
                        group_results[group_id] = []
                        continue
                    preflight_report = dict((web_search_preflight.get("reports") or {}).get(group_id) or {})
                    if group.websearch and preflight_report.get("available") is not True:
                        error_message = (
                            "web_search preflight failed for group "
                            f"`{group_id}`: {preflight_report.get('error') or 'web_search unavailable'}"
                        )
                        group_results[group_id] = materialize_failure_results(
                            group=group,
                            records=group_records,
                            output_root=output_root,
                            error_message=error_message,
                        )
                        record_group_progress_failure(
                            progress_writer,
                            group_id=group_id,
                            records=group_records,
                            error_message=error_message,
                        )
                        continue
                    config_path = config_pool.config_for_group(group)
                    spec = effective_experiment_specs.get(group_id)
                    single_agent = (
                        spec.resolve_single_agent_id(args.single_agent_id_override)
                        if spec is not None
                        else experiments.DEFAULT_SINGLE_AGENT
                    )
                    future = executor.submit(
                        execute_group,
                        group=group,
                        records=group_records,
                        output_root=output_root,
                        single_timeout=args.single_timeout,
                        chemqa_timeout=args.chemqa_timeout,
                        judge=judge,
                        config_path=config_path,
                        single_agent=single_agent,
                        chemqa_root=Path(args.chemqa_root).expanduser().resolve(),
                        chemqa_model_profile=args.chemqa_model_profile,
                        review_rounds=args.review_rounds,
                        rebuttal_rounds=args.rebuttal_rounds,
                        single_convergence_policy=single_convergence_policy,
                        chemqa_convergence_policy=chemqa_convergence_policy,
                        single_timeout_retries=single_timeout_retries,
                        single_timeout_retry_backoff_seconds=single_timeout_retry_backoff_seconds,
                        single_agent_thinking=args.single_agent_thinking,
                        no_timeout=bool(getattr(args, "no_timeout", False)),
                        workspace_manager=workspace_manager,
                        experiment_specs=effective_experiment_specs,
                        skill_health_summary=skill_health_summary,
                        progress_writer=progress_writer,
                        cancellation_token=cancellation_token,
                        process_registry=process_registry,
                    )
                    future_map[future] = group_id

                for future in as_completed(future_map):
                    group_id = future_map[future]
                    try:
                        group_results[group_id] = future.result()
                    except Exception as exc:
                        group = experiments.EXPERIMENT_GROUPS[group_id]
                        error_message = f"Group `{group_id}` failed before returning results: {exc}"
                        group_results[group_id] = materialize_failure_results(
                            group=group,
                            records=pending_records_by_group[group_id],
                            output_root=output_root,
                            error_message=error_message,
                        )
                        record_group_progress_failure(
                            progress_writer,
                            group_id=group_id,
                            records=pending_records_by_group[group_id],
                            error_message=error_message,
                        )
            gc.collect()
            completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            run_state.write_wave_status(
                output_root,
                wave_index=wave_index,
                wave_group_ids=wave_group_ids,
                status="cancelled" if cancellation_token.is_cancelled else "completed",
                started_at=started_at,
                completed_at=completed_at,
                per_record_counts=run_state.count_per_record_outputs(output_root, group_ids=wave_group_ids),
                inter_wave_delay_seconds=args.inter_wave_delay_seconds,
            )
            if cancellation_token.is_cancelled:
                break
            if wave_index < len(group_waves) and args.inter_wave_delay_seconds > 0:
                cancellation_token.wait(args.inter_wave_delay_seconds)
    finally:
        if cancellation_token.is_cancelled:
            reason = cancellation_token.reason
            progress_writer.run_cancelling(reason=reason.to_payload() if reason is not None else {})
            outcome = process_registry.terminate_all(force=cancellation_token.request_count > 1)
            cancellation_errors.extend(outcome.errors)
            if not outcome.completed:
                cancellation_errors.append(
                    {
                        "stage": "owned_process_cleanup",
                        "active_process_count": outcome.active_process_count,
                        "message": "Owned benchmark processes remained after cancellation cleanup.",
                    }
                )
        try:
            runner_adapters.run_pending_cleanroom_cleanup()
        except Exception as exc:
            if not cancellation_token.is_cancelled:
                raise
            cancellation_errors.append(
                {"stage": "cleanroom_cleanup", "error": f"{type(exc).__name__}: {exc}"}
            )
        finally:
            restore_signal_handlers(previous_signal_handlers)

    if cancellation_token.is_cancelled:
        for wave_index, wave_group_ids in enumerate(group_waves, start=1):
            wave_path = output_root / "waves" / f"wave-{wave_index:02d}.json"
            if wave_path.exists():
                continue
            cancelled_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            run_state.write_wave_status(
                output_root,
                wave_index=wave_index,
                wave_group_ids=wave_group_ids,
                status="cancelled",
                started_at=cancelled_at,
                completed_at=cancelled_at,
                per_record_counts=run_state.count_per_record_outputs(output_root, group_ids=wave_group_ids),
                inter_wave_delay_seconds=args.inter_wave_delay_seconds,
            )
        for group_id in group_ids:
            existing = {item.record_id for item in group_results.get(group_id, [])}
            group = experiments.EXPERIMENT_GROUPS[group_id]
            for record in pending_records_by_group[group_id]:
                if record.record_id in existing:
                    continue
                entry = _orchestration.build_cancelled_group_record_result(
                    group=group,
                    record=record,
                    build_error_group_record_result_fn=build_error_result,
                )
                group_results.setdefault(group_id, []).append(entry)
                run_state.save_json(
                    output_root / "per-record" / group_id / f"{run_state.slugify(record.record_id)}.json",
                    asdict(entry),
                )
                progress_writer.record_cancelled(group_id, record.record_id)
            if any(item.run_lifecycle_status == "cancelled" for item in group_results.get(group_id, [])):
                progress_writer.group_cancelled(group_id)
            elif not pending_records_by_group[group_id]:
                progress_writer.group_completed(group_id, status="completed")
        for entries in group_results.values():
            for entry in entries:
                if entry.run_lifecycle_status != "cancelled":
                    continue
                isolation = (entry.runner_meta or {}).get("workspace_isolation")
                isolation = isolation if isinstance(isolation, dict) else {}
                cleanup = isolation.get("cleanup")
                cleanup = cleanup if isinstance(cleanup, dict) else {}
                if isolation.get("archive_ok") is False:
                    cancellation_errors.append(
                        {
                            "stage": "workspace_seal",
                            "group_id": entry.group_id,
                            "record_id": entry.record_id,
                            "error": isolation.get("archive_error") or "workspace archive failed",
                        }
                    )
                if int(cleanup.get("failed_count") or 0) > 0:
                    cancellation_errors.append(
                        {
                            "stage": "workspace_cleanup",
                            "group_id": entry.group_id,
                            "record_id": entry.record_id,
                            "failed_count": int(cleanup.get("failed_count") or 0),
                        }
                    )

    aggregate_group_ids = run_state.resolve_aggregate_group_ids(
        group_ids,
        output_root=output_root,
        merge_existing_per_record=args.merge_existing_per_record,
    )
    if args.merge_existing_per_record:
        results = run_state.load_results_from_output_root(output_root, group_ids=aggregate_group_ids)
    else:
        results: list[_GroupRecordResult] = []
        for group_id in group_ids:
            results.extend(group_results.get(group_id, []))

    run_state.apply_verifier_grounded_reporting_references(
        results,
        release_config=verifier_release_config,
    )
    for item in results:
        run_state.save_json(
            output_root / "per-record" / item.group_id / f"{run_state.slugify(item.record_id)}.json",
            asdict(item),
        )
    summary = aggregate_results(results)
    workspace_policies: dict[str, dict[str, Any]] = {}
    for item in results:
        isolation = (item.runner_meta or {}).get("workspace_isolation") or {}
        if not isinstance(isolation, dict):
            continue
        policy = isolation.get("policy")
        digest = str(isolation.get("policy_digest") or "")
        if isinstance(policy, dict) and digest:
            workspace_policies[digest] = policy
        slots = isolation.get("slots")
        if isinstance(slots, dict):
            for slot in slots.values():
                if not isinstance(slot, dict):
                    continue
                slot_policy = slot.get("policy")
                slot_digest = str(slot.get("policy_digest") or "")
                if isinstance(slot_policy, dict) and slot_digest:
                    workspace_policies[slot_digest] = slot_policy
    payload = {
        "schema_version": 3,
        "status": (
            "cancelled_with_errors"
            if cancellation_token.is_cancelled and cancellation_errors
            else "cancelled"
            if cancellation_token.is_cancelled
            else "completed"
        ),
        "status_axes_description": {
            "run_lifecycle_status": "completed|failed|cancelled",
            "protocol_completion_status": "completed|failed|missing|not_applicable",
            "answer_availability": "native_final|recovered_candidate|preview_only|missing",
            "answer_reliability": "native|high_confidence_recovered|low_confidence_recovered|none",
            "evaluable": "whether a record has a trustworthy scoreable answer",
            "scored": "whether evaluator execution occurred",
            "recovery_mode": "none|candidate_submission|run-status-final-answer-preview|archived_final_answer|protocol_reconstruction",
            "workspace_isolation": {
                "audit_execution_status": "complete|unavailable",
                "boundary_status": "clean|warning|violated|unknown",
                "contamination_status": "clear|confirmed|indeterminate",
                "adjudication": "scoreable|scoreable_degraded|non_evaluable",
            },
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "benchmark_root": str(Path(args.benchmark_root).expanduser().resolve()),
        "dataset_files": [str(path) for path in dataset_files],
        "verifier_grounded_release": (
            verifier_release_config.identity if verifier_release_config is not None else None
        ),
        "groups": [asdict(experiments.EXPERIMENT_GROUPS[group_id]) for group_id in aggregate_group_ids],
        "run_groups": [asdict(experiments.EXPERIMENT_GROUPS[group_id]) for group_id in group_ids],
        "skill_health_summary": skill_health_summary,
        "web_search_preflight": web_search_preflight,
        "convergence_policy": convergence_policy_meta,
        "single_timeout_retry": {
            "max_retries": single_timeout_retries,
            "backoff_seconds": list(single_timeout_retry_backoff_seconds),
        },
        "timeout_mode": timeout_mode,
        "workspace_isolation": {
            "schema_version": 3,
            "run_id": run_id,
            "invocation_id": invocation_id,
            "scratch_contract_version": 2,
            "security_boundary": "runtime_guard_and_transcript_audit_not_os_sandbox",
            "forbidden_path_policy": workspace_manager.forbidden_path_policy_manifest(),
            "access_policies": [workspace_policies[key] for key in sorted(workspace_policies)],
        },
        "merge_existing_per_record": args.merge_existing_per_record,
        "random_sampling": {
            "enabled": args.random_count_per_subset is not None,
            "count_per_subset": args.random_count_per_subset,
            "seed": args.random_seed,
        },
        "records": len(records),
        "execution_plan": {
            "mode": "wave-batched",
            "max_concurrent_groups": args.max_concurrent_groups,
            "inter_wave_delay_seconds": args.inter_wave_delay_seconds,
            "waves": group_waves,
        },
        "results": [asdict(item) for item in results],
        "summary": summary,
        "errors": [
            {
                "group_id": item.group_id,
                "record_id": item.record_id,
                "error": item.error,
            }
            for item in results
            if item.error
        ],
    }
    run_state.save_json(output_root / "results.json", payload)
    run_state.remove_legacy_summary_csvs(output_root)
    runtime_manifest = {
        "terminal_status": payload["status"],
        "verifier_grounded_release": (
            verifier_release_config.identity if verifier_release_config is not None else None
        ),
        "execution_plan": {
            "mode": "wave-batched",
            "max_concurrent_groups": args.max_concurrent_groups,
            "inter_wave_delay_seconds": args.inter_wave_delay_seconds,
            "waves": group_waves,
        },
        "aggregate_groups": aggregate_group_ids,
        "run_groups": group_ids,
        "merge_existing_per_record": args.merge_existing_per_record,
        "skill_health": {
            "summary": skill_health_summary,
            "report_path": str(output_root / "skill-health.json"),
        },
        "web_search_preflight": {
            **web_search_preflight,
            "report_path": str(output_root / "web-search-preflight.json"),
        },
        "convergence_policy": convergence_policy_meta,
        "single_timeout_retry": {
            "max_retries": single_timeout_retries,
            "backoff_seconds": list(single_timeout_retry_backoff_seconds),
        },
        "timeout_mode": timeout_mode,
        "workspace_isolation": {
            "schema_version": 3,
            "run_id": run_id,
            "invocation_id": invocation_id,
            "runtime_runs_root": str(workspace_manager.runtime_root),
            "runtime_workspace_root": str(workspace_manager.invocation_runtime_root),
            "archive_root": str(workspace_manager.archive_root),
            "quarantine_root": str(workspace_manager.quarantine_root),
            "templates": workspace_manager.template_manifest(),
            "startup_recovery": workspace_startup_recovery,
            "scratch_contract_version": 2,
            "security_boundary": "runtime_guard_and_transcript_audit_not_os_sandbox",
            "forbidden_path_policy": workspace_manager.forbidden_path_policy_manifest(),
            "access_policies": [workspace_policies[key] for key in sorted(workspace_policies)],
        },
        "groups": {
            group_id: {
                "group": asdict(experiments.EXPERIMENT_GROUPS[group_id]),
                "config_path": str(config_pool.config_for_group(experiments.EXPERIMENT_GROUPS[group_id])),
                "effective_skill_allowlist": list(effective_experiment_specs[group_id].skill_allowlist or ()),
                "slot_set": experiments.CHEMQA_SLOT_SETS.get(group_id),
                "single_agent": (
                    effective_experiment_specs[group_id].resolve_single_agent_id(args.single_agent_id_override)
                    if group_id in effective_experiment_specs and experiments.EXPERIMENT_GROUPS[group_id].runner == "single_llm"
                    else None
                ),
                "single_agent_model": args.single_agent_model,
                "single_agent_thinking": (
                    args.single_agent_thinking if experiments.EXPERIMENT_GROUPS[group_id].runner == "single_llm" else None
                ),
                "no_timeout": bool(getattr(args, "no_timeout", False)) if experiments.EXPERIMENT_GROUPS[group_id].runner == "single_llm" else None,
                "chemqa_model_profile": args.chemqa_model_profile if experiments.EXPERIMENT_GROUPS[group_id].runner == "chemqa" else None,
                "selected_record_count": len(records),
                "pending_record_count": len(pending_records_by_group[group_id]),
                "skipped_existing_record_count": len(records) - len(pending_records_by_group[group_id]),
            }
            for group_id in group_ids
        },
        "judge": {
            "agent": args.judge_agent,
            "model": args.judge_model,
            "thinking": args.judge_agent_thinking,
            "config_path": str(config_pool.judge_config_path()),
        },
    }
    run_state.save_json(output_root / "runtime-manifest.json", runtime_manifest)
    if cancellation_token.is_cancelled:
        automated_evaluation_status = run_state.automated_evaluation_skipped(output_root)
        automated_evaluation_status["reason"] = "run_cancelled"
        run_state.save_json(Path(automated_evaluation_status["status_path"]), automated_evaluation_status)
        runtime_manifest["cancellation"] = {
            **(cancellation_token.reason.to_payload() if cancellation_token.reason is not None else {}),
            "request_count": cancellation_token.request_count,
            "errors": cancellation_errors,
        }
    elif bool(getattr(args, "no_analysis", False)):
        automated_evaluation_status = run_state.automated_evaluation_skipped(output_root)
        run_state.save_json(Path(automated_evaluation_status["status_path"]), automated_evaluation_status)
    else:
        try:
            automated_evaluation_status = launch_automated_evaluation(output_root)
        except Exception as exc:
            automated_evaluation_status = run_state.automated_evaluation_launch_failed(output_root, exc)
            run_state.save_json(Path(automated_evaluation_status["status_path"]), automated_evaluation_status)
    runtime_manifest["automated_evaluation"] = automated_evaluation_status
    run_state.save_json(output_root / "runtime-manifest.json", runtime_manifest)
    if cancellation_token.is_cancelled:
        progress_writer.run_cancelled(errors=cancellation_errors)
    else:
        progress_writer.run_completed(status="completed")
    print(json.dumps({"output_dir": str(output_root), "summary": summary}, indent=2, ensure_ascii=False))
    return 130 if cancellation_token.is_cancelled else 0


if __name__ == "__main__":
    raise SystemExit(main())
