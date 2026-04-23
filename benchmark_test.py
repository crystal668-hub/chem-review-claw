#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Iterable

import yaml

try:
    from benchmarking.contracts import AnswerPayload, FailureInfo, RecoveryInfo, RunStatus, RunnerResult
    from benchmarking.config_renderer import ConfigRenderError, render_run_config
    from benchmarking.datasets import (
        BenchmarkRecord,
        GradingSpec,
        RecordValidationError,
        classify_subset as classify_record_subset,
        dataset_name_from_file as dataset_name_from_record_file,
        load_records as load_benchmark_records,
        source_pair_key as record_source_pair_key,
    )
    from benchmarking.evaluation import evaluate_record, register_evaluator
    from benchmarking.experiments import ExperimentSpec
    from benchmarking.runners import build_runner
    from benchmarking.runners import ChemQARunner as _BenchmarkingChemQARunner
    from benchmarking.runners import SingleLLMRunner as _BenchmarkingSingleLLMRunner
    from benchmarking.provisioning import (
        ProvisionedAgent,
        ProvisionedExperiment,
        ensure_basic_agent_dirs,
        provision_slot_workspace,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - package-style import fallback
    if exc.name != "benchmarking":
        raise
    from workspace.benchmarking.contracts import AnswerPayload, FailureInfo, RecoveryInfo, RunStatus, RunnerResult
    from workspace.benchmarking.config_renderer import ConfigRenderError, render_run_config
    from workspace.benchmarking.datasets import (
        BenchmarkRecord,
        GradingSpec,
        RecordValidationError,
        classify_subset as classify_record_subset,
        dataset_name_from_file as dataset_name_from_record_file,
        load_records as load_benchmark_records,
        source_pair_key as record_source_pair_key,
    )
    from workspace.benchmarking.evaluation import evaluate_record, register_evaluator
    from workspace.benchmarking.experiments import ExperimentSpec
    from workspace.benchmarking.runners import build_runner
    from workspace.benchmarking.runners import ChemQARunner as _BenchmarkingChemQARunner
    from workspace.benchmarking.runners import SingleLLMRunner as _BenchmarkingSingleLLMRunner
    from workspace.benchmarking.provisioning import (
        ProvisionedAgent,
        ProvisionedExperiment,
        ensure_basic_agent_dirs,
        provision_slot_workspace,
    )

_runner_factory = build_runner

try:
    from workspace import runtime_paths
    from workspace.conformabench_judge import (
        ConformaBenchDependencyError,
        ConformaBenchJudgeError,
        evaluate_submission as evaluate_conformabench_submission,
        load_hidden_judge_spec,
        resolve_hidden_judge_spec_path,
    )
except ModuleNotFoundError:  # pragma: no cover - script entry fallback
    import runtime_paths
    from conformabench_judge import (
        ConformaBenchDependencyError,
        ConformaBenchJudgeError,
        evaluate_submission as evaluate_conformabench_submission,
        load_hidden_judge_spec,
        resolve_hidden_judge_spec_path,
    )


DEFAULT_WORKSPACE = runtime_paths.project_root
DEFAULT_BENCHMARK_ROOT = runtime_paths.benchmarks_root
DEFAULT_CHEMQA_ROOT = runtime_paths.skills_root / "chemqa-review"
DEFAULT_BENCHMARK_CLEANROOM_ROOT = runtime_paths.skills_root / "benchmark-cleanroom"
DEFAULT_OPENCLAW_ENV_FILE = runtime_paths.openclaw_env
DEFAULT_OPENCLAW_CONFIG = runtime_paths.openclaw_config
DEFAULT_OUTPUT_DIR = runtime_paths.project_state_root / "benchmark-runs"
DEFAULT_SINGLE_AGENT = "benchmark-single-web-off"
DEFAULT_SINGLE_AGENT_MODEL = "qwen3.5-plus"
DEFAULT_JUDGE_AGENT = "benchmark-judge"
DEFAULT_JUDGE_MODEL = "su8/gpt-5.4"
DEFAULT_CHEMQA_PRESET = "chemqa-review@1"
DEFAULT_CHEMQA_MODEL_PROFILE = "chemqa-review-su8-coord-qwen-ds-kimi-glm-minimax"
BASELINE_WORKSPACE_ROOT = runtime_paths.benchmark_runtime_root
CHEMQA_SLOT_SETS = {
    "chemqa_web_on": "A",
    "chemqa_web_off": "B",
}
BASELINE_AGENT_IDS = {
    "single_llm_web_on": "benchmark-single-web-on",
    "single_llm_web_off": "benchmark-single-web-off",
}
JUDGE_AGENT_ID = "benchmark-judge"
BENCHMARK_AGENT_THINKING = "high"
CHEMQA_WORKSPACE_ROOTS = {
    "A": BASELINE_WORKSPACE_ROOT / "chemqa_web_on",
    "B": BASELINE_WORKSPACE_ROOT / "chemqa_web_off",
}


def RunOutput(
    *,
    short_answer_text: str,
    full_response_text: str,
    raw: dict[str, Any],
    runner_meta: dict[str, Any],
) -> RunnerResult:
    return RunnerResult(
        status=RunStatus.COMPLETED,
        answer=AnswerPayload(
            short_answer_text=short_answer_text,
            full_response_text=full_response_text,
        ),
        raw=raw,
        runner_meta=runner_meta,
    )


def current_python() -> str:
    venv = os.environ.get("VIRTUAL_ENV", "").strip()
    if venv:
        venv_root = Path(venv).expanduser()
        for candidate in (venv_root / "bin" / "python", venv_root / "Scripts" / "python.exe"):
            if candidate.is_file():
                return str(candidate)
    return str(Path(sys.executable).expanduser())
SUBSET_ORDER = (
    "chembench",
    "conformabench",
    "frontierscience_Olympiad",
    "frontierscience_Research",
    "superchem_multimodal",
)
SUPERCHEM_SUBSETS = ("superchem_multimodal",)
FINAL_ANSWER_RE = re.compile(r"^\s*FINAL\s+ANSWER\s*[:：-]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", re.DOTALL | re.IGNORECASE)
SUPERCHM_XML_CHECKPOINT_RE = re.compile(
    r"<\s*checkpoint\b(?P<attrs>[^>]*)>(?P<body>.*?)</\s*checkpoint\s*>",
    re.IGNORECASE | re.DOTALL,
)
SUPERCHM_INLINE_CHECKPOINT_RE = re.compile(
    r"Checkpoint\s*(?P<index>\d+)\s*[:：-]\s*(?P<body>.*?)(?=(?:\n\s*Checkpoint\s*\d+\s*[:：-])|\Z)",
    re.IGNORECASE | re.DOTALL,
)
SUPERCHM_ATTR_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_-]*)\s*=\s*["\']([^"\']+)["\']')
SINGLE_LETTER_TOKEN_RE = re.compile(r"\b([A-Z])\b")
RUNTIME_BUNDLE_LOCK = threading.Lock()
CLEANROOM_REGISTRY_LOCK = threading.Lock()
CLEANROOM_PENDING_MANIFESTS: dict[str, Path] = {}
CLEANROOM_HOOKS_INSTALLED = False


class BenchmarkError(RuntimeError):
    pass


class CleanupFatalError(BenchmarkError):
    pass


@dataclass(frozen=True)
class ExperimentGroup:
    id: str
    label: str
    runner: str
    websearch: bool


EXPERIMENT_GROUPS: dict[str, ExperimentGroup] = {
    "chemqa_web_on": ExperimentGroup(
        id="chemqa_web_on",
        label="ChemQAWorkflow + 启用 websearch plugin",
        runner="chemqa",
        websearch=True,
    ),
    "chemqa_web_off": ExperimentGroup(
        id="chemqa_web_off",
        label="ChemQAWorkflow + 禁用 websearch plugin",
        runner="chemqa",
        websearch=False,
    ),
    "single_llm_web_on": ExperimentGroup(
        id="single_llm_web_on",
        label="单一 LLM + 启用 websearch plugin",
        runner="single_llm",
        websearch=True,
    ),
    "single_llm_web_off": ExperimentGroup(
        id="single_llm_web_off",
        label="单一 LLM + 禁用 websearch plugin",
        runner="single_llm",
        websearch=False,
    ),
}
EXPERIMENT_SPECS: dict[str, ExperimentSpec] = {
    "chemqa_web_on": ExperimentSpec(
        id="chemqa_web_on",
        label=EXPERIMENT_GROUPS["chemqa_web_on"].label,
        runner_kind="chemqa",
        websearch_enabled=True,
        slot_set=CHEMQA_SLOT_SETS["chemqa_web_on"],
    ),
    "chemqa_web_off": ExperimentSpec(
        id="chemqa_web_off",
        label=EXPERIMENT_GROUPS["chemqa_web_off"].label,
        runner_kind="chemqa",
        websearch_enabled=False,
        slot_set=CHEMQA_SLOT_SETS["chemqa_web_off"],
    ),
    "single_llm_web_on": ExperimentSpec(
        id="single_llm_web_on",
        label=EXPERIMENT_GROUPS["single_llm_web_on"].label,
        runner_kind="single_llm",
        websearch_enabled=True,
        single_agent_id=BASELINE_AGENT_IDS["single_llm_web_on"],
    ),
    "single_llm_web_off": ExperimentSpec(
        id="single_llm_web_off",
        label=EXPERIMENT_GROUPS["single_llm_web_off"].label,
        runner_kind="single_llm",
        websearch_enabled=False,
        single_agent_id=BASELINE_AGENT_IDS["single_llm_web_off"],
    ),
}


@dataclass
class RuntimeBundle:
    bundle_dir: Path
    question_markdown: Path
    image_files: list[Path]

    def to_meta(self) -> dict[str, Any]:
        return {
            "bundle_dir": str(self.bundle_dir),
            "question_markdown": str(self.question_markdown),
            "image_files": [str(path) for path in self.image_files],
        }


@dataclass
class EvaluationResult:
    eval_kind: str
    score: float
    max_score: float
    normalized_score: float
    passed: bool
    primary_metric: str
    primary_metric_direction: str
    details: dict[str, Any]


@dataclass
class GroupRecordResult:
    group_id: str
    group_label: str
    runner: str
    websearch: bool
    record_id: str
    subset: str
    dataset: str
    source_file: str
    eval_kind: str
    prompt: str
    reference_answer: str
    answer_text: str
    evaluation: dict[str, Any]
    runner_meta: dict[str, Any]
    raw: dict[str, Any]
    elapsed_seconds: float
    error: str | None = None
    short_answer_text: str = ""
    full_response_text: str = ""


def format_timestamp(epoch: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(epoch or time.time()))


def cleanroom_skill_root() -> Path:
    return DEFAULT_BENCHMARK_CLEANROOM_ROOT


def cleanroom_runtime_lease_module_path() -> Path:
    return cleanroom_skill_root() / "scripts" / "runtime_lease.py"


def load_cleanroom_runtime_lease_module() -> Any:
    module_path = cleanroom_runtime_lease_module_path()
    spec = importlib.util.spec_from_file_location("benchmark_cleanroom_runtime_lease", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark cleanroom runtime_lease.py from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


try:
    cleanroom_runtime_lease = load_cleanroom_runtime_lease_module()
except Exception:
    cleanroom_runtime_lease = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run four-group ChemQA / single-LLM benchmark experiments.")
    parser.add_argument("--benchmark-root", default=str(DEFAULT_BENCHMARK_ROOT), help="benchmarks/ 根目录")
    parser.add_argument("--chemqa-root", default=str(DEFAULT_CHEMQA_ROOT), help="chemqa-review skill 根目录")
    parser.add_argument("--openclaw-config", default=str(DEFAULT_OPENCLAW_CONFIG), help="基础 OpenClaw 配置文件")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="结果输出目录")
    parser.add_argument(
        "--exact-output-dir",
        help="若提供，则直接把该目录作为本次输出根目录，而不是自动创建 benchmark-时间戳 子目录",
    )
    parser.add_argument(
        "--merge-existing-per-record",
        action="store_true",
        help="聚合结果时合并输出目录中已存在的 per-record 结果，适合断点续跑/部分重跑",
    )
    parser.add_argument(
        "--groups",
        default=",".join(EXPERIMENT_GROUPS.keys()),
        help="要运行的实验组，逗号分隔。默认四组全跑",
    )
    parser.add_argument(
        "--datasets",
        help="仅运行指定数据集，逗号分隔；默认扫描 benchmarks/*/data/*.jsonl",
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
        "--single-agent-id-override",
        help="覆盖 single_llm 组的 agent id；未提供时按实验组规范使用默认 baseline agent",
    )
    parser.add_argument(
        "--single-agent-model",
        default=DEFAULT_SINGLE_AGENT_MODEL,
        help="单一 LLM baseline runtime model，默认锁定为 qwen3.5-plus",
    )
    parser.add_argument(
        "--chemqa-model-profile",
        default=DEFAULT_CHEMQA_MODEL_PROFILE,
        help="ChemQAWorkflow 所用 model profile，默认使用当前 benchmark 固定 profile",
    )
    parser.add_argument("--judge-agent", default=DEFAULT_JUDGE_AGENT, help="rubric / 语义评测所用 judge agent id")
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="judge runtime model，默认锁定为 su8/gpt-5.4",
    )
    parser.add_argument("--single-timeout", type=int, default=900, help="单一 LLM 每题超时秒数")
    parser.add_argument("--chemqa-timeout", type=int, default=1800, help="ChemQAWorkflow 每题超时秒数")
    parser.add_argument("--judge-timeout", type=int, default=300, help="Judge 每次评测超时秒数")
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


def require_cleanroom_runtime_lease() -> Any:
    if cleanroom_runtime_lease is None:
        raise BenchmarkError(
            f"benchmark-cleanroom runtime helpers are unavailable under {cleanroom_skill_root()}"
        )
    return cleanroom_runtime_lease


def cleanup_manifest_path(output_root: Path, run_id: str) -> Path:
    module = require_cleanroom_runtime_lease()
    return module.manifest_path(output_root, run_id)


def build_cleanup_manifest_payload(
    *,
    run_id: str,
    benchmark_kind: str,
    group_id: str,
    output_root: Path,
    launch_home: Path | None = None,
    clawteam_data_dir: Path | None = None,
    session_assignments: dict[str, str] | None = None,
    control_roots: list[Path] | None = None,
    generated_roots: list[Path] | None = None,
    artifact_roots: list[Path] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    module = require_cleanroom_runtime_lease()
    lease_dir = output_root / "cleanroom" / "leases"
    return module.build_manifest_payload(
        run_id=run_id,
        benchmark_kind=benchmark_kind,
        group_id=group_id,
        output_root=output_root,
        launch_home=launch_home or "",
        clawteam_data_dir=clawteam_data_dir or "",
        session_assignments=session_assignments or {},
        control_roots=[str(path) for path in (control_roots or [])],
        generated_roots=[str(path) for path in (generated_roots or [])],
        artifact_roots=[str(path) for path in (artifact_roots or [])],
        lease_dir=lease_dir,
        extra=extra or {},
    )


def write_cleanup_manifest(path: Path, payload: dict[str, Any]) -> Path:
    module = require_cleanroom_runtime_lease()
    path.parent.mkdir(parents=True, exist_ok=True)
    return module.write_manifest(path, payload)


def update_cleanup_manifest(path: Path, patch: dict[str, Any]) -> dict[str, Any]:
    module = require_cleanroom_runtime_lease()
    return module.update_manifest(path, patch)


def register_pending_cleanup_manifest(path: Path) -> None:
    global CLEANROOM_HOOKS_INSTALLED
    with CLEANROOM_REGISTRY_LOCK:
        CLEANROOM_PENDING_MANIFESTS[str(path)] = path
        if CLEANROOM_HOOKS_INSTALLED:
            return
        atexit.register(run_pending_cleanroom_cleanup)
        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, _cleanroom_signal_handler)
            except Exception:
                continue
        CLEANROOM_HOOKS_INSTALLED = True


def unregister_pending_cleanup_manifest(path: Path) -> None:
    with CLEANROOM_REGISTRY_LOCK:
        CLEANROOM_PENDING_MANIFESTS.pop(str(path), None)


def iter_pending_cleanup_manifests() -> list[Path]:
    with CLEANROOM_REGISTRY_LOCK:
        return list(CLEANROOM_PENDING_MANIFESTS.values())


def _cleanroom_signal_handler(signum: int, _frame: Any) -> None:
    try:
        run_pending_cleanroom_cleanup()
    finally:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


def invoke_cleanroom_cleanup(
    *,
    manifest_path: Path,
    grace_seconds: float = 5.0,
    kill_after_seconds: float = 10.0,
) -> dict[str, Any]:
    cleanup_script = cleanroom_skill_root() / "scripts" / "cleanup_benchmark_run.py"
    command = [
        current_python(),
        str(cleanup_script),
        "--manifest",
        str(manifest_path),
        "--grace-seconds",
        str(grace_seconds),
        "--kill-after-seconds",
        str(kill_after_seconds),
        "--json",
    ]
    result = run_subprocess(command, timeout=max(30, int(grace_seconds + kill_after_seconds + 20)))
    payload = parse_json_stdout(result, command)
    if not isinstance(payload, dict):
        raise BenchmarkError(f"benchmark cleanroom cleanup did not return an object for manifest `{manifest_path}`")
    if not payload.get("success"):
        raise BenchmarkError(f"benchmark cleanroom cleanup failed for `{manifest_path}`: {payload}")
    return payload


def run_pending_cleanroom_cleanup() -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for manifest_path in iter_pending_cleanup_manifests():
        try:
            report = invoke_cleanroom_cleanup(manifest_path=manifest_path)
            reports.append(report)
        except Exception:
            continue
        finally:
            unregister_pending_cleanup_manifest(manifest_path)
    return reports


def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def slugify(value: str, *, limit: int = 64) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    cleaned = cleaned or "item"
    if len(cleaned) <= limit:
        return cleaned
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[: limit - 9]}-{digest}".strip("-")


def logical_slot_ids() -> tuple[str, ...]:
    return ("debate-coordinator", "debate-1", "debate-2", "debate-3", "debate-4", "debate-5")


def actual_slot_ids(slot_set: str) -> dict[str, str]:
    normalized = str(slot_set).strip()
    prefix = f"debate{normalized}"
    return {
        "debate-coordinator": f"{prefix}-coordinator",
        "debate-1": f"{prefix}-1",
        "debate-2": f"{prefix}-2",
        "debate-3": f"{prefix}-3",
        "debate-4": f"{prefix}-4",
        "debate-5": f"{prefix}-5",
    }


def slot_role_map(slot_set: str) -> dict[str, str]:
    slots = actual_slot_ids(slot_set)
    return {
        "debate-coordinator": slots["debate-coordinator"],
        "proposer-1": slots["debate-1"],
        "proposer-2": slots["debate-2"],
        "proposer-3": slots["debate-3"],
        "proposer-4": slots["debate-4"],
        "proposer-5": slots["debate-5"],
    }


def slot_agents_template_path() -> Path:
    return runtime_paths.skills_root / "debateclaw-v1" / "scripts" / "templates" / "debate-slot-AGENTS.md"


def load_slot_agents_template() -> str:
    path = slot_agents_template_path()
    return path.read_text(encoding="utf-8").rstrip() + "\n"


def _render_run_config_or_raise(
    *,
    base_payload: dict[str, Any],
    spec: ExperimentSpec,
    provisioned: ProvisionedExperiment,
    judge_model: str,
    runner_model: str,
) -> dict[str, Any]:
    try:
        return render_run_config(
            base_payload=base_payload,
            spec=spec,
            provisioned=provisioned,
            judge_model=judge_model,
            runner_model=runner_model,
        )
    except ConfigRenderError as exc:
        raise BenchmarkError(str(exc)) from exc


def build_run_scoped_config_payload(
    base_payload: dict[str, Any],
    *,
    group: ExperimentGroup,
    single_agent_model: str,
    judge_model: str,
    single_agent_id_override: str | None = None,
) -> dict[str, Any]:
    judge_workspace = BASELINE_WORKSPACE_ROOT / JUDGE_AGENT_ID
    judge_agent_dir = runtime_paths.agents_root / JUDGE_AGENT_ID / "agent"
    ensure_basic_agent_dirs(judge_workspace, judge_agent_dir)
    judge = ProvisionedAgent(
        agent_id=JUDGE_AGENT_ID,
        workspace=judge_workspace,
        agent_dir=judge_agent_dir,
    )
    runner_agents: list[ProvisionedAgent] = []
    spec = EXPERIMENT_SPECS.get(
        group.id,
        ExperimentSpec(
            id=group.id,
            label=group.label,
            runner_kind=group.runner,
            websearch_enabled=group.websearch,
        ),
    )

    if group.id == "benchmark-judge-runtime":
        return _render_run_config_or_raise(
            base_payload=base_payload,
            spec=spec,
            provisioned=ProvisionedExperiment(judge=judge, runner_agents=()),
            judge_model=judge_model,
            runner_model=single_agent_model,
        )

    if group.runner == "single_llm":
        agent_id = spec.resolve_single_agent_id(single_agent_id_override)
        if not agent_id:
            raise BenchmarkError(f"Experiment group `{group.id}` missing single-agent id in experiment spec.")
        workspace = BASELINE_WORKSPACE_ROOT / agent_id
        agent_dir = runtime_paths.agents_root / agent_id / "agent"
        ensure_basic_agent_dirs(workspace, agent_dir)
        runner_agents.append(
            ProvisionedAgent(
                agent_id=agent_id,
                workspace=workspace,
                agent_dir=agent_dir,
            )
        )
        single_spec = ExperimentSpec(
            id=spec.id,
            label=spec.label,
            runner_kind=spec.runner_kind,
            websearch_enabled=spec.websearch_enabled,
            single_agent_id=agent_id,
            slot_set=spec.slot_set,
        )
        return _render_run_config_or_raise(
            base_payload=base_payload,
            spec=single_spec,
            provisioned=ProvisionedExperiment(judge=judge, runner_agents=tuple(runner_agents)),
            judge_model=judge_model,
            runner_model=single_agent_model,
        )

    slot_set = CHEMQA_SLOT_SETS[group.id]
    workspace_root = CHEMQA_WORKSPACE_ROOTS[slot_set]
    slot_map = actual_slot_ids(slot_set)
    for logical_slot_id, actual_slot_id in slot_map.items():
        workspace = workspace_root / actual_slot_id
        agent_dir = runtime_paths.agents_root / actual_slot_id / "agent"
        ensure_basic_agent_dirs(agent_dir)
        provision_slot_workspace(
            workspace=workspace,
            workspace_root=workspace_root,
            slot_id=actual_slot_id,
            agents_template_text=load_slot_agents_template(),
        )
        runner_agents.append(
            ProvisionedAgent(
                agent_id=actual_slot_id,
                workspace=workspace,
                agent_dir=agent_dir,
            )
        )
    chemqa_spec = ExperimentSpec(
        id=group.id,
        label=group.label,
        runner_kind=group.runner,
        websearch_enabled=group.websearch,
        slot_set=slot_set,
    )
    return _render_run_config_or_raise(
        base_payload=base_payload,
        spec=chemqa_spec,
        provisioned=ProvisionedExperiment(judge=judge, runner_agents=tuple(runner_agents)),
        judge_model=judge_model,
        runner_model=single_agent_model,
    )


def run_subprocess(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        env=env,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def ensure_success(result: subprocess.CompletedProcess[str], command: list[str]) -> None:
    if result.returncode != 0:
        raise BenchmarkError(
            "Command failed\n"
            f"command: {' '.join(command)}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def parse_json_stdout(result: subprocess.CompletedProcess[str], command: list[str]) -> Any:
    ensure_success(result, command)
    output = result.stdout.strip() or result.stderr.strip()
    if not output:
        raise BenchmarkError(f"Empty stdout/stderr from command: {' '.join(command)}")
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        try:
            return safe_json_extract(output)
        except Exception as exc:
            raise BenchmarkError(
                "JSON decode failed\n"
                f"command: {' '.join(command)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            ) from exc


def deep_copy_jsonish(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def unwrap_agent_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result") if isinstance(payload, dict) else None
    if isinstance(result, dict):
        return result
    return payload if isinstance(payload, dict) else {}


def build_temp_openclaw_config_payload(base_payload: dict[str, Any], *, enable_websearch: bool) -> dict[str, Any]:
    payload = deep_copy_jsonish(base_payload)
    tools = payload.setdefault("tools", {})
    web = tools.setdefault("web", {})
    search = web.setdefault("search", {})
    search["enabled"] = enable_websearch

    plugins = payload.setdefault("plugins", {})
    entries = plugins.setdefault("entries", {})
    duckduckgo = entries.setdefault("duckduckgo", {})
    duckduckgo["enabled"] = enable_websearch
    duckduckgo.setdefault("config", {})
    return payload


class ConfigPool:
    def __init__(
        self,
        *,
        base_config_path: Path,
        output_root: Path,
        single_agent_model: str | None = None,
        judge_model: str | None = None,
        single_agent_id_override: str | None = None,
    ) -> None:
        self.base_config_path = base_config_path
        self.output_root = output_root
        self._payload = json.loads(base_config_path.read_text(encoding="utf-8"))
        self._config_dir = output_root / "runtime-config"
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._group_paths: dict[str, Path] = {}
        self._judge_path: Path | None = None
        discovered_single = self._discover_agent_model("debate-1") or "su8/gpt-5.4"
        discovered_judge = self._discover_agent_model("debate-coordinator") or "su8/gpt-5.4"
        self._single_agent_model = str(single_agent_model or discovered_single).strip() or discovered_single
        self._judge_model = str(judge_model or discovered_judge).strip() or discovered_judge
        self._single_agent_id_override = single_agent_id_override

    def _discover_agent_model(self, agent_id: str) -> str | None:
        agents = ((self._payload.get("agents") or {}).get("list") or [])
        for entry in agents:
            if isinstance(entry, dict) and str(entry.get("id", "")) == agent_id:
                model = str(entry.get("model") or "").strip()
                if model:
                    return model
        return None

    def config_for_group(self, group: ExperimentGroup) -> Path:
        existing = self._group_paths.get(group.id)
        if existing is not None:
            return existing
        payload = build_run_scoped_config_payload(
            self._payload,
            group=group,
            single_agent_model=self._single_agent_model,
            judge_model=self._judge_model,
            single_agent_id_override=self._single_agent_id_override,
        )
        path = self._config_dir / f"{group.id}-openclaw.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._group_paths[group.id] = path
        return path

    def judge_config_path(self) -> Path:
        if self._judge_path is not None:
            return self._judge_path
        judge_group = ExperimentGroup(
            id="benchmark-judge-runtime",
            label="benchmark judge runtime",
            runner="single_llm",
            websearch=False,
        )
        payload = build_run_scoped_config_payload(
            self._payload,
            group=judge_group,
            single_agent_model=self._single_agent_model,
            judge_model=self._judge_model,
        )
        path = self._config_dir / "benchmark-judge-openclaw.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._judge_path = path
        return path

    def cleanup(self) -> None:
        return


def discover_dataset_files(root: Path) -> list[Path]:
    return sorted(path.resolve() for path in root.glob("*/data/*.jsonl") if path.is_file())


def dataset_name_from_file(path: Path) -> str:
    return dataset_name_from_record_file(path)


def load_records(paths: Iterable[Path]) -> list[BenchmarkRecord]:
    try:
        return load_benchmark_records(paths)
    except RecordValidationError as exc:
        raise BenchmarkError(str(exc)) from exc


def classify_subset(record: BenchmarkRecord) -> str:
    return classify_record_subset(record)


def source_pair_key(record: BenchmarkRecord) -> str:
    return record_source_pair_key(record)


def sample_superchem_pairs(
    grouped: dict[str, list[BenchmarkRecord]],
    *,
    per_subset_count: int,
    seed: int,
) -> list[BenchmarkRecord]:
    if not all(grouped.get(subset) for subset in SUPERCHEM_SUBSETS):
        return []

    by_uuid: dict[str, dict[str, BenchmarkRecord]] = {}
    for subset in SUPERCHEM_SUBSETS:
        for record in grouped.get(subset, []):
            by_uuid.setdefault(source_pair_key(record), {})[subset] = record

    paired = [pair for pair in by_uuid.values() if all(subset in pair for subset in SUPERCHEM_SUBSETS)]
    if not paired:
        return []
    if len(paired) < per_subset_count:
        raise BenchmarkError(f"SUPERChem 成对题目仅有 {len(paired)} 题，无法随机抽取 {per_subset_count} 题。")

    rng = random.Random(seed)
    sampled_pairs = rng.sample(paired, per_subset_count)
    sampled: list[BenchmarkRecord] = []
    for pair in sampled_pairs:
        for subset in SUPERCHEM_SUBSETS:
            sampled.append(pair[subset])
    return sampled



def sample_records_per_subset(records: list[BenchmarkRecord], *, per_subset_count: int, seed: int) -> list[BenchmarkRecord]:
    if per_subset_count <= 0:
        raise BenchmarkError("--random-count-per-subset 必须是正整数")

    grouped: dict[str, list[BenchmarkRecord]] = {}
    for record in records:
        grouped.setdefault(classify_subset(record), []).append(record)

    available_supported = [subset for subset in SUBSET_ORDER if grouped.get(subset)]
    if not available_supported:
        raise BenchmarkError("当前选定的数据范围内没有可用于按子集抽样的记录。")

    rng = random.Random(seed)
    sampled: list[BenchmarkRecord] = []
    handled_subsets: set[str] = set()
    superchem_sampled = sample_superchem_pairs(grouped, per_subset_count=per_subset_count, seed=seed)
    if superchem_sampled:
        sampled.extend(superchem_sampled)
        handled_subsets.update(SUPERCHEM_SUBSETS)
    for subset in available_supported:
        if subset in handled_subsets:
            continue
        subset_records = grouped[subset]
        if len(subset_records) < per_subset_count:
            raise BenchmarkError(
                f"子集 `{subset}` 仅有 {len(subset_records)} 题，无法随机抽取 {per_subset_count} 题。"
            )
        sampled.extend(rng.sample(subset_records, per_subset_count))
    return sampled



def apply_offset_limit(records: list[BenchmarkRecord], *, offset: int = 0, limit: int | None = None) -> list[BenchmarkRecord]:
    if offset < 0:
        raise BenchmarkError("--offset 不能为负数")
    sliced = records[offset:]
    if limit is not None:
        if limit < 0:
            raise BenchmarkError("--limit 不能为负数")
        sliced = sliced[:limit]
    return sliced


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


CHEMQA_RUN_LIFECYCLE_STATUSES = {"planned", "running", "done"}
CHEMQA_TERMINAL_STATES = {"completed", "failed", "cancelled"}


def normalize_chemqa_run_status(payload: dict[str, Any] | None) -> dict[str, Any]:
    normalized = deep_copy_jsonish(payload or {})
    legacy_status = str(normalized.get("status") or "").strip()
    status = legacy_status
    terminal_state = str(normalized.get("terminal_state") or "").strip()
    terminal_reason_code = str(normalized.get("terminal_reason_code") or "").strip()
    terminal_reason = str(normalized.get("terminal_reason") or normalized.get("reason") or "").strip()

    artifact_collection = normalized.get("artifact_collection")
    if isinstance(artifact_collection, dict):
        artifact_collection_payload = deep_copy_jsonish(artifact_collection)
    else:
        artifact_collection_payload = {}
    artifact_collection_status = str(artifact_collection_payload.get("status") or "").strip()

    if legacy_status == "completed":
        status = "done"
        terminal_state = terminal_state or "completed"
        artifact_collection_status = artifact_collection_status or "ok"
    elif legacy_status == "completed_with_artifact_errors":
        status = "done"
        terminal_state = terminal_state or "completed"
        terminal_reason_code = terminal_reason_code or "artifact_collection_error"
        artifact_collection_status = "error"
    elif legacy_status == "stalled":
        status = "done"
        terminal_state = terminal_state or "failed"
        terminal_reason_code = terminal_reason_code or "stalled"
    elif legacy_status == "terminal_failure":
        status = "done"
        terminal_state = terminal_state or "failed"
        terminal_reason_code = terminal_reason_code or "terminal_failure"
    elif legacy_status == "failed":
        status = "done"
        terminal_state = terminal_state or "failed"
    elif legacy_status == "abandoned":
        status = "done"
        terminal_state = terminal_state or "cancelled"
        terminal_reason_code = terminal_reason_code or "abandoned"
    elif legacy_status == "cancelled":
        status = "done"
        terminal_state = terminal_state or "cancelled"
        terminal_reason_code = terminal_reason_code or "cancelled"
    elif legacy_status == "done":
        status = "done"
    elif legacy_status not in CHEMQA_RUN_LIFECYCLE_STATUSES:
        status = status or ""

    if status == "done" and not terminal_state:
        if terminal_reason_code in {"abandoned", "cancelled"}:
            terminal_state = "cancelled"
        elif artifact_collection_status == "error":
            terminal_state = "completed"

    if status == "done" and terminal_state == "completed":
        artifact_collection_status = artifact_collection_status or ("error" if normalized.get("artifact_collection_error") else "ok")

    if artifact_collection_status:
        artifact_collection_payload["status"] = artifact_collection_status
        normalized["artifact_collection"] = artifact_collection_payload
    elif "artifact_collection" in normalized and not artifact_collection_payload:
        normalized.pop("artifact_collection", None)

    normalized["status"] = status
    if terminal_state:
        normalized["terminal_state"] = terminal_state
    else:
        normalized.pop("terminal_state", None)
    if terminal_reason_code:
        normalized["terminal_reason_code"] = terminal_reason_code
    else:
        normalized.pop("terminal_reason_code", None)
    if terminal_reason:
        normalized["terminal_reason"] = terminal_reason
    elif "terminal_reason" in normalized:
        normalized.pop("terminal_reason", None)

    if legacy_status and legacy_status != status:
        normalized["legacy_status"] = legacy_status
    elif "legacy_status" in normalized and not normalized["legacy_status"]:
        normalized.pop("legacy_status", None)

    return normalized


def is_chemqa_terminal_status(payload: dict[str, Any] | None) -> bool:
    normalized = normalize_chemqa_run_status(payload)
    return str(normalized.get("status") or "") == "done"


def is_chemqa_success_status(payload: dict[str, Any] | None) -> bool:
    normalized = normalize_chemqa_run_status(payload)
    return (
        str(normalized.get("status") or "") == "done"
        and str(normalized.get("terminal_state") or "") == "completed"
    )


def normalize_loose(text: str) -> str:
    text = normalize_space(text).lower()
    text = text.replace("µ", "u")
    text = re.sub(r"[\s\.,;:!?'\"`~()\[\]{}<>]+", "", text)
    return text


def last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def extract_final_answer_line(text: str) -> str:
    matches = FINAL_ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return ""


def extract_candidate_short_answer(text: str) -> str:
    final_answer = extract_final_answer_line(text)
    if final_answer:
        return final_answer
    last_line = last_nonempty_line(text)
    if last_line and len(last_line) <= 200:
        return last_line
    return normalize_space(text)


def parse_numeric_scalar(text: str) -> float | None:
    if not text:
        return None
    candidate = extract_final_answer_line(text) or text
    candidate = candidate.replace("×10^", "e").replace("x10^", "e")
    matches = NUMBER_RE.findall(candidate)
    if not matches:
        return None
    token = matches[0].replace(",", "")
    try:
        return float(token)
    except ValueError:
        return None


def safe_json_extract(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise BenchmarkError("Cannot extract JSON from empty judge response.")
    for candidate in (stripped,):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    match = JSON_BLOCK_RE.search(stripped)
    if match:
        return json.loads(match.group(1))

    lines = stripped.splitlines()
    for index, line in enumerate(lines):
        candidate = line.lstrip()
        if candidate.startswith("{") or candidate.startswith("["):
            fragment = "\n".join(lines[index:]).strip()
            for end in range(len(fragment), 0, -1):
                try:
                    return json.loads(fragment[:end])
                except json.JSONDecodeError:
                    continue
            break

    brace_positions = [idx for idx in (stripped.find("{"), stripped.rfind("{")) if idx != -1]
    for start in brace_positions:
        fragment = stripped[start:]
        for end in range(len(fragment), 0, -1):
            try:
                return json.loads(fragment[:end])
            except json.JSONDecodeError:
                continue
    raise BenchmarkError(f"Judge response did not contain parseable JSON:\n{text}")


def maybe_json_loads(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def superchem_valid_options(record: BenchmarkRecord) -> tuple[str, ...]:
    options = record.grading.config.get("options") or record.payload.get("options") or {}
    if isinstance(options, dict):
        letters = [str(key).strip().upper() for key in options.keys() if str(key).strip()]
        if letters:
            return tuple(sorted(set(letters)))
    return tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def parse_superchem_option_answer(text: str, *, valid_options: Iterable[str]) -> str:
    valid = tuple(dict.fromkeys(str(item).strip().upper() for item in valid_options if str(item).strip()))
    valid_set = set(valid)
    if not valid_set:
        raise BenchmarkError("SUPERChem valid option set is empty.")

    def extract_letters(candidate: Any) -> list[str]:
        if candidate is None:
            return []
        if isinstance(candidate, dict):
            for key in ("answer", "final_answer", "finalAnswer", "choice", "choices"):
                if key in candidate:
                    return extract_letters(candidate[key])
            letters = [str(key).strip().upper() for key in candidate.keys()]
            return [letter for letter in letters if letter in valid_set]
        if isinstance(candidate, list):
            letters: list[str] = []
            for item in candidate:
                letters.extend(extract_letters(item))
            return letters

        raw = str(candidate).strip().upper()
        if not raw:
            return []
        token_matches = [match for match in SINGLE_LETTER_TOKEN_RE.findall(raw) if match in valid_set]
        if token_matches:
            return token_matches
        compact = re.sub(r"[^A-Z]", "", raw)
        if compact and all(letter in valid_set for letter in compact):
            return list(compact)
        return []

    candidates = [
        extract_final_answer_line(text),
        last_nonempty_line(text),
        text,
    ]
    json_payload = maybe_json_loads(text)
    if json_payload is not None:
        candidates.insert(0, json_payload)
    for candidate in candidates:
        letters = extract_letters(candidate)
        if letters:
            return "|".join(letter for letter in valid if letter in set(letters))
    return ""


def parse_superchem_checkpoint_weight(attrs: str) -> float:
    weight = 1.0
    for key, value in SUPERCHM_ATTR_RE.findall(attrs):
        if key.lower() in {"weight", "points", "score"}:
            try:
                weight = float(value)
            except ValueError:
                weight = 1.0
    return max(weight, 0.0)


def parse_superchem_checkpoints(text: str) -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    for index, match in enumerate(SUPERCHM_XML_CHECKPOINT_RE.finditer(text or ""), start=1):
        body = normalize_space(match.group("body"))
        if not body:
            continue
        checkpoints.append(
            {
                "index": index,
                "weight": parse_superchem_checkpoint_weight(match.group("attrs") or ""),
                "text": body,
            }
        )
    if checkpoints:
        return checkpoints

    for match in SUPERCHM_INLINE_CHECKPOINT_RE.finditer(text or ""):
        body = normalize_space(match.group("body"))
        if not body:
            continue
        checkpoints.append(
            {
                "index": int(match.group("index")),
                "weight": 1.0,
                "text": body,
            }
        )
    return checkpoints


def superchem_image_paths(record: BenchmarkRecord) -> list[Path]:
    payload = record.payload
    paths: list[Path] = []
    for item in payload.get("question_image_paths") or []:
        text = str(item or "").strip()
        if text:
            paths.append(Path(text).expanduser().resolve())
    option_paths = payload.get("option_image_paths") or {}
    if isinstance(option_paths, dict):
        for items in option_paths.values():
            for item in items or []:
                text = str(item or "").strip()
                if text:
                    paths.append(Path(text).expanduser().resolve())
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def build_superchem_question_markdown(record: BenchmarkRecord, *, image_relpaths: list[str]) -> str:
    payload = record.payload
    options = payload.get("options") or {}
    lines = [
        "# SUPERChem Benchmark Record",
        f"Record ID: {record.record_id}",
        f"Source UUID: {source_pair_key(record)}",
        f"Modality: {payload.get('modality') or 'text_only'}",
        "",
        "Question:",
        str(payload.get("question") or record.prompt).strip(),
        "",
        "Options:",
    ]
    if isinstance(options, dict):
        for key in sorted(options):
            value = str(options.get(key) or "").strip() or "[see image]"
            lines.append(f"- {key}. {value}")
    if image_relpaths:
        lines.extend(["", "Local images to inspect:"])
        for item in image_relpaths:
            lines.append(f"- {item}")
    return "\n".join(lines).strip() + "\n"


def ensure_runtime_bundle(record: BenchmarkRecord, *, bundle_root: Path) -> RuntimeBundle | None:
    if record.dataset != "superchem":
        return None
    bundle_dir = bundle_root / slugify(record.record_id, limit=80)
    question_markdown = bundle_dir / "question.md"
    image_dir = bundle_dir / "images"
    image_files: list[Path] = []

    with RUNTIME_BUNDLE_LOCK:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        image_relpaths: list[str] = []
        for index, source_path in enumerate(superchem_image_paths(record), start=1):
            extension = source_path.suffix or ".bin"
            target_path = image_dir / f"img{index:02d}{extension}"
            if source_path.is_file():
                shutil.copy2(source_path, target_path)
                image_files.append(target_path)
                image_relpaths.append(str(target_path.relative_to(bundle_dir)))
            else:
                image_relpaths.append(str(source_path))
        question_markdown.write_text(
            build_superchem_question_markdown(record, image_relpaths=image_relpaths),
            encoding="utf-8",
        )
    return RuntimeBundle(bundle_dir=bundle_dir, question_markdown=question_markdown, image_files=image_files)


def build_single_llm_prompt(
    record: BenchmarkRecord,
    *,
    websearch_enabled: bool,
    input_bundle: RuntimeBundle | None = None,
) -> str:
    instructions = [
        "You are answering a chemistry benchmark question.",
        "Be careful, concise, and do not fabricate missing facts.",
    ]
    if websearch_enabled:
        instructions.append("You may use web search if it is genuinely helpful.")
    else:
        instructions.append("Do not use web search or external browsing.")

    if record.eval_kind == "superchem_multiple_choice_rpf":
        instructions.append("This is a chemistry multiple-choice question.")
        instructions.append("Show concise reasoning, then end with exactly one line formatted as: FINAL ANSWER: <option letters>.")
        instructions.append("If multiple options are correct, separate the letters with `|`.")
        if input_bundle is not None:
            instructions.append(f"Local file bundle: {input_bundle.bundle_dir}")
            instructions.append(f"Read the question bundle file first: {input_bundle.question_markdown}")
            if input_bundle.image_files:
                instructions.append("Inspect the local image files referenced in the bundle before answering.")
    elif record.eval_kind == "chembench_open_ended":
        instructions.append("Show brief reasoning if needed, then end with exactly one line formatted as: FINAL ANSWER: <answer>.")
    elif record.eval_kind == "frontierscience_olympiad":
        instructions.append("End with exactly one line formatted as: FINAL ANSWER: <answer>.")
    elif record.eval_kind == "conformabench_constructive":
        instructions.append("Propose one chemically valid molecule and end with exactly one line formatted as: FINAL ANSWER: <SMILES>.")
    else:
        instructions.append("Provide a complete answer. If you include a final answer line, use: FINAL ANSWER: <answer>.")

    return "\n".join(instructions) + "\n\nQUESTION:\n" + record.prompt.strip()


def build_chemqa_goal(
    record: BenchmarkRecord,
    *,
    websearch_enabled: bool,
    input_bundle: RuntimeBundle | None = None,
) -> str:
    instructions = [
        "Solve the following chemistry benchmark question.",
        "Return a final answer that is faithful to the prompt.",
    ]
    if websearch_enabled:
        instructions.append("Web search may be used if helpful.")
    else:
        instructions.append("Do not use web search or external browsing.")
    if record.eval_kind == "superchem_multiple_choice_rpf":
        instructions.append("This is a multiple-choice chemistry question.")
        instructions.append("End with a line `FINAL ANSWER: <option letters>`.")
        instructions.append("If multiple options are correct, separate the letters with `|`.")
        if input_bundle is not None:
            instructions.append(f"Use the local file bundle at `{input_bundle.bundle_dir}`.")
            instructions.append(f"Open `{input_bundle.question_markdown}` first and inspect any referenced images.")
    elif record.eval_kind == "conformabench_constructive":
        instructions.append("End with exactly one line `FINAL ANSWER: <SMILES>`.")
    elif record.eval_kind in {"chembench_open_ended", "frontierscience_olympiad"}:
        instructions.append("If appropriate, end with a line `FINAL ANSWER: <answer>`.")
    return "\n".join(instructions) + "\n\nQUESTION:\n" + record.prompt.strip()


def summarize_payloads(payloads: list[dict[str, Any]]) -> str:
    texts = [str(item.get("text") or "").strip() for item in payloads if str(item.get("text") or "").strip()]
    return "\n\n".join(texts).strip()


def normalize_answer_tracks(*, short_answer_text: str = "", full_response_text: str = "") -> tuple[str, str]:
    short_text = str(short_answer_text or "").strip()
    full_text = str(full_response_text or "").strip()
    if not short_text and full_text:
        short_text = extract_candidate_short_answer(full_text)
    if not full_text and short_text:
        full_text = f"FINAL ANSWER: {short_text}"
    return short_text, full_text


def render_chemqa_submission_rationale(final_submission: dict[str, Any], *, final_answer_text: str = "") -> str:
    parts: list[str] = []
    summary = normalize_space(str(final_submission.get("summary") or ""))
    if summary:
        parts.extend(["Summary:", summary])

    submission_trace = list(final_submission.get("submission_trace") or [])
    if submission_trace:
        parts.append("")
        parts.append("Reasoning / submission trace:")
        for item in submission_trace:
            if not isinstance(item, dict):
                continue
            step = normalize_space(str(item.get("step") or item.get("phase") or "reasoning"))
            detail = normalize_space(str(item.get("detail") or item.get("summary") or item.get("finding") or ""))
            status = normalize_space(str(item.get("status") or ""))
            bullet = f"- {step}"
            if status:
                bullet += f" [{status}]"
            if detail:
                bullet += f": {detail}"
            parts.append(bullet)

    claim_anchors = list(final_submission.get("claim_anchors") or [])
    if claim_anchors:
        parts.append("")
        parts.append("Claim anchors:")
        for item in claim_anchors:
            if not isinstance(item, dict):
                continue
            claim = normalize_space(str(item.get("claim") or ""))
            anchor = normalize_space(str(item.get("anchor") or ""))
            if claim:
                parts.append(f"- {anchor + ': ' if anchor else ''}{claim}")

    evidence_limits = list(final_submission.get("evidence_limits") or [])
    if evidence_limits:
        parts.append("")
        parts.append("Evidence limits:")
        for item in evidence_limits:
            text = normalize_space(str(item or ""))
            if text:
                parts.append(f"- {text}")

    final_answer = normalize_space(final_answer_text or str(final_submission.get("direct_answer") or ""))
    if final_answer:
        parts.append("")
        parts.append(f"FINAL ANSWER: {final_answer}")

    return "\n".join(part for part in parts if part is not None).strip()


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}



def build_chemqa_response_from_submission(*, final_submission: dict[str, Any], final_answer_text: str = "") -> tuple[str, str]:
    short_answer_text = normalize_space(final_answer_text or str(final_submission.get("direct_answer") or ""))
    full_response_text = render_chemqa_submission_rationale(final_submission, final_answer_text=short_answer_text)
    return normalize_answer_tracks(short_answer_text=short_answer_text, full_response_text=full_response_text)



def build_chemqa_full_response(*, qa_result: dict[str, Any]) -> tuple[str, str]:
    artifact_paths = dict(qa_result.get("artifact_paths") or {})
    short_answer_text = normalize_space(str(qa_result.get("final_answer") or ""))
    final_submission_path = str(artifact_paths.get("final_submission") or "").strip()
    if final_submission_path:
        path = Path(final_submission_path)
        if path.is_file():
            try:
                final_submission = json.loads(path.read_text(encoding="utf-8"))
                return build_chemqa_response_from_submission(final_submission=final_submission, final_answer_text=short_answer_text)
            except Exception:
                pass
    final_answer_path = str(artifact_paths.get("final_answer") or "").strip()
    if final_answer_path:
        path = Path(final_answer_path)
        if path.is_file():
            fallback_text = path.read_text(encoding="utf-8").strip()
            return normalize_answer_tracks(short_answer_text=short_answer_text, full_response_text=fallback_text)
    return normalize_answer_tracks(short_answer_text=short_answer_text, full_response_text="")


class JudgeClient:
    def __init__(
        self,
        *,
        judge_agent: str,
        timeout_seconds: int,
        config_path: Path,
    ) -> None:
        self.judge_agent = judge_agent
        self.timeout_seconds = timeout_seconds
        self.config_path = config_path
        self._lock = threading.Lock()

    def evaluate_json(self, prompt: str) -> dict[str, Any]:
        session_id = f"benchmark-judge-{uuid.uuid4().hex[:12]}"
        command = [
            "openclaw",
            "agent",
            "--local",
            "--agent",
            self.judge_agent,
            "--session-id",
            session_id,
            "--message",
            prompt,
            "--thinking",
            BENCHMARK_AGENT_THINKING,
            "--timeout",
            str(self.timeout_seconds),
            "--json",
        ]
        env = os.environ.copy()
        env["OPENCLAW_CONFIG_PATH"] = str(self.config_path)
        with self._lock:
            result = run_subprocess(command, env=env, timeout=self.timeout_seconds + 30)
            payload = parse_json_stdout(result, command)
        result_payload = unwrap_agent_payload(payload)
        reply = summarize_payloads(list((result_payload.get("payloads") or [])))
        parsed = safe_json_extract(reply)
        if not isinstance(parsed, dict):
            raise BenchmarkError(f"Judge must return a JSON object, got: {reply}")
        return parsed


class SingleLLMRunner(_BenchmarkingSingleLLMRunner):
    def __init__(
        self,
        *,
        agent_id: str,
        timeout_seconds: int,
        config_path: Path,
        runtime_bundle_root: Path,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            timeout_seconds=timeout_seconds,
            config_path=config_path,
            runtime_bundle_root=runtime_bundle_root,
            run_subprocess=run_subprocess,
            parse_json_stdout=parse_json_stdout,
            unwrap_agent_payload=unwrap_agent_payload,
            summarize_payloads=summarize_payloads,
            normalize_answer_tracks=normalize_answer_tracks,
            ensure_runtime_bundle=ensure_runtime_bundle,
            build_single_llm_prompt=build_single_llm_prompt,
            slugify=slugify,
            benchmark_agent_thinking=BENCHMARK_AGENT_THINKING,
        )


class ChemQARunner(_BenchmarkingChemQARunner):
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
    ) -> None:
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
            current_python=current_python,
            run_subprocess=run_subprocess,
            parse_json_stdout=parse_json_stdout,
            deep_copy_jsonish=deep_copy_jsonish,
            ensure_runtime_bundle=ensure_runtime_bundle,
            build_chemqa_goal=build_chemqa_goal,
            cleanup_manifest_path=cleanup_manifest_path,
            build_cleanup_manifest_payload=build_cleanup_manifest_payload,
            write_cleanup_manifest=write_cleanup_manifest,
            register_pending_cleanup_manifest=register_pending_cleanup_manifest,
            update_cleanup_manifest=update_cleanup_manifest,
            invoke_cleanroom_cleanup=invoke_cleanroom_cleanup,
            unregister_pending_cleanup_manifest=unregister_pending_cleanup_manifest,
            now_stamp=now_stamp,
            slugify=slugify,
            default_chemqa_preset=DEFAULT_CHEMQA_PRESET,
            default_openclaw_env_file=DEFAULT_OPENCLAW_ENV_FILE,
            actual_slot_ids=actual_slot_ids,
            chemqa_workspace_roots=CHEMQA_WORKSPACE_ROOTS,
            normalize_chemqa_run_status=normalize_chemqa_run_status,
            is_chemqa_terminal_status=is_chemqa_terminal_status,
            is_chemqa_success_status=is_chemqa_success_status,
            build_chemqa_full_response=build_chemqa_full_response,
            build_chemqa_response_from_submission=build_chemqa_response_from_submission,
            load_yaml_mapping=load_yaml_mapping,
            normalize_space=normalize_space,
            benchmark_error_factory=BenchmarkError,
            cleanup_error_factory=CleanupFatalError,
            benchmark_agent_thinking=BENCHMARK_AGENT_THINKING,
        )

    def _wait_for_terminal_status(self, run_id: str, *, timeout_seconds: int) -> dict[str, Any]:
        if not hasattr(self, "_is_chemqa_terminal_status"):
            self._is_chemqa_terminal_status = is_chemqa_terminal_status
        if not hasattr(self, "_normalize_chemqa_run_status"):
            self._normalize_chemqa_run_status = normalize_chemqa_run_status
        if not hasattr(self, "_benchmark_error_factory"):
            self._benchmark_error_factory = BenchmarkError
        return super()._wait_for_terminal_status(run_id, timeout_seconds=timeout_seconds)

    def _candidate_protocol_dirs(self, run_id: str, run_status: dict[str, Any]) -> list[Path]:
        if not hasattr(self, "_actual_slot_ids"):
            self._actual_slot_ids = actual_slot_ids
        if not hasattr(self, "_chemqa_workspace_roots"):
            self._chemqa_workspace_roots = CHEMQA_WORKSPACE_ROOTS
        return super()._candidate_protocol_dirs(run_id, run_status)

    def _build_candidate_submission_fallback(self, run_id: str, run_status: dict[str, Any]) -> tuple[str, str, dict[str, Any]] | None:
        if not hasattr(self, "_load_yaml_mapping"):
            self._load_yaml_mapping = load_yaml_mapping
        if not hasattr(self, "_build_chemqa_response_from_submission"):
            self._build_chemqa_response_from_submission = build_chemqa_response_from_submission
        if not hasattr(self, "_normalize_space"):
            self._normalize_space = normalize_space
        return super()._build_candidate_submission_fallback(run_id, run_status)


def build_runner(*, runner_kind: str, **kwargs):
    return _runner_factory(
        runner_kind=runner_kind,
        chemqa_runner_cls=ChemQARunner,
        single_llm_runner_cls=SingleLLMRunner,
        **kwargs,
    )


def evaluate_chembench_open_ended(
    record: BenchmarkRecord,
    *,
    short_answer_text: str,
    full_response_text: str,
    judge: JudgeClient | None = None,
) -> EvaluationResult:
    _ = judge
    expected = str(record.grading.reference_answer or record.payload.get("target") or record.reference_answer)
    predicted_short, _ = normalize_answer_tracks(short_answer_text=short_answer_text, full_response_text=full_response_text)
    expected_norm = normalize_loose(expected)
    predicted_norm = normalize_loose(predicted_short)

    expected_num = parse_numeric_scalar(expected)
    predicted_num = parse_numeric_scalar(predicted_short)
    exact_match = predicted_norm == expected_norm
    relative_tolerance = record.grading.config.get("relative_tolerance")
    mae = None
    mse = None
    within_relative_tolerance = None
    if expected_num is not None and predicted_num is not None:
        mae = abs(predicted_num - expected_num)
        mse = mae * mae
        if relative_tolerance is not None:
            denom = max(abs(expected_num), 1e-12)
            within_relative_tolerance = mae <= abs(float(relative_tolerance)) * denom
        if mae <= 1e-12:
            exact_match = True
        if within_relative_tolerance:
            exact_match = True

    preferred = str(record.grading.config.get("preferred_score") or "exact_str_match")
    if preferred == "mae" and mae is not None:
        score = mae
        normalized_score = 1.0 / (1.0 + mae)
        direction = "lower_is_better"
    elif preferred == "mse" and mse is not None:
        score = mse
        normalized_score = 1.0 / (1.0 + mse)
        direction = "lower_is_better"
    else:
        score = 1.0 if exact_match else 0.0
        normalized_score = score
        direction = "higher_is_better"
        preferred = "exact_str_match"

    return EvaluationResult(
        eval_kind=record.eval_kind,
        score=float(score),
        max_score=1.0,
        normalized_score=float(normalized_score),
        passed=bool(exact_match),
        primary_metric=preferred,
        primary_metric_direction=direction,
        details={
            "expected": expected,
            "predicted_short": predicted_short,
            "exact_match": exact_match,
            "expected_numeric": expected_num,
            "predicted_numeric": predicted_num,
            "mae": mae,
            "mse": mse,
            "relative_tolerance": relative_tolerance,
            "within_relative_tolerance": within_relative_tolerance,
        },
    )


def heuristic_semantic_match(expected: str, predicted: str) -> bool | None:
    expected_short = extract_candidate_short_answer(expected)
    predicted_short = extract_candidate_short_answer(predicted)
    if not expected_short or not predicted_short:
        return None
    expected_num = parse_numeric_scalar(expected_short)
    predicted_num = parse_numeric_scalar(predicted_short)
    if expected_num is not None and predicted_num is not None:
        return math.isclose(expected_num, predicted_num, rel_tol=1e-4, abs_tol=1e-8)
    expected_norm = normalize_loose(expected_short)
    predicted_norm = normalize_loose(predicted_short)
    if expected_norm == predicted_norm:
        return True
    if expected_norm and expected_norm in predicted_norm:
        return True
    if predicted_norm and predicted_norm in expected_norm:
        return True
    return None


def evaluate_frontierscience_olympiad(
    record: BenchmarkRecord,
    *,
    short_answer_text: str,
    full_response_text: str,
    judge: JudgeClient,
) -> EvaluationResult:
    expected = record.grading.reference_answer
    predicted, full_text = normalize_answer_tracks(short_answer_text=short_answer_text, full_response_text=full_response_text)
    heuristic = heuristic_semantic_match(expected, predicted)
    if heuristic is not None:
        return EvaluationResult(
            eval_kind=record.eval_kind,
            score=1.0 if heuristic else 0.0,
            max_score=1.0,
            normalized_score=1.0 if heuristic else 0.0,
            passed=bool(heuristic),
            primary_metric="semantic_match",
            primary_metric_direction="higher_is_better",
            details={
                "method": "heuristic",
                "expected": expected,
                "predicted_short": predicted,
            },
        )

    prompt = f"""
You are evaluating a chemistry olympiad benchmark answer.
Decide whether the candidate answer matches the reference answer semantically.
Ignore harmless formatting differences, punctuation, capitalization, and equivalent chemical naming.
Do not give partial credit.
Return strict JSON only.

Required JSON schema:
{{
  "correct": true,
  "score": 1.0,
  "rationale": "brief explanation",
  "expected_answer": "...",
  "candidate_answer": "..."
}}

QUESTION:
{record.prompt}

REFERENCE ANSWER:
{expected}

CANDIDATE SHORT ANSWER:
{predicted}

CANDIDATE FULL RESPONSE:
{full_text}
""".strip()
    judged = judge.evaluate_json(prompt)
    correct = bool(judged.get("correct"))
    score = 1.0 if correct else 0.0
    return EvaluationResult(
        eval_kind=record.eval_kind,
        score=score,
        max_score=1.0,
        normalized_score=score,
        passed=correct,
        primary_metric="semantic_match",
        primary_metric_direction="higher_is_better",
        details={
            "method": "judge",
            "expected": expected,
            "predicted_short": predicted,
            "judge": judged,
        },
    )


def parse_frontierscience_research_rubric(text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("Points:"):
            i += 1
            continue
        match = re.match(r"Points:\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*Item:\s*(.*)", line)
        if not match:
            i += 1
            continue
        points = float(match.group(1))
        description_parts = [match.group(2).strip()]
        i += 1
        while i < len(lines) and not lines[i].strip().startswith("Points:"):
            description_parts.append(lines[i].rstrip())
            i += 1
        description = "\n".join(part for part in description_parts if part is not None).strip()
        items.append({"points": points, "description": description})
    return items


def evaluate_conformabench_constructive(
    record: BenchmarkRecord,
    *,
    short_answer_text: str,
    full_response_text: str,
    judge: JudgeClient,
) -> EvaluationResult:
    short_text, full_text = normalize_answer_tracks(short_answer_text=short_answer_text, full_response_text=full_response_text)
    final_answer = extract_final_answer_line(full_text) or short_text
    hidden_ref = str(record.grading.config.get("hidden_judge_spec_ref") or record.payload.get("hidden_judge_spec_ref") or "").strip()
    if not hidden_ref:
        raise BenchmarkError(f"ConformaBench record is missing hidden_judge_spec_ref: {record.record_id}")
    hidden_path = resolve_hidden_judge_spec_path(record.source_file, hidden_ref)
    hidden_spec = load_hidden_judge_spec(hidden_path)
    try:
        gate_details = evaluate_conformabench_submission(final_answer_smiles=final_answer, hidden_spec=hidden_spec)
    except ConformaBenchDependencyError as exc:
        raise BenchmarkError(str(exc)) from exc
    except ConformaBenchJudgeError as exc:
        raise BenchmarkError(f"ConformaBench judge failed for `{record.record_id}`: {exc}") from exc

    passed = bool(gate_details.get("passed"))
    score = 1.0 if passed else 0.0
    details = {
        "method": "conformabench_rdkit_gate",
        "hidden_judge_spec_ref": hidden_ref,
        "hidden_judge_spec_path": str(hidden_path),
        **gate_details,
    }

    rubric_items = parse_frontierscience_research_rubric(record.grading.reference_answer)
    if passed and rubric_items:
        rubric_lines = [f"{idx + 1}. [{item['points']} points] {item['description']}" for idx, item in enumerate(rubric_items)]
        max_score = float(sum(item["points"] for item in rubric_items))
        candidate_response = full_text or short_text
        prompt = f"""
You are grading a chemistry benchmark explanation against a point rubric.
The submitted molecule has already passed a deterministic RDKit structure/geometry gate.
For each rubric item, award either 0 or the item's full points only.
Return strict JSON only.

Required JSON schema:
{{
  "items": [
    {{"index": 1, "awarded": 1.0, "max_points": 1.0, "met": true, "rationale": "brief"}}
  ],
  "total_awarded": 0.0,
  "max_points": {max_score},
  "summary": "brief overall summary"
}}

QUESTION:
{record.prompt}

RUBRIC ITEMS:
{os.linesep.join(rubric_lines)}

CANDIDATE ANSWER:
{candidate_response}
""".strip()
        try:
            details["rubric"] = judge.evaluate_json(prompt)
        except Exception as exc:
            details["rubric_error"] = str(exc)

    return EvaluationResult(
        eval_kind=record.eval_kind,
        score=score,
        max_score=1.0,
        normalized_score=score,
        passed=passed,
        primary_metric="rdkit_gate_pass",
        primary_metric_direction="higher_is_better",
        details=details,
    )


def evaluate_frontierscience_research(
    record: BenchmarkRecord,
    *,
    short_answer_text: str,
    full_response_text: str,
    judge: JudgeClient,
) -> EvaluationResult:
    rubric_items = parse_frontierscience_research_rubric(record.grading.reference_answer)
    if not rubric_items:
        raise BenchmarkError(f"No rubric items parsed for record: {record.record_id}")
    rubric_lines = [f"{idx + 1}. [{item['points']} points] {item['description']}" for idx, item in enumerate(rubric_items)]
    max_score = float(sum(item["points"] for item in rubric_items))
    short_text, full_text = normalize_answer_tracks(short_answer_text=short_answer_text, full_response_text=full_response_text)
    candidate_response = full_text or short_text
    prompt = f"""
You are grading a chemistry research benchmark response against a point rubric.
For each rubric item, award either 0 or the item's full points only.
Do not invent extra rubric items.
Return strict JSON only.

Required JSON schema:
{{
  "items": [
    {{"index": 1, "awarded": 1.0, "max_points": 1.0, "met": true, "rationale": "brief"}}
  ],
  "total_awarded": 0.0,
  "max_points": {max_score},
  "summary": "brief overall summary"
}}

QUESTION:
{record.prompt}

RUBRIC ITEMS:
{os.linesep.join(rubric_lines)}

CANDIDATE ANSWER:
{candidate_response}
""".strip()
    judged = judge.evaluate_json(prompt)
    judged_items = judged.get("items")
    if not isinstance(judged_items, list):
        raise BenchmarkError(f"Judge response missing items list: {judged}")

    awarded_items: list[dict[str, Any]] = []
    total_awarded = 0.0
    for idx, rubric_item in enumerate(rubric_items, start=1):
        judged_item = next((item for item in judged_items if int(item.get("index", -1)) == idx), None)
        if not isinstance(judged_item, dict):
            awarded = 0.0
            rationale = "Judge omitted this rubric item; treated as unmet."
            met = False
        else:
            met = bool(judged_item.get("met"))
            awarded = float(judged_item.get("awarded") or 0.0)
            max_points = float(rubric_item["points"])
            awarded = max(0.0, min(max_points, awarded))
            if met and not math.isclose(awarded, max_points, rel_tol=1e-9, abs_tol=1e-9):
                awarded = max_points
            if not met:
                awarded = 0.0
            rationale = str(judged_item.get("rationale") or "")
        total_awarded += awarded
        awarded_items.append(
            {
                "index": idx,
                "awarded": awarded,
                "max_points": float(rubric_item["points"]),
                "met": met,
                "description": rubric_item["description"],
                "rationale": rationale,
            }
        )

    normalized_score = 0.0 if max_score <= 0 else total_awarded / max_score
    return EvaluationResult(
        eval_kind=record.eval_kind,
        score=total_awarded,
        max_score=max_score,
        normalized_score=normalized_score,
        passed=normalized_score > 0.0,
        primary_metric="rubric_points",
        primary_metric_direction="higher_is_better",
        details={
            "judge": judged,
            "rubric_items": awarded_items,
            "summary": judged.get("summary"),
        },
    )


def evaluate_superchem_multiple_choice_rpf(
    record: BenchmarkRecord,
    *,
    short_answer_text: str,
    full_response_text: str,
    judge: JudgeClient,
) -> EvaluationResult:
    valid_options = superchem_valid_options(record)
    short_text, full_text = normalize_answer_tracks(short_answer_text=short_answer_text, full_response_text=full_response_text)
    expected = parse_superchem_option_answer(record.grading.reference_answer, valid_options=valid_options) or record.reference_answer
    predicted = parse_superchem_option_answer(short_text, valid_options=valid_options)
    answer_accuracy = 1.0 if predicted and predicted == expected else 0.0

    checkpoints = parse_superchem_checkpoints(str(record.grading.config.get("reference_reasoning") or record.payload.get("reference_reasoning") or ""))
    if not checkpoints:
        raise BenchmarkError(f"No SUPERChem checkpoints parsed for record: {record.record_id}")

    rendered_checkpoints = [
        f"{item['index']}. [weight={item['weight']}] {item['text']}"
        for item in checkpoints
    ]
    prompt = f"""
You are scoring a chemistry candidate response against expert reasoning checkpoints from SUPERChem.
For each checkpoint, mark it matched only if the candidate response clearly covers the same reasoning step or conclusion.
Do not award partial matches.
Return strict JSON only.

Required JSON schema:
{{
  "items": [
    {{"index": 1, "matched": true, "rationale": "brief"}}
  ],
  "summary": "brief overall summary"
}}

QUESTION:
{record.prompt}

REFERENCE CHECKPOINTS:
{os.linesep.join(rendered_checkpoints)}

CANDIDATE RESPONSE:
{full_text}
""".strip()
    judged = judge.evaluate_json(prompt)
    judged_items = judged.get("items")
    if not isinstance(judged_items, list):
        raise BenchmarkError(f"Judge response missing checkpoint items list: {judged}")

    total_weight = float(sum(float(item["weight"]) for item in checkpoints))
    matched_weight = 0.0
    checkpoint_matches: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        judged_item = next((item for item in judged_items if int(item.get("index", -1)) == checkpoint["index"]), None)
        matched = bool(judged_item.get("matched")) if isinstance(judged_item, dict) else False
        rationale = "" if not isinstance(judged_item, dict) else str(judged_item.get("rationale") or "")
        if matched:
            matched_weight += float(checkpoint["weight"])
        checkpoint_matches.append(
            {
                "index": checkpoint["index"],
                "weight": float(checkpoint["weight"]),
                "matched": matched,
                "text": checkpoint["text"],
                "rationale": rationale,
            }
        )
    rpf = 0.0 if total_weight <= 0 else matched_weight / total_weight
    return EvaluationResult(
        eval_kind=record.eval_kind,
        score=answer_accuracy,
        max_score=1.0,
        normalized_score=answer_accuracy,
        passed=bool(answer_accuracy),
        primary_metric="answer_accuracy",
        primary_metric_direction="higher_is_better",
        details={
            "parsed_reference": expected,
            "parsed_prediction": predicted,
            "answer_accuracy": answer_accuracy,
            "rpf": rpf,
            "checkpoint_matches": checkpoint_matches,
            "judge": judged,
        },
    )


def evaluate_generic_semantic(
    record: BenchmarkRecord,
    *,
    short_answer_text: str,
    full_response_text: str,
    judge: JudgeClient,
) -> EvaluationResult:
    expected = record.grading.reference_answer
    predicted, full_text = normalize_answer_tracks(short_answer_text=short_answer_text, full_response_text=full_response_text)
    heuristic = heuristic_semantic_match(expected, predicted)
    if heuristic is not None:
        score = 1.0 if heuristic else 0.0
        return EvaluationResult(
            eval_kind=record.eval_kind,
            score=score,
            max_score=1.0,
            normalized_score=score,
            passed=bool(heuristic),
            primary_metric="semantic_match",
            primary_metric_direction="higher_is_better",
            details={"method": "heuristic", "expected": expected, "predicted_short": predicted},
        )

    prompt = f"""
You are evaluating whether a benchmark candidate answer matches a reference answer.
Return strict JSON only.

Required JSON schema:
{{
  "correct": true,
  "score": 1.0,
  "rationale": "brief explanation"
}}

QUESTION:
{record.prompt}

REFERENCE ANSWER:
{expected}

CANDIDATE SHORT ANSWER:
{predicted}

CANDIDATE FULL RESPONSE:
{full_text}
""".strip()
    judged = judge.evaluate_json(prompt)
    correct = bool(judged.get("correct"))
    score = 1.0 if correct else 0.0
    return EvaluationResult(
        eval_kind=record.eval_kind,
        score=score,
        max_score=1.0,
        normalized_score=score,
        passed=correct,
        primary_metric="semantic_match",
        primary_metric_direction="higher_is_better",
        details={"method": "judge", "judge": judged, "expected": expected, "predicted_short": predicted},
    )


def evaluate_answer(
    record: BenchmarkRecord,
    *,
    short_answer_text: str,
    full_response_text: str,
    judge: JudgeClient,
) -> EvaluationResult:
    return evaluate_record(
        record,
        short_answer_text=short_answer_text,
        full_response_text=full_response_text,
        judge=judge,
    )


register_evaluator("chembench_open_ended", evaluate_chembench_open_ended)
register_evaluator("conformabench_constructive", evaluate_conformabench_constructive)
register_evaluator("frontierscience_olympiad", evaluate_frontierscience_olympiad)
register_evaluator("frontierscience_research", evaluate_frontierscience_research)
register_evaluator("superchem_multiple_choice_rpf", evaluate_superchem_multiple_choice_rpf)
register_evaluator("generic_semantic", evaluate_generic_semantic)


def average_optional_metric(items: list[GroupRecordResult], key: str) -> float | None:
    values: list[float] = []
    for item in items:
        details = item.evaluation.get("details") or {}
        value = details.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def aggregate_bucket(items: list[GroupRecordResult]) -> dict[str, Any]:
    return {
        "count": len(items),
        "pass_count": sum(1 for item in items if item.evaluation["passed"]),
        "avg_score": sum(float(item.evaluation["score"]) for item in items) / len(items),
        "avg_normalized_score": sum(float(item.evaluation["normalized_score"]) for item in items) / len(items),
        "avg_elapsed_seconds": sum(float(item.elapsed_seconds) for item in items) / len(items),
        "avg_answer_accuracy": average_optional_metric(items, "answer_accuracy"),
        "avg_rpf": average_optional_metric(items, "rpf"),
    }



def materialize_group_failure_results(
    *,
    group: ExperimentGroup,
    records: list[BenchmarkRecord],
    output_root: Path,
    error_message: str,
) -> list[GroupRecordResult]:
    group_results = [
        build_error_group_record_result(group=group, record=record, error_message=error_message)
        for record in records
    ]
    for entry in group_results:
        save_json(output_root / "per-record" / group.id / f"{slugify(entry.record_id)}.json", asdict(entry))
    return group_results



def aggregate_results(results: list[GroupRecordResult]) -> dict[str, Any]:
    grouped: dict[str, list[GroupRecordResult]] = {}
    for item in results:
        grouped.setdefault(item.group_id, []).append(item)

    summary_groups: dict[str, Any] = {}
    summary_group_subset: dict[str, dict[str, Any]] = {}
    for group_id, items in grouped.items():
        by_eval_kind: dict[str, list[GroupRecordResult]] = {}
        by_subset: dict[str, list[GroupRecordResult]] = {}
        for item in items:
            by_eval_kind.setdefault(item.eval_kind, []).append(item)
            by_subset.setdefault(item.subset, []).append(item)
        bucket = aggregate_bucket(items)
        summary_groups[group_id] = {
            "group_label": items[0].group_label,
            "runner": items[0].runner,
            "websearch": items[0].websearch,
            **bucket,
            "by_eval_kind": {
                eval_kind: {
                    key: value
                    for key, value in aggregate_bucket(eval_items).items()
                }
                for eval_kind, eval_items in by_eval_kind.items()
            },
            "by_subset": {
                subset: {
                    key: value
                    for key, value in aggregate_bucket(subset_items).items()
                }
                for subset, subset_items in by_subset.items()
            },
        }
        for subset, subset_items in by_subset.items():
            summary_group_subset[f"{group_id}::{subset}"] = {
                "group_id": group_id,
                "group_label": items[0].group_label,
                "runner": items[0].runner,
                "websearch": items[0].websearch,
                "subset": subset,
                **aggregate_bucket(subset_items),
            }

    return {
        "group_order": list(grouped.keys()),
        "groups": summary_groups,
        "group_subset": summary_group_subset,
    }



def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})



def export_csv_reports(output_root: Path, summary: dict[str, Any], group_ids: list[str]) -> None:
    summary_rows = []
    for group_id in group_ids:
        group_summary = summary["groups"].get(group_id)
        if not group_summary:
            continue
        summary_rows.append(
            {
                "group_id": group_id,
                "runner": group_summary["runner"],
                "websearch": group_summary["websearch"],
                "count": group_summary["count"],
                "pass_count": group_summary["pass_count"],
                "avg_normalized_score": group_summary["avg_normalized_score"],
                "avg_answer_accuracy": group_summary.get("avg_answer_accuracy"),
                "avg_rpf": group_summary.get("avg_rpf"),
            }
        )
    write_csv(
        output_root / "summary_by_group.csv",
        summary_rows,
        [
            "group_id",
            "runner",
            "websearch",
            "count",
            "pass_count",
            "avg_normalized_score",
            "avg_answer_accuracy",
            "avg_rpf",
        ],
    )

    subset_rows = []
    for key in sorted(summary.get("group_subset", {})):
        row = summary["group_subset"][key]
        subset_rows.append(
            {
                "group_id": row["group_id"],
                "runner": row["runner"],
                "websearch": row["websearch"],
                "subset": row["subset"],
                "count": row["count"],
                "pass_count": row["pass_count"],
                "avg_normalized_score": row["avg_normalized_score"],
                "avg_answer_accuracy": row.get("avg_answer_accuracy"),
                "avg_rpf": row.get("avg_rpf"),
            }
        )
    write_csv(
        output_root / "summary_by_group_and_subset.csv",
        subset_rows,
        [
            "group_id",
            "runner",
            "websearch",
            "subset",
            "count",
            "pass_count",
            "avg_normalized_score",
            "avg_answer_accuracy",
            "avg_rpf",
        ],
    )


def select_group_ids(raw: str) -> list[str]:
    group_ids = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in group_ids if item not in EXPERIMENT_GROUPS]
    if unknown:
        raise BenchmarkError(f"Unknown group ids: {', '.join(unknown)}")
    if not group_ids:
        raise BenchmarkError("No experiment groups selected.")
    return group_ids


def select_dataset_files(args: argparse.Namespace) -> list[Path]:
    root = Path(args.benchmark_root).expanduser().resolve()
    if args.files:
        files = [Path(item.strip()).expanduser().resolve() for item in args.files.split(",") if item.strip()]
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            raise BenchmarkError(f"Missing benchmark files: {', '.join(missing)}")
        return files

    discovered = discover_dataset_files(root)
    if args.datasets:
        wanted = {item.strip() for item in args.datasets.split(",") if item.strip()}
        discovered = [path for path in discovered if dataset_name_from_file(path) in wanted]
    return discovered


def print_dataset_listing(paths: list[Path]) -> None:
    payload = [
        {
            "dataset": dataset_name_from_file(path),
            "path": str(path),
        }
        for path in paths
    ]
    print(json.dumps(payload, indent=2, ensure_ascii=False))



def print_selected_records(records: list[BenchmarkRecord]) -> None:
    payload = [
        {
            "record_id": record.record_id,
            "subset": classify_subset(record),
            "dataset": record.dataset,
            "eval_kind": record.eval_kind,
            "source_file": record.source_file,
            "prompt_preview": normalize_space(record.prompt)[:200],
        }
        for record in records
    ]
    print(json.dumps(payload, indent=2, ensure_ascii=False))



def build_group_waves(group_ids: list[str], *, max_concurrent_groups: int) -> list[list[str]]:
    if max_concurrent_groups <= 0:
        raise BenchmarkError("--max-concurrent-groups 必须是正整数")
    web_on_groups = [group_id for group_id in group_ids if EXPERIMENT_GROUPS[group_id].websearch]
    web_off_groups = [group_id for group_id in group_ids if not EXPERIMENT_GROUPS[group_id].websearch]
    waves: list[list[str]] = []
    for bucket in (web_on_groups, web_off_groups):
        for index in range(0, len(bucket), max_concurrent_groups):
            waves.append(bucket[index : index + max_concurrent_groups])
    return waves



def count_per_record_outputs(output_root: Path, *, group_ids: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    per_record_root = output_root / "per-record"
    for group_id in group_ids:
        group_dir = per_record_root / group_id
        counts[group_id] = len(list(group_dir.glob("*.json"))) if group_dir.is_dir() else 0
    return counts



def load_group_record_result(path: Path) -> GroupRecordResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return GroupRecordResult(**payload)



def resolve_aggregate_group_ids(
    selected_group_ids: list[str],
    *,
    output_root: Path,
    merge_existing_per_record: bool,
) -> list[str]:
    if not merge_existing_per_record:
        return list(selected_group_ids)
    present = set(selected_group_ids)
    per_record_root = output_root / "per-record"
    for group_id in EXPERIMENT_GROUPS:
        group_dir = per_record_root / group_id
        if group_dir.is_dir() and any(group_dir.glob("*.json")):
            present.add(group_id)
    return [group_id for group_id in EXPERIMENT_GROUPS if group_id in present]



def load_results_from_output_root(output_root: Path, *, group_ids: list[str]) -> list[GroupRecordResult]:
    results: list[GroupRecordResult] = []
    for group_id in group_ids:
        group_dir = output_root / "per-record" / group_id
        if not group_dir.is_dir():
            continue
        for path in sorted(group_dir.glob("*.json")):
            results.append(load_group_record_result(path))
    return results



def write_wave_status(
    output_root: Path,
    *,
    wave_index: int,
    wave_group_ids: list[str],
    status: str,
    started_at: str,
    completed_at: str | None = None,
    per_record_counts: dict[str, int] | None = None,
    inter_wave_delay_seconds: int | None = None,
) -> None:
    payload: dict[str, Any] = {
        "wave_index": wave_index,
        "groups": wave_group_ids,
        "status": status,
        "started_at": started_at,
    }
    if completed_at is not None:
        payload["completed_at"] = completed_at
    if per_record_counts is not None:
        payload["per_record_counts"] = per_record_counts
    if inter_wave_delay_seconds is not None:
        payload["inter_wave_delay_seconds"] = inter_wave_delay_seconds
    save_json(output_root / "waves" / f"wave-{wave_index:02d}.json", payload)



def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")



def build_execution_error_evaluation(record: BenchmarkRecord, *, error_message: str) -> EvaluationResult:
    return EvaluationResult(
        eval_kind=record.eval_kind,
        score=0.0,
        max_score=1.0,
        normalized_score=0.0,
        passed=False,
        primary_metric="execution_error",
        primary_metric_direction="higher_is_better",
        details={
            "method": "execution_error",
            "error": error_message,
        },
    )



def build_error_group_record_result(
    *,
    group: ExperimentGroup,
    record: BenchmarkRecord,
    error_message: str,
    elapsed_seconds: float = 0.0,
    answer_text: str = "",
    short_answer_text: str = "",
    full_response_text: str = "",
    runner_meta: dict[str, Any] | None = None,
    raw: dict[str, Any] | None = None,
) -> GroupRecordResult:
    evaluation = build_execution_error_evaluation(record, error_message=error_message)
    meta = deep_copy_jsonish(runner_meta or {})
    meta.setdefault("error", error_message)
    payload = deep_copy_jsonish(raw or {"error": error_message})
    short_text, full_text = normalize_answer_tracks(short_answer_text=short_answer_text, full_response_text=full_response_text)
    compatible_answer_text = answer_text or full_text or short_text
    return GroupRecordResult(
        group_id=group.id,
        group_label=group.label,
        runner=group.runner,
        websearch=group.websearch,
        record_id=record.record_id,
        subset=classify_subset(record),
        dataset=record.dataset,
        source_file=record.source_file,
        eval_kind=record.eval_kind,
        prompt=record.prompt,
        reference_answer=record.reference_answer,
        answer_text=compatible_answer_text,
        evaluation=asdict(evaluation),
        runner_meta=meta,
        raw=payload,
        elapsed_seconds=elapsed_seconds,
        error=error_message,
        short_answer_text=short_text,
        full_response_text=full_text,
    )



def run_group(
    *,
    group: ExperimentGroup,
    records: list[BenchmarkRecord],
    output_root: Path,
    single_timeout: int,
    chemqa_timeout: int,
    judge: JudgeClient,
    config_path: Path,
    single_agent: str,
    chemqa_root: Path,
    chemqa_model_profile: str,
    review_rounds: int | None,
    rebuttal_rounds: int | None,
) -> list[GroupRecordResult]:
    def ensure_compatible_runner_result(run_result: Any) -> None:
        missing: list[str] = []
        should_score = getattr(run_result, "should_score", None)
        if not callable(should_score):
            missing.append("callable should_score()")
        answer = getattr(run_result, "answer", None)
        if answer is None:
            missing.append("answer")
        else:
            if not hasattr(answer, "short_answer_text"):
                missing.append("answer.short_answer_text")
            if not hasattr(answer, "full_response_text"):
                missing.append("answer.full_response_text")
        if not isinstance(getattr(run_result, "runner_meta", None), dict):
            missing.append("runner_meta: dict")
        if not isinstance(getattr(run_result, "raw", None), dict):
            missing.append("raw: dict")
        if not hasattr(run_result, "status"):
            missing.append("status")
        failure = getattr(run_result, "failure", None)
        if failure is not None and not hasattr(failure, "message"):
            missing.append("failure.message")
        if missing:
            raise BenchmarkError(
                f"Runner `{group.runner}` returned incompatible result object `{type(run_result).__name__}`; "
                f"missing/invalid fields: {', '.join(missing)}"
            )

    def status_label(run_result: Any) -> str:
        status = getattr(run_result, "status", None)
        if status is None:
            return "unknown"
        return str(getattr(status, "value", status))

    runtime_bundle_root = output_root / "input-bundles"
    try:
        if group.runner == "chemqa":
            runner = build_runner(
                runner_kind=group.runner,
                chemqa_root=chemqa_root,
                timeout_seconds=chemqa_timeout,
                config_path=config_path,
                slot_set=CHEMQA_SLOT_SETS[group.id],
                review_rounds=review_rounds,
                rebuttal_rounds=rebuttal_rounds,
                model_profile=chemqa_model_profile,
                runtime_bundle_root=runtime_bundle_root,
                launch_workspace_root=output_root / "chemqa-launch",
            )
        else:
            runner = build_runner(
                runner_kind=group.runner,
                agent_id=single_agent,
                timeout_seconds=single_timeout,
                config_path=config_path,
                runtime_bundle_root=runtime_bundle_root,
            )
    except Exception as exc:
        error_message = f"Failed to initialize runner for group `{group.id}`: {exc}"
        group_results = [
            build_error_group_record_result(group=group, record=record, error_message=error_message)
            for record in records
        ]
        for entry in group_results:
            save_json(output_root / "per-record" / group.id / f"{slugify(entry.record_id)}.json", asdict(entry))
        return group_results

    group_results: list[GroupRecordResult] = []
    for record in records:
        started = time.time()
        try:
            run_result = runner.run(record, group)
            ensure_compatible_runner_result(run_result)
            if run_result.should_score():
                evaluation = evaluate_answer(
                    record,
                    short_answer_text=run_result.answer.short_answer_text,
                    full_response_text=run_result.answer.full_response_text,
                    judge=judge,
                )
                answer_text = run_result.answer.full_response_text or run_result.answer.short_answer_text
                entry = GroupRecordResult(
                    group_id=group.id,
                    group_label=group.label,
                    runner=group.runner,
                    websearch=group.websearch,
                    record_id=record.record_id,
                    subset=classify_subset(record),
                    dataset=record.dataset,
                    source_file=record.source_file,
                    eval_kind=record.eval_kind,
                    prompt=record.prompt,
                    reference_answer=record.reference_answer,
                    answer_text=answer_text,
                    evaluation=asdict(evaluation),
                    runner_meta=run_result.runner_meta,
                    raw=run_result.raw,
                    elapsed_seconds=time.time() - started,
                    error=None,
                    short_answer_text=run_result.answer.short_answer_text,
                    full_response_text=run_result.answer.full_response_text,
                )
            else:
                runner_meta_error = str((run_result.runner_meta or {}).get("error") or "").strip()
                failure = getattr(run_result, "failure", None)
                failure_message = str(getattr(failure, "message", "") or "").strip()
                error_message = (
                    runner_meta_error
                    or failure_message
                    or f"Record `{record.record_id}` finished in non-success terminal status `{status_label(run_result)}`"
                )
                entry = build_error_group_record_result(
                    group=group,
                    record=record,
                    error_message=error_message,
                    elapsed_seconds=time.time() - started,
                    short_answer_text=run_result.answer.short_answer_text,
                    full_response_text=run_result.answer.full_response_text,
                    runner_meta=run_result.runner_meta,
                    raw=run_result.raw,
                )
        except Exception as exc:
            elapsed = time.time() - started
            error_message = f"Record `{record.record_id}` failed in group `{group.id}`: {exc}"
            entry = build_error_group_record_result(
                group=group,
                record=record,
                error_message=error_message,
                elapsed_seconds=elapsed,
                runner_meta={
                    "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                },
            )
        group_results.append(entry)
        save_json(output_root / "per-record" / group.id / f"{slugify(record.record_id)}.json", asdict(entry))
    return group_results



def main() -> int:
    args = parse_args()
    group_ids = select_group_ids(args.groups)
    dataset_files = select_dataset_files(args)
    if args.list_datasets:
        print_dataset_listing(dataset_files)
        return 0
    if not dataset_files:
        raise BenchmarkError("No benchmark files discovered.")

    all_records = load_records(dataset_files)
    if args.random_count_per_subset is not None:
        selected_pool = sample_records_per_subset(
            all_records,
            per_subset_count=args.random_count_per_subset,
            seed=args.random_seed,
        )
    else:
        selected_pool = all_records

    records = apply_offset_limit(selected_pool, offset=args.offset, limit=args.limit)
    if not records:
        raise BenchmarkError("No benchmark records selected.")
    if args.print_selected_records:
        print_selected_records(records)
        return 0

    if args.exact_output_dir:
        output_root = Path(args.exact_output_dir).expanduser().resolve()
    else:
        output_root = Path(args.output_dir).expanduser().resolve() / f"benchmark-{now_stamp()}"
    ensure_dir(output_root)

    config_pool = ConfigPool(
        base_config_path=Path(args.openclaw_config).expanduser().resolve(),
        output_root=output_root,
        single_agent_model=args.single_agent_model,
        judge_model=args.judge_model,
        single_agent_id_override=args.single_agent_id_override,
    )
    judge = JudgeClient(
        judge_agent=args.judge_agent,
        timeout_seconds=args.judge_timeout,
        config_path=config_pool.judge_config_path(),
    )
    group_waves = build_group_waves(group_ids, max_concurrent_groups=args.max_concurrent_groups)

    group_results: dict[str, list[GroupRecordResult]] = {}
    try:
        for wave_index, wave_group_ids in enumerate(group_waves, start=1):
            started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            write_wave_status(
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
                    group = EXPERIMENT_GROUPS[group_id]
                    config_path = config_pool.config_for_group(group)
                    spec = EXPERIMENT_SPECS.get(group_id)
                    single_agent = (
                        spec.resolve_single_agent_id(args.single_agent_id_override)
                        if spec is not None
                        else DEFAULT_SINGLE_AGENT
                    )
                    future = executor.submit(
                        run_group,
                        group=group,
                        records=records,
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
                    )
                    future_map[future] = group_id

                for future in as_completed(future_map):
                    group_id = future_map[future]
                    try:
                        group_results[group_id] = future.result()
                    except Exception as exc:
                        group = EXPERIMENT_GROUPS[group_id]
                        error_message = f"Group `{group_id}` failed before returning results: {exc}"
                        group_results[group_id] = materialize_group_failure_results(
                            group=group,
                            records=records,
                            output_root=output_root,
                            error_message=error_message,
                        )
            gc.collect()
            completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            write_wave_status(
                output_root,
                wave_index=wave_index,
                wave_group_ids=wave_group_ids,
                status="completed",
                started_at=started_at,
                completed_at=completed_at,
                per_record_counts=count_per_record_outputs(output_root, group_ids=wave_group_ids),
                inter_wave_delay_seconds=args.inter_wave_delay_seconds,
            )
            if wave_index < len(group_waves) and args.inter_wave_delay_seconds > 0:
                time.sleep(args.inter_wave_delay_seconds)
    finally:
        run_pending_cleanroom_cleanup()

    aggregate_group_ids = resolve_aggregate_group_ids(
        group_ids,
        output_root=output_root,
        merge_existing_per_record=args.merge_existing_per_record,
    )
    if args.merge_existing_per_record:
        results = load_results_from_output_root(output_root, group_ids=aggregate_group_ids)
    else:
        results: list[GroupRecordResult] = []
        for group_id in group_ids:
            results.extend(group_results.get(group_id, []))

    summary = aggregate_results(results)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "benchmark_root": str(Path(args.benchmark_root).expanduser().resolve()),
        "dataset_files": [str(path) for path in dataset_files],
        "groups": [asdict(EXPERIMENT_GROUPS[group_id]) for group_id in aggregate_group_ids],
        "run_groups": [asdict(EXPERIMENT_GROUPS[group_id]) for group_id in group_ids],
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
    save_json(output_root / "results.json", payload)
    export_csv_reports(output_root, summary, aggregate_group_ids)
    save_json(
        output_root / "runtime-manifest.json",
        {
            "execution_plan": {
                "mode": "wave-batched",
                "max_concurrent_groups": args.max_concurrent_groups,
                "inter_wave_delay_seconds": args.inter_wave_delay_seconds,
                "waves": group_waves,
            },
            "aggregate_groups": aggregate_group_ids,
            "run_groups": group_ids,
            "merge_existing_per_record": args.merge_existing_per_record,
            "groups": {
                group_id: {
                    "group": asdict(EXPERIMENT_GROUPS[group_id]),
                    "config_path": str(config_pool.config_for_group(EXPERIMENT_GROUPS[group_id])),
                    "slot_set": CHEMQA_SLOT_SETS.get(group_id),
                    "single_agent": (
                        EXPERIMENT_SPECS[group_id].resolve_single_agent_id(args.single_agent_id_override)
                        if group_id in EXPERIMENT_SPECS and EXPERIMENT_GROUPS[group_id].runner == "single_llm"
                        else None
                    ),
                    "single_agent_model": args.single_agent_model,
                    "chemqa_model_profile": args.chemqa_model_profile if EXPERIMENT_GROUPS[group_id].runner == "chemqa" else None,
                }
                for group_id in group_ids
            },
            "judge": {
                "agent": args.judge_agent,
                "model": args.judge_model,
                "config_path": str(config_pool.judge_config_path()),
            },
        },
    )
    print(json.dumps({"output_dir": str(output_root), "summary": summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
