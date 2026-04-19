#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


WORKSPACE_ROOT = Path("/home/dministrator/.openclaw/workspace")
DEFAULT_BENCHMARK_ROOT = WORKSPACE_ROOT / "temp-benchmarks" / "representative15"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "state" / "benchmark-rl-runs"
DEFAULT_OPENCLAW_CONFIG = Path.home() / ".openclaw" / "openclaw.json"
DEFAULT_DEBATECLAW_ROOT = Path("/home/dministrator/.openclaw/skills/debateclaw-v1")
DEFAULT_COLLECTOR_AGENT = "benchmark-rl-collector"
DEFAULT_COLLECTOR_MODEL = "packy/gpt-5.4"
DEFAULT_JUDGE_AGENT = "benchmark-judge"
DEFAULT_JUDGE_MODEL = "su8/gpt-5.4"
BENCHMARK_AGENT_THINKING = "high"


def load_benchmark_test_module() -> Any:
    module_path = Path(__file__).resolve().parent / "benchmark_test.py"
    spec = importlib.util.spec_from_file_location("benchmark_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark_test.py from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


benchmark_test = load_benchmark_test_module()


class BenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExperimentGroup:
    id: str
    label: str
    runner: str
    websearch: bool


@dataclass
class BenchmarkRecord:
    record_id: str
    dataset: str
    source_file: str
    eval_kind: str
    prompt: str
    reference_answer: str
    payload: dict[str, Any]


@dataclass
class RunOutput:
    short_answer_text: str
    full_response_text: str
    raw: dict[str, Any]
    runner_meta: dict[str, Any]


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


EXPERIMENT_GROUPS: dict[str, ExperimentGroup] = {
    "review_loop_web_on": ExperimentGroup(
        id="review_loop_web_on",
        label="DebateClaw review-loop@1 + 启用 websearch",
        runner="review_loop",
        websearch=True,
    ),
    "review_loop_web_off": ExperimentGroup(
        id="review_loop_web_off",
        label="DebateClaw review-loop@1 + 禁用 websearch",
        runner="review_loop",
        websearch=False,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DebateClaw review-loop batch benchmarks.")
    parser.add_argument("--benchmark-root", default=str(DEFAULT_BENCHMARK_ROOT), help="benchmark 根目录")
    parser.add_argument("--openclaw-config", default=str(DEFAULT_OPENCLAW_CONFIG), help="基础 OpenClaw 配置文件")
    parser.add_argument("--debateclaw-root", default=str(DEFAULT_DEBATECLAW_ROOT), help="debateclaw-v1 根目录")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="结果输出目录")
    parser.add_argument("--exact-output-dir", help="若提供，则直接把该目录作为本次输出根目录")
    parser.add_argument("--datasets", help="仅运行指定数据集，逗号分隔")
    parser.add_argument("--files", help="仅运行指定 jsonl 文件，逗号分隔，优先级高于 --datasets")
    parser.add_argument("--limit", type=int, help="最多运行多少条题目")
    parser.add_argument("--offset", type=int, default=0, help="跳过前多少条题目")
    parser.add_argument("--websearch", choices=("on", "off"), required=True, help="本次运行是否启用 websearch")
    parser.add_argument("--review-rounds", type=int, help="review rounds 覆盖值")
    parser.add_argument("--rebuttal-rounds", type=int, help="rebuttal rounds 覆盖值")
    parser.add_argument("--proposer-count", type=int, help="proposer 数量覆盖值")
    parser.add_argument("--model-profile", help="review-loop model profile 覆盖值")
    parser.add_argument("--rl-timeout", type=int, default=3600, help="每题 review-loop 超时秒数")
    parser.add_argument("--collector-timeout", type=int, default=300, help="outer collector 超时秒数")
    parser.add_argument("--judge-timeout", type=int, default=300, help="benchmark judge 超时秒数")
    parser.add_argument("--collector-agent", default=DEFAULT_COLLECTOR_AGENT, help="outer collector agent id")
    parser.add_argument("--collector-model", default=DEFAULT_COLLECTOR_MODEL, help="outer collector runtime model")
    parser.add_argument("--judge-agent", default=DEFAULT_JUDGE_AGENT, help="benchmark judge agent id")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="benchmark judge runtime model")
    parser.add_argument("--list-datasets", action="store_true", help="列出可发现的数据集文件后退出")
    parser.add_argument("--print-selected-records", action="store_true", help="打印本次实际选中的题目清单后退出")
    return parser.parse_args()


def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def slugify(value: str, *, limit: int = 64) -> str:
    return benchmark_test.slugify(value, limit=limit)


def normalize_space(text: str) -> str:
    return benchmark_test.normalize_space(text)


def deep_copy_jsonish(value: Any) -> Any:
    return benchmark_test.deep_copy_jsonish(value)


def unwrap_agent_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return benchmark_test.unwrap_agent_payload(payload)


def run_subprocess(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    timeout: int | None = None,
) -> Any:
    return benchmark_test.run_subprocess(command, env=env, cwd=cwd, timeout=timeout)


def parse_json_stdout(result: Any, command: list[str]) -> Any:
    return benchmark_test.parse_json_stdout(result, command)


def detect_clawteam_team_flag(*, cwd: Path) -> str:
    result = run_subprocess(["clawteam", "launch", "--help"], cwd=cwd, timeout=30)
    help_text = (result.stdout or "") + "\n" + (result.stderr or "")
    if "--team-name" in help_text:
        return "--team-name"
    return "--team"


def save_json(path: Path, payload: Any) -> None:
    benchmark_test.save_json(path, payload)


def ensure_dir(path: Path) -> None:
    benchmark_test.ensure_dir(path)


def dataset_name_from_file(path: Path) -> str:
    return path.parent.parent.name


def discover_dataset_files(root: Path) -> list[Path]:
    return sorted(path.resolve() for path in root.glob("*/data/*.jsonl") if path.is_file())


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


def extract_prompt(payload: dict[str, Any]) -> str:
    for key in ("prompt", "problem", "input", "question"):
        text = str(payload.get(key) or "").strip()
        if text:
            return text
    return ""


def extract_reference_answer(payload: dict[str, Any]) -> str:
    for key in ("answer", "target"):
        text = str(payload.get(key) or "").strip()
        if text:
            return text
    return ""


def load_records(paths: Iterable[Path]) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for path in paths:
        dataset = dataset_name_from_file(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                record_id = str(payload.get("id") or f"{dataset}-{len(records)}")
                prompt = extract_prompt(payload)
                if not prompt:
                    raise BenchmarkError(f"Missing prompt/problem/input/question field in record: {record_id}")
                reference_answer = extract_reference_answer(payload)
                if not reference_answer:
                    raise BenchmarkError(f"Missing answer/target field in record: {record_id}")
                eval_kind = str(payload.get("eval_kind") or "generic_semantic").strip() or "generic_semantic"
                records.append(
                    BenchmarkRecord(
                        record_id=record_id,
                        dataset=dataset,
                        source_file=str(path),
                        eval_kind=eval_kind,
                        prompt=prompt,
                        reference_answer=reference_answer,
                        payload=payload,
                    )
                )
    return records


def apply_offset_limit(records: list[BenchmarkRecord], *, offset: int = 0, limit: int | None = None) -> list[BenchmarkRecord]:
    return benchmark_test.apply_offset_limit(records, offset=offset, limit=limit)


def classify_subset(record: BenchmarkRecord) -> str:
    return benchmark_test.classify_subset(record)  # type: ignore[arg-type]


def print_dataset_listing(paths: list[Path]) -> None:
    payload = [{"dataset": dataset_name_from_file(path), "path": str(path)} for path in paths]
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


def benchmark_group_id_for_websearch(value: str) -> str:
    return "review_loop_web_on" if value == "on" else "review_loop_web_off"


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
        ["group_id", "runner", "websearch", "count", "pass_count", "avg_normalized_score", "avg_answer_accuracy", "avg_rpf"],
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
        ["group_id", "runner", "websearch", "subset", "count", "pass_count", "avg_normalized_score", "avg_answer_accuracy", "avg_rpf"],
    )


class ReviewLoopConfigPool:
    def __init__(
        self,
        *,
        base_config_path: Path,
        output_root: Path,
        debateclaw_root: Path,
        collector_agent: str,
        collector_model: str,
        judge_agent: str,
        judge_model: str,
    ) -> None:
        self.base_config_path = base_config_path
        self.output_root = output_root
        self.debateclaw_root = debateclaw_root
        self.collector_agent = collector_agent
        self.collector_model = collector_model
        self.judge_agent = judge_agent
        self.judge_model = judge_model
        self._payload = json.loads(base_config_path.read_text(encoding="utf-8"))
        self._config_dir = output_root / "runtime-config"
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._group_paths: dict[str, Path] = {}
        self._collector_path: Path | None = None
        self._judge_path: Path | None = None
        self._slot_workspace_root = output_root / "review-loop-workspaces"

    def _discover_agent_model(self, agent_id: str) -> str | None:
        agents = ((self._payload.get("agents") or {}).get("list") or [])
        for entry in agents:
            if isinstance(entry, dict) and str(entry.get("id", "")) == agent_id:
                model = str(entry.get("model") or "").strip()
                if model:
                    return model
        return None

    def _ensure_basic_agent_dirs(self, *paths: Path) -> None:
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)

    def _ensure_agent_dir_with_models(self, target: Path, *model_sources: Path) -> Path:
        target.mkdir(parents=True, exist_ok=True)
        target_models = target / "models.json"
        if target_models.is_file():
            return target
        for source in model_sources:
            source_models = source / "models.json"
            if source.is_dir() and source_models.is_file():
                shutil.copy2(source_models, target_models)
                return target
        return target

    def _upsert_agent_entry(
        self,
        payload: dict[str, Any],
        *,
        agent_id: str,
        workspace: Path,
        agent_dir: Path,
        model: str,
    ) -> None:
        agents = payload.setdefault("agents", {})
        entries = agents.setdefault("list", [])
        if not isinstance(entries, list):
            raise BenchmarkError("OpenClaw config agents.list is not a list")
        normalized_workspace = str(workspace.resolve())
        normalized_agent_dir = str(agent_dir.resolve())
        for entry in entries:
            if isinstance(entry, dict) and str(entry.get("id", "")) == agent_id:
                entry["name"] = agent_id
                entry["workspace"] = normalized_workspace
                entry["agentDir"] = normalized_agent_dir
                entry["model"] = model
                entry.pop("thinking", None)
                return
        entries.append(
            {
                "id": agent_id,
                "name": agent_id,
                "workspace": normalized_workspace,
                "agentDir": normalized_agent_dir,
                "model": model,
            }
        )

    def _model_ref_to_runtime_model(self, model_ref: str) -> str:
        model_def_path = self.debateclaw_root / "control" / "models" / f"{model_ref}.json"
        model_def = json.loads(model_def_path.read_text(encoding="utf-8"))
        provider_ref = str(model_def.get("provider_ref") or "").strip()
        remote_model_id = str(model_def.get("remote_model_id") or "").strip()
        if not provider_ref or not remote_model_id:
            raise BenchmarkError(f"Invalid model definition for `{model_ref}`")
        return f"{provider_ref}/{remote_model_id}"

    def _build_group_payload(self, group: ExperimentGroup, *, model_profile: str) -> dict[str, Any]:
        payload = benchmark_test.build_temp_openclaw_config_payload(self._payload, enable_websearch=group.websearch)
        profile_path = self.debateclaw_root / "control" / "model-profiles" / f"{model_profile}.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        slot_models = dict(profile.get("slot_models") or {})
        if "debate-coordinator" not in slot_models:
            raise BenchmarkError(f"Model profile `{model_profile}` is missing debate-coordinator")

        for slot_id, slot_payload in slot_models.items():
            workspace = self._slot_workspace_root / group.id / slot_id
            agent_dir = Path.home() / ".openclaw" / "agents" / slot_id / "agent"
            if slot_id.startswith("debate-"):
                if slot_id == "debate-coordinator":
                    benchmark_test.ensure_slot_workspace(workspace, slot_id=slot_id, workspace_root=workspace.parent)
                else:
                    benchmark_test.ensure_slot_workspace(workspace, slot_id=slot_id, workspace_root=workspace.parent)
            self._ensure_basic_agent_dirs(agent_dir)
            runtime_model = self._model_ref_to_runtime_model(str(slot_payload.get("model_ref") or ""))
            self._upsert_agent_entry(
                payload,
                agent_id=slot_id,
                workspace=workspace,
                agent_dir=agent_dir,
                model=runtime_model,
            )

        collector_workspace = self.output_root / "collector-workspace"
        collector_agent_dir = self._ensure_agent_dir_with_models(
            Path.home() / ".openclaw" / "agents" / self.collector_agent / "agent",
            Path.home() / ".openclaw" / "agents" / self.judge_agent / "agent",
            Path.home() / ".openclaw" / "agents" / "main" / "agent",
        )
        self._ensure_basic_agent_dirs(collector_workspace, collector_agent_dir)
        self._upsert_agent_entry(
            payload,
            agent_id=self.collector_agent,
            workspace=collector_workspace,
            agent_dir=collector_agent_dir,
            model=self.collector_model,
        )

        judge_workspace = self.output_root / "judge-workspace"
        judge_agent_dir = self._ensure_agent_dir_with_models(
            Path.home() / ".openclaw" / "agents" / self.judge_agent / "agent",
            Path.home() / ".openclaw" / "agents" / "main" / "agent",
        )
        self._ensure_basic_agent_dirs(judge_workspace, judge_agent_dir)
        self._upsert_agent_entry(
            payload,
            agent_id=self.judge_agent,
            workspace=judge_workspace,
            agent_dir=judge_agent_dir,
            model=self.judge_model,
        )
        return payload

    def config_for_group(self, group: ExperimentGroup, *, model_profile: str) -> Path:
        existing = self._group_paths.get(group.id)
        if existing is not None:
            return existing
        payload = self._build_group_payload(group, model_profile=model_profile)
        path = self._config_dir / f"{group.id}-openclaw.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._group_paths[group.id] = path
        return path

    def judge_config_path(self) -> Path:
        if self._judge_path is not None:
            return self._judge_path
        payload = benchmark_test.build_temp_openclaw_config_payload(self._payload, enable_websearch=False)
        judge_workspace = self.output_root / "judge-workspace"
        judge_agent_dir = self._ensure_agent_dir_with_models(
            Path.home() / ".openclaw" / "agents" / self.judge_agent / "agent",
            Path.home() / ".openclaw" / "agents" / "main" / "agent",
        )
        self._ensure_basic_agent_dirs(judge_workspace, judge_agent_dir)
        self._upsert_agent_entry(
            payload,
            agent_id=self.judge_agent,
            workspace=judge_workspace,
            agent_dir=judge_agent_dir,
            model=self.judge_model,
        )
        path = self._config_dir / "benchmark-judge-openclaw.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._judge_path = path
        return path


class JudgeClient:
    def __init__(self, *, judge_agent: str, timeout_seconds: int, config_path: Path) -> None:
        self.judge_agent = judge_agent
        self.timeout_seconds = timeout_seconds
        self.config_path = config_path

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
        result = run_subprocess(command, env=env, timeout=self.timeout_seconds + 30)
        payload = parse_json_stdout(result, command)
        result_payload = unwrap_agent_payload(payload)
        reply = benchmark_test.summarize_payloads(list((result_payload.get("payloads") or [])))
        parsed = benchmark_test.safe_json_extract(reply)
        if not isinstance(parsed, dict):
            raise BenchmarkError(f"Judge must return a JSON object, got: {reply}")
        return parsed


class OuterCollectorClient:
    def __init__(
        self,
        *,
        collector_agent: str,
        timeout_seconds: int,
        config_path: Path,
    ) -> None:
        self.collector_agent = collector_agent
        self.timeout_seconds = timeout_seconds
        self.config_path = config_path

    def _prompt(self, *, record: BenchmarkRecord, summary_payload: dict[str, Any], candidate_payloads: list[dict[str, Any]], coordinator_summary_text: str) -> str:
        schema = {
            "final_answer": "string",
            "short_answer": "string",
            "full_response_text": "string",
            "selected_candidate": "string",
            "decision_mode": "single_candidate_direct|multi_candidate_select|multi_candidate_synthesize",
            "decision_rationale": "string",
            "candidate_summaries": [{"candidate": "string", "summary": "string"}],
            "source_summary_path": "string",
        }
        return (
            "You are the outer collector/final decider for a DebateClaw review-loop benchmark run.\n"
            "Your job is only to read the completed debate trace and produce a single benchmark-ready final answer.\n"
            "You are not grading benchmark correctness.\n"
            "Return strict JSON only.\n\n"
            f"Required JSON schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
            f"BENCHMARK RECORD:\n{json.dumps({'record_id': record.record_id, 'eval_kind': record.eval_kind, 'prompt': record.prompt}, ensure_ascii=False, indent=2)}\n\n"
            f"DEBATE SUMMARY:\n{json.dumps(summary_payload, ensure_ascii=False, indent=2)}\n\n"
            f"CANDIDATE PAYLOADS:\n{json.dumps(candidate_payloads, ensure_ascii=False, indent=2)}\n\n"
            f"COORDINATOR SUMMARY:\n{coordinator_summary_text or '[missing]'}\n\n"
            "Rules:\n"
            "- If there is exactly one surviving candidate, prefer its latest defended answer.\n"
            "- If there are multiple surviving candidates, select or minimally synthesize among them.\n"
            "- `full_response_text` must include exactly one `FINAL ANSWER: ...` line.\n"
            "- `short_answer` should be the benchmark-ready short answer.\n"
            "- Do not mention grading, scoring, rubric, or benchmark correctness.\n"
        )

    def decide(
        self,
        *,
        record: BenchmarkRecord,
        summary_payload: dict[str, Any],
        candidate_payloads: list[dict[str, Any]],
        coordinator_summary_text: str,
    ) -> dict[str, Any]:
        prompt = self._prompt(
            record=record,
            summary_payload=summary_payload,
            candidate_payloads=candidate_payloads,
            coordinator_summary_text=coordinator_summary_text,
        )
        session_id = f"benchmark-rl-collector-{uuid.uuid4().hex[:12]}"
        command = [
            "openclaw",
            "agent",
            "--local",
            "--agent",
            self.collector_agent,
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
        result = run_subprocess(command, env=env, timeout=self.timeout_seconds + 30)
        payload = parse_json_stdout(result, command)
        result_payload = unwrap_agent_payload(payload)
        reply = benchmark_test.summarize_payloads(list((result_payload.get("payloads") or [])))
        parsed = benchmark_test.safe_json_extract(reply)
        if not isinstance(parsed, dict):
            raise BenchmarkError(f"Collector must return a JSON object, got: {reply}")
        parsed.setdefault("collector_prompt", prompt)
        parsed.setdefault("collector_reply", reply)
        return parsed


def extract_final_answer_line(text: str) -> str:
    return benchmark_test.extract_final_answer_line(text)


def normalize_answer_tracks(*, short_answer_text: str = "", full_response_text: str = "") -> tuple[str, str]:
    return benchmark_test.normalize_answer_tracks(short_answer_text=short_answer_text, full_response_text=full_response_text)


def build_execution_error_evaluation(record: BenchmarkRecord, *, error_message: str) -> Any:
    return benchmark_test.build_execution_error_evaluation(record, error_message=error_message)  # type: ignore[arg-type]


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


def aggregate_results(results: list[GroupRecordResult]) -> dict[str, Any]:
    return benchmark_test.aggregate_results(results)  # type: ignore[arg-type]


def evaluate_answer(
    record: BenchmarkRecord,
    *,
    short_answer_text: str,
    full_response_text: str,
    judge: JudgeClient,
) -> Any:
    return benchmark_test.evaluate_answer(
        record,
        short_answer_text=short_answer_text,
        full_response_text=full_response_text,
        judge=judge,
    )


def ensure_runtime_bundle(record: BenchmarkRecord, *, bundle_root: Path) -> Any:
    return benchmark_test.ensure_runtime_bundle(record, bundle_root=bundle_root)  # type: ignore[arg-type]


def build_review_loop_goal(
    record: BenchmarkRecord,
    *,
    websearch_enabled: bool,
    input_bundle: Any | None = None,
) -> str:
    instructions = [
        "Solve the following chemistry benchmark question.",
        "The final answer must be faithful to the benchmark prompt.",
        "The debate team should produce evidence-first candidate proposals and stress-test them through review and rebuttal.",
        "The outer collector will read the completed debate trace and extract a single final answer.",
        "Every serious candidate should preserve a clear final answer line in the form `FINAL ANSWER: <answer>`.",
    ]
    if websearch_enabled:
        instructions.append("Web search may be used if genuinely helpful.")
    else:
        instructions.append("Do not use web search or external browsing.")

    if record.eval_kind == "superchem_multiple_choice_rpf":
        instructions.append("This is a chemistry multiple-choice question.")
        instructions.append("Candidate answers should end with `FINAL ANSWER: <option letters>`.")
        instructions.append("If multiple options are correct, separate letters with `|`.")
        if input_bundle is not None:
            instructions.append(f"Use the local file bundle at `{input_bundle.bundle_dir}`.")
            instructions.append(f"Read `{input_bundle.question_markdown}` first and inspect all referenced local images before deciding.")
    elif record.eval_kind in {"chembench_open_ended", "frontierscience_olympiad"}:
        instructions.append("Candidate answers should end with `FINAL ANSWER: <answer>`.")
    else:
        instructions.append("The accepted final answer should be extractable as a single `FINAL ANSWER: <answer>` line.")

    return "\n".join(instructions) + "\n\nQUESTION:\n" + record.prompt.strip()


def hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def coordinator_summary_path_for_run(run_id: str, debateclaw_root: Path) -> Path | None:
    candidate_roots = [
        debateclaw_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id / "coordinator-summary.md",
        debateclaw_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id / "debate-coordinator" / "coordinator-summary.md",
    ]
    for path in candidate_roots:
        if path.is_file():
            return path
    return None


def proposal_payloads_for_candidates(summary_payload: dict[str, Any]) -> list[dict[str, Any]]:
    proposals = list(summary_payload.get("proposals") or [])
    final_candidates = set(str(item) for item in (summary_payload.get("final_candidates") or []))
    reviews = list(summary_payload.get("reviews") or [])
    rebuttals = list(summary_payload.get("rebuttals") or [])
    payloads: list[dict[str, Any]] = []
    for proposal in proposals:
        proposer = str(proposal.get("proposer") or "")
        if proposer not in final_candidates:
            continue
        payloads.append(
            {
                "candidate": proposer,
                "proposal": proposal,
                "review_history": [item for item in reviews if str(item.get("target_proposer") or "") == proposer],
                "rebuttal_history": [item for item in rebuttals if str(item.get("proposer") or "") == proposer],
                "attack_registry": [item for item in (summary_payload.get("attack_registry") or []) if str(item.get("target_proposer") or "") == proposer],
            }
        )
    return payloads


def extract_answer_from_candidate_payload(candidate_payload: dict[str, Any]) -> tuple[str, str]:
    latest_rebuttal_body = ""
    rebuttal_history = list(candidate_payload.get("rebuttal_history") or [])
    if rebuttal_history:
        latest_rebuttal_body = str(rebuttal_history[-1].get("body") or "")
        answer = extract_final_answer_line(latest_rebuttal_body)
        if answer:
            return normalize_answer_tracks(short_answer_text=answer, full_response_text=latest_rebuttal_body)
    proposal_body = str((candidate_payload.get("proposal") or {}).get("body") or "")
    answer = extract_final_answer_line(proposal_body)
    if answer:
        return normalize_answer_tracks(short_answer_text=answer, full_response_text=proposal_body)
    if latest_rebuttal_body:
        return normalize_answer_tracks(full_response_text=latest_rebuttal_body)
    return normalize_answer_tracks(full_response_text=proposal_body)


def fallback_collect_answer(summary_payload: dict[str, Any]) -> dict[str, Any] | None:
    candidate_payloads = proposal_payloads_for_candidates(summary_payload)
    if not candidate_payloads:
        return None
    first = candidate_payloads[0]
    short_answer, full_response = extract_answer_from_candidate_payload(first)
    if not short_answer and not full_response:
        return None
    return {
        "final_answer": short_answer,
        "short_answer": short_answer,
        "full_response_text": full_response or (f"FINAL ANSWER: {short_answer}" if short_answer else ""),
        "selected_candidate": first.get("candidate", ""),
        "decision_mode": "single_candidate_direct" if len(candidate_payloads) == 1 else "fallback_first_candidate",
        "decision_rationale": "Fallback extraction from surviving candidate proposal/rebuttal body.",
        "candidate_summaries": [{"candidate": item.get("candidate", ""), "summary": ""} for item in candidate_payloads],
        "source_summary_path": "",
    }


class ReviewLoopRunner:
    def __init__(
        self,
        *,
        debateclaw_root: Path,
        timeout_seconds: int,
        config_path: Path,
        collector: OuterCollectorClient,
        runtime_bundle_root: Path,
        template_output_dir: Path,
        launch_home_dir: Path,
        clawteam_data_dir: Path,
        review_rounds: int | None,
        rebuttal_rounds: int | None,
        proposer_count: int | None,
        model_profile: str | None,
    ) -> None:
        self.debateclaw_root = debateclaw_root
        self.timeout_seconds = timeout_seconds
        self.config_path = config_path
        self.collector = collector
        self.runtime_bundle_root = runtime_bundle_root
        self.template_output_dir = template_output_dir
        self.launch_home_dir = launch_home_dir
        self.clawteam_data_dir = clawteam_data_dir
        self.review_rounds = review_rounds
        self.rebuttal_rounds = rebuttal_rounds
        self.proposer_count = proposer_count
        self.model_profile = model_profile
        self.compile_script = debateclaw_root / "scripts" / "compile_runplan.py"
        self.materialize_script = debateclaw_root / "scripts" / "materialize_runplan.py"
        self.debate_state_script = debateclaw_root / "scripts" / "debate_state.py"
        self.runtime_helper_dir = Path.home() / ".clawteam" / "debateclaw" / "bin"
        self.real_openclaw_env_file = Path.home() / ".openclaw" / ".env"
        self.launch_openclaw_dir = self.launch_home_dir / ".openclaw"
        self.launch_openclaw_config_path = self.launch_openclaw_dir / "openclaw.json"
        self.template_output_dir.mkdir(parents=True, exist_ok=True)
        self.launch_home_dir.mkdir(parents=True, exist_ok=True)
        self.launch_openclaw_dir.mkdir(parents=True, exist_ok=True)
        self.clawteam_data_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_launch_home(self) -> None:
        self.launch_openclaw_dir.mkdir(parents=True, exist_ok=True)
        self.launch_openclaw_config_path.write_text(self.config_path.read_text(encoding="utf-8"), encoding="utf-8")

    def _summary_command(self, run_id: str) -> list[str]:
        return [
            "python3",
            str(self.debate_state_script),
            "summary",
            "--team",
            run_id,
            "--json",
            "--include-bodies",
        ]

    def _status_summary(self, run_id: str) -> dict[str, Any]:
        command = self._summary_command(run_id)
        env = os.environ.copy()
        env["OPENCLAW_CONFIG_PATH"] = str(self.config_path)
        env["CLAWTEAM_DATA_DIR"] = str(self.clawteam_data_dir)
        result = run_subprocess(command, env=env, cwd=self.debateclaw_root, timeout=120)
        return parse_json_stdout(result, command)

    def _wait_for_done(self, run_id: str) -> dict[str, Any]:
        deadline = time.time() + self.timeout_seconds
        last_payload: dict[str, Any] = {}
        while time.time() < deadline:
            last_payload = self._status_summary(run_id)
            if str(last_payload.get("status") or "") == "done" and str(last_payload.get("terminal_state") or "") in {"completed", "failed"}:
                return last_payload
            time.sleep(5)
        raise BenchmarkError(
            f"Review-loop run `{run_id}` did not reach terminal state within {self.timeout_seconds}s. Last summary: {last_payload}"
        )

    def _launch(self, *, goal: str, run_id: str, additional_file_workspace: str | None) -> dict[str, Any]:
        compile_command = [
            "python3",
            str(self.compile_script),
            "--root",
            str(self.debateclaw_root),
            "--preset",
            "review-loop@1",
            "--goal",
            goal,
            "--run-id",
            run_id,
        ]
        if additional_file_workspace:
            compile_command.extend(["--additional-file-workspace", additional_file_workspace])
        if self.model_profile:
            compile_command.extend(["--model-profile", self.model_profile])
        if self.proposer_count is not None:
            compile_command.extend(["--proposer-count", str(self.proposer_count)])
        if self.review_rounds is not None:
            compile_command.extend(["--review-rounds", str(self.review_rounds)])
        if self.rebuttal_rounds is not None:
            compile_command.extend(["--rebuttal-rounds", str(self.rebuttal_rounds)])
        self._prepare_launch_home()
        env = os.environ.copy()
        env["OPENCLAW_CONFIG_PATH"] = str(self.config_path)
        env["CLAWTEAM_DATA_DIR"] = str(self.clawteam_data_dir)
        env["HOME"] = str(self.launch_home_dir)
        if self.real_openclaw_env_file.is_file():
            env["OPENCLAW_ENV_FILE"] = str(self.real_openclaw_env_file)
        compile_result = run_subprocess(compile_command, env=env, cwd=self.debateclaw_root, timeout=120)
        compiled = parse_json_stdout(compile_result, compile_command)

        materialize_command = [
            "python3",
            str(self.materialize_script),
            "--root",
            str(self.debateclaw_root),
            "--run-id",
            run_id,
            "--template-dir",
            str(self.template_output_dir),
            "--runtime-dir",
            str(self.runtime_helper_dir),
        ]
        materialize_result = run_subprocess(materialize_command, env=env, cwd=self.debateclaw_root, timeout=180)
        materialized = parse_json_stdout(materialize_result, materialize_command)

        launch_command: list[str] | None = None
        launch_command_json = str(materialized.get("launch_command_json") or "").strip()
        if launch_command_json:
            parsed = json.loads(launch_command_json)
            if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                launch_command = list(parsed)
        if launch_command is None:
            template_name = str(materialized.get("template_name") or "").strip()
            if not template_name:
                template_name = str((materialized.get("template_name") or compiled.get("run_id") or "")).strip()
            if not template_name:
                raise BenchmarkError(f"Unable to determine clawteam template name for run `{run_id}`.")
            launch_command = [
                "clawteam",
                "launch",
                template_name,
                detect_clawteam_team_flag(cwd=self.debateclaw_root),
                run_id,
                "--goal",
                goal,
                "--backend",
                str(compiled.get("launch_spec", {}).get("backend") or "subprocess"),
            ]

        launch_result = run_subprocess(launch_command, env=env, cwd=self.debateclaw_root, timeout=180)
        if launch_result.returncode != 0:
            raise BenchmarkError(
                "Command failed\n"
                f"command: {' '.join(launch_command)}\n"
                f"returncode: {launch_result.returncode}\n"
                f"stdout:\n{launch_result.stdout}\n"
                f"stderr:\n{launch_result.stderr}"
            )
        return {
            "preset": "review-loop@1",
            "goal": goal,
            "run_id": run_id,
            "compile": compiled,
            "materialize": materialized,
            "launch_command": launch_command,
            "launched": {
                "command": launch_command,
                "returncode": launch_result.returncode,
                "stdout": launch_result.stdout,
                "stderr": launch_result.stderr,
            },
        }

    def run(self, record: BenchmarkRecord, group: ExperimentGroup) -> RunOutput:
        input_bundle = ensure_runtime_bundle(record, bundle_root=self.runtime_bundle_root)
        goal = build_review_loop_goal(record, websearch_enabled=group.websearch, input_bundle=input_bundle)
        run_id = f"benchmark-{group.id}-{slugify(record.record_id, limit=40)}-{hash_text(goal)}-{time.strftime('%Y%m%d-%H%M%S')}"
        launch_payload = self._launch(
            goal=goal,
            run_id=run_id,
            additional_file_workspace=str(input_bundle.bundle_dir) if input_bundle is not None else None,
        )
        summary_payload = self._wait_for_done(run_id)
        candidate_payloads = proposal_payloads_for_candidates(summary_payload)
        coordinator_summary_path = coordinator_summary_path_for_run(run_id, self.debateclaw_root)
        coordinator_summary_text = coordinator_summary_path.read_text(encoding="utf-8") if coordinator_summary_path else ""

        collector_meta: dict[str, Any] = {}
        if str(summary_payload.get("terminal_state") or "") == "failed" or not candidate_payloads:
            collector_result = None
        else:
            try:
                collector_result = self.collector.decide(
                    record=record,
                    summary_payload=summary_payload,
                    candidate_payloads=candidate_payloads,
                    coordinator_summary_text=coordinator_summary_text,
                )
                collector_meta["mode"] = "collector"
            except Exception as exc:
                collector_meta["mode"] = "collector_failed"
                collector_meta["error"] = str(exc)
                collector_result = None

        if collector_result is None:
            fallback = fallback_collect_answer(summary_payload)
            if fallback is None:
                raise BenchmarkError(
                    f"Review-loop run `{run_id}` did not produce a collectible final answer. terminal_state={summary_payload.get('terminal_state')}"
                )
            collector_result = fallback
            collector_meta.setdefault("mode", "fallback")

        short_answer_text, full_response_text = normalize_answer_tracks(
            short_answer_text=str(collector_result.get("short_answer") or collector_result.get("final_answer") or ""),
            full_response_text=str(collector_result.get("full_response_text") or ""),
        )
        runner_meta = {
            "run_id": run_id,
            "launch": launch_payload,
            "summary_payload": summary_payload,
            "summary_json_path": "",
            "clawteam_data_dir": str(self.clawteam_data_dir),
            "terminal_state": summary_payload.get("terminal_state"),
            "final_candidates": summary_payload.get("final_candidates"),
            "collector": {
                **collector_meta,
                "result": collector_result,
            },
        }
        if coordinator_summary_path is not None:
            runner_meta["summary_json_path"] = str(coordinator_summary_path)
        if input_bundle is not None:
            runner_meta["runtime_bundle"] = input_bundle.to_meta()
        return RunOutput(
            short_answer_text=short_answer_text,
            full_response_text=full_response_text,
            raw={
                "launch": launch_payload,
                "summary": summary_payload,
                "candidate_payloads": candidate_payloads,
                "coordinator_summary_text": coordinator_summary_text,
                "collector": collector_result,
            },
            runner_meta=runner_meta,
        )


def run_group(
    *,
    group: ExperimentGroup,
    records: list[BenchmarkRecord],
    output_root: Path,
    debateclaw_root: Path,
    rl_timeout: int,
    collector_timeout: int,
    judge: JudgeClient,
    config_path: Path,
    collector_agent: str,
    review_rounds: int | None,
    rebuttal_rounds: int | None,
    proposer_count: int | None,
    model_profile: str | None,
) -> list[GroupRecordResult]:
    runtime_bundle_root = output_root / "input-bundles"
    collector = OuterCollectorClient(
        collector_agent=collector_agent,
        timeout_seconds=collector_timeout,
        config_path=config_path,
    )
    runner = ReviewLoopRunner(
        debateclaw_root=debateclaw_root,
        timeout_seconds=rl_timeout,
        config_path=config_path,
        collector=collector,
        runtime_bundle_root=runtime_bundle_root,
        template_output_dir=output_root / "clawteam-home" / ".clawteam" / "templates",
        launch_home_dir=output_root / "clawteam-home",
        clawteam_data_dir=output_root / "clawteam-data",
        review_rounds=review_rounds,
        rebuttal_rounds=rebuttal_rounds,
        proposer_count=proposer_count,
        model_profile=model_profile,
    )
    group_results: list[GroupRecordResult] = []
    for record in records:
        started = time.time()
        try:
            run_output = runner.run(record, group)
            evaluation = evaluate_answer(
                record,
                short_answer_text=run_output.short_answer_text,
                full_response_text=run_output.full_response_text,
                judge=judge,
            )
            elapsed = time.time() - started
            answer_text = run_output.full_response_text or run_output.short_answer_text
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
                runner_meta=run_output.runner_meta,
                raw=run_output.raw,
                elapsed_seconds=elapsed,
                error=None,
                short_answer_text=run_output.short_answer_text,
                full_response_text=run_output.full_response_text,
            )
        except Exception as exc:
            elapsed = time.time() - started
            error_message = f"Record `{record.record_id}` failed in group `{group.id}`: {exc}"
            entry = build_error_group_record_result(
                group=group,
                record=record,
                error_message=error_message,
                elapsed_seconds=elapsed,
                runner_meta={"traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))},
            )
        group_results.append(entry)
        save_json(output_root / "per-record" / group.id / f"{slugify(record.record_id)}.json", asdict(entry))
    return group_results


def main() -> int:
    args = parse_args()
    dataset_files = select_dataset_files(args)
    if args.list_datasets:
        print_dataset_listing(dataset_files)
        return 0
    if not dataset_files:
        raise BenchmarkError("No benchmark files discovered.")

    all_records = load_records(dataset_files)
    records = apply_offset_limit(all_records, offset=args.offset, limit=args.limit)
    if not records:
        raise BenchmarkError("No benchmark records selected.")
    if args.print_selected_records:
        print_selected_records(records)
        return 0

    group_id = benchmark_group_id_for_websearch(args.websearch)
    group = EXPERIMENT_GROUPS[group_id]

    if args.exact_output_dir:
        output_root = Path(args.exact_output_dir).expanduser().resolve()
    else:
        output_root = Path(args.output_dir).expanduser().resolve() / f"benchmark-rl-{now_stamp()}"
    ensure_dir(output_root)

    debateclaw_root = Path(args.debateclaw_root).expanduser().resolve()
    model_profile = args.model_profile or "review-loop-default"
    config_pool = ReviewLoopConfigPool(
        base_config_path=Path(args.openclaw_config).expanduser().resolve(),
        output_root=output_root,
        debateclaw_root=debateclaw_root,
        collector_agent=args.collector_agent,
        collector_model=args.collector_model,
        judge_agent=args.judge_agent,
        judge_model=args.judge_model,
    )
    config_path = config_pool.config_for_group(group, model_profile=model_profile)
    judge = JudgeClient(
        judge_agent=args.judge_agent,
        timeout_seconds=args.judge_timeout,
        config_path=config_pool.judge_config_path(),
    )
    results = run_group(
        group=group,
        records=records,
        output_root=output_root,
        debateclaw_root=debateclaw_root,
        rl_timeout=args.rl_timeout,
        collector_timeout=args.collector_timeout,
        judge=judge,
        config_path=config_path,
        collector_agent=args.collector_agent,
        review_rounds=args.review_rounds,
        rebuttal_rounds=args.rebuttal_rounds,
        proposer_count=args.proposer_count,
        model_profile=model_profile,
    )
    summary = aggregate_results(results)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "benchmark_root": str(Path(args.benchmark_root).expanduser().resolve()),
        "dataset_files": [str(path) for path in dataset_files],
        "group": asdict(group),
        "records": len(records),
        "results": [asdict(item) for item in results],
        "summary": summary,
        "errors": [{"group_id": item.group_id, "record_id": item.record_id, "error": item.error} for item in results if item.error],
    }
    save_json(output_root / "results.json", payload)
    export_csv_reports(output_root, summary, [group_id])
    save_json(
        output_root / "runtime-manifest.json",
        {
            "group": asdict(group),
            "websearch": args.websearch,
            "config_path": str(config_path),
            "model_profile": model_profile,
            "review_rounds": args.review_rounds,
            "rebuttal_rounds": args.rebuttal_rounds,
            "proposer_count": args.proposer_count,
            "collector": {
                "agent": args.collector_agent,
                "model": args.collector_model,
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
