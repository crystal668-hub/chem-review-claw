#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_WORKSPACE = Path("/home/dministrator/.openclaw/workspace")
DEFAULT_BENCHMARK_ROOT = Path("/home/dministrator/.openclaw/benchmarks")
DEFAULT_CHEMQA_ROOT = Path("/home/dministrator/.openclaw/skills/chemqa-review")
DEFAULT_OPENCLAW_CONFIG = Path.home() / ".openclaw" / "openclaw.json"
DEFAULT_OUTPUT_DIR = DEFAULT_WORKSPACE / "state" / "benchmark-runs"
DEFAULT_SINGLE_AGENT = "debate-1"
DEFAULT_JUDGE_AGENT = "debate-coordinator"
DEFAULT_CHEMQA_PRESET = "chemqa-review@1"
DEFAULT_CHEMQA_MODEL_PROFILE = "chemqa-review-su8-coord-packy-reviewers"
SUBSET_ORDER = (
    "chembench",
    "frontierscience_Olympiad",
    "frontierscience_Research",
)
FINAL_ANSWER_RE = re.compile(r"^\s*FINAL\s+ANSWER\s*[:：-]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", re.DOTALL | re.IGNORECASE)


class BenchmarkError(RuntimeError):
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
    text: str
    raw: dict[str, Any]
    runner_meta: dict[str, Any]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run four-group ChemQA / single-LLM benchmark experiments.")
    parser.add_argument("--benchmark-root", default=str(DEFAULT_BENCHMARK_ROOT), help="benchmarks/ 根目录")
    parser.add_argument("--chemqa-root", default=str(DEFAULT_CHEMQA_ROOT), help="chemqa-review skill 根目录")
    parser.add_argument("--openclaw-config", default=str(DEFAULT_OPENCLAW_CONFIG), help="基础 OpenClaw 配置文件")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="结果输出目录")
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
        help="按子集随机抽样时，每个子集抽取多少题；当前支持 chembench / frontierscience_Olympiad / frontierscience_Research",
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
    parser.add_argument("--single-agent", default=DEFAULT_SINGLE_AGENT, help="单一 LLM 基线所用 agent id")
    parser.add_argument(
        "--chemqa-model-profile",
        default=DEFAULT_CHEMQA_MODEL_PROFILE,
        help="ChemQAWorkflow 所用 model profile，默认使用当前已验证可运行的 profile",
    )
    parser.add_argument("--judge-agent", default=DEFAULT_JUDGE_AGENT, help="rubric / 语义评测所用 judge agent id")
    parser.add_argument("--single-timeout", type=int, default=900, help="单一 LLM 每题超时秒数")
    parser.add_argument("--chemqa-timeout", type=int, default=3600, help="ChemQAWorkflow 每题超时秒数")
    parser.add_argument("--judge-timeout", type=int, default=300, help="Judge 每次评测超时秒数")
    parser.add_argument("--review-rounds", type=int, help="ChemQA review rounds 覆盖值")
    parser.add_argument("--rebuttal-rounds", type=int, help="ChemQA rebuttal rounds 覆盖值")
    parser.add_argument("--keep-temp-configs", action="store_true", help="保留临时 OpenClaw 配置文件")
    parser.add_argument("--list-datasets", action="store_true", help="列出可发现的数据集文件后退出")
    return parser.parse_args()


def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def slugify(value: str, *, limit: int = 64) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    cleaned = cleaned or "item"
    if len(cleaned) <= limit:
        return cleaned
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[: limit - 9]}-{digest}".strip("-")


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
    stdout = result.stdout.strip()
    if not stdout:
        raise BenchmarkError(f"Empty stdout from command: {' '.join(command)}")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise BenchmarkError(
            "JSON decode failed\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        ) from exc


def deep_copy_jsonish(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


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
    def __init__(self, *, base_config_path: Path, keep: bool) -> None:
        self.base_config_path = base_config_path
        self.keep = keep
        self._payload = json.loads(base_config_path.read_text(encoding="utf-8"))
        self._paths: dict[bool, Path] = {}

    def config_for_websearch(self, enabled: bool) -> Path:
        if enabled in self._paths:
            return self._paths[enabled]
        payload = build_temp_openclaw_config_payload(self._payload, enable_websearch=enabled)
        fd, temp_name = tempfile.mkstemp(
            prefix=f"openclaw-benchmark-{'web' if enabled else 'no-web'}-",
            suffix=".json",
        )
        path = Path(temp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        self._paths[enabled] = path
        return path

    def cleanup(self) -> None:
        if self.keep:
            return
        for path in self._paths.values():
            path.unlink(missing_ok=True)


def discover_dataset_files(root: Path) -> list[Path]:
    return sorted(path.resolve() for path in root.glob("*/data/*.jsonl") if path.is_file())


def dataset_name_from_file(path: Path) -> str:
    return path.parent.parent.name


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
                prompt = str(payload.get("prompt") or payload.get("problem") or payload.get("input") or payload.get("question") or "").strip()
                if not prompt:
                    raise BenchmarkError(f"Missing prompt/problem field in record: {record_id}")
                reference_answer = str(payload.get("answer") or payload.get("target") or "").strip()
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


def classify_subset(record: BenchmarkRecord) -> str:
    if record.dataset == "chembench":
        return "chembench"
    if record.dataset == "frontierscience":
        track = str(record.payload.get("track") or "").strip().lower()
        if track == "olympiad" or record.eval_kind == "frontierscience_olympiad":
            return "frontierscience_Olympiad"
        if track == "research" or record.eval_kind == "frontierscience_research":
            return "frontierscience_Research"
    return f"{record.dataset}:{record.eval_kind}"



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
    for subset in available_supported:
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
    start = min((idx for idx in [stripped.find("{"), stripped.find("[")] if idx != -1), default=-1)
    if start != -1:
        fragment = stripped[start:]
        for end in range(len(fragment), 0, -1):
            try:
                return json.loads(fragment[:end])
            except json.JSONDecodeError:
                continue
    raise BenchmarkError(f"Judge response did not contain parseable JSON:\n{text}")


def build_single_llm_prompt(record: BenchmarkRecord, *, websearch_enabled: bool) -> str:
    instructions = [
        "You are answering a chemistry benchmark question.",
        "Be careful, concise, and do not fabricate missing facts.",
    ]
    if websearch_enabled:
        instructions.append("You may use web search if it is genuinely helpful.")
    else:
        instructions.append("Do not use web search or external browsing.")

    if record.eval_kind == "chembench_open_ended":
        instructions.append("Show brief reasoning if needed, then end with exactly one line formatted as: FINAL ANSWER: <answer>.")
    elif record.eval_kind == "frontierscience_olympiad":
        instructions.append("End with exactly one line formatted as: FINAL ANSWER: <answer>.")
    else:
        instructions.append("Provide a complete answer. If you include a final answer line, use: FINAL ANSWER: <answer>.")

    return "\n".join(instructions) + "\n\nQUESTION:\n" + record.prompt.strip()


def build_chemqa_goal(record: BenchmarkRecord, *, websearch_enabled: bool) -> str:
    instructions = [
        "Solve the following chemistry benchmark question.",
        "Return a final answer that is faithful to the prompt.",
    ]
    if websearch_enabled:
        instructions.append("Web search may be used if helpful.")
    else:
        instructions.append("Do not use web search or external browsing.")
    if record.eval_kind in {"chembench_open_ended", "frontierscience_olympiad"}:
        instructions.append("If appropriate, end with a line `FINAL ANSWER: <answer>`.")
    return "\n".join(instructions) + "\n\nQUESTION:\n" + record.prompt.strip()


def summarize_payloads(payloads: list[dict[str, Any]]) -> str:
    texts = [str(item.get("text") or "").strip() for item in payloads if str(item.get("text") or "").strip()]
    return "\n\n".join(texts).strip()


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

    def evaluate_json(self, prompt: str) -> dict[str, Any]:
        session_id = f"benchmark-judge-{uuid.uuid4().hex[:12]}"
        command = [
            "openclaw",
            "agent",
            "--agent",
            self.judge_agent,
            "--session-id",
            session_id,
            "--message",
            prompt,
            "--thinking",
            "medium",
            "--timeout",
            str(self.timeout_seconds),
            "--json",
        ]
        env = os.environ.copy()
        env["OPENCLAW_CONFIG_PATH"] = str(self.config_path)
        result = run_subprocess(command, env=env, timeout=self.timeout_seconds + 30)
        payload = parse_json_stdout(result, command)
        reply = summarize_payloads(list(((payload.get("result") or {}).get("payloads") or [])))
        parsed = safe_json_extract(reply)
        if not isinstance(parsed, dict):
            raise BenchmarkError(f"Judge must return a JSON object, got: {reply}")
        return parsed


class SingleLLMRunner:
    def __init__(
        self,
        *,
        agent_id: str,
        timeout_seconds: int,
        config_pool: ConfigPool,
    ) -> None:
        self.agent_id = agent_id
        self.timeout_seconds = timeout_seconds
        self.config_pool = config_pool

    def run(self, record: BenchmarkRecord, group: ExperimentGroup) -> RunOutput:
        prompt = build_single_llm_prompt(record, websearch_enabled=group.websearch)
        session_id = f"benchmark-{group.id}-{slugify(record.record_id, limit=40)}-{uuid.uuid4().hex[:8]}"
        command = [
            "openclaw",
            "agent",
            "--agent",
            self.agent_id,
            "--session-id",
            session_id,
            "--message",
            prompt,
            "--thinking",
            "low",
            "--timeout",
            str(self.timeout_seconds),
            "--json",
        ]
        env = os.environ.copy()
        env["OPENCLAW_CONFIG_PATH"] = str(self.config_pool.config_for_websearch(group.websearch))
        result = run_subprocess(command, env=env, timeout=self.timeout_seconds + 30)
        payload = parse_json_stdout(result, command)
        payloads = list(((payload.get("result") or {}).get("payloads") or []))
        text = summarize_payloads(payloads)
        runner_meta = deep_copy_jsonish((payload.get("result") or {}).get("meta") or {})
        return RunOutput(text=text, raw=payload, runner_meta=runner_meta)


class ChemQARunner:
    def __init__(
        self,
        *,
        chemqa_root: Path,
        timeout_seconds: int,
        config_pool: ConfigPool,
        review_rounds: int | None,
        rebuttal_rounds: int | None,
        model_profile: str,
    ) -> None:
        self.chemqa_root = chemqa_root
        self.timeout_seconds = timeout_seconds
        self.config_pool = config_pool
        self.review_rounds = review_rounds
        self.rebuttal_rounds = rebuttal_rounds
        self.model_profile = model_profile
        self.launch_script = chemqa_root / "scripts" / "launch_from_preset.py"
        self.collect_script = chemqa_root / "scripts" / "collect_artifacts.py"

    def _wait_for_terminal_status(self, run_id: str, *, timeout_seconds: int) -> dict[str, Any]:
        status_path = self.chemqa_root / "control" / "run-status" / f"{run_id}.json"
        deadline = time.time() + timeout_seconds
        last_status: dict[str, Any] = {}
        while time.time() < deadline:
            if status_path.is_file():
                last_status = json.loads(status_path.read_text(encoding="utf-8"))
                status = str(last_status.get("status") or "")
                if status in {"completed", "failed", "abandoned", "cancelled"}:
                    return last_status
            time.sleep(5)
        raise BenchmarkError(
            f"ChemQA run `{run_id}` did not reach a terminal state within {timeout_seconds}s. Last status: {last_status}"
        )

    def _ensure_artifacts(self, run_id: str, *, env: dict[str, str]) -> Path:
        artifact_dir = self.chemqa_root / "generated" / "artifacts" / run_id
        qa_result_path = artifact_dir / "qa_result.json"
        if qa_result_path.is_file():
            return qa_result_path

        protocol_dir = self.chemqa_root / "generated" / "clawteam-data" / "teams" / run_id
        protocol_path = protocol_dir / "chemqa_review_protocol.yaml"
        if not protocol_path.is_file():
            raise BenchmarkError(
                f"ChemQA run `{run_id}` finished without qa_result.json and protocol file was not found at {protocol_path}"
            )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "python3",
            str(self.collect_script),
            "--skill-root",
            str(self.chemqa_root),
            "--source-dir",
            str(protocol_dir),
            "--output-dir",
            str(artifact_dir),
        ]
        result = run_subprocess(command, env=env, cwd=self.chemqa_root, timeout=120)
        parse_json_stdout(result, command)
        if not qa_result_path.is_file():
            raise BenchmarkError(f"Failed to rebuild ChemQA artifacts for run `{run_id}`")
        return qa_result_path

    def run(self, record: BenchmarkRecord, group: ExperimentGroup) -> RunOutput:
        run_id = f"benchmark-{group.id}-{slugify(record.record_id, limit=40)}-{now_stamp()}"
        goal = build_chemqa_goal(record, websearch_enabled=group.websearch)
        command = [
            "python3",
            str(self.launch_script),
            "--root",
            str(self.chemqa_root),
            "--preset",
            DEFAULT_CHEMQA_PRESET,
            "--goal",
            goal,
            "--run-id",
            run_id,
            "--model-profile",
            self.model_profile,
            "--launch-mode",
            "run",
        ]
        if self.review_rounds is not None:
            command.extend(["--review-rounds", str(self.review_rounds)])
        if self.rebuttal_rounds is not None:
            command.extend(["--rebuttal-rounds", str(self.rebuttal_rounds)])

        env = os.environ.copy()
        env["OPENCLAW_CONFIG_PATH"] = str(self.config_pool.config_for_websearch(group.websearch))
        env["OPENCLAW_DEBATE_TRUSTED_PLUGINS"] = "duckduckgo" if group.websearch else "__none__"
        result = run_subprocess(command, env=env, cwd=self.chemqa_root, timeout=self.timeout_seconds)
        payload = parse_json_stdout(result, command)
        run_status = self._wait_for_terminal_status(run_id, timeout_seconds=self.timeout_seconds)
        terminal_status = str(run_status.get("status") or "")
        if terminal_status != "completed":
            raise BenchmarkError(f"ChemQA run `{run_id}` ended with non-success status: {run_status}")
        qa_result_path = self._ensure_artifacts(run_id, env=env)
        qa_result = json.loads(qa_result_path.read_text(encoding="utf-8"))
        text = str(qa_result.get("final_answer") or "").strip()
        if not text:
            text = (Path(qa_result["artifact_paths"]["final_answer"]).read_text(encoding="utf-8")).strip()
        runner_meta = {
            "run_id": run_id,
            "launch": payload,
            "qa_result_path": str(qa_result_path),
            "acceptance_status": qa_result.get("acceptance_status"),
            "terminal_state": qa_result.get("terminal_state"),
            "run_status": run_status,
        }
        return RunOutput(text=text, raw=qa_result, runner_meta=runner_meta)


def evaluate_chembench_open_ended(record: BenchmarkRecord, answer_text: str) -> EvaluationResult:
    expected = str(record.payload.get("target") or record.reference_answer)
    predicted_short = extract_candidate_short_answer(answer_text)
    expected_norm = normalize_loose(expected)
    predicted_norm = normalize_loose(predicted_short)

    expected_num = parse_numeric_scalar(expected)
    predicted_num = parse_numeric_scalar(predicted_short)
    exact_match = predicted_norm == expected_norm
    relative_tolerance = record.payload.get("relative_tolerance")
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

    preferred = str(record.payload.get("preferred_score") or "exact_str_match")
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
    answer_text: str,
    *,
    judge: JudgeClient,
) -> EvaluationResult:
    expected = record.reference_answer
    predicted = extract_candidate_short_answer(answer_text)
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

CANDIDATE ANSWER:
{predicted}
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


def evaluate_frontierscience_research(
    record: BenchmarkRecord,
    answer_text: str,
    *,
    judge: JudgeClient,
) -> EvaluationResult:
    rubric_items = parse_frontierscience_research_rubric(record.reference_answer)
    if not rubric_items:
        raise BenchmarkError(f"No rubric items parsed for record: {record.record_id}")
    rubric_lines = [f"{idx + 1}. [{item['points']} points] {item['description']}" for idx, item in enumerate(rubric_items)]
    max_score = float(sum(item["points"] for item in rubric_items))
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
{answer_text}
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


def evaluate_generic_semantic(
    record: BenchmarkRecord,
    answer_text: str,
    *,
    judge: JudgeClient,
) -> EvaluationResult:
    expected = record.reference_answer
    predicted = extract_candidate_short_answer(answer_text)
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

CANDIDATE ANSWER:
{predicted}
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


def evaluate_answer(record: BenchmarkRecord, answer_text: str, *, judge: JudgeClient) -> EvaluationResult:
    if record.eval_kind == "chembench_open_ended":
        return evaluate_chembench_open_ended(record, answer_text)
    if record.eval_kind == "frontierscience_olympiad":
        return evaluate_frontierscience_olympiad(record, answer_text, judge=judge)
    if record.eval_kind == "frontierscience_research":
        return evaluate_frontierscience_research(record, answer_text, judge=judge)
    return evaluate_generic_semantic(record, answer_text, judge=judge)


def aggregate_results(results: list[GroupRecordResult]) -> dict[str, Any]:
    grouped: dict[str, list[GroupRecordResult]] = {}
    for item in results:
        grouped.setdefault(item.group_id, []).append(item)

    summary_groups: dict[str, Any] = {}
    for group_id, items in grouped.items():
        by_eval_kind: dict[str, list[GroupRecordResult]] = {}
        for item in items:
            by_eval_kind.setdefault(item.eval_kind, []).append(item)
        summary_groups[group_id] = {
            "group_label": items[0].group_label,
            "runner": items[0].runner,
            "websearch": items[0].websearch,
            "count": len(items),
            "pass_count": sum(1 for item in items if item.evaluation["passed"]),
            "avg_score": sum(float(item.evaluation["score"]) for item in items) / len(items),
            "avg_normalized_score": sum(float(item.evaluation["normalized_score"]) for item in items) / len(items),
            "avg_elapsed_seconds": sum(float(item.elapsed_seconds) for item in items) / len(items),
            "by_eval_kind": {
                eval_kind: {
                    "count": len(eval_items),
                    "pass_count": sum(1 for item in eval_items if item.evaluation["passed"]),
                    "avg_score": sum(float(item.evaluation["score"]) for item in eval_items) / len(eval_items),
                    "avg_normalized_score": sum(float(item.evaluation["normalized_score"]) for item in eval_items) / len(eval_items),
                }
                for eval_kind, eval_items in by_eval_kind.items()
            },
        }

    return {
        "group_order": list(grouped.keys()),
        "groups": summary_groups,
    }


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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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

    output_root = Path(args.output_dir).expanduser().resolve() / f"benchmark-{now_stamp()}"
    ensure_dir(output_root)

    config_pool = ConfigPool(base_config_path=Path(args.openclaw_config).expanduser().resolve(), keep=args.keep_temp_configs)
    judge = JudgeClient(
        judge_agent=args.judge_agent,
        timeout_seconds=args.judge_timeout,
        config_path=config_pool.config_for_websearch(False),
    )
    single_runner = SingleLLMRunner(
        agent_id=args.single_agent,
        timeout_seconds=args.single_timeout,
        config_pool=config_pool,
    )
    chemqa_runner = ChemQARunner(
        chemqa_root=Path(args.chemqa_root).expanduser().resolve(),
        timeout_seconds=args.chemqa_timeout,
        config_pool=config_pool,
        review_rounds=args.review_rounds,
        rebuttal_rounds=args.rebuttal_rounds,
        model_profile=args.chemqa_model_profile,
    )

    results: list[GroupRecordResult] = []
    try:
        for group_id in group_ids:
            group = EXPERIMENT_GROUPS[group_id]
            for record in records:
                started = time.time()
                if group.runner == "chemqa":
                    run_output = chemqa_runner.run(record, group)
                else:
                    run_output = single_runner.run(record, group)
                evaluation = evaluate_answer(record, run_output.text, judge=judge)
                elapsed = time.time() - started
                entry = GroupRecordResult(
                    group_id=group.id,
                    group_label=group.label,
                    runner=group.runner,
                    websearch=group.websearch,
                    record_id=record.record_id,
                    dataset=record.dataset,
                    source_file=record.source_file,
                    eval_kind=record.eval_kind,
                    prompt=record.prompt,
                    reference_answer=record.reference_answer,
                    answer_text=run_output.text,
                    evaluation=asdict(evaluation),
                    runner_meta=run_output.runner_meta,
                    raw=run_output.raw,
                    elapsed_seconds=elapsed,
                )
                results.append(entry)
                save_json(output_root / "per-record" / group.id / f"{slugify(record.record_id)}.json", asdict(entry))
    finally:
        config_pool.cleanup()

    summary = aggregate_results(results)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "benchmark_root": str(Path(args.benchmark_root).expanduser().resolve()),
        "dataset_files": [str(path) for path in dataset_files],
        "groups": [asdict(EXPERIMENT_GROUPS[group_id]) for group_id in group_ids],
        "random_sampling": {
            "enabled": args.random_count_per_subset is not None,
            "count_per_subset": args.random_count_per_subset,
            "seed": args.random_seed,
        },
        "records": len(records),
        "results": [asdict(item) for item in results],
        "summary": summary,
    }
    save_json(output_root / "results.json", payload)
    print(json.dumps({"output_dir": str(output_root), "summary": summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
