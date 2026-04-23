# Benchmark Architecture Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the benchmark stack so execution status, experiment definition, runtime provisioning, dataset parsing, and evaluation are explicit contracts instead of hidden cross-file conventions.

**Architecture:** Introduce a small `benchmarking/` package at repo root and move shared contracts into it first. Keep `benchmark_test.py` and `benchmark_rl.py` as thin CLI entrypoints that compose shared modules, while preserving the current report formats during the migration. The migration should stop after each task in a working state; no task should require a flag day rewrite.

**Tech Stack:** Python 3.12, stdlib `dataclasses`/`enum`/`argparse`, existing `openclaw` subprocess integration, existing `unittest` test suite, `uv` for test execution.

---

## Planned File Structure

**Create**
- `benchmarking/__init__.py`: stable public exports for shared benchmark modules
- `benchmarking/contracts.py`: run status, answer payload, failure/recovery metadata, shared result contracts
- `benchmarking/experiments.py`: explicit experiment spec objects and CLI override resolution
- `benchmarking/provisioning.py`: runtime workspace/bootstrap side effects only
- `benchmarking/config_renderer.py`: pure config rendering from spec + provisioned runtime
- `benchmarking/datasets.py`: validated record loading and grading spec construction
- `benchmarking/evaluation.py`: evaluator registry and score dispatch
- `benchmarking/runners/__init__.py`: runner factory exports
- `benchmarking/runners/single_llm.py`: single-agent execution runner
- `benchmarking/runners/chemqa.py`: ChemQA execution runner with explicit recovery semantics
- `benchmarking/reporting.py`: `GroupRecordResult` materialization and aggregation helpers
- `tests/test_benchmark_contracts.py`: new contract-level tests
- `tests/test_benchmark_config_runtime.py`: new config/provisioning tests
- `tests/test_benchmark_datasets.py`: new dataset/evaluator registry tests

**Modify**
- `benchmark_test.py`: reduce to CLI orchestration and compatibility adapters
- `benchmark_rl.py`: import from `benchmarking/` directly; remove dynamic file loader
- `tests/test_benchmark_test.py`: update runner expectations to use explicit run status
- `tests/test_benchmark_rl.py`: update shared helper imports and regression coverage

## Migration Rules

- Preserve `results.json`, `runtime-manifest.json`, and `per-record/*.json` shapes until Task 5 completes.
- Do not score any runner output unless the runner explicitly says the run is scoreable.
- Keep filesystem writes out of config rendering code.
- Remove fake CLI flags instead of keeping dead compatibility shims.

### Task 1: Introduce Explicit Result and Experiment Contracts

**Files:**
- Create: `benchmarking/__init__.py`
- Create: `benchmarking/contracts.py`
- Create: `benchmarking/experiments.py`
- Test: `tests/test_benchmark_contracts.py`

- [x] **Step 1: Write the failing tests**

```python
import unittest

from benchmarking.contracts import (
    AnswerPayload,
    FailureInfo,
    RecoveryInfo,
    RunStatus,
    RunnerResult,
)
from benchmarking.experiments import ExperimentSpec


class BenchmarkContractsTests(unittest.TestCase):
    def test_runner_result_only_scores_completed_or_scored_recovery(self) -> None:
        completed = RunnerResult(
            status=RunStatus.COMPLETED,
            answer=AnswerPayload(short_answer_text="42", full_response_text="FINAL ANSWER: 42"),
            raw={},
            runner_meta={},
        )
        recovered = RunnerResult(
            status=RunStatus.RECOVERED,
            answer=AnswerPayload(short_answer_text="42", full_response_text="FINAL ANSWER: 42"),
            raw={},
            runner_meta={},
            recovery=RecoveryInfo(source="proposer-1-proposal", scored=False, details={}),
        )
        failed = RunnerResult(
            status=RunStatus.FAILED,
            answer=AnswerPayload(),
            raw={},
            runner_meta={},
            failure=FailureInfo(code="terminal_failure", message="review stalled", details={}),
        )

        self.assertTrue(completed.should_score())
        self.assertFalse(recovered.should_score())
        self.assertFalse(failed.should_score())

    def test_experiment_spec_resolves_single_agent_override_explicitly(self) -> None:
        spec = ExperimentSpec(
            id="single_llm_web_off",
            label="Single LLM without web",
            runner_kind="single_llm",
            websearch_enabled=False,
            single_agent_id="benchmark-single-web-off",
        )

        self.assertEqual("benchmark-single-web-off", spec.resolve_single_agent_id(None))
        self.assertEqual("custom-single-agent", spec.resolve_single_agent_id("custom-single-agent"))


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_benchmark_contracts -v`
Expected: `ModuleNotFoundError` for `benchmarking.contracts`

- [x] **Step 3: Write minimal implementation**

```python
# benchmarking/contracts.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class RunStatus(StrEnum):
    COMPLETED = "completed"
    RECOVERED = "recovered"
    FAILED = "failed"


@dataclass(frozen=True)
class AnswerPayload:
    short_answer_text: str = ""
    full_response_text: str = ""


@dataclass(frozen=True)
class FailureInfo:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveryInfo:
    source: str
    scored: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunnerResult:
    status: RunStatus
    answer: AnswerPayload
    raw: dict[str, Any]
    runner_meta: dict[str, Any]
    failure: FailureInfo | None = None
    recovery: RecoveryInfo | None = None

    def should_score(self) -> bool:
        if self.status is RunStatus.COMPLETED:
            return True
        return bool(self.status is RunStatus.RECOVERED and self.recovery and self.recovery.scored)
```

```python
# benchmarking/experiments.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    label: str
    runner_kind: str
    websearch_enabled: bool
    single_agent_id: str | None = None
    slot_set: str | None = None

    def resolve_single_agent_id(self, override: str | None) -> str | None:
        candidate = (override or "").strip()
        if candidate:
            return candidate
        return self.single_agent_id
```

```python
# benchmarking/__init__.py
from .contracts import AnswerPayload, FailureInfo, RecoveryInfo, RunnerResult, RunStatus
from .experiments import ExperimentSpec

__all__ = [
    "AnswerPayload",
    "ExperimentSpec",
    "FailureInfo",
    "RecoveryInfo",
    "RunnerResult",
    "RunStatus",
]
```

- [x] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_benchmark_contracts -v`
Expected: `OK`

- [x] **Step 5: Commit**

```bash
git add benchmarking/__init__.py benchmarking/contracts.py benchmarking/experiments.py tests/test_benchmark_contracts.py
git commit -m "refactor: add explicit benchmark contracts"
```

### Task 2: Separate Pure Config Rendering from Runtime Provisioning

**Files:**
- Create: `benchmarking/provisioning.py`
- Create: `benchmarking/config_renderer.py`
- Test: `tests/test_benchmark_config_runtime.py`
- Modify: `benchmark_test.py`

- [x] **Step 1: Write the failing tests**

```python
import json
import tempfile
import unittest
from pathlib import Path

from benchmarking.config_renderer import render_run_config
from benchmarking.experiments import ExperimentSpec
from benchmarking.provisioning import ProvisionedAgent, ProvisionedExperiment, provision_slot_workspace


class BenchmarkConfigRuntimeTests(unittest.TestCase):
    def test_render_run_config_is_pure_and_does_not_mutate_base_payload(self) -> None:
        base = {
            "agents": {"list": []},
            "tools": {"web": {"search": {"enabled": False}}},
            "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
        }
        spec = ExperimentSpec(
            id="single_llm_web_on",
            label="Single LLM with web",
            runner_kind="single_llm",
            websearch_enabled=True,
            single_agent_id="benchmark-single-web-on",
        )
        provisioned = ProvisionedExperiment(
            judge=ProvisionedAgent("benchmark-judge", Path("/tmp/judge"), Path("/tmp/agents/judge")),
            runner_agents=(ProvisionedAgent("benchmark-single-web-on", Path("/tmp/single"), Path("/tmp/agents/single")),),
        )

        rendered = render_run_config(base_payload=base, spec=spec, provisioned=provisioned, judge_model="su8/gpt-5.4", runner_model="qwen3.5-plus")

        self.assertEqual([], base["agents"]["list"])
        self.assertTrue(rendered["tools"]["web"]["search"]["enabled"])
        self.assertTrue(rendered["plugins"]["entries"]["duckduckgo"]["enabled"])

    def test_provision_slot_workspace_creates_agents_md_and_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "debateA-1"
            workspace_root = workspace.parent

            provision_slot_workspace(workspace=workspace, workspace_root=workspace_root, slot_id="debateA-1", agents_template_text="# demo\\n")

            self.assertTrue((workspace / "AGENTS.md").is_file())
            sentinel = json.loads((workspace / ".debateclaw-slot.json").read_text(encoding="utf-8"))
            self.assertEqual("debateA-1", sentinel["slot"])
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_benchmark_config_runtime -v`
Expected: `ModuleNotFoundError` for `benchmarking.config_renderer`

- [x] **Step 3: Write minimal implementation**

```python
# benchmarking/provisioning.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProvisionedAgent:
    agent_id: str
    workspace: Path
    agent_dir: Path


@dataclass(frozen=True)
class ProvisionedExperiment:
    judge: ProvisionedAgent
    runner_agents: tuple[ProvisionedAgent, ...]


def provision_slot_workspace(*, workspace: Path, workspace_root: Path, slot_id: str, agents_template_text: str) -> None:
    workspace_root.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "AGENTS.md").write_text(agents_template_text, encoding="utf-8")
    (workspace / ".debateclaw-slot.json").write_text(
        json.dumps(
            {
                "kind": "debateclaw-slot-workspace",
                "version": 1,
                "slot": slot_id,
                "workspace": str(workspace.resolve()),
                "workspace_root": str(workspace_root.resolve()),
                "managed_by": "debateclaw",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\\n",
        encoding="utf-8",
    )
```

```python
# benchmarking/config_renderer.py
from __future__ import annotations

import json
from typing import Any

from benchmarking.experiments import ExperimentSpec
from benchmarking.provisioning import ProvisionedExperiment


def render_run_config(
    *,
    base_payload: dict[str, Any],
    spec: ExperimentSpec,
    provisioned: ProvisionedExperiment,
    judge_model: str,
    runner_model: str,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(base_payload, ensure_ascii=False))
    payload.setdefault("tools", {}).setdefault("web", {}).setdefault("search", {})["enabled"] = spec.websearch_enabled
    payload.setdefault("plugins", {}).setdefault("entries", {}).setdefault("duckduckgo", {})["enabled"] = spec.websearch_enabled
    payload["plugins"]["entries"]["duckduckgo"].setdefault("config", {})
    entries = payload.setdefault("agents", {}).setdefault("list", [])
    entries.append(
        {
            "id": provisioned.judge.agent_id,
            "name": provisioned.judge.agent_id,
            "workspace": str(provisioned.judge.workspace.resolve()),
            "agentDir": str(provisioned.judge.agent_dir.resolve()),
            "model": judge_model,
        }
    )
    for agent in provisioned.runner_agents:
        entries.append(
            {
                "id": agent.agent_id,
                "name": agent.agent_id,
                "workspace": str(agent.workspace.resolve()),
                "agentDir": str(agent.agent_dir.resolve()),
                "model": runner_model,
            }
        )
    return payload
```

- [x] **Step 4: Wire `benchmark_test.py` through the new helpers**

```python
try:
    from benchmarking.config_renderer import render_run_config
    from benchmarking.provisioning import ProvisionedAgent, ProvisionedExperiment, provision_slot_workspace
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from workspace.benchmarking.config_renderer import render_run_config
    from workspace.benchmarking.provisioning import ProvisionedAgent, ProvisionedExperiment, provision_slot_workspace


def build_run_scoped_config_payload(...):
    provisioned = build_provisioned_experiment(...)
    return render_run_config(
        base_payload=base_payload,
        spec=spec,
        provisioned=provisioned,
        judge_model=judge_model,
        runner_model=single_agent_model,
    )
```

- [x] **Step 5: Run tests to verify they pass**

Run: `uv run python -m unittest tests.test_benchmark_config_runtime tests.test_benchmark_test -v`
Expected: `OK`

- [x] **Step 6: Commit**

```bash
git add benchmarking/provisioning.py benchmarking/config_renderer.py benchmark_test.py tests/test_benchmark_config_runtime.py tests/test_benchmark_test.py
git commit -m "refactor: split benchmark config rendering from provisioning"
```

### Task 3: Replace Loose Record Payload Coupling with Validated Record and Grading Specs

**Files:**
- Create: `benchmarking/datasets.py`
- Create: `benchmarking/evaluation.py`
- Test: `tests/test_benchmark_datasets.py`
- Modify: `benchmark_test.py`

- [x] **Step 1: Write the failing tests**

```python
import json
import tempfile
import unittest
from pathlib import Path

from benchmarking.datasets import GradingSpec, load_records
from benchmarking.evaluation import EVALUATORS, evaluate_record


class BenchmarkDatasetsTests(unittest.TestCase):
    def test_load_records_builds_chembench_grading_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chembench" / "data" / "sample.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "id": "chem-1",
                        "prompt": "Q",
                        "target": "42",
                        "eval_kind": "chembench_open_ended",
                        "relative_tolerance": 0.1,
                    }
                )
                + "\\n",
                encoding="utf-8",
            )

            record = load_records([path])[0]

            self.assertEqual("chembench_open_ended", record.grading.kind)
            self.assertEqual("42", record.grading.reference_answer)
            self.assertEqual(0.1, record.grading.config["relative_tolerance"])

    def test_evaluate_record_uses_registry_dispatch(self) -> None:
        self.assertIn("chembench_open_ended", EVALUATORS)
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_benchmark_datasets -v`
Expected: `ModuleNotFoundError` for `benchmarking.datasets`

- [x] **Step 3: Write minimal implementation**

```python
# benchmarking/datasets.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class GradingSpec:
    kind: str
    reference_answer: str
    subset: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkRecord:
    record_id: str
    dataset: str
    source_file: str
    prompt: str
    grading: GradingSpec
    raw_payload: dict[str, Any]


def dataset_name_from_file(path: Path) -> str:
    return path.parent.parent.name


def classify_subset(record: BenchmarkRecord) -> str:
    if record.dataset == "chembench":
        return "chembench"
    if record.dataset == "conformabench":
        return "conformabench"
    if record.dataset == "frontierscience":
        track = str(record.raw_payload.get("track") or "").strip().lower()
        if track == "olympiad" or record.grading.kind == "frontierscience_olympiad":
            return "frontierscience_Olympiad"
        if track == "research" or record.grading.kind == "frontierscience_research":
            return "frontierscience_Research"
    if record.dataset == "superchem":
        return "superchem_multimodal"
    return f"{record.dataset}:{record.grading.kind}"


def build_grading_spec(*, dataset: str, source_file: str, prompt: str, payload: dict[str, Any]) -> GradingSpec:
    reference_answer = str(payload.get("answer") or payload.get("target") or "").strip()
    eval_kind = str(payload.get("eval_kind") or "generic_semantic").strip() or "generic_semantic"
    draft = BenchmarkRecord(
        record_id="preview",
        dataset=dataset,
        source_file=source_file,
        prompt=prompt,
        grading=GradingSpec(kind=eval_kind, reference_answer=reference_answer, subset="preview", config={}),
        raw_payload=payload,
    )
    return GradingSpec(
        kind=eval_kind,
        reference_answer=reference_answer,
        subset=classify_subset(draft),
        config={
            "preferred_score": payload.get("preferred_score"),
            "relative_tolerance": payload.get("relative_tolerance"),
            "track": payload.get("track"),
            "options": payload.get("options"),
            "reference_reasoning": payload.get("reference_reasoning"),
        },
    )


def load_records(paths: Iterable[Path]) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for path in paths:
        dataset = dataset_name_from_file(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                prompt = str(payload.get("prompt") or payload.get("problem") or payload.get("input") or payload.get("question") or "").strip()
                grading = build_grading_spec(dataset=dataset, source_file=str(path), prompt=prompt, payload=payload)
                record = BenchmarkRecord(
                    record_id=str(payload.get("id") or f"{dataset}-{len(records)}"),
                    dataset=dataset,
                    source_file=str(path),
                    prompt=prompt,
                    grading=grading,
                    raw_payload=payload,
                )
                records.append(record)
    return records
```

```python
# benchmarking/evaluation.py
from __future__ import annotations

from typing import Callable

from benchmarking.datasets import BenchmarkRecord


Evaluator = Callable[[BenchmarkRecord, str, str, object], object]
EVALUATORS: dict[str, Evaluator] = {}


def register_evaluator(kind: str, evaluator: Evaluator) -> None:
    EVALUATORS[kind] = evaluator


def evaluate_record(record: BenchmarkRecord, *, short_answer_text: str, full_response_text: str, judge: object) -> object:
    evaluator = EVALUATORS[record.grading.kind]
    return evaluator(record, short_answer_text, full_response_text, judge)
```

- [x] **Step 4: Port existing evaluators in `benchmark_test.py` to use `record.grading` instead of `record.payload`**

```python
expected = record.grading.reference_answer
relative_tolerance = record.grading.config.get("relative_tolerance")
checkpoints = parse_superchem_checkpoints(str(record.grading.config.get("reference_reasoning") or ""))
```

```python
register_evaluator("chembench_open_ended", evaluate_chembench_open_ended)
register_evaluator("conformabench_constructive", evaluate_conformabench_constructive)
register_evaluator("frontierscience_olympiad", evaluate_frontierscience_olympiad)
register_evaluator("frontierscience_research", evaluate_frontierscience_research)
register_evaluator("superchem_multiple_choice_rpf", evaluate_superchem_multiple_choice_rpf)
register_evaluator("generic_semantic", evaluate_generic_semantic)
```

- [x] **Step 5: Run tests to verify they pass**

Run: `uv run python -m unittest tests.test_benchmark_datasets tests.test_benchmark_test tests.test_benchmark_rl -v`
Expected: `OK`

- [x] **Step 6: Commit**

```bash
git add benchmarking/datasets.py benchmarking/evaluation.py benchmark_test.py tests/test_benchmark_datasets.py tests/test_benchmark_test.py tests/test_benchmark_rl.py
git commit -m "refactor: add validated benchmark record loading and evaluator registry"
```

### Task 4: Make Runner Status Explicit and Stop Scoring Hidden Failures

**Files:**
- Create: `benchmarking/runners/__init__.py`
- Create: `benchmarking/runners/single_llm.py`
- Create: `benchmarking/runners/chemqa.py`
- Modify: `benchmark_test.py`
- Test: `tests/test_benchmark_test.py`

- [x] **Step 1: Write the failing tests**

```python
def test_run_group_marks_unscored_recovery_as_execution_error(self) -> None:
    record = benchmark_test.BenchmarkRecord(
        record_id="r1",
        dataset="chembench",
        source_file="/tmp/demo.jsonl",
        eval_kind="chembench_open_ended",
        prompt="Q",
        reference_answer="42",
        payload={},
    )

    class StubRunner:
        def run(self, record, group):
            return benchmark_test.RunnerResult(
                status=benchmark_test.RunStatus.RECOVERED,
                answer=benchmark_test.AnswerPayload(short_answer_text="42", full_response_text="FINAL ANSWER: 42"),
                raw={"fallback": True},
                runner_meta={"fallback_used": True},
                recovery=benchmark_test.RecoveryInfo(source="proposal", scored=False, details={}),
            )

    original_factory = benchmark_test.build_runner
    benchmark_test.build_runner = lambda **kwargs: StubRunner()
    try:
        results = benchmark_test.run_group(
            group=benchmark_test.EXPERIMENT_GROUPS["single_llm_web_off"],
            records=[record],
            output_root=Path(self.tempdir.name),
            single_timeout=10,
            chemqa_timeout=10,
            judge=JudgeStub({}),
            config_path=Path(self.tempdir.name) / "cfg.json",
            single_agent="benchmark-single-web-off",
            chemqa_root=Path(self.tempdir.name),
            chemqa_model_profile="unused",
            review_rounds=None,
            rebuttal_rounds=None,
        )
    finally:
        benchmark_test.build_runner = original_factory

    self.assertEqual("execution_error", results[0].evaluation["primary_metric"])
    self.assertIsNotNone(results[0].error)
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_benchmark_test -v`
Expected: `AttributeError` for missing `build_runner` or missing `RunnerResult`

- [x] **Step 3: Extract runners and factory**

```python
# benchmarking/runners/__init__.py
from benchmarking.runners.chemqa import ChemQARunner
from benchmarking.runners.single_llm import SingleLLMRunner


def build_runner(*, runner_kind: str, **kwargs):
    if runner_kind == "chemqa":
        return ChemQARunner(**kwargs)
    if runner_kind == "single_llm":
        return SingleLLMRunner(**kwargs)
    raise ValueError(f"Unsupported runner kind: {runner_kind}")
```

```python
# benchmarking/runners/single_llm.py
from benchmarking.contracts import AnswerPayload, RunStatus, RunnerResult


return RunnerResult(
    status=RunStatus.COMPLETED,
    answer=AnswerPayload(short_answer_text=short_answer_text, full_response_text=full_response_text),
    raw=payload,
    runner_meta=runner_meta,
)
```

```python
# benchmarking/runners/chemqa.py
from benchmarking.contracts import AnswerPayload, FailureInfo, RecoveryInfo, RunStatus, RunnerResult


if not is_chemqa_success_status(run_status):
    if fallback_payload is not None:
        short_answer_text, full_response_text, fallback_meta = fallback_payload
        return RunnerResult(
            status=RunStatus.RECOVERED,
            answer=AnswerPayload(short_answer_text=short_answer_text, full_response_text=full_response_text),
            raw={"run_status": run_status, "fallback": fallback_meta},
            runner_meta=runner_meta,
            recovery=RecoveryInfo(source=str(fallback_meta["fallback_source"]), scored=False, details=fallback_meta),
        )
    return RunnerResult(
        status=RunStatus.FAILED,
        answer=AnswerPayload(),
        raw={"run_status": run_status},
        runner_meta=runner_meta,
        failure=FailureInfo(
            code=terminal_reason_code or "chemqa_non_success_terminal_status",
            message=f"ChemQA run ended with non-success status: {terminal_state or legacy_status or 'unknown'}",
            details={"run_status": run_status},
        ),
    )
```

- [x] **Step 4: Gate scoring in `run_group()`**

```python
run_result = runner.run(record, group)
if not run_result.should_score():
    failure_message = (
        run_result.failure.message
        if run_result.failure is not None
        else f"Record `{record.record_id}` finished in non-scoreable status `{run_result.status}`"
    )
    entry = build_error_group_record_result(
        group=group,
        record=record,
        error_message=failure_message,
        elapsed_seconds=elapsed,
        runner_meta=run_result.runner_meta,
        raw=run_result.raw,
        short_answer_text=run_result.answer.short_answer_text,
        full_response_text=run_result.answer.full_response_text,
    )
else:
    evaluation = evaluate_answer(
        record,
        short_answer_text=run_result.answer.short_answer_text,
        full_response_text=run_result.answer.full_response_text,
        judge=judge,
    )
```

- [x] **Step 5: Run tests to verify they pass**

Run: `uv run python -m unittest tests.test_benchmark_test tests.test_benchmark_rl -v`
Expected: `OK`

- [x] **Step 6: Commit**

```bash
git add benchmarking/runners/__init__.py benchmarking/runners/single_llm.py benchmarking/runners/chemqa.py benchmark_test.py tests/test_benchmark_test.py tests/test_benchmark_rl.py
git commit -m "refactor: make benchmark runner status explicit"
```

### Task 5: Make the CLI Truthful and Move Shared Logic out of `benchmark_test.py`

**Files:**
- Modify: `benchmark_test.py`
- Modify: `benchmark_rl.py`
- Modify: `tests/test_benchmark_test.py`
- Modify: `tests/test_benchmark_rl.py`

- [x] **Step 1: Write the failing tests**

```python
def test_select_single_agent_override_applies_to_single_llm_groups(self) -> None:
    spec = benchmark_test.EXPERIMENT_SPECS["single_llm_web_on"]
    self.assertEqual("custom-single-agent", spec.resolve_single_agent_id("custom-single-agent"))

def test_benchmark_rl_uses_shared_benchmarking_modules(self) -> None:
    self.assertEqual("benchmarking.datasets", benchmark_rl.load_records.__module__)
    self.assertEqual("benchmarking.evaluation", benchmark_rl.evaluate_answer.__module__)
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_benchmark_test tests.test_benchmark_rl -v`
Expected: current modules still resolve through `benchmark_test` wrappers

- [x] **Step 3: Replace fake CLI flags and hard-coded single-agent override behavior**

```python
parser.add_argument(
    "--single-agent-id-override",
    help="若提供，则覆盖所有 single_llm 实验组的 agent id",
)
```

```python
single_agent_id = spec.resolve_single_agent_id(args.single_agent_id_override)
if spec.runner_kind == "single_llm" and not single_agent_id:
    raise BenchmarkError(f"Experiment `{spec.id}` is missing a single-agent id")
```

```python
# Remove this dead flag entirely
# parser.add_argument("--keep-temp-configs", action="store_true", help="保留临时 OpenClaw 配置文件")
```

- [x] **Step 4: Remove dynamic file loading from `benchmark_rl.py`**

```python
try:
    from benchmarking.datasets import BenchmarkRecord, classify_subset, load_records
    from benchmarking.evaluation import evaluate_record as evaluate_answer
    from benchmarking.reporting import aggregate_results, build_error_group_record_result
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from workspace.benchmarking.datasets import BenchmarkRecord, classify_subset, load_records
    from workspace.benchmarking.evaluation import evaluate_record as evaluate_answer
    from workspace.benchmarking.reporting import aggregate_results, build_error_group_record_result
```

```python
# Delete these entirely
def load_benchmark_test_module() -> Any:
    ...


benchmark_test = load_benchmark_test_module()
```

- [x] **Step 5: Collapse `benchmark_test.py` to orchestration-only imports**

```python
from benchmarking.contracts import AnswerPayload, RunStatus, RunnerResult
from benchmarking.datasets import BenchmarkRecord, classify_subset, load_records
from benchmarking.evaluation import evaluate_record
from benchmarking.experiments import ExperimentSpec
from benchmarking.reporting import aggregate_results, build_error_group_record_result
from benchmarking.runners import build_runner
```

Keep only CLI parsing, wave scheduling, judge client, and report writing in `benchmark_test.py`.

- [x] **Step 6: Run full regression**

Run: `uv run python -m unittest tests.test_benchmark_contracts tests.test_benchmark_config_runtime tests.test_benchmark_datasets tests.test_benchmark_test tests.test_benchmark_rl -v`
Expected: `OK`

- [x] **Step 7: Commit**

```bash
git add benchmark_test.py benchmark_rl.py tests/test_benchmark_test.py tests/test_benchmark_rl.py
git commit -m "refactor: slim benchmark CLIs and use shared modules"
```

### Task 6: Add Reporting Compatibility Checks and Finish the Migration

**Files:**
- Create: `benchmarking/reporting.py`
- Modify: `benchmark_test.py`
- Modify: `benchmark_rl.py`
- Test: `tests/test_benchmark_test.py`
- Test: `tests/test_benchmark_rl.py`

- [x] **Step 1: Write the failing regression tests**

```python
def test_results_json_keeps_legacy_top_level_shape(self) -> None:
    summary = benchmark_test.aggregate_results(sample_results)
    self.assertIn("group_order", summary)
    self.assertIn("groups", summary)
    self.assertIn("group_subset", summary)

def test_group_record_result_preserves_error_and_runner_meta_fields(self) -> None:
    entry = benchmark_test.build_error_group_record_result(
        group=benchmark_test.EXPERIMENT_GROUPS["single_llm_web_off"],
        record=record,
        error_message="runner failed",
        runner_meta={"traceback": "demo"},
    )
    self.assertEqual("runner failed", entry.error)
    self.assertEqual("demo", entry.runner_meta["traceback"])
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_benchmark_test tests.test_benchmark_rl -v`
Expected: failures while shared reporting helpers do not exist

- [x] **Step 3: Extract shared reporting helpers**

```python
# benchmarking/reporting.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any


def aggregate_results(results: list[GroupRecordResult]) -> dict[str, Any]:
    ...


def materialize_group_failure_results(...):
    ...


def build_error_group_record_result(...):
    ...
```

Move `aggregate_results`, `materialize_group_failure_results`, `build_error_group_record_result`, `average_optional_metric`, and `aggregate_bucket` here without changing their serialized output.

- [x] **Step 4: Run full regression**

Run: `uv run python -m unittest tests.test_benchmark_contracts tests.test_benchmark_config_runtime tests.test_benchmark_datasets tests.test_benchmark_test tests.test_benchmark_rl -v`
Expected: `OK`

- [x] **Step 5: Commit**

```bash
git add benchmarking/reporting.py benchmark_test.py benchmark_rl.py tests/test_benchmark_test.py tests/test_benchmark_rl.py
git commit -m "refactor: extract shared benchmark reporting"
```

## Rollout Notes

- Do not remove the old `BenchmarkRecord` dataclass from `benchmark_test.py` until both CLI entrypoints use the new `benchmarking.datasets.BenchmarkRecord`.
- Keep `evaluate_answer()` as a thin adapter during Tasks 3-4, then delete it after Task 5 once all callers use `evaluate_record()`.
- `ChemQARunner` recovery runs should be emitted into output artifacts for auditability, but must remain unscored unless an explicit future policy says otherwise.
- If a compatibility adapter grows beyond 20-30 lines, stop and migrate the remaining caller in the same task instead of creating a second hidden API layer.

## Verification Checklist

- `benchmark_test.py` no longer contains runner implementation classes.
- `benchmark_rl.py` no longer dynamically loads `benchmark_test.py`.
- Config rendering functions do not write files.
- Provisioning functions do not mutate config payloads.
- No CLI flag exists without an observable effect in tests.
- Non-success ChemQA runs no longer produce `error=None` scored entries.

## Completion Status

Completed on branch `benchmark-refactor`.

- Task 1 committed as `ea76d04 refactor: add explicit benchmark contracts`.
- Task 2 committed as `2df18d2 refactor: split benchmark config rendering from provisioning`.
- Task 3 committed as `b48cd40 refactor: add validated benchmark record loading and evaluator registry`.
- Task 4 committed as `5f286c5 refactor: make benchmark runner status explicit`.
- Task 5 committed as `95217d8 refactor: slim benchmark CLIs and use shared modules`.
- Task 6 committed as `f41048d refactor: extract shared benchmark reporting`.

Final verification performed after Task 6:

- `uv run python -m py_compile benchmarking/reporting.py benchmark_test.py benchmark_rl.py` passed.
- Task 6 focused reporting tests passed.
- `uv run python -m unittest tests.test_benchmark_contracts tests.test_benchmark_config_runtime tests.test_benchmark_datasets -v` passed.
- `uv run python -m unittest tests.test_benchmark_test tests.test_benchmark_rl -v` ran 90 tests: 89 passed, with one known environment/fixture failure in `test_evaluate_conformabench_constructive_handles_current_rdkit_environment` caused by missing `rdkit` and missing ConformaBench hidden judge spec at `benchmarks/conformabench/items/conformabench-0001/hidden_judge_spec.yaml`.
