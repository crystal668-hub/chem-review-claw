# Benchmark Result Contract and Skill Runtime Health Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent malformed OpenClaw stdout from becoming scoreable benchmark answers, and make benchmark-advertised chemistry skills execute through a fixed workspace `uv run` runtime with startup health checks.

**Architecture:** Add a strict result-contract adapter between OpenClaw stdout and `SingleLLMRunner`, so only schema-valid `payloads` reach answer extraction and evaluator input. Add a benchmark skill runtime layer that provides `uv run` script execution, structured unavailable/failure payloads, and a startup health report used to filter advertised skills and record diagnostics.

**Tech Stack:** Python 3.12, `uv run`, pytest, existing OpenClaw CLI wrapper, existing benchmark `ExperimentSpec` and runtime config rendering.

---

## Scope

This plan modifies the `skill-autonomous-discovery-audit` worktree at `/Users/xutao/.config/superpowers/worktrees/workspace/skill-autonomous-discovery-audit`.

This plan covers two required behaviors:

1. Strict stdout/result schema validation:
   - Valid answer payload source shape is `{"result": {"payloads": [{"text": "..."}], "meta": {...}}}` or the equivalent unwrapped top-level `{"payloads": [{"text": "..."}], "meta": {...}}`.
   - `payloads` must be a list of dicts.
   - Each payload item used for answer extraction must contain non-empty string `text`.
   - Non-conforming stdout is retained only in diagnostics and never passed to `summarize_payloads()` or the evaluator.
   - If no valid answer source remains, the single-LLM runner returns `RunStatus.FAILED` with `failure.code == "agent_result_contract_invalid"` instead of scoring an empty answer.

2. Fixed skill runtime and health checks:
   - Benchmark skill script execution is documented and routed through a workspace runner that invokes target scripts with `uv run python`.
   - Benchmark startup checks each allowlisted skill for declared Python imports, executables, API keys, data files, and network providers.
   - Unavailable skills are removed from the effective skills-on allowlist before config rendering and are reported in runtime manifest and per-run audit metadata.
   - Script execution failures through the runner return structured JSON with `available=false`, `error_kind`, and `reason`.

Transcript answer recovery is intentionally not implemented in this plan. The contract fix prevents invalid stdout from corrupting answer scoring; transcript fallback can be planned separately if we want invalid-stdout records to remain scoreable from session files.

## File Structure

- Create `benchmarking/result_contract.py`
  - Owns strict OpenClaw agent result schema validation.
  - Produces a normalized `AgentResultContract` object with valid payloads, meta, and diagnostics.

- Modify `benchmarking/single_llm_openclaw_wrapper.py`
  - Replaces broad `safe_json_extract()` result handling with schema-aware stdout parsing.
  - Emits invalid stdout only under `result.meta.stdout_diagnostics`.

- Modify `benchmarking/runners/single_llm.py`
  - Uses the result contract before answer extraction.
  - Fails non-scoreably when stdout does not contain schema-valid payloads.

- Create `benchmarking/skill_runtime.py`
  - Provides `WorkspaceUvSkillRunner`.
  - Builds `uv run python <script> ...` commands.
  - Normalizes process, import, timeout, and provider failures into structured payloads.

- Create `benchmarking/skill_health.py`
  - Defines health requirement dataclasses and startup health checks.
  - Covers every benchmark allowlisted skill via default checks plus explicit per-skill overrides.

- Modify `benchmarking/skill_tree.py`
  - Adds available-skill filtering for rendered discovery text.
  - Keeps the full inventory loader unchanged.

- Modify `benchmarking/prompts.py`
  - Accepts an optional health/available skill context and no longer says all benchmark skills are available when startup health filtered them.

- Modify `benchmark_test.py`
  - Allows effective experiment specs to replace static skill allowlists after health check before runtime config generation.
  - Writes `skill-health.json` and includes the summary in `runtime-manifest.json`.

- Modify `pyproject.toml`
  - Adds direct dependencies used by benchmark-visible skill scripts that currently fail when imports are missing: `beautifulsoup4` for `bs4` and `sympy`.

- Add tests:
  - `tests/test_benchmark_result_contract.py`
  - `tests/test_skill_runtime.py`
  - `tests/test_skill_health.py`
  - Extend `tests/test_single_llm_session_wrapper.py`
  - Extend `tests/test_benchmark_test.py`
  - Extend `tests/test_benchmark_skill_tree.py`

---

### Task 1: Strict OpenClaw Result Contract

**Files:**
- Create: `benchmarking/result_contract.py`
- Test: `tests/test_benchmark_result_contract.py`

- [ ] **Step 1: Write failing result-contract tests**

Create `tests/test_benchmark_result_contract.py`:

```python
from __future__ import annotations

import json

from benchmarking.result_contract import normalize_agent_result_payload, parse_agent_stdout


def test_accepts_wrapped_result_payloads() -> None:
    payload = {"result": {"payloads": [{"text": "Reasoning\nFINAL ANSWER: 7.59"}], "meta": {"toolSummary": {"calls": 1}}}}

    result = normalize_agent_result_payload(payload)

    assert result.valid is True
    assert result.payloads == [{"text": "Reasoning\nFINAL ANSWER: 7.59"}]
    assert result.meta["toolSummary"]["calls"] == 1
    assert result.diagnostics["schema_valid"] is True


def test_accepts_unwrapped_result_payloads() -> None:
    payload = {"payloads": [{"text": "FINAL ANSWER: A"}], "meta": {}}

    result = normalize_agent_result_payload(payload)

    assert result.valid is True
    assert result.payloads == [{"text": "FINAL ANSWER: A"}]


def test_rejects_tool_argument_object_as_answer_payload() -> None:
    payload = {"query": "benzene NMR", "count": 5, "region": "us-en", "meta": {"session_isolation": {"session_isolation_ok": True}}}

    result = normalize_agent_result_payload(payload)

    assert result.valid is False
    assert result.payloads == []
    assert result.meta == {"session_isolation": {"session_isolation_ok": True}}
    assert result.diagnostics["schema_valid"] is False
    assert result.diagnostics["reason"] == "missing_payloads"
    assert result.diagnostics["invalid_stdout_payload"]["query"] == "benzene NMR"


def test_rejects_payloads_with_non_text_items() -> None:
    payload = {"result": {"payloads": [{"tool": "exec"}, {"text": ""}], "meta": {}}}

    result = normalize_agent_result_payload(payload)

    assert result.valid is False
    assert result.payloads == []
    assert result.diagnostics["reason"] == "invalid_payload_item"


def test_parse_agent_stdout_recovers_valid_embedded_agent_result_only() -> None:
    stdout = "warning before json\n" + json.dumps({"result": {"payloads": [{"text": "FINAL ANSWER: B"}], "meta": {}}})

    result = parse_agent_stdout(stdout)

    assert result.valid is True
    assert result.payloads == [{"text": "FINAL ANSWER: B"}]
    assert result.diagnostics["parse_mode"] == "embedded_agent_result"


def test_parse_agent_stdout_does_not_promote_embedded_tool_json() -> None:
    stdout = "tool event\n" + json.dumps({"query": "bad candidate", "count": 3})

    result = parse_agent_stdout(stdout)

    assert result.valid is False
    assert result.payloads == []
    assert result.diagnostics["schema_valid"] is False
    assert result.diagnostics["parse_mode"] == "diagnostic_only"
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
uv run pytest tests/test_benchmark_result_contract.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarking.result_contract'`.

- [ ] **Step 3: Implement `benchmarking/result_contract.py`**

Create `benchmarking/result_contract.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentResultContract:
    valid: bool
    payloads: list[dict[str, str]]
    meta: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def normalize_agent_result_payload(payload: Any) -> AgentResultContract:
    if not isinstance(payload, dict):
        return _invalid("not_object", payload, meta={})

    candidate = payload.get("result") if isinstance(payload.get("result"), dict) else payload
    meta = candidate.get("meta") if isinstance(candidate, dict) and isinstance(candidate.get("meta"), dict) else {}
    if not isinstance(candidate, dict) or "payloads" not in candidate:
        return _invalid("missing_payloads", payload, meta=meta)

    raw_payloads = candidate.get("payloads")
    if not isinstance(raw_payloads, list):
        return _invalid("payloads_not_list", payload, meta=meta)

    normalized: list[dict[str, str]] = []
    for item in raw_payloads:
        if not isinstance(item, dict):
            return _invalid("invalid_payload_item", payload, meta=meta)
        text = str(item.get("text") or "").strip()
        if not text:
            return _invalid("invalid_payload_item", payload, meta=meta)
        normalized.append({"text": text})

    return AgentResultContract(
        valid=True,
        payloads=normalized,
        meta=dict(meta),
        diagnostics={"schema_valid": True, "payload_count": len(normalized)},
    )


def parse_agent_stdout(output: str) -> AgentResultContract:
    stripped = str(output or "").strip()
    if not stripped:
        return AgentResultContract(
            valid=False,
            payloads=[],
            meta={},
            diagnostics={"schema_valid": False, "reason": "empty_output", "parse_mode": "diagnostic_only"},
        )

    try:
        direct = json.loads(stripped)
    except json.JSONDecodeError:
        direct = None
    if direct is not None:
        result = normalize_agent_result_payload(direct)
        result.diagnostics["parse_mode"] = "direct_json"
        return result

    for candidate in _embedded_json_objects(stripped):
        result = normalize_agent_result_payload(candidate)
        if result.valid:
            result.diagnostics["parse_mode"] = "embedded_agent_result"
            return result

    return AgentResultContract(
        valid=False,
        payloads=[],
        meta={},
        diagnostics={
            "schema_valid": False,
            "reason": "no_valid_agent_result_json",
            "parse_mode": "diagnostic_only",
            "stdout_excerpt": stripped[:4000],
        },
    )


def contract_to_payload(contract: AgentResultContract) -> dict[str, Any]:
    return {
        "result": {
            "payloads": contract.payloads,
            "meta": {
                **contract.meta,
                "stdout_diagnostics": dict(contract.diagnostics),
            },
        }
    }


def _invalid(reason: str, payload: Any, *, meta: dict[str, Any]) -> AgentResultContract:
    diagnostics: dict[str, Any] = {"schema_valid": False, "reason": reason}
    if isinstance(payload, dict):
        diagnostics["invalid_stdout_payload"] = payload
    else:
        diagnostics["invalid_stdout_payload_repr"] = repr(payload)[:1000]
    return AgentResultContract(valid=False, payloads=[], meta=dict(meta), diagnostics=diagnostics)


def _embedded_json_objects(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    candidates: list[Any] = []
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        candidates.append(value)
    return candidates
```

- [ ] **Step 4: Run tests and confirm they pass**

Run:

```bash
uv run pytest tests/test_benchmark_result_contract.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add benchmarking/result_contract.py tests/test_benchmark_result_contract.py
git commit -m "feat: add strict benchmark result contract"
```

---

### Task 2: Apply Result Contract to Single-LLM Wrapper and Runner

**Files:**
- Modify: `benchmarking/single_llm_openclaw_wrapper.py`
- Modify: `benchmarking/runners/single_llm.py`
- Test: `tests/test_single_llm_session_wrapper.py`
- Test: `tests/test_benchmark_test.py`

- [ ] **Step 1: Add wrapper tests for invalid stdout diagnostics**

Append to `tests/test_single_llm_session_wrapper.py`:

```python
    def test_main_keeps_invalid_stdout_as_diagnostics_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = self.write_config(root)
            fake_args = argparse.Namespace(
                agent="benchmark-single",
                config_file=str(config_path),
                session_id="session-a",
                message="Q",
                thinking="high",
                timeout=30,
                json=True,
            )
            completed = subprocess.CompletedProcess(
                ["openclaw"],
                0,
                stdout=json.dumps({"query": "not an agent result", "count": 3}),
                stderr="",
            )
            audit = {
                "requested_session_id": "session-a",
                "agent_id": "benchmark-single",
                "session_store_path": str(root / "sessions.json"),
                "preflight_removed_stale_main_entry": False,
                "preflight_previous_session_id": "",
                "postflight_entry_session_id": "session-a",
                "postflight_entry_session_file": str(root / "session-a.jsonl"),
                "session_isolation_ok": True,
            }

            with mock.patch.object(wrapper, "parse_args", return_value=fake_args):
                with mock.patch.object(wrapper, "reset_agent_main_session_if_stale", return_value=audit):
                    with mock.patch.object(wrapper, "run_openclaw", return_value=completed):
                        with mock.patch.object(wrapper, "inspect_postflight_session", return_value=audit):
                            with mock.patch("builtins.print") as print_mock:
                                exit_code = wrapper.main()

            self.assertEqual(0, exit_code)
            payload = json.loads(print_mock.call_args.args[0])
            result = payload["result"]
            self.assertEqual([], result["payloads"])
            diagnostics = result["meta"]["stdout_diagnostics"]
            self.assertFalse(diagnostics["schema_valid"])
            self.assertEqual("missing_payloads", diagnostics["reason"])
            self.assertEqual("not an agent result", diagnostics["invalid_stdout_payload"]["query"])
            self.assertTrue(result["meta"]["session_isolation"]["session_isolation_ok"])
```

- [ ] **Step 2: Add runner test for non-scoreable invalid contract**

Append to the `BenchmarkTestCase` class in `tests/test_benchmark_test.py` near existing `SingleLLMRunner` tests:

```python
    def test_single_llm_runner_rejects_invalid_stdout_payloads_without_scoring_empty_answer(self) -> None:
        original_run_subprocess = benchmark_test.run_subprocess
        original_ensure_runtime_bundle = benchmark_test.ensure_runtime_bundle
        try:
            benchmark_test.ensure_runtime_bundle = lambda record, bundle_root: None

            def fake_run_subprocess(command: list[str], *, env=None, cwd=None, timeout=None):
                return benchmark_test.subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "result": {
                                "payloads": [],
                                "meta": {
                                    "stdout_diagnostics": {
                                        "schema_valid": False,
                                        "reason": "missing_payloads",
                                        "invalid_stdout_payload": {"query": "tool args"},
                                    },
                                    "session_isolation": {"session_isolation_ok": True},
                                },
                            }
                        }
                    ),
                    stderr="",
                )

            benchmark_test.run_subprocess = fake_run_subprocess
            runner = benchmark_test.SingleLLMRunner(
                agent_id="benchmark-single-skills-on",
                timeout_seconds=30,
                config_path=Path("/tmp/single.json"),
                runtime_bundle_root=Path("/tmp"),
            )
            record = benchmark_test.BenchmarkRecord(
                record_id="demo",
                dataset="chembench",
                source_file="/tmp/demo.jsonl",
                eval_kind="chembench_open_ended",
                prompt="What is 2+3?",
                reference_answer="5",
                payload={},
            )

            out = runner.run(record, benchmark_test.EXPERIMENT_GROUPS["single_llm_skills_on"])

            self.assertEqual(benchmark_test.RunStatus.FAILED, out.status)
            assert out.failure is not None
            self.assertEqual("agent_result_contract_invalid", out.failure.code)
            self.assertEqual("", out.answer.full_response_text)
            self.assertFalse(out.should_score())
            self.assertEqual("missing_payloads", out.runner_meta["stdout_diagnostics"]["reason"])
        finally:
            benchmark_test.run_subprocess = original_run_subprocess
            benchmark_test.ensure_runtime_bundle = original_ensure_runtime_bundle
```

- [ ] **Step 3: Run targeted tests and confirm failure**

Run:

```bash
uv run pytest tests/test_single_llm_session_wrapper.py::SingleLLMSessionWrapperTests::test_main_keeps_invalid_stdout_as_diagnostics_only tests/test_benchmark_test.py::BenchmarkTestCase::test_single_llm_runner_rejects_invalid_stdout_payloads_without_scoring_empty_answer -q
```

Expected: FAIL because wrapper still promotes any parsed dict and runner still treats empty payloads as completed.

- [ ] **Step 4: Update wrapper to use result contract**

Modify imports in `benchmarking/single_llm_openclaw_wrapper.py`:

```python
from benchmarking.result_contract import contract_to_payload, parse_agent_stdout
```

Replace `parse_openclaw_json_output()` body:

```python
def parse_openclaw_json_output(output: str) -> Any:
    return contract_to_payload(parse_agent_stdout(output))
```

Keep `merge_isolation_audit()` unchanged so session isolation metadata is merged into the normalized payload.

- [ ] **Step 5: Update runner to fail invalid contracts**

Modify `benchmarking/runners/single_llm.py` after `result_payload = self._unwrap_agent_payload(payload)`:

```python
        runner_meta = dict(result_payload.get("meta") or {})
        stdout_diagnostics = runner_meta.get("stdout_diagnostics")
        if isinstance(stdout_diagnostics, dict) and stdout_diagnostics.get("schema_valid") is False:
            message = (
                "Single-LLM OpenClaw stdout did not contain a schema-valid agent result payload: "
                + str(stdout_diagnostics.get("reason") or "invalid_stdout")
            )
            runner_meta["error"] = message
            runner_meta["stdout_diagnostics"] = dict(stdout_diagnostics)
            runner_meta["skill_use_audit"] = build_skill_use_audit(
                skills_enabled=bool(getattr(group, "skills_enabled", True)),
                configured_skills=self.configured_skills,
                runner_meta=runner_meta,
                final_response_text="",
            )
            if input_bundle is not None:
                runner_meta["runtime_bundle"] = input_bundle.to_meta()
            return RunnerResult(
                status=RunStatus.FAILED,
                answer=AnswerPayload(),
                raw=payload,
                runner_meta=runner_meta,
                failure=FailureInfo(
                    code="agent_result_contract_invalid",
                    message=message,
                    details=dict(stdout_diagnostics),
                ),
            )
```

Then keep the existing payload summarization path, but remove the later duplicate `runner_meta = dict(result_payload.get("meta") or {})` assignment.

- [ ] **Step 6: Run targeted tests**

Run:

```bash
uv run pytest tests/test_benchmark_result_contract.py tests/test_single_llm_session_wrapper.py tests/test_benchmark_test.py::BenchmarkTestCase::test_single_llm_runner_rejects_invalid_stdout_payloads_without_scoring_empty_answer tests/test_benchmark_test.py::BenchmarkTestCase::test_single_llm_runner_invokes_openclaw_with_high_thinking -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

Run:

```bash
git add benchmarking/single_llm_openclaw_wrapper.py benchmarking/runners/single_llm.py tests/test_single_llm_session_wrapper.py tests/test_benchmark_test.py
git commit -m "fix: reject invalid single-llm stdout payloads"
```

---

### Task 3: Fixed Workspace `uv run` Skill Script Runner

**Files:**
- Create: `benchmarking/skill_runtime.py`
- Create: `scripts/run_skill.py`
- Test: `tests/test_skill_runtime.py`

- [ ] **Step 1: Write failing skill-runtime tests**

Create `tests/test_skill_runtime.py`:

```python
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from benchmarking.skill_runtime import WorkspaceUvSkillRunner, classify_skill_process_failure


def test_builds_workspace_uv_python_command() -> None:
    runner = WorkspaceUvSkillRunner(workspace_root=Path("/repo"), uv_executable="/usr/bin/uv")

    command = runner.build_command(Path("/repo/skills/chem-calculator/scripts/ksp_solver.py"), ["--json"])

    assert command == ["/usr/bin/uv", "run", "python", "/repo/skills/chem-calculator/scripts/ksp_solver.py", "--json"]


def test_missing_module_stderr_is_structured_missing_dependency() -> None:
    payload = classify_skill_process_failure(
        returncode=1,
        stdout="",
        stderr="ModuleNotFoundError: No module named 'bs4'",
        command=["uv", "run", "python", "script.py"],
    )

    assert payload["available"] is False
    assert payload["error_kind"] == "missing_dependency"
    assert payload["reason"] == "missing Python module: bs4"


def test_http_provider_error_is_structured_provider_failure() -> None:
    payload = classify_skill_process_failure(
        returncode=1,
        stdout="",
        stderr="requests.exceptions.HTTPError: 403 Client Error: Forbidden",
        command=["uv", "run", "python", "script.py"],
    )

    assert payload["available"] is False
    assert payload["error_kind"] == "provider_failure"
    assert "403" in payload["reason"]


def test_runner_returns_structured_invalid_output_payload() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        script = root / "script.py"
        script.write_text("print('not json')\n", encoding="utf-8")

        def fake_run(command, *, cwd=None, env=None, text=None, capture_output=None, check=None, timeout=None):
            return subprocess.CompletedProcess(command, 0, stdout="not json\n", stderr="")

        runner = WorkspaceUvSkillRunner(workspace_root=root, uv_executable="uv", run_subprocess=fake_run)
        payload = runner.run_script(script, [])

    assert payload["available"] is False
    assert payload["error_kind"] == "invalid_output"
    assert payload["command"][:3] == ["uv", "run", "python"]
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
uv run pytest tests/test_skill_runtime.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarking.skill_runtime'`.

- [ ] **Step 3: Implement `benchmarking/skill_runtime.py`**

Create `benchmarking/skill_runtime.py`:

```python
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


RunSubprocess = Callable[..., subprocess.CompletedProcess[str]]


MODULE_NOT_FOUND_RE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")


@dataclass(frozen=True)
class WorkspaceUvSkillRunner:
    workspace_root: Path
    uv_executable: str | None = None
    run_subprocess: RunSubprocess = subprocess.run
    timeout_seconds: int = 60

    def resolved_uv(self) -> str:
        executable = self.uv_executable or shutil.which("uv")
        if not executable:
            raise FileNotFoundError("uv executable not found in PATH")
        return executable

    def build_command(self, script_path: Path, args: list[str]) -> list[str]:
        return [self.resolved_uv(), "run", "python", str(script_path), *args]

    def run_script(self, script_path: Path, args: list[str], *, env: dict[str, str] | None = None) -> dict[str, Any]:
        command = self.build_command(script_path, args)
        merged_env = os.environ.copy()
        merged_env["PYTHONNOUSERSITE"] = "1"
        merged_env["OPENCLAW_SKILL_RUNNER"] = "workspace_uv"
        if env:
            merged_env.update(env)
        try:
            completed = self.run_subprocess(
                command,
                cwd=str(self.workspace_root),
                env=merged_env,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            return _unavailable("missing_executable", str(exc), command=command)
        except subprocess.TimeoutExpired:
            return _unavailable("provider_failure", f"skill script timed out after {self.timeout_seconds}s", command=command)

        if completed.returncode != 0:
            return classify_skill_process_failure(
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                command=command,
            )
        try:
            payload = json.loads((completed.stdout or "").strip())
        except json.JSONDecodeError:
            return _unavailable(
                "invalid_output",
                "skill script completed but did not emit a JSON object",
                command=command,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        if not isinstance(payload, dict):
            return _unavailable("invalid_output", "skill script JSON output was not an object", command=command)
        payload.setdefault("available", True)
        payload.setdefault("runner", "workspace_uv")
        payload.setdefault("command", command)
        return payload


def classify_skill_process_failure(*, returncode: int, stdout: str, stderr: str, command: list[str]) -> dict[str, Any]:
    combined = f"{stdout}\n{stderr}"
    missing = MODULE_NOT_FOUND_RE.search(combined)
    if missing:
        return _unavailable(
            "missing_dependency",
            f"missing Python module: {missing.group(1)}",
            command=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )
    if any(token in combined for token in ("403", "401", "HTTPError", "SSLError", "ConnectionError", "fetch failed")):
        return _unavailable(
            "provider_failure",
            _single_line(combined) or "external provider request failed",
            command=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )
    return _unavailable(
        "script_error",
        _single_line(combined) or f"skill script exited with return code {returncode}",
        command=command,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _unavailable(
    error_kind: str,
    reason: str,
    *,
    command: list[str],
    returncode: int | None = None,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "available": False,
        "error_kind": error_kind,
        "reason": reason,
        "runner": "workspace_uv",
        "command": command,
    }
    if returncode is not None:
        payload["returncode"] = returncode
    if stdout:
        payload["stdout_excerpt"] = stdout[:2000]
    if stderr:
        payload["stderr_excerpt"] = stderr[:2000]
    return payload


def _single_line(text: str) -> str:
    return " ".join(str(text or "").split())[:500]
```

- [ ] **Step 4: Add CLI entrypoint for agent-visible skill execution**

Create `scripts/run_skill.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarking.skill_runtime import WorkspaceUvSkillRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an OpenClaw chemistry skill script through workspace uv.")
    parser.add_argument("--workspace-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--script", required=True, help="Absolute or workspace-relative script path.")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    script_path = Path(args.script).expanduser()
    if not script_path.is_absolute():
        script_path = (workspace_root / script_path).resolve()
    script_args = list(args.script_args)
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]
    payload = WorkspaceUvSkillRunner(workspace_root=workspace_root).run_script(script_path, script_args)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload.get("available") is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests**

Run:

```bash
uv run pytest tests/test_skill_runtime.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add benchmarking/skill_runtime.py scripts/run_skill.py tests/test_skill_runtime.py
git commit -m "feat: add workspace uv skill runner"
```

---

### Task 4: Startup Skill Health Checks

**Files:**
- Create: `benchmarking/skill_health.py`
- Modify: `pyproject.toml`
- Test: `tests/test_skill_health.py`

- [ ] **Step 1: Write failing health-check tests**

Create `tests/test_skill_health.py`:

```python
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from benchmarking.skill_health import (
    HealthRequirement,
    check_skill_health,
    health_requirements_for_allowlist,
    summarize_skill_health,
)
from benchmarking.skill_tree import benchmark_skill_allowlist


def test_health_requirements_cover_every_allowlisted_skill() -> None:
    requirements = health_requirements_for_allowlist(benchmark_skill_allowlist())

    assert set(requirements) == set(benchmark_skill_allowlist())


def test_missing_python_module_marks_skill_unavailable() -> None:
    def fake_run(command, *, cwd=None, env=None, text=None, capture_output=None, check=None, timeout=None):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="ModuleNotFoundError: No module named 'bs4'")

    requirement = HealthRequirement(skill="paper-access", python_modules=("bs4",))
    report = check_skill_health(requirement, workspace_root=Path("/repo"), run_subprocess=fake_run)

    assert report["skill"] == "paper-access"
    assert report["available"] is False
    assert report["checks"]["python_modules"]["bs4"]["ok"] is False
    assert report["unavailable_reasons"][0]["kind"] == "missing_dependency"


def test_missing_api_key_marks_skill_unavailable() -> None:
    requirement = HealthRequirement(skill="materials-project", api_keys=("MP_API_KEY",))
    report = check_skill_health(requirement, workspace_root=Path("/repo"), env={})

    assert report["available"] is False
    assert report["checks"]["api_keys"]["MP_API_KEY"]["ok"] is False
    assert report["unavailable_reasons"][0]["kind"] == "missing_api_key"


def test_missing_data_file_marks_skill_unavailable() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        requirement = HealthRequirement(skill="demo", data_files=("skills/demo/SKILL.md",))
        report = check_skill_health(requirement, workspace_root=root)

    assert report["available"] is False
    assert report["checks"]["data_files"]["skills/demo/SKILL.md"]["ok"] is False
    assert report["unavailable_reasons"][0]["kind"] == "missing_data_file"


def test_summary_lists_available_and_unavailable_skills() -> None:
    reports = {
        "ok-skill": {"skill": "ok-skill", "available": True, "unavailable_reasons": []},
        "bad-skill": {"skill": "bad-skill", "available": False, "unavailable_reasons": [{"kind": "missing_dependency"}]},
    }

    summary = summarize_skill_health(reports)

    assert summary["available_skill_count"] == 1
    assert summary["unavailable_skill_count"] == 1
    assert summary["available_skills"] == ["ok-skill"]
    assert summary["unavailable_skills"] == ["bad-skill"]
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
uv run pytest tests/test_skill_health.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarking.skill_health'`.

- [ ] **Step 3: Add direct dependencies for the workspace uv environment**

Modify `pyproject.toml` `[project].dependencies`:

```toml
dependencies = [
    "PyYAML==6.0.3",
    "requests==2.33.1",
    "beautifulsoup4==4.14.3",
    "sympy==1.14.0",
]
```

Run:

```bash
uv lock
```

Expected: `uv.lock` updates successfully.

- [ ] **Step 4: Implement health checks**

Create `benchmarking/skill_health.py`:

```python
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


RunSubprocess = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class HealthRequirement:
    skill: str
    python_modules: tuple[str, ...] = ()
    executables: tuple[str, ...] = ()
    api_keys: tuple[str, ...] = ()
    data_files: tuple[str, ...] = ()
    network_urls: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


REQUIREMENT_OVERRIDES: dict[str, HealthRequirement] = {
    "rdkit": HealthRequirement(skill="rdkit", python_modules=("rdkit",), data_files=("skills/rdkit/SKILL.md",)),
    "chem-calculator": HealthRequirement(skill="chem-calculator", python_modules=("sympy",), data_files=("skills/chem-calculator/SKILL.md",)),
    "paper-retrieval": HealthRequirement(
        skill="paper-retrieval",
        python_modules=("requests",),
        data_files=("skills/paper-retrieval/SKILL.md",),
        network_urls=("https://api.openalex.org/works?per-page=1", "https://api.crossref.org/works?rows=0"),
    ),
    "paper-access": HealthRequirement(
        skill="paper-access",
        python_modules=("requests", "bs4"),
        data_files=("skills/paper-access/SKILL.md",),
        network_urls=("https://api.crossref.org/works?rows=0",),
    ),
    "paper-parse": HealthRequirement(
        skill="paper-parse",
        python_modules=("fitz",),
        executables=("pdfinfo",),
        data_files=("skills/paper-parse/SKILL.md",),
    ),
    "pubchem": HealthRequirement(
        skill="pubchem",
        python_modules=("requests",),
        data_files=("skills/pubchem/SKILL.md",),
        network_urls=("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/water/property/MolecularFormula/JSON",),
    ),
    "materials-project": HealthRequirement(
        skill="materials-project",
        python_modules=("mp_api",),
        api_keys=("MP_API_KEY",),
        data_files=("skills/materials-project/SKILL.md",),
    ),
    "pymatgen": HealthRequirement(skill="pymatgen", python_modules=("pymatgen",), data_files=("skills/pymatgen/SKILL.md",)),
    "ase": HealthRequirement(skill="ase", python_modules=("ase",), data_files=("skills/ase/SKILL.md",)),
    "cclib": HealthRequirement(skill="cclib", python_modules=("cclib",), data_files=("skills/cclib/SKILL.md",)),
    "chembl-database": HealthRequirement(
        skill="chembl-database",
        python_modules=("chembl_webresource_client",),
        data_files=("skills/chembl-database/SKILL.md",),
        network_urls=("https://www.ebi.ac.uk/chembl/api/data/status.json",),
    ),
}


def health_requirements_for_allowlist(allowlist: Iterable[str]) -> dict[str, HealthRequirement]:
    requirements: dict[str, HealthRequirement] = {}
    for skill in allowlist:
        key = str(skill)
        requirements[key] = REQUIREMENT_OVERRIDES.get(
            key,
            HealthRequirement(skill=key, data_files=(f"skills/{key}/SKILL.md",)),
        )
    return requirements


def check_skill_health(
    requirement: HealthRequirement,
    *,
    workspace_root: Path,
    env: dict[str, str] | None = None,
    run_subprocess: RunSubprocess = subprocess.run,
    network_timeout_seconds: int = 3,
) -> dict[str, Any]:
    environment = os.environ.copy()
    if env is not None:
        environment = dict(env)
    checks: dict[str, Any] = {"python_modules": {}, "executables": {}, "api_keys": {}, "data_files": {}, "network": {}}
    unavailable: list[dict[str, str]] = []

    for module in requirement.python_modules:
        command = ["uv", "run", "python", "-c", f"import {module}"]
        completed = run_subprocess(command, cwd=str(workspace_root), env=environment, text=True, capture_output=True, check=False, timeout=network_timeout_seconds)
        ok = completed.returncode == 0
        checks["python_modules"][module] = {"ok": ok, "returncode": completed.returncode}
        if not ok:
            unavailable.append({"kind": "missing_dependency", "name": module, "reason": (completed.stderr or completed.stdout).strip()[:500]})

    for executable in requirement.executables:
        ok = shutil.which(executable) is not None
        checks["executables"][executable] = {"ok": ok}
        if not ok:
            unavailable.append({"kind": "missing_executable", "name": executable, "reason": f"{executable} not found in PATH"})

    for key in requirement.api_keys:
        ok = bool(str(environment.get(key) or "").strip())
        checks["api_keys"][key] = {"ok": ok}
        if not ok:
            unavailable.append({"kind": "missing_api_key", "name": key, "reason": f"{key} is not set"})

    for relative_path in requirement.data_files:
        ok = (workspace_root / relative_path).is_file()
        checks["data_files"][relative_path] = {"ok": ok}
        if not ok:
            unavailable.append({"kind": "missing_data_file", "name": relative_path, "reason": f"{relative_path} does not exist"})

    for url in requirement.network_urls:
        command = [
            "uv",
            "run",
            "python",
            "-c",
            "import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=3).read(64)",
            url,
        ]
        completed = run_subprocess(command, cwd=str(workspace_root), env=environment, text=True, capture_output=True, check=False, timeout=network_timeout_seconds + 2)
        ok = completed.returncode == 0
        checks["network"][url] = {"ok": ok, "returncode": completed.returncode}
        if not ok:
            unavailable.append({"kind": "provider_failure", "name": url, "reason": (completed.stderr or completed.stdout).strip()[:500]})

    return {
        "skill": requirement.skill,
        "available": not unavailable,
        "checks": checks,
        "unavailable_reasons": unavailable,
    }


def check_all_skill_health(
    allowlist: Iterable[str],
    *,
    workspace_root: Path,
    env: dict[str, str] | None = None,
    run_subprocess: RunSubprocess = subprocess.run,
) -> dict[str, dict[str, Any]]:
    return {
        skill: check_skill_health(requirement, workspace_root=workspace_root, env=env, run_subprocess=run_subprocess)
        for skill, requirement in health_requirements_for_allowlist(allowlist).items()
    }


def summarize_skill_health(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    available = sorted(skill for skill, report in reports.items() if report.get("available") is True)
    unavailable = sorted(skill for skill, report in reports.items() if report.get("available") is not True)
    return {
        "available_skill_count": len(available),
        "unavailable_skill_count": len(unavailable),
        "available_skills": available,
        "unavailable_skills": unavailable,
    }
```

- [ ] **Step 5: Run health tests**

Run:

```bash
uv run pytest tests/test_skill_health.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

Run:

```bash
git add pyproject.toml uv.lock benchmarking/skill_health.py tests/test_skill_health.py
git commit -m "feat: add benchmark skill health checks"
```

---

### Task 5: Wire Health Checks Into Benchmark Startup, Config, Prompt, and Audit

**Files:**
- Modify: `benchmark_test.py`
- Modify: `benchmarking/prompts.py`
- Modify: `benchmarking/skill_tree.py`
- Modify: `benchmarking/skill_audit.py`
- Test: `tests/test_benchmark_test.py`
- Test: `tests/test_benchmark_skill_tree.py`
- Test: `tests/test_benchmark_skill_audit.py`

- [ ] **Step 1: Add tests for effective allowlist filtering**

Add to `tests/test_benchmark_test.py`:

```python
    def test_effective_experiment_specs_filter_unavailable_skills(self) -> None:
        health_reports = {
            "rdkit": {"available": True},
            "paper-access": {"available": False, "unavailable_reasons": [{"kind": "missing_dependency", "name": "bs4"}]},
        }
        specs = {
            "single_llm_skills_on": benchmark_test.ExperimentSpec(
                id="single_llm_skills_on",
                label="demo",
                runner_kind="single_llm",
                websearch_enabled=True,
                skills_enabled=True,
                single_agent_id="benchmark-single-skills-on",
                skill_allowlist=("rdkit", "paper-access"),
            )
        }

        effective = benchmark_test.build_effective_experiment_specs(specs, skill_health_reports=health_reports)

        self.assertEqual(("rdkit",), effective["single_llm_skills_on"].skill_allowlist)
```

Add to `tests/test_benchmark_skill_tree.py`:

```python
def test_top_level_skill_tree_reflects_health_filtered_availability() -> None:
    rendered = render_top_level_skill_tree(available_skills={"rdkit", "paper-access"})

    assert "Only health-checked skills in this run are available" in rendered
    assert "molecular-structure-identity" in rendered
    assert "literature-evidence" in rendered
```

Add to `tests/test_benchmark_skill_audit.py`:

```python
def test_skill_use_audit_includes_health_summary() -> None:
    audit = build_skill_use_audit(
        skills_enabled=True,
        configured_skills=("rdkit",),
        runner_meta={},
        final_response_text="FINAL ANSWER: A",
        skill_health_summary={"available_skill_count": 1, "unavailable_skill_count": 2},
    )

    assert audit["skill_health_summary"]["available_skill_count"] == 1
    assert audit["skill_health_summary"]["unavailable_skill_count"] == 2
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
uv run pytest tests/test_benchmark_test.py::BenchmarkTestCase::test_effective_experiment_specs_filter_unavailable_skills tests/test_benchmark_skill_tree.py::test_top_level_skill_tree_reflects_health_filtered_availability tests/test_benchmark_skill_audit.py::test_skill_use_audit_includes_health_summary -q
```

Expected: FAIL because the new functions/signatures do not exist.

- [ ] **Step 3: Make runtime config context accept effective specs**

Modify `benchmark_test.py`:

```python
from dataclasses import asdict, dataclass, replace
```

Add helper:

```python
def build_effective_experiment_specs(
    specs: dict[str, ExperimentSpec],
    *,
    skill_health_reports: dict[str, dict[str, Any]],
) -> dict[str, ExperimentSpec]:
    available = {skill for skill, report in skill_health_reports.items() if report.get("available") is True}
    effective: dict[str, ExperimentSpec] = {}
    for group_id, spec in specs.items():
        if spec.skills_enabled and spec.skill_allowlist:
            filtered = tuple(skill for skill in spec.skill_allowlist if skill in available)
            effective[group_id] = replace(spec, skill_allowlist=filtered)
        else:
            effective[group_id] = spec
    return effective
```

Change `runtime_config_context()` signature:

```python
def runtime_config_context(experiment_specs: dict[str, ExperimentSpec] | None = None) -> RuntimeConfigContext:
```

Inside it, pass:

```python
experiment_specs=experiment_specs or EXPERIMENT_SPECS,
```

Change `ConfigPool.__init__()` to accept:

```python
experiment_specs: dict[str, ExperimentSpec] | None = None,
```

and pass:

```python
context=runtime_config_context(experiment_specs=experiment_specs),
```

- [ ] **Step 4: Run skill health at benchmark startup**

Modify imports in `benchmark_test.py`:

```python
from benchmarking.skill_health import check_all_skill_health, summarize_skill_health
```

In `main()`, after `ensure_dir(output_root)` and before `ConfigPool(...)`:

```python
    skill_health_reports = check_all_skill_health(BENCHMARK_SKILLS_ALLOWLIST, workspace_root=runtime_paths.project_root)
    skill_health_summary = summarize_skill_health(skill_health_reports)
    save_json(output_root / "skill-health.json", {"summary": skill_health_summary, "skills": skill_health_reports})
    effective_experiment_specs = build_effective_experiment_specs(
        EXPERIMENT_SPECS,
        skill_health_reports=skill_health_reports,
    )
```

Pass `experiment_specs=effective_experiment_specs` into `ConfigPool`.

In `run_group()`, add parameters:

```python
    experiment_specs: dict[str, ExperimentSpec],
    skill_health_summary: dict[str, Any],
```

Use:

```python
configured_skills=tuple(experiment_specs[group.id].skill_allowlist or ()),
skill_health_summary=skill_health_summary,
```

when building `SingleLLMRunner`.

Pass these values from `main()` to `executor.submit(...)`.

- [ ] **Step 5: Pass health summary through SingleLLMRunner and audit**

Modify `benchmarking/runners/single_llm.py` constructor:

```python
        skill_health_summary: dict[str, Any] | None = None,
```

Store:

```python
        self.skill_health_summary = dict(skill_health_summary or {})
```

Pass to `build_skill_use_audit(...)`:

```python
            skill_health_summary=self.skill_health_summary,
```

Modify `benchmark_test.SingleLLMRunner.__init__()` to accept and forward `skill_health_summary`.

Modify `benchmarking/skill_audit.py` signature:

```python
    skill_health_summary: dict[str, Any] | None = None,
```

and include:

```python
        "skill_health_summary": dict(skill_health_summary or {}),
```

- [ ] **Step 6: Make prompt tree reflect filtered availability**

Modify `benchmarking/skill_tree.py`:

```python
def render_top_level_skill_tree(available_skills: set[str] | None = None) -> str:
    lines = [
        "Skill capability tree:",
        "First choose a capability domain, then a skill family, then call a concrete skill only when it helps answer the record.",
    ]
    if available_skills is None:
        lines.append("All benchmark skills remain available; this tree is a discovery aid, not a router or allowlist filter.")
    else:
        lines.append("Only health-checked skills in this run are available; unavailable skills were omitted from the runtime allowlist.")
    for domain in SKILL_TREE:
        families: list[str] = []
        for family in domain["families"]:
            family_skills = set(str(skill) for skill in family["skills"])
            if available_skills is not None and not (family_skills & available_skills):
                continue
            families.append(f"`{family['id']}`")
        if families:
            lines.append(f"- `{domain['id']}`: {domain['label']} Families: {', '.join(families)}.")
    lines.append("When a family is relevant, run local skill scripts through `scripts/run_skill.py`, which executes target scripts with workspace `uv run python`.")
    lines.append("Use tool outputs, artifact paths, or cited retrieved evidence in the answer when a skill contributes.")
    return "\n".join(lines)
```

Modify `benchmarking/prompts.py`:

```python
def build_single_llm_prompt(..., available_skills: set[str] | None = None) -> str:
```

and call:

```python
instructions.append(render_top_level_skill_tree(available_skills=available_skills))
```

Modify `SingleLLMRunner.run()` to pass `available_skills=set(self.configured_skills)` to `_build_single_llm_prompt`.

- [ ] **Step 7: Include health in runtime manifest**

In `benchmark_test.py`, add to `results.json` payload:

```python
        "skill_health_summary": skill_health_summary,
```

Add to `runtime-manifest.json` payload:

```python
            "skill_health": {
                "summary": skill_health_summary,
                "report_path": str(output_root / "skill-health.json"),
            },
```

In each runtime manifest group object, add:

```python
                    "effective_skill_allowlist": list(effective_experiment_specs[group_id].skill_allowlist or ()),
```

- [ ] **Step 8: Run targeted tests**

Run:

```bash
uv run pytest tests/test_benchmark_test.py::BenchmarkTestCase::test_effective_experiment_specs_filter_unavailable_skills tests/test_benchmark_skill_tree.py tests/test_benchmark_skill_audit.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit Task 5**

Run:

```bash
git add benchmark_test.py benchmarking/prompts.py benchmarking/skill_tree.py benchmarking/skill_audit.py tests/test_benchmark_test.py tests/test_benchmark_skill_tree.py tests/test_benchmark_skill_audit.py
git commit -m "feat: filter benchmark skills by startup health"
```

---

### Task 6: Structured Unavailable Results for Agent-Invoked Skill Scripts

**Files:**
- Modify: `scripts/run_skill.py`
- Modify: `benchmarking/skill_runtime.py`
- Test: `tests/test_skill_runtime.py`

- [ ] **Step 1: Add CLI test for structured unavailable payload**

Append to `tests/test_skill_runtime.py`:

```python
def test_run_skill_cli_prints_structured_missing_dependency_payload() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "benchmarking").mkdir()
        script = root / "skills" / "demo" / "scripts" / "needs_missing.py"
        script.parent.mkdir(parents=True)
        script.write_text("raise ModuleNotFoundError(\"No module named 'missing_demo'\")\n", encoding="utf-8")

        runner_script = Path(__file__).resolve().parents[1] / "scripts" / "run_skill.py"
        completed = subprocess.run(
            ["uv", "run", "python", str(runner_script), "--workspace-root", str(root), "--script", str(script)],
            text=True,
            capture_output=True,
            check=False,
        )

    payload = json.loads(completed.stdout)
    assert completed.returncode == 2
    assert payload["available"] is False
    assert payload["error_kind"] == "missing_dependency"
    assert payload["reason"] == "missing Python module: missing_demo"
```

- [ ] **Step 2: Run test and confirm current behavior**

Run:

```bash
uv run pytest tests/test_skill_runtime.py::test_run_skill_cli_prints_structured_missing_dependency_payload -q
```

Expected: PASS if Task 3 implementation already handles this path; FAIL if stderr formatting needs adjustment.

- [ ] **Step 3: Harden `scripts/run_skill.py` for top-level runner failures**

If the test fails before JSON is printed, wrap runner construction/execution in `scripts/run_skill.py`:

```python
    try:
        payload = WorkspaceUvSkillRunner(workspace_root=workspace_root).run_script(script_path, script_args)
    except Exception as exc:
        payload = {
            "available": False,
            "error_kind": "script_error",
            "reason": str(exc),
            "runner": "workspace_uv",
            "command": [],
        }
```

Keep the final `print(json.dumps(payload, ensure_ascii=False))` and return code rule.

- [ ] **Step 4: Run all skill runtime tests**

Run:

```bash
uv run pytest tests/test_skill_runtime.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 6**

Run:

```bash
git add scripts/run_skill.py benchmarking/skill_runtime.py tests/test_skill_runtime.py
git commit -m "fix: structure skill runner unavailable results"
```

---

### Task 7: Documentation, Dev Spec, and Verification

**Files:**
- Modify: `GLOBAL_DEV_SPEC.md`
- Modify: `docs/2026-05-09-single-llm-benchmark-blockers.md`

- [ ] **Step 1: Update `GLOBAL_DEV_SPEC.md`**

Update the single-agent and skill inventory sections so they say:

```markdown
- `DONE`: Run a single-agent OpenClaw baseline through a benchmark wrapper that validates OpenClaw stdout against a strict agent result schema before answer extraction. Invalid stdout is retained as diagnostics and marks the record non-scoreable with `agent_result_contract_invalid` instead of being evaluated as an empty answer.
- `DONE`: Run benchmark skill health checks before skills-on groups. Startup health checks verify declared Python imports through workspace `uv run`, executables, API keys, data files, and network providers; unavailable skills are removed from effective runtime allowlists and reported in `skill-health.json` plus `runtime-manifest.json`.
- `DONE`: Provide a fixed skill script runner via `scripts/run_skill.py`; agent-invoked skill scripts run through workspace `uv run python` and return structured unavailable payloads such as `missing_dependency`, `missing_executable`, `missing_api_key`, and `provider_failure`.
```

- [ ] **Step 2: Update blocker document with resolution note**

Append to `docs/2026-05-09-single-llm-benchmark-blockers.md`:

```markdown
## Resolution Plan Status

The implementation plan in `docs/superpowers/plans/2026-05-09-benchmark-result-contract-skill-runtime-health.md` addresses the two merge-blocking architecture issues:

- malformed OpenClaw stdout can no longer enter answer extraction or evaluation as payload data;
- benchmark skills-on runs use startup health checks and a workspace `uv run` skill runner so unavailable skill paths become explicit diagnostics instead of repeated ambiguous tool failures.
```

- [ ] **Step 3: Run focused tests**

Run:

```bash
uv run pytest tests/test_benchmark_result_contract.py tests/test_single_llm_session_wrapper.py tests/test_skill_runtime.py tests/test_skill_health.py tests/test_benchmark_skill_tree.py tests/test_benchmark_skill_audit.py -q
```

Expected: PASS.

- [ ] **Step 4: Run broader benchmark tests touched by this work**

Run:

```bash
uv run pytest tests/test_benchmark_test.py tests/test_benchmark_config_runtime.py -q
```

Expected: PASS.

- [ ] **Step 5: Run dependency confirmation**

Run:

```bash
uv run python - <<'PY'
mods = ["requests", "bs4", "sympy", "rdkit"]
for name in mods:
    try:
        __import__(name)
        print(name, "ok")
    except Exception as exc:
        print(name, "missing", repr(exc))
PY
```

Expected minimum after `uv sync --extra chem --group dev`: `requests ok`, `bs4 ok`, `sympy ok`, `rdkit ok`.

- [ ] **Step 6: Run formatter-free sanity import**

Run:

```bash
uv run python - <<'PY'
from benchmarking.result_contract import parse_agent_stdout
from benchmarking.skill_health import health_requirements_for_allowlist
from benchmarking.skill_runtime import WorkspaceUvSkillRunner
print(parse_agent_stdout('{"result":{"payloads":[{"text":"FINAL ANSWER: A"}],"meta":{}}}').valid)
print(bool(health_requirements_for_allowlist(["rdkit"])))
print(WorkspaceUvSkillRunner.__name__)
PY
```

Expected:

```text
True
True
WorkspaceUvSkillRunner
```

- [ ] **Step 7: Commit Task 7**

Run:

```bash
git add GLOBAL_DEV_SPEC.md docs/2026-05-09-single-llm-benchmark-blockers.md
git commit -m "docs: document benchmark result and skill runtime fixes"
```

---

## Final Verification

After all tasks are complete, run:

```bash
uv run pytest tests/test_benchmark_result_contract.py tests/test_single_llm_session_wrapper.py tests/test_skill_runtime.py tests/test_skill_health.py tests/test_benchmark_skill_tree.py tests/test_benchmark_skill_audit.py tests/test_benchmark_config_runtime.py tests/test_benchmark_test.py -q
```

Expected: PASS.

Then run the same 12-record smoke benchmark from the blocker document:

```bash
uv run python benchmark_test.py \
  --groups single_llm_skills_on \
  --files /Users/xutao/.openclaw/workspace/temp-benchmarks/frontierscience/data/frontierscience_chemistry_pool.jsonl,/Users/xutao/.openclaw/workspace/temp-benchmarks/hle/data/hle_chemistry_pool.jsonl,/Users/xutao/.openclaw/workspace/temp-benchmarks/superchem/data/superchem_pool.jsonl \
  --single-agent-model openai/gpt-5.4 \
  --judge-model su8/gpt-5.4 \
  --max-concurrent-groups 1 \
  --inter-wave-delay-seconds 0 \
  --exact-output-dir state/benchmark-runs/temp-benchmark-20260509-contract-health-rerun
```

Expected post-run checks:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
root = Path("state/benchmark-runs/temp-benchmark-20260509-contract-health-rerun")
health = json.loads((root / "skill-health.json").read_text())
print("available", health["summary"]["available_skill_count"])
print("unavailable", health["summary"]["unavailable_skill_count"])
for p in sorted((root / "per-record" / "single_llm_skills_on").glob("*.json")):
    d = json.loads(p.read_text())
    diag = ((d.get("runner_meta") or {}).get("stdout_diagnostics") or {})
    if diag.get("schema_valid") is False:
        assert d["execution_error_kind"] == "agent_result_contract_invalid"
        assert d["scored"] is False
print("contract checks ok")
PY
```

Expected: invalid stdout records are not scored as empty wrong answers, `skill-health.json` exists, and runtime manifest includes the effective filtered skill allowlist.
