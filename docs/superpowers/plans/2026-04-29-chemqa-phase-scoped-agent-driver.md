# ChemQA Phase-Scoped Agent Driver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ChemQA role execution phase-scoped so a normal OpenClaw turn without an artifact no longer becomes an immediate lane failure.

**Architecture:** Keep the existing DebateClaw protocol state machine and typed artifact validators, but insert a phase executor layer inside the ChemQA driver. The executor will reuse the same session id across repeated turns, validate artifacts after every turn, classify missing or invalid artifacts as in-progress phase states, and only fail once phase budgets or hard-error conditions are exhausted. Add wrapper turn-result sidecar output so the driver can distinguish normal stops, aborts, timeouts, and hard wrapper failures without guessing from return codes alone.

**Tech Stack:** Python, `unittest`, existing ChemQA transport validators, DebateClaw wrapper/session store, JSON sidecars

---

### Task 1: Lock Down Multi-Turn Phase Semantics With Failing Tests

**Files:**
- Modify: `skills/chemqa-review/tests/test_chemqa_review_runtime.py`
- Read: `skills/chemqa-review/scripts/chemqa_review_openclaw_driver.py`
- Read: `skills/chemqa-review/scripts/materialize_runplan.py`
- Read: `skills/debateclaw-v1/scripts/openclaw_debate_agent.py`

- [ ] **Step 1: Add a failing proposer test for missing artifact on the first normal turn**

Add a test that drives `ChemQAReviewDriver.ensure_candidate_submission()` with a fake `call_model()` implementation where:

```python
turn 1 -> returns normal stop, writes no proposal.yaml
turn 2 -> writes a valid proposal.yaml
```

The test must assert:

```python
len(turn_calls) == 2
record_lane_failure_calls == []
submit_proposal_calls == 1
"required artifact" in second_prompt
"phase is still in progress" in second_prompt
```

- [ ] **Step 2: Add a failing coordinator fallback test for aborted refinement**

Add a test that drives `ChemQAReviewDriver.generate_protocol_with_model()` with:

```python
driver.last_turn_outcome = driver_module.TurnOutcome(
    returncode=0,
    stop_reason="aborted",
    aborted=True,
)
raise driver_module.DriverError("Coordinator aborted before writing the refined protocol.")
```

The test must assert:

```python
protocol_payload == deterministic_protocol
generation_mode == "deterministic_fallback"
```

- [ ] **Step 3: Add a failing prompt-generation test for compact snapshot runtime rooting**

Add a test that calls:

```python
prompt = materialize_runplan.render_role_prompt(...)
```

and asserts:

```python
"--runtime-dir /tmp/runtime" in prompt
```

- [ ] **Step 4: Add a failing run-status test for role-phase diagnostics**

Add a test that sets:

```python
driver.current_role_phase_state = driver_module.PhaseAttemptState(...)
```

then calls `sync_run_status(...)` and asserts:

```python
payload["role_phase"]["classification"] == "waiting_for_artifact"
payload["role_phase"]["last_artifact"]["state"] == "missing"
```

- [ ] **Step 5: Add a failing wrapper-sidecar test**

Add a wrapper `main()` test that patches:

```python
session_store_path_for_slot(...)
subprocess.run(...)
```

and asserts that a requested `turn_result_file` is written with:

```json
{
  "stop_reason": "stop",
  "tool_call_count": 1,
  "transcript_path": "...jsonl"
}
```

- [ ] **Step 6: Run the focused runtime test file and confirm the new tests fail**

Run:

```bash
/Users/xutao/.openclaw/workspace/.venv/bin/python -m pytest /Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py -q
```

Expected: FAIL, with failures in the newly added phase-loop, fallback, prompt, run-status, and sidecar tests.

### Task 2: Implement the Phase Executor and Structured Turn Diagnostics

**Files:**
- Modify: `skills/chemqa-review/scripts/chemqa_review_openclaw_driver.py`
- Modify: `skills/debateclaw-v1/scripts/openclaw_debate_agent.py`

- [ ] **Step 1: Introduce phase-state dataclasses in the ChemQA driver**

Add focused dataclasses near the top of `chemqa_review_openclaw_driver.py`:

```python
@dataclass
class TurnOutcome: ...

@dataclass
class ArtifactOutcome: ...

@dataclass
class PhaseAttemptState: ...

@dataclass
class ArtifactContract: ...
```

These types must have `as_payload()` helpers so run-status and blocker payloads can write JSON-ready dictionaries.

- [ ] **Step 2: Teach `call_model()` to capture structured turn results**

Update `call_model()` so it:

```python
- passes --turn-result-file <workspace>/.chemqa-turn-result.json to the wrapper
- returns TurnOutcome on normal completion
- stores self.last_turn_outcome on every path
- raises DriverError only after attaching timeout / hard-error diagnostics
```

- [ ] **Step 3: Extend the wrapper to write an optional turn-result sidecar**

Update `openclaw_debate_agent.py` so it:

```python
- accepts --turn-result-file
- records started_at / completed_at
- resolves the session JSONL path from sessions.json
- parses messages written during this wrapper invocation
- writes stop_reason, tool_call_count, transcript_path, assistant_text_tail, stdout/stderr previews, and returncode
```

- [ ] **Step 4: Replace one-turn artifact generation with a phase loop**

Refactor artifact generation into a phase executor that:

```python
- reuses the same session id across repeated turns
- checks artifact state after each turn
- classifies missing/invalid/stale separately
- continues when the artifact is still repairable
- fails only on exhausted budgets, wall timeout, or hard errors
```

Keep `attempt_model_artifact()` as the public driver entrypoint, but make it call the new executor.

- [ ] **Step 5: Preserve duplicate-candidate handling on top of the new phase executor**

Keep `ensure_candidate_submission()` responsible for submit-time duplicate rejection, but remove the current single-turn artifact restriction so a valid candidate can take multiple turns before the submit step.

- [ ] **Step 6: Harden coordinator refinement fallback**

Update `generate_protocol_with_model()` so:

```python
- timeout + valid artifact -> "model_timeout_salvaged"
- aborted turn -> deterministic fallback
- invalid rewrite -> deterministic fallback
- missing artifact after normal stop -> deterministic fallback after budget exhaustion
```

### Task 3: Wire Runtime Prompt and Run-Status Reporting

**Files:**
- Modify: `skills/chemqa-review/scripts/materialize_runplan.py`
- Modify: `skills/chemqa-review/scripts/control_store.py`
- Modify: `skills/chemqa-review/scripts/chemqa_review_openclaw_driver.py`

- [ ] **Step 1: Fix the compact snapshot prompt command**

Update `render_role_prompt()` so the compact snapshot command is rendered as:

```text
python chemqa_review_state_snapshot.py --skill-root ... --runtime-dir <runtime_root> --team {team_name} --agent {agent_name}
```

- [ ] **Step 2: Preserve role-phase payloads in run-status**

Add `FileControlStore.get_run_status()` and update driver status synchronization so:

```python
payload["role_phase"] = current_phase_state.as_payload()
```

is preserved while the benchmark is still running.

- [ ] **Step 3: Surface role-phase diagnostics in blocker and driver-error payloads**

Update blocker and driver-error outputs to include the latest phase executor payload when available.

### Task 4: Verify, Update Docs, and Commit

**Files:**
- Modify: `GLOBAL_DEV_SPEC.md`

- [ ] **Step 1: Run the focused runtime tests after implementation**

Run:

```bash
/Users/xutao/.openclaw/workspace/.venv/bin/python -m pytest /Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py -q
```

Expected: PASS.

- [ ] **Step 2: Run an additional ChemQA regression target if the runtime test file passes**

Run:

```bash
/Users/xutao/.openclaw/workspace/.venv/bin/python -m pytest /Users/xutao/.openclaw/workspace/tests/test_chemqa_epoch_flow.py -q
```

Expected: PASS, or a narrowly explained pre-existing failure unrelated to this change.

- [ ] **Step 3: Update the global development spec**

Update `GLOBAL_DEV_SPEC.md` so it reflects:

```text
- phase-scoped ChemQA role execution
- role-phase diagnostics in run-status
- compact snapshot runtime-dir prompt fix
- wrapper turn-result sidecar support
```

- [ ] **Step 4: Review the final diff and commit**

Run:

```bash
git -C /Users/xutao/.openclaw/workspace status --short
git -C /Users/xutao/.openclaw/workspace diff --stat
git -C /Users/xutao/.openclaw/workspace add docs/superpowers/plans/2026-04-29-chemqa-phase-scoped-agent-driver.md skills/chemqa-review/scripts/chemqa_review_openclaw_driver.py skills/chemqa-review/scripts/materialize_runplan.py skills/chemqa-review/scripts/control_store.py skills/chemqa-review/tests/test_chemqa_review_runtime.py skills/debateclaw-v1/scripts/openclaw_debate_agent.py GLOBAL_DEV_SPEC.md
git -C /Users/xutao/.openclaw/workspace commit -m "fix: make chemqa agent driver phase scoped"
```

Expected: clean commit containing the driver, wrapper, prompt, test, plan, and spec changes.
