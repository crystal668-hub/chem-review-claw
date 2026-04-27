# ChemQA Unattended Benchmark Natural Completion Repair Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:systematic-debugging` before any code changes. After this plan is approved for execution, use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement it task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the remaining runtime behaviors that cause ChemQA benchmark runs to end as `execution_error` or enter persistent rollback / recovery loops, so the benchmark can proceed without manual intervention to a judgeable completed result.

**Architecture:** Treat this as a runtime-state and recovery-semantics repair, not a scoring redesign. Fix the control contract in this order: first, only the coordinator may publish terminal failure to `run-status`; second, recovery/rollback must converge naturally instead of oscillating between `rebuttal` and `propose`; third, malformed-but-schema-valid candidate artifacts must stop counting as actionable progress; fourth, tune stop-loss budgets only after the state machine semantics are correct. Validation must cover both the duplicate-proposal `0001` path and the rollback / wait-forever `0002` path, then finish with an unattended ConformaBench rerun.

**Tech Stack:** Python 3.12, ChemQA benchmark harness in `/Users/xutao/.config/superpowers/worktrees/openclaw/chemqa-recovery-status`, active runtime scripts in `/Users/xutao/.openclaw/workspace/skills/chemqa-review`, DebateClaw SQLite state, JSON/YAML protocol artifacts, `unittest`, real benchmark CLI reruns.

---

## Problem Statement

最新一次额外 benchmark 测试没有复现之前的“completed/rejected 被 runner 错算成 execution_error”主问题，但暴露出两个更直接的无人值守推进阻塞点：

1. `conformabench-0001` 是真实运行失败，不是评分误判。其 `per-record` 结果显示 `terminal_state=failed`、`terminal_reason_code=stalled`，并没有 archived completed `qa_result.json` 可供 reconciliation。
2. `conformabench-0002` 进入了持续回卷 / 非收敛恢复。它曾推进到 review/rebuttal，并且 `last_recovery` 记录了合成 review 与 `advance`，但顶层 `run-status` 又回到了 `phase=propose`，同时 `next_action.message` 仍在等待 `proposer-1`，即使 epoch 2 proposal 已经存在归档。
3. 当前 stop-loss 默认值极激进：`stale_timeout_seconds=300`、`max_model_attempts=1`、`lane_retry_budget=2`、`phase_repair_budget=1`、`max_respawns_per_role_phase_signature=1`。它们会把暂时性恢复不充分直接升级成 terminal failure 或持续回卷。
4. `check_candidate_submission()` 目前只要求 `direct_answer` 非空、`summary` 非空、`submission_trace` 形状合法，因此像 `0002` epoch 2 这种“`direct_answer` 是大段 prose、summary 混入整个修订叙述”的 artifact 仍会被视为有效 proposal，从而让系统误以为 propose 阶段已经恢复成功。

本计划只修复会阻断 benchmark 自然推进到 judgeable 结果的运行时问题，不扩展 acceptance policy，也不重写 scoring。

## Confirmed Evidence

### Runtime evidence from the extra rerun

- `conformabench-0001` 结果文件：
  - `/Users/xutao/.openclaw/workspace/state/benchmark-runs/conformabench-top3-qwen-web-on-rerun-20260426/per-record/chemqa_web_on/conformabench-0001.json`
  - `primary_metric=execution_error`
  - `error="ChemQA run ended with non-success status: failed"`
  - `runner_meta.terminal_state="failed"`
  - `runner_meta.terminal_reason_code="stalled"`
  - `short_answer_text="N#Cc1ccccc1CO"`
  - `reconciled_from_archived_artifacts=null`
- `conformabench-0001` cleanup report confirms the same aggressive stop-loss args were used by the real driver command:
  - `/Users/xutao/.openclaw/workspace/state/benchmark-runs/conformabench-top3-qwen-web-on-rerun-20260426/cleanroom/reports/benchmark-chemqa_web_on-conformabench-0001-20260426-003747.cleanup-report.json`
- `conformabench-0002` `run-status` shows split progress surfaces:
  - `/Users/xutao/.openclaw/workspace/skills/chemqa-review/control/run-status/benchmark-chemqa_web_on-conformabench-0002-20260426-004748.json`
  - top-level: `status="running"`, `phase="propose"`, `next_action.message="Waiting for proposer-1 to submit the candidate before advancing."`
  - `last_recovery.status="done"`
  - `last_recovery.actions` contains four review submissions plus `advance debate-coordinator`
  - `last_recovery.state.phase="rebuttal"`
  - `last_recovery.state.epoch=2`
- `conformabench-0002` spawn registry proves proposer-1 was respawned with the same tight budgets:
  - `/Users/xutao/.openclaw/workspace/skills/chemqa-review/generated/clawteam-data/runs/benchmark-chemqa_web_on-conformabench-0002-20260426-004748/teams/benchmark-chemqa_web_on-conformabench-0002-20260426-004748/spawn_registry.json`
- `conformabench-0002` workspace and archived epoch-2 proposal show a malformed-but-accepted candidate artifact:
  - `/Users/xutao/.openclaw/benchmark/workspaces/chemqa_web_on/debateA-1/proposal.yaml`
  - `/Users/xutao/.openclaw/workspace/skills/chemqa-review/generated/clawteam-data/runs/benchmark-chemqa_web_on-conformabench-0002-20260426-004748/teams/benchmark-chemqa_web_on-conformabench-0002-20260426-004748/debate/artifacts/proposals/epoch-002/proposer-1.md`
  - `direct_answer` is prose beginning with `Revised proposal for 2-fluoroethylamine (NCCF) as the target molecule...`, not a scalar final answer.
- `conformabench-0002` epoch-1 rebuttal explicitly concedes failure, but recovery still leaves the system waiting on a fresh candidate rather than naturally converging to either a new valid epoch-2 proposal or a terminal coordinator decision.

### Code-path evidence already confirmed

- Active benchmark launches are still pointed at `/Users/xutao/.openclaw/workspace/skills/chemqa-review`, not the worktree copy.
- The worktree copy and active workspace copy are already divergent for at least:
  - `/Users/xutao/.openclaw/workspace/skills/chemqa-review/scripts/recover_run.py`
  - `/Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py`
- The worktree copy of `recover_run.py` already changed one behavior (`payload["status"] = "running" if stalled else "done"`), but the active benchmark rerun still executed the workspace copy, where stalled recovery currently writes top-level `status="done"`, `terminal_state="failed"`, `terminal_reason_code="stalled"` to `run-status`.
- `chemqa_review_openclaw_driver.py` currently treats either of these as “progress happened”:
  - post-recovery phase signature changed
  - or `recovery.get("status") == "done"`
  Because active `recover_run.py` returns `{"status": "done"}` even when it merely stopped after one stagnant cycle, the driver can reset the progress clock on a false success signal.
- `repair_invalid_review_state()` mutates DebateClaw state directly from `review` to next `epoch` `propose` if the active candidate proposal is missing. This is the most likely rollback source for `0002`.
- DebateClaw duplicate proposal rejection is still hard-fail strict:
  - `debate_state.py submit_proposal()` rejects any identical proposal fingerprint for the same proposer across epochs with `Proposal matches a prior submission from epoch ...`.
- Scoring-side rejection-blob handling has already been fixed in `benchmark_test.py` via `extract_chemqa_scoreable_answer()`. That issue is no longer the immediate unattended-completion blocker.

## Scope

### In Scope

- Active runtime scripts under `/Users/xutao/.openclaw/workspace/skills/chemqa-review`
- Matching benchmark harness code under `/Users/xutao/.config/superpowers/worktrees/openclaw/chemqa-recovery-status`
- Recovery semantics, rollback semantics, candidate artifact validation semantics, and stop-loss configuration
- New regression tests covering `0001` duplicate-proposal failure semantics and `0002` rollback/non-convergence semantics
- A final unattended real rerun on ConformaBench top 3

### Out of Scope

- Changing ChemQA acceptance policy or review rubric
- Reworking ConformaBench scoring semantics beyond existing rejected-blob fix
- Refactoring unrelated DebateClaw architecture
- Prompt redesign unrelated to the observed runtime failure modes
- General “make answers better” changes that are not required for unattended completion

## Root-Cause Hypotheses

### H1: `recover_run.py` still has terminal-failure authority it should not have

With `--max-steps 1`, a single non-converging recovery probe currently writes `run-status.status="done"`, `terminal_state="failed"`, `terminal_reason_code="stalled"`. That is an authority violation: recovery is a diagnostic / repair helper, not the coordinator.

### H2: The driver interprets “recovery finished its probe” as “the run made progress”

`maybe_handle_stagnation()` resets the progress clock when `recovery.get("status") == "done"`, even if no phase or artifact state changed. Combined with H1, this can create alternating false-success / false-failure behavior.

### H3: Rollback from `review` to next-epoch `propose` is under-specified and non-convergent

`repair_invalid_review_state()` jumps the engine to the next epoch and `phase="propose"`, but there is no paired guarantee that proposer-1 will regenerate a truly new valid candidate. In `0002`, this leaves the run stranded waiting for proposer-1 after a rollback that recovery itself triggered.

### H4: Candidate artifact validation is too permissive for benchmark final-answer tasks

`check_candidate_submission()` only validates presence and shape, not whether `direct_answer` is plausibly a scalar answer rather than an epoch narrative. That lets semantically broken proposals count as successful progress.

### H5: The configured stop-loss budgets are too tight for real recovery latency

Even after semantics are corrected, `max_model_attempts=1`, `phase_repair_budget=1`, and `max_respawns_per_role_phase_signature=1` are likely too small for unattended recovery on this benchmark profile.

## Design Principles

1. Terminal failure must be published by the coordinator after observing real engine terminal state, not by the recovery helper.
2. A one-step recovery probe may report `progress_made=false`; it must not synthesize a completed terminal status just because the probe returned.
3. Rollback must either converge to a genuinely new actionable candidate or escalate to a real coordinator-owned terminal failure. It must not bounce indefinitely.
4. Progress should be counted from state change, artifact registration, or task/lane transition, not from a helper script exiting normally.
5. Parameter tuning comes after semantics are correct. Otherwise tuning only hides authority bugs.
6. Fix the active code path first. Editing only the divergent worktree copy is a no-op for real benchmark runs.

## File Map and Responsibilities

### Active runtime files to modify

- Modify: `/Users/xutao/.openclaw/workspace/skills/chemqa-review/scripts/recover_run.py`
  - Remove or constrain recovery-owned terminal failure publication for stalled one-step probes.
  - Return a more truthful recovery payload contract.
- Modify: `/Users/xutao/.openclaw/workspace/skills/chemqa-review/scripts/chemqa_review_openclaw_driver.py`
  - Stop treating `recovery.status == "done"` as implicit progress.
  - Gate stagnation reset on actual state change.
  - Add convergence guards around rollback / duplicate-proposal paths.
- Modify: `/Users/xutao/.openclaw/workspace/skills/chemqa-review/scripts/chemqa_review_artifacts.py`
  - Tighten candidate validation/salvage so semantically broken `direct_answer` payloads do not count as valid benchmark answers.
- Modify: `/Users/xutao/.openclaw/workspace/skills/chemqa-review/scripts/compile_runplan.py`
  - Adjust stop-loss defaults only after the semantic fixes and only minimally.

### Possible runtime file to inspect before deciding whether to edit

- Inspect: `/Users/xutao/.openclaw/workspace/skills/debateclaw-v1/scripts/debate_state.py`
  - Only change if duplicate-proposal hard rejection cannot be handled correctly in the driver/recovery layer.
  - Prefer not to relax fingerprint rules globally unless the smaller coordinator/runtime fix proves insufficient.

### Benchmark-harness files to keep aligned

- Inspect or modify if needed: `/Users/xutao/.config/superpowers/worktrees/openclaw/chemqa-recovery-status/benchmarking/runners/chemqa.py`
  - Only if new recovery semantics require runner-side handling changes.
- Keep as verification target: `/Users/xutao/.config/superpowers/worktrees/openclaw/chemqa-recovery-status/benchmark_test.py`
  - No new scoring fix is expected here for this task.

### Tests to modify

- Modify: `/Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py`
  - This is the active runtime test suite that should track the code actually executed by benchmark runs.
- Optionally mirror or re-sync after implementation:
  - `/Users/xutao/.config/superpowers/worktrees/openclaw/chemqa-recovery-status/skills/chemqa-review/tests/test_chemqa_review_runtime.py`
- Keep benchmark-scoring regression intact:
  - `/Users/xutao/.config/superpowers/worktrees/openclaw/chemqa-recovery-status/tests/test_benchmark_test.py`

## Implementation Tasks

### Task 1: Lock the Active Edit Target Before Touching Behavior

**Files:**
- Read: `/Users/xutao/.openclaw/workspace/skills/chemqa-review/scripts/recover_run.py`
- Read: `/Users/xutao/.config/superpowers/worktrees/openclaw/chemqa-recovery-status/skills/chemqa-review/scripts/recover_run.py`
- Read: `/Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py`
- Read: `/Users/xutao/.config/superpowers/worktrees/openclaw/chemqa-recovery-status/skills/chemqa-review/tests/test_chemqa_review_runtime.py`

- [ ] **Step 1: Confirm the active benchmark path is the workspace skill root**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

path = Path('/Users/xutao/.config/superpowers/worktrees/openclaw/chemqa-recovery-status/benchmark_test.py')
spec = spec_from_file_location('benchmark_test_active_path_check', path)
mod = module_from_spec(spec)
spec.loader.exec_module(mod)
print(mod.DEFAULT_CHEMQA_ROOT)
PY
```

Expected:
- Prints `/Users/xutao/.openclaw/workspace/skills/chemqa-review`

- [ ] **Step 2: Confirm active/runtime file divergence before editing**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
pairs = [
    ('recover_run.py', '/Users/xutao/.openclaw/workspace/skills/chemqa-review/scripts/recover_run.py', '/Users/xutao/.config/superpowers/worktrees/openclaw/chemqa-recovery-status/skills/chemqa-review/scripts/recover_run.py'),
    ('test_chemqa_review_runtime.py', '/Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py', '/Users/xutao/.config/superpowers/worktrees/openclaw/chemqa-recovery-status/skills/chemqa-review/tests/test_chemqa_review_runtime.py'),
]
for label, a, b in pairs:
    pa, pb = Path(a), Path(b)
    print(label, pa.read_text(encoding='utf-8') == pb.read_text(encoding='utf-8'))
PY
```

Expected:
- At least `recover_run.py` reports `False`
- This confirms implementation must start in the active workspace tree

- [ ] **Step 3: Treat the workspace tree as the implementation source of truth for this repair**

No code block for this step.
Expected:
- All runtime edits and runtime tests in later tasks target `/Users/xutao/.openclaw/workspace/skills/chemqa-review/...`
- Worktree copies are only re-synced after active runtime verification passes

### Task 2: Remove Recovery-Owned Terminal Failure Semantics

**Files:**
- Modify: `/Users/xutao/.openclaw/workspace/skills/chemqa-review/scripts/recover_run.py`
- Test: `/Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py`

- [ ] **Step 1: Write the failing regression test for a stalled one-step recovery probe**

Add a test near the existing `test_write_run_status_done_failed_can_set_stalled_reason_code` section asserting:

```python
def test_recover_run_stalled_single_step_does_not_publish_done_failed(self) -> None:
    ...
    self.assertEqual("running", payload["status"])
    self.assertNotIn("terminal_state", payload)
    self.assertNotIn("terminal_reason_code", payload)
```

The scenario should mirror the current one-step recovery path:
- `max_steps=1`
- engine status remains not done
- no phase signature change
- recoverer exits with a stalled probe result

Expected before implementation:
- FAIL because active `recover_run.py` currently writes `status="done"`, `terminal_state="failed"`, `terminal_reason_code="stalled"`

- [ ] **Step 2: Run only that regression test**

Run:

```bash
uv run python -m pytest /Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py -k stalled_single_step_does_not_publish_done_failed -v
```

Expected:
- FAIL

- [ ] **Step 3: Change `recover_run.py` so a non-terminal stalled probe remains non-terminal in `run-status`**

Required behavior:
- If the engine itself is not done, recovery may write `status="running"`
- It may record `progress_made`, `recovery_cycles_without_progress`, `actions`, and `blockers`
- It must not set `terminal_state`, `terminal_reason_code`, or `terminal_reason` for a mere probe-level stall
- JSON stdout from `recover_run.py` may still include an internal diagnostic marker like `stalled=true`, but that is not a global terminal publication

Implementation note:
- Preserve the existing path where a truly done engine writes `status="done"` and terminal metadata copied from engine state
- Do not remove `repair_invalid_review_state()` terminal-failure publication for `max_epochs` exhaustion yet; that is a real engine-state mutation and will be revisited only if evidence later requires it

- [ ] **Step 4: Run the targeted regression test again**

Run:

```bash
uv run python -m pytest /Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py -k stalled_single_step_does_not_publish_done_failed -v
```

Expected:
- PASS

- [ ] **Step 5: Update or replace the old stalled-reason test to reflect the new authority boundary**

The existing test `test_write_run_status_done_failed_can_set_stalled_reason_code` encodes the old wrong behavior. Replace it with one of:
- a test that coordinator-owned terminal failure writes `terminal_reason_code`, or
- a lower-level test proving `write_run_status()` only adds terminal fields when `status="done"`

Run:

```bash
uv run python -m pytest /Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py -k 'stalled_reason_code or write_run_status' -v
```

Expected:
- PASS with the new semantics

### Task 3: Make the Driver Count Only Real Progress

**Files:**
- Modify: `/Users/xutao/.openclaw/workspace/skills/chemqa-review/scripts/chemqa_review_openclaw_driver.py`
- Test: `/Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py`

- [ ] **Step 1: Write the failing regression test that recovery probe completion alone does not reset stagnation**

Add a test shaped like:

```python
def test_maybe_handle_stagnation_does_not_mark_progress_when_recovery_reports_done_without_state_change(self) -> None:
    ...
    self.assertEqual([], progress_marks)
    self.assertEqual(expected_repair_cycles, driver.repair_cycles_without_progress)
```

Fixture requirements:
- `run_recovery_cycle()` returns `{"status": "done", "progress_made": False, "blockers": []}`
- pre/post phase signature is identical
- no lane exits, no advance, no artifact registration

Expected before implementation:
- FAIL because current code treats `recovery.get("status") == "done"` as progress

- [ ] **Step 2: Run the targeted failing test**

Run:

```bash
uv run python -m pytest /Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py -k recovery_reports_done_without_state_change -v
```

Expected:
- FAIL

- [ ] **Step 3: Change `maybe_handle_stagnation()` to rely on actual progress signals**

Required behavior:
- Progress may be credited when post-recovery phase signature changed
- Or when recovery payload explicitly says `progress_made=true`
- Or when refreshed `next_action` / artifact-bearing state proves the blocked action became actionable
- `recovery.status == "done"` by itself must no longer count as progress

Implementation note:
- Keep this change local to stagnation accounting
- Do not simultaneously tune `phase_repair_budget` in this step

- [ ] **Step 4: Re-run the targeted test plus the nearby reviewer-exit stagnation test**

Run:

```bash
uv run python -m pytest /Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py -k 'recovery_reports_done_without_state_change or maybe_handle_stagnation_marks_missing_reviewer_exited_and_continues' -v
```

Expected:
- Both PASS

### Task 4: Stop Rollback From Converging to “Wait Forever for proposer-1”

**Files:**
- Modify: `/Users/xutao/.openclaw/workspace/skills/chemqa-review/scripts/chemqa_review_openclaw_driver.py`
- Modify if required: `/Users/xutao/.openclaw/workspace/skills/chemqa-review/scripts/recover_run.py`
- Test: `/Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py`
- Inspect only unless strictly needed: `/Users/xutao/.openclaw/workspace/skills/debateclaw-v1/scripts/debate_state.py`

- [ ] **Step 1: Write a failing rollback regression based on the `0002` shape**

Add a test shaped like:

```python
def test_repair_invalid_review_state_followed_by_existing_epoch2_candidate_does_not_loop_wait_forever(self) -> None:
    ...
```

Minimum assertion contract:
- If rollback moved the engine to `epoch=2`, `phase=propose`, and a valid epoch-2 candidate already exists in workspace/capture/archive-equivalent source, the next driver action should submit or reuse that candidate rather than remain indefinitely in `wait for proposer-1`
- If no truly usable candidate exists, the driver must accumulate a bounded failure path rather than oscillate silently

Implementation note:
- Reuse the existing recovery fixture style from `test_recover_propose_uses_captured_candidate_when_workspace_file_is_missing`
- Keep the test narrow: one epoch transition, one candidate owner, one expected convergence behavior

- [ ] **Step 2: Run the targeted rollback regression**

Run:

```bash
uv run python -m pytest /Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py -k rollback_wait_forever -v
```

Expected:
- FAIL

- [ ] **Step 3: Implement the smallest convergence guard that fixes the loop**

Preferred fix order:
1. In the driver, detect that rollback placed the run into `propose` but a valid candidate submission is already available from workspace/capture and `current_proposal()` is still missing; explicitly call the candidate submission path.
2. If submission fails due to DebateClaw duplicate fingerprint rejection and the candidate body is identical to a prior epoch, treat that as evidence that proposer-1 did not produce a genuinely new epoch candidate. Do not just spin. Either:
   - trigger a bounded regeneration attempt, or
   - count a real lane failure toward a terminal coordinator decision.
3. Only if this cannot be expressed cleanly in driver/recovery code should DebateClaw duplicate semantics be reconsidered.

Do not implement all three at once. Start with driver-side convergence.

- [ ] **Step 4: Re-run the rollback regression and adjacent propose-recovery tests**

Run:

```bash
uv run python -m pytest /Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py -k 'rollback_wait_forever or recover_propose' -v
```

Expected:
- PASS

### Task 5: Tighten Candidate Validation So Narrative Blobs Do Not Count as Valid Answers

**Files:**
- Modify: `/Users/xutao/.openclaw/workspace/skills/chemqa-review/scripts/chemqa_review_artifacts.py`
- Modify: `/Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py`

- [ ] **Step 1: Add a failing artifact-validation regression modeled on `0002` epoch-2 proposal**

Add a test like:

```python
def test_candidate_submission_rejects_narrative_direct_answer_for_scalar_answer_tasks(self) -> None:
    candidate = """
artifact_kind: candidate_submission
phase: propose
owner: proposer-1
direct_answer: Revised proposal for 2-fluoroethylamine (NCCF) as the target molecule. This epoch
summary: ...
submission_trace:
- step: structural_reasoning
  status: success
  detail: ...
""".strip()
    checked = transport.check_candidate_submission(candidate, owner='proposer-1')
    self.assertFalse(checked.ok)
```

Expected before implementation:
- FAIL because the current checker accepts any non-empty `direct_answer`

- [ ] **Step 2: Run the targeted validation regression plus neighboring candidate-recovery tests**

Run:

```bash
uv run python -m pytest /Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py -k 'narrative_direct_answer or candidate_' -v
```

Expected:
- New regression FAILS
- Existing candidate recovery tests still PASS before the implementation change

- [ ] **Step 3: Add a narrow semantic guard for candidate `direct_answer`**

Required behavior:
- Continue allowing short scalar answers and short SMILES-like strings
- Continue allowing legacy markdown extraction when the extracted answer is still scalar-like
- Reject obviously narrative `direct_answer` values that look like sentence-level revision summaries, especially those beginning with phrases like `Revised proposal ...`, multi-clause prose, or embedded step/item narratives

Implementation constraints:
- Keep this heuristic narrow and benchmark-safe
- Do not overfit to exact `0002` text only
- Do not introduce a chemistry-specific validator that would reject valid non-SMILES tasks elsewhere unless the current runtime is ChemQA-specific at this point in the pipeline

- [ ] **Step 4: Re-run the targeted validation regression and candidate recovery tests**

Run:

```bash
uv run python -m pytest /Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py -k 'narrative_direct_answer or recover_propose or candidate_markdown_inline_direct_answer_is_recovered' -v
```

Expected:
- PASS

### Task 6: Raise Stop-Loss Budgets Only After Semantics Are Correct

**Files:**
- Modify: `/Users/xutao/.openclaw/workspace/skills/chemqa-review/scripts/compile_runplan.py`
- Test: `/Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py`

- [ ] **Step 1: Write a minimal configuration regression for the stop-loss defaults**

Add or update a test that inspects the compiled runplan / materialized command map and asserts the exact stop-loss values passed to the driver.

Start from the current assertions near `test_run_recovery_cycle_uses_single_step_and_passes_respawn_budget` and the compile/materialize tests around lines `419-456`.

Target defaults for the first parameter experiment should be conservative, for example:
- `max_model_attempts = 2`
- `phase_repair_budget = 2`
- `max_respawns_per_role_phase_signature = 2`
- keep `stale_timeout_seconds = 300` initially unless evidence shows it is too short
- keep `lane_retry_budget = 2` initially unless later evidence shows otherwise

Expected before implementation:
- FAIL if the test expects the new values

- [ ] **Step 2: Update `compile_runplan.py` stop-loss defaults to the chosen minimal experiment values**

Change only the runplan stop-loss snapshot under `chemqa_review.stop_loss`.
Do not simultaneously edit unrelated prompt or workflow config.

- [ ] **Step 3: Re-run the runplan/materialization tests**

Run:

```bash
uv run python -m pytest /Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py -k 'stop_loss or materialize or command_map or max_respawns_per_role_phase_signature' -v
```

Expected:
- PASS

### Task 7: Run the Full Active Runtime Regression Slice

**Files:**
- Test only: `/Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py`

- [ ] **Step 1: Run the focused runtime suite covering all touched behaviors**

Run:

```bash
uv run python -m pytest /Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py -k 'recover_run or stagnation or candidate_submission or stop_loss or rollback' -v
```

Expected:
- PASS

- [ ] **Step 2: Run the full active runtime suite if the focused slice passes**

Run:

```bash
uv run python -m pytest /Users/xutao/.openclaw/workspace/skills/chemqa-review/tests/test_chemqa_review_runtime.py -v
```

Expected:
- PASS

- [ ] **Step 3: Run the benchmark-scoring regression that fixed the earlier rejected-blob issue**

Run:

```bash
uv run python -m pytest /Users/xutao/.config/superpowers/worktrees/openclaw/chemqa-recovery-status/tests/test_benchmark_test.py -k 'rejected_blob or build_chemqa_full_response' -v
```

Expected:
- PASS

### Task 8: Perform Unattended Real Benchmark Verification

**Files:**
- Read runtime output under a fresh benchmark output dir
- No source edits in this task

- [ ] **Step 1: Choose a fresh output directory for the repair verification rerun**

Use exactly:

```text
/Users/xutao/.openclaw/workspace/state/benchmark-runs/conformabench-top3-qwen-web-on-natural-completion-rerun-20260426
```

- [ ] **Step 2: Remove any stale directory at that path before running**

Run:

```bash
rm -rf /Users/xutao/.openclaw/workspace/state/benchmark-runs/conformabench-top3-qwen-web-on-natural-completion-rerun-20260426
```

Expected:
- Exit `0`

- [ ] **Step 3: Print the selected records before the real rerun**

Run:

```bash
uv run python /Users/xutao/.config/superpowers/worktrees/openclaw/chemqa-recovery-status/benchmark_test.py \
  --benchmark-root /Users/xutao/.openclaw/workspace/benchmarks \
  --chemqa-root /Users/xutao/.openclaw/workspace/skills/chemqa-review \
  --openclaw-config /Users/xutao/.openclaw/workspace/.openclaw/config.json \
  --groups chemqa_web_on \
  --datasets conformabench \
  --offset 0 \
  --limit 3 \
  --chemqa-model-profile chemqa-review-su8-coord-qwen-ds-kimi-glm-minimax \
  --exact-output-dir /Users/xutao/.openclaw/workspace/state/benchmark-runs/conformabench-top3-qwen-web-on-natural-completion-rerun-20260426 \
  --print-selected-records
```

Expected:
- Exactly `conformabench-0001`, `conformabench-0002`, `conformabench-0003`

- [ ] **Step 4: Run the unattended real benchmark rerun**

Run:

```bash
uv run python /Users/xutao/.config/superpowers/worktrees/openclaw/chemqa-recovery-status/benchmark_test.py \
  --benchmark-root /Users/xutao/.openclaw/workspace/benchmarks \
  --chemqa-root /Users/xutao/.openclaw/workspace/skills/chemqa-review \
  --openclaw-config /Users/xutao/.openclaw/workspace/.openclaw/config.json \
  --groups chemqa_web_on \
  --datasets conformabench \
  --offset 0 \
  --limit 3 \
  --max-concurrent-groups 1 \
  --inter-wave-delay-seconds 0 \
  --chemqa-model-profile chemqa-review-su8-coord-qwen-ds-kimi-glm-minimax \
  --exact-output-dir /Users/xutao/.openclaw/workspace/state/benchmark-runs/conformabench-top3-qwen-web-on-natural-completion-rerun-20260426
```

Expected:
- Command exits on its own without manual recovery intervention
- `results.json` and three `per-record` JSON files materialize

- [ ] **Step 5: Summarize the three per-record outcomes**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
base = Path('/Users/xutao/.openclaw/workspace/state/benchmark-runs/conformabench-top3-qwen-web-on-natural-completion-rerun-20260426/per-record/chemqa_web_on')
for record_id in ('conformabench-0001', 'conformabench-0002', 'conformabench-0003'):
    payload = json.loads((base / f'{record_id}.json').read_text(encoding='utf-8'))
    meta = payload.get('runner_meta') or {}
    print('==', record_id, '==')
    print('primary_metric =', payload['evaluation']['primary_metric'])
    print('error =', payload.get('error'))
    print('terminal_state =', meta.get('terminal_state'))
    print('terminal_reason_code =', meta.get('terminal_reason_code'))
    print('acceptance_status =', meta.get('acceptance_status'))
    print('reconciled_from_archived_artifacts =', meta.get('reconciled_from_archived_artifacts'))
    print()
PY
```

Expected:
- `0001` no longer fails for the old duplicate-proposal / false-stall path without a coordinator-owned terminal reason
- `0002` no longer sits in a rollback loop waiting forever for proposer-1
- Each record ends either as a judgeable completed result or as a real coordinator-owned terminal failure with clear reason, not a recovery-poisoned split-brain state

- [ ] **Step 6: If any record still fails, capture the exact new failure surface before changing code again**

Run:

```bash
ps -axo pid=,ppid=,command= | rg 'benchmark_test.py|chemqa_review_openclaw_driver.py|recover_run.py|launch_from_preset.py|benchmark-chemqa_web_on-conformabench-000[123]'
```

Then inspect the new run-status, spawn registry, and archived protocol for the failing record only.

Expected:
- No speculative follow-up fixes without fresh evidence

### Task 9: Clean Up Residual Benchmark Processes

**Files:**
- No code files; live process table only

- [ ] **Step 1: List residual benchmark-related processes**

Run:

```bash
ps -axo pid=,ppid=,command= | rg 'benchmark_test.py|chemqa_review_openclaw_driver.py|recover_run.py|launch_from_preset.py|benchmark-chemqa_web_on-conformabench-000[123]'
```

Expected:
- Either no matches, or a precise list of leftovers tied to this rerun

- [ ] **Step 2: Kill only the leftover processes from this rerun if any remain**

Run:

```bash
kill <pid1> <pid2> ...
```

If required:

```bash
kill -9 <pid1> <pid2> ...
```

Expected:
- Only benchmark processes from this rerun are terminated

- [ ] **Step 3: Re-check the process table**

Run:

```bash
ps -axo pid=,ppid=,command= | rg 'benchmark_test.py|chemqa_review_openclaw_driver.py|recover_run.py|launch_from_preset.py|benchmark-chemqa_web_on-conformabench-000[123]'
```

Expected:
- No matching residual processes remain

## Acceptance Criteria

This repair is complete only if all of the following are true:

1. A one-step stalled recovery probe no longer publishes `run-status.status="done"` / `terminal_state="failed"` on its own.
2. The coordinator no longer treats `recovery.status == "done"` as progress unless the state actually changed or recovery explicitly reports real progress.
3. The `0002`-class rollback path no longer loops in `propose` waiting forever for proposer-1 after recovery-induced epoch transition.
4. `check_candidate_submission()` no longer accepts obviously narrative epoch-summary blobs as valid `direct_answer` payloads.
5. Runtime regressions pass in the active workspace test suite.
6. The benchmark-scoring regression for rejected blobs still passes.
7. A real unattended rerun of ConformaBench top 3 completes without manual intervention and no longer reproduces the observed `execution_error` / persistent rollback failure class.
8. No residual benchmark processes remain after verification.

## Risks and Open Questions

- The cleanest fix may still require a DebateClaw-level distinction between “duplicate proposal because no new epoch candidate exists” and “duplicate proposal because proposer retried the exact same artifact by mistake”. The plan intentionally defers that until driver-side convergence is proven insufficient.
- Tightening `check_candidate_submission()` too aggressively could break non-SMILES tasks if the runtime is shared more widely than expected. Keep the heuristic narrow and validate against existing recovery tests.
- Because active runtime and worktree copies are already divergent, there is a real risk of fixing the wrong tree. Do not skip Task 1.
- If the real rerun reveals a third independent failure class after these fixes, stop and write a new diagnosis addendum rather than silently extending scope.

## Execution Notes

- Use `apply_patch` for all manual edits.
- Do not amend unrelated files or revert existing user changes.
- Keep commits small and aligned to the task boundaries above.
- Do not claim success until the real unattended benchmark rerun and process cleanup are both complete.
