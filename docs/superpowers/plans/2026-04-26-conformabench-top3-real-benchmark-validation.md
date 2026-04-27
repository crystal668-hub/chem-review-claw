# ConformaBench Top 3 Real Benchmark Validation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run the real ChemQA benchmark on the first three ConformaBench records with the same ChemQA model profile as the prior experiment, then inspect whether the previously observed false execution-failure pattern recurs and kill any leftover benchmark processes.

**Architecture:** Keep this as a pure runtime validation pass, not an implementation pass. Use the existing `benchmark_test.py` entrypoint with `group=chemqa_web_on`, `offset=0`, `limit=3`, and the prior ChemQA model profile `chemqa-review-su8-coord-qwen-ds-kimi-glm-minimax`. After the run, compare the resulting per-record statuses and archived artifacts for records `0001`–`0003`, then inspect the process table for any surviving benchmark / ChemQA / driver processes and terminate only those tied to this validation run.

**Tech Stack:** Python CLI benchmark harness, OpenClaw runtime, shell process inspection (`ps`, `pgrep`, `pkill`)

---

### Task 1: Fix the Validation Target and Output Location

**Files:**
- Create: `docs/superpowers/plans/2026-04-26-conformabench-top3-real-benchmark-validation.md`
- Read: `benchmark_test.py`
- Read: `workspace/state/benchmark-runs/conformabench-small-qwen-web-on/results.json`

- [ ] **Step 1: Confirm the prior ChemQA model profile and target records**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

results_path = Path("/Users/xutao/.openclaw/workspace/state/benchmark-runs/conformabench-small-qwen-web-on/results.json")
data = json.loads(results_path.read_text(encoding="utf-8"))
print("chemqa_model_profile =", data["results"][0]["runner_meta"]["launch"]["compile"]["request_snapshot"]["overrides"]["model_profile"])
print("run_groups =", [item["id"] for item in data["run_groups"]])
PY
```

Expected:
- `chemqa_model_profile = chemqa-review-su8-coord-qwen-ds-kimi-glm-minimax`
- Prior run groups include `chemqa_web_on`

- [ ] **Step 2: Choose a fresh exact output directory for this validation run**

Use this path exactly:

```text
/Users/xutao/.openclaw/workspace/state/benchmark-runs/conformabench-top3-qwen-web-on-rerun-20260426
```

This must be a fresh directory so we do not merge with the older `conformabench-small-qwen-web-on` results.

- [ ] **Step 3: Remove any stale directory at the exact output path before starting**

Run:

```bash
rm -rf /Users/xutao/.openclaw/workspace/state/benchmark-runs/conformabench-top3-qwen-web-on-rerun-20260426
```

Expected:
- Command exits `0`
- The target output path does not exist before the new run starts

---

### Task 2: Run the Real Benchmark on ConformaBench Records 0001–0003

**Files:**
- Read: `benchmark_test.py`
- Write: output under `/Users/xutao/.openclaw/workspace/state/benchmark-runs/conformabench-top3-qwen-web-on-rerun-20260426`

- [ ] **Step 1: Print the selected records before running**

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
  --exact-output-dir /Users/xutao/.openclaw/workspace/state/benchmark-runs/conformabench-top3-qwen-web-on-rerun-20260426 \
  --print-selected-records
```

Expected:
- Output lists exactly `conformabench-0001`, `conformabench-0002`, `conformabench-0003`
- No benchmark execution starts in this step

- [ ] **Step 2: Run the real ChemQA benchmark**

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
  --exact-output-dir /Users/xutao/.openclaw/workspace/state/benchmark-runs/conformabench-top3-qwen-web-on-rerun-20260426
```

Expected:
- The command exits on its own with a JSON payload containing `output_dir`
- Results are materialized under the exact output directory
- Per-record JSON files exist for `conformabench-0001`, `conformabench-0002`, and `conformabench-0003`

- [ ] **Step 3: Snapshot the top-level result files**

Run:

```bash
find /Users/xutao/.openclaw/workspace/state/benchmark-runs/conformabench-top3-qwen-web-on-rerun-20260426 -maxdepth 2 -type f | sort
```

Expected:
- Includes `results.json`
- Includes `runtime-manifest.json`
- Includes `per-record/chemqa_web_on/conformabench-0001.json`
- Includes `per-record/chemqa_web_on/conformabench-0002.json`
- Includes `per-record/chemqa_web_on/conformabench-0003.json`

---

### Task 3: Inspect Whether the Prior Failure Pattern Recurred

**Files:**
- Read: `/Users/xutao/.openclaw/workspace/state/benchmark-runs/conformabench-top3-qwen-web-on-rerun-20260426/results.json`
- Read: `/Users/xutao/.openclaw/workspace/state/benchmark-runs/conformabench-top3-qwen-web-on-rerun-20260426/per-record/chemqa_web_on/*.json`
- Read: archived artifact directories under `/Users/xutao/.openclaw/workspace/state/benchmark-runs/conformabench-top3-qwen-web-on-rerun-20260426/artifacts/chemqa_web_on/`

- [ ] **Step 1: Summarize the three per-record outcomes**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

base = Path("/Users/xutao/.openclaw/workspace/state/benchmark-runs/conformabench-top3-qwen-web-on-rerun-20260426/per-record/chemqa_web_on")
for record_id in ("conformabench-0001", "conformabench-0002", "conformabench-0003"):
    payload = json.loads((base / f"{record_id}.json").read_text(encoding="utf-8"))
    runner_meta = payload.get("runner_meta") or {}
    print("==", record_id, "==")
    print("primary_metric =", payload["evaluation"]["primary_metric"])
    print("error =", payload.get("error"))
    print("short_answer_text =", repr(payload.get("short_answer_text", "")))
    print("terminal_state =", runner_meta.get("terminal_state"))
    print("terminal_reason_code =", runner_meta.get("terminal_reason_code"))
    print("acceptance_status =", runner_meta.get("acceptance_status"))
    print("reconciled_from_archived_artifacts =", runner_meta.get("reconciled_from_archived_artifacts"))
    print()
PY
```

Expected:
- Produces a compact summary for all three records
- Lets us directly see whether a `completed/rejected` run still lands as `execution_error`

- [ ] **Step 2: If any record shows `execution_error`, inspect its archived protocol and qa_result**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

per_record = Path("/Users/xutao/.openclaw/workspace/state/benchmark-runs/conformabench-top3-qwen-web-on-rerun-20260426/per-record/chemqa_web_on")
for record_id in ("conformabench-0001", "conformabench-0002", "conformabench-0003"):
    payload = json.loads((per_record / f"{record_id}.json").read_text(encoding="utf-8"))
    if payload["evaluation"]["primary_metric"] != "execution_error":
        continue
    runner_meta = payload.get("runner_meta") or {}
    protocol_path = runner_meta.get("archived_protocol_path")
    qa_result_path = runner_meta.get("qa_result_path")
    print("== execution_error artifact check:", record_id, "==")
    print("archived_protocol_path =", protocol_path)
    print("qa_result_path =", qa_result_path)
    if protocol_path and Path(protocol_path).is_file():
        protocol_text = Path(protocol_path).read_text(encoding="utf-8")
        print("protocol terminal_state line =", next((line for line in protocol_text.splitlines() if line.startswith("terminal_state:")), ""))
        print("protocol acceptance_status line =", next((line for line in protocol_text.splitlines() if line.startswith("acceptance_status:")), ""))
    if qa_result_path and Path(qa_result_path).is_file():
        qa_result = json.loads(Path(qa_result_path).read_text(encoding="utf-8"))
        print("qa_result terminal_state =", qa_result.get("terminal_state"))
        print("qa_result acceptance_status =", qa_result.get("acceptance_status"))
        print("qa_result final_answer type =", type(qa_result.get("final_answer")).__name__)
    print()
PY
```

Expected:
- Either no records need this inspection, or the inspection exposes the exact mismatch surface again

- [ ] **Step 3: Capture whether the prior false-failure pattern reproduced**

Decision rule:
- If a record has `evaluation.primary_metric == "execution_error"` while archived `protocol` or `qa_result` says `terminal_state == "completed"`, the old split-brain pattern reproduced.
- If completed rejections are no longer labeled `execution_error`, the specific issue under test did not reproduce in this rerun.

Record the answer in the final report record-by-record.

---

### Task 4: Check for Residual Benchmark Processes and Kill Them

**Files:**
- No code files; inspect live process table only

- [ ] **Step 1: List candidate residual processes**

Run:

```bash
ps -axo pid=,ppid=,command= | rg 'benchmark_test.py|chemqa_review_openclaw_driver.py|recover_run.py|launch_from_preset.py|benchmark-chemqa_web_on-conformabench-000[123]'
```

Expected:
- Either no matching processes remain, or the output clearly identifies leftover benchmark-related processes

- [ ] **Step 2: Kill only the leftover processes tied to this validation run**

If Step 1 shows matching live processes, run:

```bash
ps -axo pid=,command= | rg 'benchmark_test.py|chemqa_review_openclaw_driver.py|recover_run.py|launch_from_preset.py|benchmark-chemqa_web_on-conformabench-000[123]'
```

Then kill the listed PIDs explicitly with:

```bash
kill <pid1> <pid2> ...
```

If any survive `kill`, follow with:

```bash
kill -9 <pid1> <pid2> ...
```

Expected:
- Only residual benchmark processes for this run are terminated
- No unrelated long-lived user processes are touched

- [ ] **Step 3: Re-check process table**

Run:

```bash
ps -axo pid=,ppid=,command= | rg 'benchmark_test.py|chemqa_review_openclaw_driver.py|recover_run.py|launch_from_preset.py|benchmark-chemqa_web_on-conformabench-000[123]'
```

Expected:
- No matching processes remain

---

## Self-Review

- Spec coverage: This plan covers the requested real benchmark rerun on the first three ConformaBench records, preserves the prior ChemQA model profile, checks whether the earlier anomaly recurs, and performs explicit residual-process cleanup.
- Placeholder scan: No TODO/TBD markers remain. All commands, paths, and decision criteria are concrete.
- Type consistency: Output inspection consistently uses `per-record/chemqa_web_on/*.json`, archived artifact paths from `runner_meta`, and the exact ChemQA model profile from the previous run.
