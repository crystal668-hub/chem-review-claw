# ChemQA Rejected Completed Answer Sanitization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent completed-but-rejected ChemQA runs from feeding structured rejection blobs or narrative placeholders into ConformaBench scoring as if they were real final answers.

**Architecture:** Keep the fix narrowly inside ChemQA answer reconstruction. `build_chemqa_full_response()` should distinguish between accepted-answer payloads and rejected/no-answer payloads, only emitting a scoreable short answer when the archived artifacts actually contain one. Rejected completed runs must still return a readable full response for diagnostics, but their short answer track must stay empty unless a real scalar answer exists.

**Tech Stack:** Python, `unittest`, existing benchmark runner / reporting utilities

---

### Task 1: Lock Down Rejected-Completed Reconstruction With Failing Tests

**Files:**
- Modify: `tests/test_benchmark_test.py`
- Read: `benchmark_test.py`

- [ ] **Step 1: Add a failing unit test for `build_chemqa_full_response()` when `qa_result.final_answer` is a structured rejection blob**

Add this test near the existing `build_chemqa_full_response` coverage in `tests/test_benchmark_test.py`:

```python
    def test_build_chemqa_full_response_rejected_blob_does_not_return_blob_as_short_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            rejection_blob = {
                "accepted_owner": "",
                "answer": None,
                "direct_answer": None,
                "summary": "No candidate submission achieved acceptance.",
            }
            final_answer_path = temp_dir / "final_answer.md"
            final_answer_path.write_text(json.dumps(rejection_blob, ensure_ascii=False, indent=2), encoding="utf-8")
            qa_result = {
                "final_answer": json.dumps(rejection_blob, ensure_ascii=False, indent=2),
                "acceptance_status": "rejected",
                "terminal_state": "completed",
                "artifact_paths": {
                    "final_answer": str(final_answer_path),
                },
            }

            short_text, full_text = benchmark_test.build_chemqa_full_response(qa_result=qa_result)

            self.assertEqual("", short_text)
            self.assertIn("No candidate submission achieved acceptance.", full_text)
            self.assertNotIn("FINAL ANSWER:", full_text)
```

- [ ] **Step 2: Add a failing runner-level regression test for reconciled rejected runs**

Add this test immediately after `test_chemqa_runner_reconciles_failed_run_status_with_completed_archived_rejection` in `tests/test_benchmark_test.py`:

```python
    def test_chemqa_runner_reconciled_rejected_run_does_not_expose_blob_as_short_answer(self) -> None:
        original_run_subprocess = benchmark_test.run_subprocess
        original_ensure_runtime_bundle = benchmark_test.ensure_runtime_bundle
        original_wait_for_terminal_status = benchmark_test.ChemQARunner._wait_for_terminal_status
        original_collect_artifacts = benchmark_test.ChemQARunner._collect_artifacts_from_source
        original_invoke_cleanroom_cleanup = benchmark_test.invoke_cleanroom_cleanup
        try:
            benchmark_test.run_subprocess = lambda command, *, env=None, cwd=None, timeout=None: benchmark_test.subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"run_id": "demo", "launch_mode": "run", "launched": {"returncode": 0}}),
                stderr="",
            )
            benchmark_test.ensure_runtime_bundle = lambda record, bundle_root: None
            benchmark_test.invoke_cleanroom_cleanup = lambda manifest_path: {"status": "cleaned", "manifest_path": str(manifest_path)}
            benchmark_test.ChemQARunner._wait_for_terminal_status = lambda self, run_id, timeout_seconds: {
                "status": "done",
                "terminal_state": "failed",
                "terminal_reason_code": "stalled",
                "artifact_collection": {},
                "protocol_path": str(self.chemqa_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id / "chemqa_review_protocol.yaml"),
            }

            def fake_collect_artifacts(self, *, source_dir, output_dir, env):
                output_dir.mkdir(parents=True, exist_ok=True)
                rejection_blob = {
                    "accepted_owner": "",
                    "answer": None,
                    "direct_answer": None,
                    "summary": "No candidate submission achieved acceptance.",
                }
                qa_result_path = output_dir / "qa_result.json"
                qa_result_path.write_text(
                    json.dumps(
                        {
                            "final_answer": json.dumps(rejection_blob, ensure_ascii=False, indent=2),
                            "artifact_paths": {
                                "qa_result": str(qa_result_path),
                                "final_answer": str(output_dir / "final_answer.md"),
                            },
                            "acceptance_status": "rejected",
                            "terminal_state": "completed",
                        }
                    ),
                    encoding="utf-8",
                )
                (output_dir / "final_answer.md").write_text(
                    json.dumps(rejection_blob, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

            benchmark_test.ChemQARunner._collect_artifacts_from_source = fake_collect_artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                output_root = Path(tmpdir) / "benchmark-output"
                launch_root = output_root / "chemqa-launch"
                chemqa_root = Path(tmpdir) / "chemqa-root"
                runner = benchmark_test.ChemQARunner(
                    chemqa_root=chemqa_root,
                    timeout_seconds=30,
                    config_path=Path(tmpdir) / "config.json",
                    slot_set="A",
                    review_rounds=None,
                    rebuttal_rounds=None,
                    model_profile="profile-x",
                    runtime_bundle_root=Path(tmpdir) / "bundles",
                    launch_workspace_root=launch_root,
                )
                record = benchmark_test.BenchmarkRecord(
                    record_id="conformabench-0001",
                    dataset="conformabench",
                    source_file="/tmp/demo.jsonl",
                    eval_kind="conformabench_constructive",
                    prompt="Design a molecule.",
                    reference_answer="Points: 1.0, Item: ok",
                    payload={},
                )
                run_id = "benchmark-chemqa_web_on-conformabench-0001-20260424-000000"
                protocol_dir = chemqa_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id
                protocol_dir.mkdir(parents=True, exist_ok=True)
                (protocol_dir / "chemqa_review_protocol.yaml").write_text(
                    "question: Demo\nacceptance_status: rejected\nterminal_state: completed\nfinal_answer: \"\"\n",
                    encoding="utf-8",
                )
                runner._now_stamp = lambda: "20260424-000000"

                out = runner.run(record, benchmark_test.EXPERIMENT_GROUPS["chemqa_web_on"])

                self.assertEqual(benchmark_test.RunStatus.COMPLETED, out.status)
                self.assertEqual("", out.short_answer_text)
                self.assertIn("No candidate submission achieved acceptance.", out.full_response_text)
            finally:
                benchmark_test.run_subprocess = original_run_subprocess
                benchmark_test.ensure_runtime_bundle = original_ensure_runtime_bundle
                benchmark_test.invoke_cleanroom_cleanup = original_invoke_cleanroom_cleanup
                benchmark_test.ChemQARunner._wait_for_terminal_status = original_wait_for_terminal_status
                benchmark_test.ChemQARunner._collect_artifacts_from_source = original_collect_artifacts
```

- [ ] **Step 3: Run the two new tests and verify they fail for the right reason**

Run:

```bash
python3 -m pytest /Users/xutao/.config/superpowers/worktrees/openclaw/chemqa-recovery-status/tests/test_benchmark_test.py -k 'rejected_blob_does_not_return_blob_as_short_answer or reconciled_rejected_run_does_not_expose_blob_as_short_answer' -v
```

Expected:
- Both tests fail.
- Failure shows the current code returns the serialized rejection blob or a non-empty placeholder through `short_answer_text`.

---

### Task 2: Sanitize Rejected Completed Answer Reconstruction

**Files:**
- Modify: `benchmark_test.py`
- Test: `tests/test_benchmark_test.py`

- [ ] **Step 1: Add a small helper that extracts only scoreable scalar ChemQA answers**

Insert this helper above `build_chemqa_full_response()` in `benchmark_test.py`:

```python
def extract_chemqa_scoreable_answer(value: Any) -> str:
    if isinstance(value, str):
        stripped = normalize_space(value)
        if not stripped:
            return ""
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except Exception:
                return stripped
            return extract_chemqa_scoreable_answer(parsed)
        return stripped
    if isinstance(value, dict):
        for key in ("direct_answer", "answer", "value", "final_answer"):
            candidate = extract_chemqa_scoreable_answer(value.get(key))
            if candidate:
                return candidate
        return ""
    return ""
```
```

- [ ] **Step 2: Change `build_chemqa_full_response()` to use the helper and preserve diagnostics without inventing an answer**

Replace the body of `build_chemqa_full_response()` in `benchmark_test.py` with:

```python
def build_chemqa_full_response(*, qa_result: dict[str, Any]) -> tuple[str, str]:
    artifact_paths = dict(qa_result.get("artifact_paths") or {})
    short_answer_text = extract_chemqa_scoreable_answer(qa_result.get("final_answer"))
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
```

Behavior target:
- Accepted scalar answer: unchanged.
- Rejected dict / JSON blob / empty answer: `short_answer_text == ""`.
- Diagnostic narrative in `final_answer.md`: preserved in `full_response_text`.

- [ ] **Step 3: Run the focused tests again and verify they pass**

Run:

```bash
python3 -m pytest /Users/xutao/.config/superpowers/worktrees/openclaw/chemqa-recovery-status/tests/test_benchmark_test.py -k 'build_chemqa_full_response_uses_final_submission_rationale or rejected_blob_does_not_return_blob_as_short_answer or reconciled_rejected_run_does_not_expose_blob_as_short_answer or reconciles_failed_run_status_with_completed_archived_rejection' -v
```

Expected:
- All selected tests pass.
- Existing accepted-answer reconstruction test still passes.
- Existing reconciled rejection test still passes with its updated expectations.

---

### Task 3: Verify No Regression In Execution-Error Materialization Path

**Files:**
- Test: `tests/test_benchmark_test.py`
- Read: `benchmarking/contracts.py`
- Read: `benchmarking/reporting.py`

- [ ] **Step 1: Run the recovery/error-materialization tests that guard unscored fallback behavior**

Run:

```bash
python3 -m pytest /Users/xutao/.config/superpowers/worktrees/openclaw/chemqa-recovery-status/tests/test_benchmark_test.py -k 'run_group_marks_unscored_recovery_as_execution_error or structural_unscored_recovery_without_failure_attr_uses_runner_meta_error' -v
```

Expected:
- Both tests pass unchanged.
- This confirms the new answer sanitization did not alter the separate `RECOVERED` / execution-error reporting contract.

- [ ] **Step 2: Run the full targeted benchmark test slice for this bug family**

Run:

```bash
python3 -m pytest /Users/xutao/.config/superpowers/worktrees/openclaw/chemqa-recovery-status/tests/test_benchmark_test.py -k 'chemqa_full_response or chemqa_runner_reconciles_failed_run_status_with_completed_archived_rejection or run_group_marks_unscored_recovery_as_execution_error or structural_unscored_recovery_without_failure_attr_uses_runner_meta_error' -v
```

Expected:
- All selected tests pass.
- No new failures in nearby ChemQA benchmark behavior.

- [ ] **Step 3: Record the outcome**

Capture in the final report:
- Which tests were added
- Which commands were run
- Whether rejected completed runs now keep an empty short answer track while preserving readable diagnostics

---

## Self-Review

- Spec coverage: This plan covers only the P1 review finding about rejected completed runs leaking invalid answers into ConformaBench scoring. It does not attempt to fix archival provenance mismatch or recovery JSON semantics.
- Placeholder scan: No TODO/TBD markers remain; every code change and test command is concrete.
- Type consistency: The helper returns `str`, `build_chemqa_full_response()` still returns `tuple[str, str]`, and the runner-level expectations remain `RunStatus.COMPLETED` for reconciled completed rejections.
