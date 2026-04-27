# ChemQA Evaluable Answer Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ChemQA benchmark recovery preserve evaluable answers whenever a trustworthy answer can be reconstructed, and expose the new evaluability semantics through per-record results and aggregate reporting.

**Architecture:** Keep the first rollout additive. Extend runner contracts and ChemQA fallback classification so degraded executions can still return scoreable `RECOVERED` results. Then extend `GroupRecordResult`, aggregation, JSON, and CSV outputs with explicit evaluability and degradation fields while preserving legacy score fields during the migration window.

**Tech Stack:** Python, `unittest`, existing benchmark runner / reporting utilities, CSV export, JSON result manifests

---

### Task 1: Lock Down Recovery Contract and Scoreability With Failing Tests

**Files:**
- Modify: `tests/test_benchmark_test.py`
- Read: `benchmarking/contracts.py`
- Read: `benchmark_test.py`

- [ ] **Step 1: Add a failing contract-level test for scoreable recovered answers**

Add this test near the existing `RunnerResult.should_score()`-related coverage in `tests/test_benchmark_test.py`:

```python
    def test_runner_result_should_score_when_recovery_is_evaluable(self) -> None:
        result = benchmark_test.RunnerResult(
            status=benchmark_test.RunStatus.RECOVERED,
            answer=benchmark_test.AnswerPayload(
                short_answer_text="CCO",
                full_response_text="FINAL ANSWER: CCO",
            ),
            raw={"run_status": {"status": "done", "terminal_state": "failed"}},
            runner_meta={"run_id": "demo-run"},
            recovery=benchmark_test.RecoveryInfo(
                source="candidate_submission",
                scored=True,
                details={
                    "evaluable": True,
                    "reliability": "high_confidence_recovered",
                    "recovery_mode": "candidate_submission",
                },
            ),
        )

        self.assertTrue(result.should_score())
```

- [ ] **Step 2: Add a failing ChemQA runner test for failed workflow but evaluable recovered answer**

Add this test after `test_chemqa_runner_archives_protocol_and_rebuilds_qa_result_for_failed_terminal_run` in `tests/test_benchmark_test.py`:

```python
    def test_chemqa_runner_failed_terminal_with_candidate_fallback_returns_scored_recovered_result(self) -> None:
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
                (output_dir / "final_answer.md").write_text("FINAL ANSWER: CCO\n", encoding="utf-8")

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
                    record_id="chembench-0001",
                    dataset="chembench",
                    source_file="/tmp/demo.jsonl",
                    eval_kind="chembench_open_ended",
                    prompt="Return ethanol.",
                    reference_answer="CCO",
                    payload={},
                )
                run_id = "benchmark-chemqa_web_on-chembench-0001-20260427-000000"
                protocol_dir = chemqa_root / "generated" / "clawteam-data" / "runs" / run_id / "teams" / run_id
                protocol_dir.mkdir(parents=True, exist_ok=True)
                (protocol_dir / "chemqa_review_protocol.yaml").write_text(
                    "\\n".join(
                        [
                            "artifact_kind: coordinator_protocol",
                            "artifact_contract_version: react-reviewed-v2",
                            "terminal_state: failed",
                            "acceptance_status: failed",
                            "candidate_submission:",
                            "  owner: proposer-1",
                            "  direct_answer: CCO",
                            "  summary: recovered answer",
                            "  submission_trace:",
                            "    - step: structural_reasoning",
                            "      status: success",
                            "      detail: reconstructed from proposer artifact",
                        ]
                    ) + "\\n",
                    encoding="utf-8",
                )
                runner._now_stamp = lambda: "20260427-000000"

                out = runner.run(record, benchmark_test.EXPERIMENT_GROUPS["chemqa_web_on"])

                self.assertEqual(benchmark_test.RunStatus.RECOVERED, out.status)
                self.assertEqual("CCO", out.short_answer_text)
                self.assertIn("FINAL ANSWER: CCO", out.full_response_text)
                self.assertTrue(out.should_score())
                self.assertTrue(out.recovery.scored)
                self.assertEqual("candidate_submission", out.recovery.source)
        finally:
            benchmark_test.run_subprocess = original_run_subprocess
            benchmark_test.ensure_runtime_bundle = original_ensure_runtime_bundle
            benchmark_test.invoke_cleanroom_cleanup = original_invoke_cleanroom_cleanup
            benchmark_test.ChemQARunner._wait_for_terminal_status = original_wait_for_terminal_status
            benchmark_test.ChemQARunner._collect_artifacts_from_source = original_collect_artifacts
```

- [ ] **Step 3: Add a failing run-group test proving scoreable recovery reaches the evaluator**

Replace the existing unscored assumption with a new test next to `test_run_group_marks_unscored_recovery_as_execution_error` in `tests/test_benchmark_test.py`:

```python
    def test_run_group_scores_evaluable_recovery(self) -> None:
        record = benchmark_test.BenchmarkRecord(
            record_id="recovered-record",
            dataset="chembench",
            source_file="/tmp/demo.jsonl",
            eval_kind="chembench_open_ended",
            prompt="Q",
            reference_answer="fallback-answer",
            payload={},
        )
        recovered_result = benchmark_test.RunnerResult(
            status=benchmark_test.RunStatus.RECOVERED,
            answer=benchmark_test.AnswerPayload(
                short_answer_text="fallback-answer",
                full_response_text="FINAL ANSWER: fallback-answer",
            ),
            raw={"run_status": {"status": "done", "terminal_state": "failed"}},
            runner_meta={
                "run_id": "demo-run",
                "fallback_used": True,
                "fallback_source": "candidate_submission",
                "error": "ChemQA run `demo-run` terminated as failed (reason=stalled)",
            },
            recovery=benchmark_test.RecoveryInfo(
                source="candidate_submission",
                scored=True,
                details={
                    "evaluable": True,
                    "reliability": "high_confidence_recovered",
                    "recovery_mode": "candidate_submission",
                },
            ),
        )

        class StubRunner:
            def run(self, record: object, group: object) -> benchmark_test.RunnerResult:
                return recovered_result

        original_build_runner = getattr(benchmark_test, "build_runner", None)
        original_evaluate_answer = benchmark_test.evaluate_answer
        try:
            benchmark_test.build_runner = lambda **kwargs: StubRunner()
            benchmark_test.evaluate_answer = lambda *args, **kwargs: benchmark_test.EvaluationResult(
                eval_kind="chembench_open_ended",
                score=1.0,
                max_score=1.0,
                normalized_score=1.0,
                passed=True,
                primary_metric="exact_str_match",
                primary_metric_direction="higher_is_better",
                details={},
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                results = benchmark_test.run_group(
                    group=benchmark_test.EXPERIMENT_GROUPS["chemqa_web_off"],
                    records=[record],
                    output_root=Path(tmpdir),
                    single_timeout=10,
                    chemqa_timeout=10,
                    judge=object(),
                    config_path=Path(tmpdir) / "cfg.json",
                    single_agent="unused",
                    chemqa_root=Path(tmpdir),
                    chemqa_model_profile="unused",
                    review_rounds=None,
                    rebuttal_rounds=None,
                )
            entry = results[0]
            self.assertIsNone(entry.error)
            self.assertTrue(entry.evaluation["passed"])
            self.assertEqual("fallback-answer", entry.short_answer_text)
            self.assertEqual("FINAL ANSWER: fallback-answer", entry.full_response_text)
        finally:
            if original_build_runner is None:
                delattr(benchmark_test, "build_runner")
            else:
                benchmark_test.build_runner = original_build_runner
            benchmark_test.evaluate_answer = original_evaluate_answer
```

- [ ] **Step 4: Run the targeted failing tests**

Run:

```bash
python3 -m pytest /Users/xutao/.openclaw/workspace/tests/test_benchmark_test.py -k 'should_score_when_recovery_is_evaluable or failed_terminal_with_candidate_fallback_returns_scored_recovered_result or scores_evaluable_recovery' -v
```

Expected:

- the new tests fail
- at least one failure shows recovered ChemQA answers are still treated as unscored execution errors

- [ ] **Step 5: Commit the failing test baseline**

```bash
git add /Users/xutao/.openclaw/workspace/tests/test_benchmark_test.py
git commit -m "test: lock down evaluable ChemQA recovery semantics"
```

### Task 2: Make Recovered ChemQA Answers Scoreable

**Files:**
- Modify: `benchmarking/contracts.py`
- Modify: `benchmarking/runners/chemqa.py`
- Test: `tests/test_benchmark_test.py`

- [ ] **Step 1: Extend `RecoveryInfo` with explicit evaluability metadata**

Update `benchmarking/contracts.py`:

```python
@dataclass(frozen=True)
class RecoveryInfo:
    source: str
    scored: bool
    evaluable: bool = False
    reliability: str = "none"
    recovery_mode: str = "none"
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)
```

Keep `RunnerResult.should_score()` as:

```python
    def should_score(self) -> bool:
        if self.status is RunStatus.COMPLETED:
            return True
        return (
            self.status is RunStatus.RECOVERED
            and self.recovery is not None
            and self.recovery.scored
            and self.recovery.evaluable
        )
```

- [ ] **Step 2: Run the contract test and verify only contract behavior still fails downstream**

Run:

```bash
python3 -m pytest /Users/xutao/.openclaw/workspace/tests/test_benchmark_test.py -k 'should_score_when_recovery_is_evaluable' -v
```

Expected:

- the contract-level test passes
- the runner-level tests still fail

- [ ] **Step 3: Introduce a structured ChemQA recovery assessment helper**

In `benchmarking/runners/chemqa.py`, add a helper above the non-success branch:

```python
    def _assess_recovered_answer(
        self,
        *,
        run_id: str,
        run_status: dict[str, Any],
        archive_meta: dict[str, Any],
    ) -> dict[str, Any] | None:
        fallback_payload = self._build_candidate_submission_fallback(run_id, run_status)
        if fallback_payload is None:
            return None
        short_answer_text, full_response_text, fallback_meta = fallback_payload
        short_text = self._normalize_space(short_answer_text)
        if not short_text:
            return {
                "evaluable": False,
                "scored": False,
                "reliability": "none",
                "recovery_mode": str(fallback_meta.get("fallback_source") or "none"),
                "reason": "empty_short_answer",
                "short_answer_text": "",
                "full_response_text": full_response_text,
                "details": fallback_meta,
            }
        recovery_mode = str(fallback_meta.get("fallback_source") or "candidate_submission")
        reliability = "high_confidence_recovered" if recovery_mode != "run-status-final-answer-preview" else "low_confidence_recovered"
        return {
            "evaluable": True,
            "scored": True,
            "reliability": reliability,
            "recovery_mode": recovery_mode,
            "reason": "",
            "short_answer_text": short_text,
            "full_response_text": full_response_text,
            "details": fallback_meta,
        }
```

- [ ] **Step 4: Replace the hardcoded `scored=False` fallback branch**

In the non-success section of `ChemQARunner.run()`, replace the existing fallback return with:

```python
                recovery_assessment = self._assess_recovered_answer(
                    run_id=run_id,
                    run_status=run_status,
                    archive_meta=archive_meta,
                )
                if recovery_assessment is not None and recovery_assessment["evaluable"]:
                    runner_meta.update(
                        {
                            "fallback_used": True,
                            **recovery_assessment["details"],
                            "evaluable": True,
                            "scored": True,
                            "recovery_mode": recovery_assessment["recovery_mode"],
                            "answer_reliability": recovery_assessment["reliability"],
                            "degraded_execution": True,
                        }
                    )
                    return RunnerResult(
                        status=RunStatus.RECOVERED,
                        answer=AnswerPayload(
                            short_answer_text=recovery_assessment["short_answer_text"],
                            full_response_text=recovery_assessment["full_response_text"],
                        ),
                        raw={"run_status": run_status, "fallback": recovery_assessment["details"]},
                        runner_meta=runner_meta,
                        recovery=RecoveryInfo(
                            source=str(recovery_assessment["recovery_mode"]),
                            scored=True,
                            evaluable=True,
                            reliability=str(recovery_assessment["reliability"]),
                            recovery_mode=str(recovery_assessment["recovery_mode"]),
                            reason=str(recovery_assessment["reason"]),
                            details=recovery_assessment["details"],
                        ),
                    )
```

Retain the existing `FAILED` return when assessment is absent or not evaluable.

- [ ] **Step 5: Run the recovery-focused tests**

Run:

```bash
python3 -m pytest /Users/xutao/.openclaw/workspace/tests/test_benchmark_test.py -k 'failed_terminal_with_candidate_fallback_returns_scored_recovered_result or scores_evaluable_recovery or run_group_marks_unscored_recovery_as_execution_error' -v
```

Expected:

- the new scoreable recovery tests pass
- the old unscored-recovery test now fails and must be updated in the next task

- [ ] **Step 6: Commit the runner recovery behavior change**

```bash
git add /Users/xutao/.openclaw/workspace/benchmarking/contracts.py /Users/xutao/.openclaw/workspace/benchmarking/runners/chemqa.py /Users/xutao/.openclaw/workspace/tests/test_benchmark_test.py
git commit -m "feat: score evaluable ChemQA recovery results"
```

### Task 3: Extend Per-Record Result Schema With Evaluability Axes

**Files:**
- Modify: `benchmarking/reporting.py`
- Modify: `benchmark_test.py`
- Test: `tests/test_benchmark_test.py`

- [ ] **Step 1: Add new fields to `GroupRecordResult`**

Update `benchmarking/reporting.py`:

```python
@dataclass
class GroupRecordResult:
    schema_version: int
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
    run_lifecycle_status: str
    protocol_completion_status: str
    protocol_acceptance_status: str | None
    answer_availability: str
    answer_reliability: str
    evaluable: bool
    scored: bool
    recovery_mode: str
    degraded_execution: bool
    execution_error_kind: str | None = None
    error: str | None = None
    short_answer_text: str = ""
    full_response_text: str = ""
```
```

- [ ] **Step 2: Add a helper that derives result axes from runner output**

In `benchmark_test.py`, above `run_group()`, add:

```python
def build_result_axes_from_runner(run_result: Any) -> dict[str, Any]:
    runner_meta = dict(getattr(run_result, "runner_meta", None) or {})
    run_status = dict(runner_meta.get("run_status") or {})
    recovery = getattr(run_result, "recovery", None)
    lifecycle_status = "completed" if getattr(run_result, "status", None) in {RunStatus.COMPLETED, RunStatus.RECOVERED} else "failed"
    protocol_completion_status = "completed" if str(runner_meta.get("terminal_state") or "") == "completed" else ("failed" if run_status else "missing")
    protocol_acceptance_status = runner_meta.get("acceptance_status")
    if recovery is not None:
        answer_availability = "recovered_candidate" if str(getattr(recovery, "recovery_mode", "") or "") != "run-status-final-answer-preview" else "preview_only"
        answer_reliability = str(getattr(recovery, "reliability", "none") or "none")
        recovery_mode = str(getattr(recovery, "recovery_mode", "none") or "none")
        evaluable = bool(getattr(recovery, "evaluable", False))
        scored = bool(run_result.should_score())
        degraded_execution = True
    else:
        answer_availability = "native_final"
        answer_reliability = "native" if getattr(run_result, "status", None) is RunStatus.COMPLETED else "none"
        recovery_mode = "none"
        evaluable = bool(run_result.should_score())
        scored = bool(run_result.should_score())
        degraded_execution = getattr(run_result, "status", None) is not RunStatus.COMPLETED
    return {
        "schema_version": 2,
        "run_lifecycle_status": lifecycle_status,
        "protocol_completion_status": protocol_completion_status,
        "protocol_acceptance_status": protocol_acceptance_status,
        "answer_availability": answer_availability,
        "answer_reliability": answer_reliability,
        "evaluable": evaluable,
        "scored": scored,
        "recovery_mode": recovery_mode,
        "degraded_execution": degraded_execution,
        "execution_error_kind": None if scored else "execution_error",
    }
```

- [ ] **Step 3: Thread the new axes into scored and error result creation**

Update `run_group()` scored branch:

```python
                axes = build_result_axes_from_runner(run_result)
                entry = GroupRecordResult(
                    **axes,
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
```

Update `build_error_group_record_result()` to set:

```python
        schema_version=2,
        run_lifecycle_status="failed",
        protocol_completion_status="missing",
        protocol_acceptance_status=None,
        answer_availability="missing",
        answer_reliability="none",
        evaluable=False,
        scored=False,
        recovery_mode="none",
        degraded_execution=False,
        execution_error_kind="execution_error",
```

- [ ] **Step 4: Add a failing schema test before updating all fixtures**

Add this test near `test_results_json_keeps_legacy_top_level_shape`:

```python
    def test_group_record_result_includes_evaluability_axes(self) -> None:
        entry = benchmark_test.GroupRecordResult(
            schema_version=2,
            group_id="g1",
            group_label="Group 1",
            runner="single_llm",
            websearch=False,
            record_id="r1",
            subset="chembench",
            dataset="d1",
            source_file="/tmp/a.jsonl",
            eval_kind="chembench_open_ended",
            prompt="Q1",
            reference_answer="1",
            answer_text="1",
            evaluation={
                "eval_kind": "chembench_open_ended",
                "score": 1.0,
                "max_score": 1.0,
                "normalized_score": 1.0,
                "passed": True,
                "primary_metric": "exact_str_match",
                "primary_metric_direction": "higher_is_better",
                "details": {},
            },
            runner_meta={},
            raw={},
            elapsed_seconds=2.0,
            run_lifecycle_status="completed",
            protocol_completion_status="completed",
            protocol_acceptance_status=None,
            answer_availability="native_final",
            answer_reliability="native",
            evaluable=True,
            scored=True,
            recovery_mode="none",
            degraded_execution=False,
        )
        self.assertEqual(2, entry.schema_version)
        self.assertTrue(entry.evaluable)
        self.assertTrue(entry.scored)
```

- [ ] **Step 5: Run the schema-focused tests and update remaining constructor call sites**

Run:

```bash
python3 -m pytest /Users/xutao/.openclaw/workspace/tests/test_benchmark_test.py -k 'group_record_result_includes_evaluability_axes or aggregate_results_groups_by_experiment or export_csv_reports_writes_summary_files or results_json_keeps_legacy_top_level_shape' -v
```

Expected:

- fixture construction fails until every `GroupRecordResult(...)` test site includes the new fields

Then update every `GroupRecordResult(...)` instantiation in `tests/test_benchmark_test.py` to include:

```python
                schema_version=2,
                run_lifecycle_status="completed",
                protocol_completion_status="completed",
                protocol_acceptance_status=None,
                answer_availability="native_final",
                answer_reliability="native",
                evaluable=True,
                scored=True,
                recovery_mode="none",
                degraded_execution=False,
```

For deliberate error fixtures, use:

```python
                run_lifecycle_status="failed",
                protocol_completion_status="missing",
                protocol_acceptance_status=None,
                answer_availability="missing",
                answer_reliability="none",
                evaluable=False,
                scored=False,
                recovery_mode="none",
                degraded_execution=False,
                execution_error_kind="execution_error",
```

- [ ] **Step 6: Commit the schema extension**

```bash
git add /Users/xutao/.openclaw/workspace/benchmarking/reporting.py /Users/xutao/.openclaw/workspace/benchmark_test.py /Users/xutao/.openclaw/workspace/tests/test_benchmark_test.py
git commit -m "feat: add evaluability axes to benchmark record results"
```

### Task 4: Add Aggregate and CSV Metrics for Evaluability and Degraded Execution

**Files:**
- Modify: `benchmarking/reporting.py`
- Modify: `benchmark_test.py`
- Test: `tests/test_benchmark_test.py`

- [ ] **Step 1: Extend aggregate buckets with operational metrics**

Update `aggregate_bucket()` in `benchmarking/reporting.py`:

```python
def aggregate_bucket(items: list[GroupRecordResult]) -> dict[str, Any]:
    return {
        "count": len(items),
        "pass_count": sum(1 for item in items if item.evaluation["passed"]),
        "run_completed_count": sum(1 for item in items if item.run_lifecycle_status == "completed"),
        "run_failed_count": sum(1 for item in items if item.run_lifecycle_status == "failed"),
        "protocol_completed_count": sum(1 for item in items if item.protocol_completion_status == "completed"),
        "protocol_failed_count": sum(1 for item in items if item.protocol_completion_status == "failed"),
        "evaluable_count": sum(1 for item in items if item.evaluable),
        "scored_count": sum(1 for item in items if item.scored),
        "recovered_evaluable_count": sum(1 for item in items if item.evaluable and item.recovery_mode != "none"),
        "native_evaluable_count": sum(1 for item in items if item.evaluable and item.recovery_mode == "none"),
        "non_evaluable_count": sum(1 for item in items if not item.evaluable),
        "degraded_execution_count": sum(1 for item in items if item.degraded_execution),
        "avg_score": sum(float(item.evaluation["score"]) for item in items) / len(items),
        "avg_normalized_score": sum(float(item.evaluation["normalized_score"]) for item in items) / len(items),
        "avg_elapsed_seconds": sum(float(item.elapsed_seconds) for item in items) / len(items),
        "avg_answer_accuracy": average_optional_metric(items, "answer_accuracy"),
        "avg_rpf": average_optional_metric(items, "rpf"),
    }
```

- [ ] **Step 2: Add a failing aggregation test for the new counters**

Add this test in `tests/test_benchmark_test.py` after `test_aggregate_results_groups_by_experiment`:

```python
    def test_aggregate_results_tracks_evaluable_and_degraded_counts(self) -> None:
        sample = [
            benchmark_test.GroupRecordResult(
                schema_version=2,
                group_id="g1",
                group_label="Group 1",
                runner="chemqa",
                websearch=True,
                record_id="r1",
                subset="chembench",
                dataset="d1",
                source_file="/tmp/a.jsonl",
                eval_kind="chembench_open_ended",
                prompt="Q1",
                reference_answer="1",
                answer_text="1",
                evaluation={"eval_kind": "chembench_open_ended", "score": 1.0, "max_score": 1.0, "normalized_score": 1.0, "passed": True, "primary_metric": "exact_str_match", "primary_metric_direction": "higher_is_better", "details": {}},
                runner_meta={},
                raw={},
                elapsed_seconds=2.0,
                run_lifecycle_status="completed",
                protocol_completion_status="failed",
                protocol_acceptance_status="rejected",
                answer_availability="recovered_candidate",
                answer_reliability="high_confidence_recovered",
                evaluable=True,
                scored=True,
                recovery_mode="candidate_submission",
                degraded_execution=True,
            ),
            benchmark_test.GroupRecordResult(
                schema_version=2,
                group_id="g1",
                group_label="Group 1",
                runner="chemqa",
                websearch=True,
                record_id="r2",
                subset="chembench",
                dataset="d1",
                source_file="/tmp/a.jsonl",
                eval_kind="chembench_open_ended",
                prompt="Q2",
                reference_answer="2",
                answer_text="",
                evaluation={"eval_kind": "chembench_open_ended", "score": 0.0, "max_score": 1.0, "normalized_score": 0.0, "passed": False, "primary_metric": "execution_error", "primary_metric_direction": "higher_is_better", "details": {}},
                runner_meta={},
                raw={},
                elapsed_seconds=4.0,
                run_lifecycle_status="failed",
                protocol_completion_status="missing",
                protocol_acceptance_status=None,
                answer_availability="missing",
                answer_reliability="none",
                evaluable=False,
                scored=False,
                recovery_mode="none",
                degraded_execution=False,
                execution_error_kind="execution_error",
                error="missing answer",
            ),
        ]
        summary = benchmark_test.aggregate_results(sample)
        bucket = summary["groups"]["g1"]
        self.assertEqual(1, bucket["evaluable_count"])
        self.assertEqual(1, bucket["scored_count"])
        self.assertEqual(1, bucket["recovered_evaluable_count"])
        self.assertEqual(1, bucket["degraded_execution_count"])
        self.assertEqual(1, bucket["non_evaluable_count"])
```

- [ ] **Step 3: Extend CSV export with new columns**

In `benchmark_test.py`, update `export_csv_reports()` so both CSVs include:

```python
                "run_completed_count": group_summary["run_completed_count"],
                "protocol_completed_count": group_summary["protocol_completed_count"],
                "evaluable_count": group_summary["evaluable_count"],
                "scored_count": group_summary["scored_count"],
                "recovered_evaluable_count": group_summary["recovered_evaluable_count"],
                "degraded_execution_count": group_summary["degraded_execution_count"],
                "non_evaluable_count": group_summary["non_evaluable_count"],
```

and add corresponding fieldnames:

```python
            "run_completed_count",
            "protocol_completed_count",
            "evaluable_count",
            "scored_count",
            "recovered_evaluable_count",
            "degraded_execution_count",
            "non_evaluable_count",
```

- [ ] **Step 4: Update the existing CSV snapshot test**

In `tests/test_benchmark_test.py`, update `test_export_csv_reports_writes_summary_files` so the input summary contains the new keys and assert the new headers exactly:

```python
                    "run_completed_count": 2,
                    "protocol_completed_count": 2,
                    "evaluable_count": 2,
                    "scored_count": 2,
                    "recovered_evaluable_count": 0,
                    "degraded_execution_count": 0,
                    "non_evaluable_count": 0,
```

and:

```python
                    "run_completed_count",
                    "protocol_completed_count",
                    "evaluable_count",
                    "scored_count",
                    "recovered_evaluable_count",
                    "degraded_execution_count",
                    "non_evaluable_count",
```

- [ ] **Step 5: Run the aggregation and CSV tests**

Run:

```bash
python3 -m pytest /Users/xutao/.openclaw/workspace/tests/test_benchmark_test.py -k 'aggregate_results_tracks_evaluable_and_degraded_counts or aggregate_results_groups_by_experiment or export_csv_reports_writes_summary_files' -v
```

Expected:

- all aggregation and CSV tests pass

- [ ] **Step 6: Commit reporting changes**

```bash
git add /Users/xutao/.openclaw/workspace/benchmarking/reporting.py /Users/xutao/.openclaw/workspace/benchmark_test.py /Users/xutao/.openclaw/workspace/tests/test_benchmark_test.py
git commit -m "feat: report ChemQA evaluability and degraded execution metrics"
```

### Task 5: Version `results.json` and Preserve Legacy Top-Level Shape

**Files:**
- Modify: `benchmark_test.py`
- Test: `tests/test_benchmark_test.py`
- Read: `GLOBAL_DEV_SPEC.md`

- [ ] **Step 1: Add `schema_version` and status-axis description to `results.json`**

Update the `payload` assembly in `benchmark_test.py`:

```python
    payload = {
        "schema_version": 2,
        "status_axes_description": {
            "run_lifecycle_status": "completed|failed|cancelled",
            "protocol_completion_status": "completed|failed|missing|not_applicable",
            "answer_availability": "native_final|recovered_candidate|preview_only|missing",
            "answer_reliability": "native|high_confidence_recovered|low_confidence_recovered|none",
            "evaluable": "whether a record has a trustworthy scoreable answer",
            "scored": "whether evaluator execution occurred",
            "recovery_mode": "none|candidate_submission|run-status-final-answer-preview|archived_final_answer|protocol_reconstruction",
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        ...
    }
```

Do not remove existing top-level keys such as:

- `results`
- `summary`
- `errors`

- [ ] **Step 2: Add a results payload regression test**

Add this test after `test_results_json_keeps_legacy_top_level_shape`:

```python
    def test_results_json_payload_adds_schema_version_without_dropping_legacy_keys(self) -> None:
        sample = [
            benchmark_test.GroupRecordResult(
                schema_version=2,
                group_id="g1",
                group_label="Group 1",
                runner="single_llm",
                websearch=False,
                record_id="r1",
                subset="chembench",
                dataset="d1",
                source_file="/tmp/a.jsonl",
                eval_kind="chembench_open_ended",
                prompt="Q1",
                reference_answer="1",
                answer_text="1",
                evaluation={"eval_kind": "chembench_open_ended", "score": 1.0, "max_score": 1.0, "normalized_score": 1.0, "passed": True, "primary_metric": "exact_str_match", "primary_metric_direction": "higher_is_better", "details": {}},
                runner_meta={},
                raw={},
                elapsed_seconds=2.0,
                run_lifecycle_status="completed",
                protocol_completion_status="completed",
                protocol_acceptance_status=None,
                answer_availability="native_final",
                answer_reliability="native",
                evaluable=True,
                scored=True,
                recovery_mode="none",
                degraded_execution=False,
            )
        ]
        summary = benchmark_test.aggregate_results(sample)
        payload = {
            "schema_version": 2,
            "status_axes_description": {"evaluable": "desc"},
            "results": [benchmark_test.asdict(item) for item in sample],
            "summary": summary,
            "errors": [],
        }
        self.assertEqual(2, payload["schema_version"])
        self.assertIn("results", payload)
        self.assertIn("summary", payload)
        self.assertIn("errors", payload)
```

- [ ] **Step 3: Run the payload regression tests**

Run:

```bash
python3 -m pytest /Users/xutao/.openclaw/workspace/tests/test_benchmark_test.py -k 'results_json_keeps_legacy_top_level_shape or results_json_payload_adds_schema_version_without_dropping_legacy_keys' -v
```

Expected:

- both tests pass

- [ ] **Step 4: Commit payload versioning**

```bash
git add /Users/xutao/.openclaw/workspace/benchmark_test.py /Users/xutao/.openclaw/workspace/tests/test_benchmark_test.py
git commit -m "feat: version benchmark results payload for evaluability schema"
```

### Task 6: Update Documentation and Run Full Verification

**Files:**
- Modify: `GLOBAL_DEV_SPEC.md`
- Read: `docs/superpowers/specs/2026-04-27-chemqa-evaluable-answer-recovery-design.md`
- Test: `tests/test_benchmark_test.py`

- [ ] **Step 1: Document the new result semantics in `GLOBAL_DEV_SPEC.md`**

Add a short section under the benchmark reporting area in `GLOBAL_DEV_SPEC.md` describing:

- `schema_version = 2`
- new per-record axes
- distinction between `evaluable`, `scored`, and `passed`
- `pass_count` remains legacy-compatible but is no longer an operational stability metric

Use this text verbatim where appropriate:

```markdown
### Benchmark Result Status Axes

Per-record benchmark outputs now separate workflow and scoring semantics.

- `run_lifecycle_status` captures whether execution completed operationally.
- `protocol_completion_status` captures whether ChemQA protocol finalization succeeded.
- `evaluable` means a trustworthy scoreable answer exists.
- `scored` means the evaluator actually ran.
- `passed` remains the benchmark task outcome only.

Recovered ChemQA answers may therefore be `evaluable=true` and `scored=true` even when the underlying workflow degraded or failed.
```

- [ ] **Step 2: Run the focused benchmark test file**

Run:

```bash
python3 -m pytest /Users/xutao/.openclaw/workspace/tests/test_benchmark_test.py -v
```

Expected:

- the full benchmark test module passes

- [ ] **Step 3: Run a smoke test over the benchmark runner helpers**

Run:

```bash
python3 -m pytest /Users/xutao/.openclaw/workspace/tests/test_benchmark_test.py -k 'chemqa_runner or aggregate_results or export_csv_reports or run_group' -v
```

Expected:

- all ChemQA runner, grouping, and export regressions pass

- [ ] **Step 4: Commit docs and verification**

```bash
git add /Users/xutao/.openclaw/workspace/GLOBAL_DEV_SPEC.md /Users/xutao/.openclaw/workspace/tests/test_benchmark_test.py
git commit -m "docs: document evaluable answer benchmark semantics"
```
