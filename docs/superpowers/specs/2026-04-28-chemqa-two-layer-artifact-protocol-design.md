# ChemQA Two-Layer Artifact and Protocol Flow Design

## 中文设计摘要

本次临时基准测试中，`chemqa_web_on` 组大量出现“未完成”和“结果不可评估”，根因不是单一超时、提示词不够强、或者个别 artifact 文件丢失，而是当前系统在架构层面把太多协议耦合在同一条执行链路里：

- 协调器同时承担协议推进、角色调度、artifact 注册、artifact 校验、最终答案抽取、运行状态更新等职责。
- benchmark runner 在工作流终止后仍要反向重建 `qa_result.json`，导致“协议看起来完成”和“最终答案可持久、可评估”之间没有原子边界。
- 现有 `direct_answer` 语义过于单一，适合短答案和数值题，但会错误约束研究型、多段型、证据型回答。
- rebuttal 的语义不够明确，系统无法稳定判断它只是回应 reviewer，还是实际修订了最终候选答案。
- artifact 生命周期没有独立状态机，导致 cleanup、fallback recovery、protocol status 和 evaluator 输入之间存在脆弱的隐式依赖。

建议暂时不改变 `one proposer + four reviewers` 的基本格局，而是把系统抽象为两层：

1. **Protocol Flow**
   只负责 `propose -> review -> rebuttal` 的流程控制，包括当前 phase、下一个行动角色、哪些角色已经提交、phase 是否可前进、是否达到轮次或恢复限制。它不再决定最终答案内容，也不直接写 benchmark 的成功产物。

2. **Artifact Flow**
   负责 artifact 生命周期，包括 typed artifact 校验、candidate view 派生、review item 开闭状态、rebuttal 是否形成答案修订、final answer projection、`qa_result.json` 持久化，以及失败时的结构化 `FailureArtifact`。

双层架构的核心边界是：Protocol Flow 只按已注册的角色提交推进 phase；Artifact Flow 负责判断提交内容是否类型正确、是否形成可用候选答案、是否需要 rebuttal、是否可最终化。Protocol Flow 不读取 artifact 正文来推断 benchmark 答案；Artifact Flow 是唯一能够产出 `FinalAnswerArtifact`、`FailureArtifact` 和 `qa_result.json` 的层。

终态需要显式分层：DebateClaw SQLite 里的 `phase=done/status=done` 只表示内部协议已经走到 `protocol_done`；ChemQA benchmark 的外部 `terminal_state=completed` 只能在 Artifact Flow 已经原子写入 final artifact、manifest 和 `qa_result.json` 后发布。这样既保留现有 DebateClaw 状态机，又避免 runner 把“协议结束”误判为“可评估结果已持久化”。

落地目标不是让 ChemQA 立刻变成更复杂的多候选综合系统，而是先让现有“集思广益”机制稳定生效：reviewer 的反馈必须通过结构化 review/rebuttal artifact 更新当前候选答案；最终评测必须消费 durable、typed、validated 的 final artifact；失败也必须留下可诊断、可统计的结构化原因。

## Goal

Refactor ChemQA runtime responsibilities into a two-layer architecture while keeping the current collaboration topology:

- one semantic candidate owner: `proposer-1`
- four fixed reviewer lanes: `proposer-2`, `proposer-3`, `proposer-4`, `proposer-5`
- the existing `propose -> review -> rebuttal` phase model

The change is not intended to make ChemQA more ambitious immediately. The immediate goal is to make ChemQA more stable by separating:

- protocol progression: who should act, in which phase, and when the run can advance
- artifact flow: what was produced, whether it is valid for the benchmark task, how revisions update the current candidate, and when final output is durable and evaluable

The target outcome is that every ChemQA run reaches a clear final artifact state:

- a validated `FinalAnswerArtifact`, or
- a structured `FailureArtifact`

Benchmark scoring should consume that final artifact state instead of reconstructing answers from transcripts, slot workspaces, or best-effort artifact search after the run is already terminal.

## Background

The temporary benchmark run at:

`/Users/xutao/.openclaw/workspace/benchmark-runs/temp-frontierscience-superchem-web-on-ab-20260428-single`

showed that `chemqa_web_on` produced many operationally failed or non-evaluable records even when agents had produced meaningful intermediate answers.

Observed failure patterns included:

1. Runs whose protocol status reached `done` / `completed`, but whose `qa_result.json` was not resolved by the runner.
2. Research-answer runs failed by candidate validation because `direct_answer` was too narrative, even though research tasks naturally require multi-part explanatory answers.
3. Reviewer feedback identified issues, but the rebuttal path did not reliably create an updated candidate view used as the final answer.
4. The coordinator and runner had to reason across protocol state, OpenClaw sessions, generated artifacts, cleanup behavior, and benchmark evaluator expectations.

These are architecture symptoms rather than isolated prompt or timeout issues. The system currently mixes protocol control, artifact validation, artifact lifecycle, answer projection, and benchmark reporting.

## Non-Goals

- Do not change the one-proposer plus four-reviewer collaboration structure in the first implementation phase.
- Do not replace DebateClaw V1 state management immediately.
- Do not introduce free-form multi-candidate synthesis as part of this design's first phase.
- Do not change benchmark evaluators or scoring logic except through cleaner final-answer projection.
- Do not remove current diagnostics, recovery metadata, or degraded-execution reporting.
- Do not treat this design as a request to implement code immediately; implementation should follow a separate plan.

## Design Principles

1. Preserve the current collaboration topology while clarifying ownership boundaries.
2. Make artifact validity independent from protocol phase progression.
3. Make finalization atomic: a ChemQA benchmark run is not externally `completed` until final artifacts are written and readable, even if the underlying DebateClaw protocol state is already `done`.
4. Treat task answer shape as typed data, not as one universal `direct_answer` string.
5. Let reviewer feedback modify the current candidate only through structured artifact transitions.
6. Keep benchmark runner behavior simple: launch, wait for terminal final artifact, score or record failure.
7. Preserve enough failed-run artifacts to diagnose systemic behavior after cleanup.

## Current Responsibility Overload

The current ChemQA path spreads responsibilities across:

- `benchmarking/runners/chemqa.py`
  - launches ChemQA
  - waits for run status
  - reconstructs artifacts
  - archives outputs
  - tries fallback recovery
  - maps results into benchmark runner contracts
- `skills/chemqa-review/scripts/chemqa_review_openclaw_driver.py`
  - runs role loops
  - calls OpenClaw
  - submits protocol artifacts
  - repairs stalled phases
  - updates run status
- `skills/debateclaw-v1/scripts/debate_state.py`
  - stores protocol state
  - advances phases
  - stores proposal/review/rebuttal rows
- `skills/chemqa-review/scripts/chemqa_review_artifacts.py`
  - validates multiple artifact types
  - reconstructs final protocol output
  - decides acceptance semantics
- `collect_artifacts.py`
  - converts protocol outputs into `qa_result.json`

This makes the coordinator and runner indirectly responsible for transcript interpretation and final-answer reconstruction. The proposed design moves those concerns into a dedicated Artifact Flow layer.

## Target Architecture

### Layer 1: Protocol Flow

Protocol Flow owns phase progression only.

It answers:

- What phase is the run in?
- Which role should act next?
- Which roles have registered submissions for the phase?
- Is the phase complete by protocol counting rules?
- Should the run advance to the next phase?
- Did the protocol exhaust its round, retry, or liveness budget?

Protocol Flow keeps the existing semantics:

```text
propose -> review -> rebuttal -> review -> ... -> done
```

It also keeps existing role semantics:

```text
proposer-1: semantic candidate owner
proposer-2: search / coverage reviewer
proposer-3: evidence trace reviewer
proposer-4: reasoning consistency reviewer
proposer-5: counterevidence reviewer
```

Protocol Flow should not decide:

- whether a candidate answer is scoreable for a benchmark type
- whether a research answer is too long
- whether a rebuttal updated the answer
- which answer string the evaluator should receive
- where `qa_result.json` should be rebuilt from
- whether final artifacts are durable

Those are Artifact Flow responsibilities.

### Layer 2: Artifact Flow

Artifact Flow owns all ChemQA artifact lifecycle semantics.

It answers:

- Which artifact type is required for the current role and phase?
- Is the artifact valid for its role, phase, and benchmark answer kind?
- What is the current candidate view after proposal and rebuttal updates?
- Which blocking review items remain open?
- Does the candidate view require another rebuttal round before the protocol may close?
- Can the run be finalized?
- What exact payload should be written to `qa_result.json`?
- What diagnostic artifact should be written if finalization fails?

Artifact Flow is the only layer allowed to produce:

- `FinalAnswerArtifact`
- `FailureArtifact`
- `qa_result.json`
- final artifact manifest

Protocol Flow may request finalization, but it does not write final benchmark output directly.

## Core Boundary Contract

Protocol Flow communicates with Artifact Flow through narrow state summaries.

The boundary is intentionally asymmetric:

- Protocol Flow is authoritative for phase counters, role scheduling, round limits, epoch changes, and submission registration.
- Artifact Flow is authoritative for artifact schema validity, answer-kind validity, semantic blocking items, candidate-view derivation, final answer projection, failure projection, and benchmark-visible terminal output.
- Protocol Flow may ask Artifact Flow whether semantic gates allow an advance or finalization, but it must not inspect artifact bodies itself.
- Artifact Flow may report semantic gates, but it must not create new phase participants or directly advance DebateClaw rounds.

Protocol Flow submits phase events:

```json
{
  "run_id": "...",
  "phase": "review",
  "role": "proposer-3",
  "artifact_path": ".../review-proposer-1.yaml",
  "event": "artifact_submitted"
}
```

Artifact Flow returns artifact state:

```json
{
  "artifact_valid": true,
  "current_candidate_valid": true,
  "semantic_blockers_open": 0,
  "requires_rebuttal": false,
  "finalizable_by_artifact_rules": true,
  "artifact_errors": []
}
```

Protocol Flow can use this state to decide whether to advance. It does not inspect raw artifact bodies except for transport-level registration.

This means phase completion has two inputs in ChemQA:

1. Protocol completion: required lanes have registered their submissions in DebateClaw state.
2. Artifact semantic gate: Artifact Flow has accepted or rejected the submitted contents for the current answer kind and phase.

The first input lives in Protocol Flow. The second input lives in Artifact Flow.

## State Authority and Storage

The first implementation should not introduce a second protocol event source.

Authoritative storage is split as follows:

| State | Owner | Storage |
| --- | --- | --- |
| Raw proposal/review/rebuttal submissions | Protocol Flow | Existing DebateClaw SQLite tables and `artifacts` table |
| Phase, epoch, review round, rebuttal round | Protocol Flow | Existing DebateClaw SQLite meta |
| Artifact validation records | Artifact Flow | Derived records under the ChemQA generated artifact directory; may be indexed by existing artifact IDs |
| Current candidate view | Artifact Flow | Derived `candidate_view` artifact plus manifest entry |
| Final answer / failure artifact | Artifact Flow | Canonical final artifact directory plus manifest |
| Benchmark-visible run terminal state | Artifact Flow via ChemQA run-status overlay | `control/run-status/<run_id>.json` |

Existing DebateClaw artifact metadata is the transport ledger. Artifact Flow consumes it and writes derived typed artifacts. It must not fork protocol truth by inventing a separate list of who acted in which phase.

The canonical final artifact directory for a run should be:

```text
workspace/skills/chemqa-review/generated/artifacts/<run_id>/
```

Required final files in that directory are:

- `artifact_manifest.json`
- either `final_answer_artifact.json` or `failure_artifact.json`
- `qa_result.json`
- compact diagnostics such as `validation_summary.json` and `candidate_view.json`

Benchmark archives may copy those files elsewhere, but these generated artifacts remain the source directory referenced by run status before cleanup.

## Terminal State Contract

There are two terminal concepts:

1. DebateClaw internal protocol terminal state.
2. ChemQA benchmark-visible terminal state.

DebateClaw may continue to set:

```json
{
  "phase": "done",
  "status": "done",
  "terminal_state": "completed"
}
```

inside its SQLite state when protocol phase progression ends. In the two-layer ChemQA design, that state means `protocol_done`, not benchmark completion.

ChemQA run status should expose a separate overlay:

```json
{
  "status": "running|done",
  "protocol_terminal_state": "completed",
  "artifact_flow_state": "finalizing|finalized|finalization_failed",
  "benchmark_terminal_state": "running|completed|failed",
  "terminal_state": "running|completed|failed"
}
```

Compatibility rule:

- The legacy run-status field `status` mirrors whether the ChemQA benchmark-visible run is terminal, not whether DebateClaw's internal protocol is done.
- While Artifact Flow is finalizing, legacy `status` must remain `running`. It must not be written as `done`, because the current runner terminal predicate treats `status == "done"` as terminal.
- Only after Artifact Flow writes and verifies final artifacts may legacy `status` become `done`.
- The legacy run-status field `terminal_state` mirrors `benchmark_terminal_state`, not DebateClaw's internal terminal state.
- While DebateClaw is `done` but Artifact Flow has not finished, ChemQA run status remains non-terminal for the runner, for example `artifact_flow_state = finalizing` and `terminal_state = running`.
- `terminal_state = completed` is published only after `FinalAnswerArtifact`, `artifact_manifest.json`, and `qa_result.json` have all been written and reopened successfully.
- `terminal_state = failed` is published after a readable `FailureArtifact` and manifest have been written, unless the system cannot write even the failure artifact. In that last case the runner reports `artifact_finalization_failed_without_failure_artifact`.

This avoids changing DebateClaw V1's core state machine in the first phase while making benchmark terminal semantics reliable.

Later implementation may tighten `is_chemqa_terminal_status()` to prefer `benchmark_terminal_state` and canonical artifact paths, but the Phase 1 invariant is still that run-status must not expose legacy `status=done` until benchmark terminal artifacts exist.

## Artifact Model

Artifact Flow should use typed artifacts with explicit versions and validation outcomes.

### Common Fields

Every artifact should carry:

```json
{
  "artifact_id": "stable id",
  "run_id": "benchmark-...",
  "artifact_kind": "candidate|review|rebuttal|candidate_view|final_answer|failure",
  "schema_version": 1,
  "role": "proposer-1",
  "phase": "propose",
  "epoch": 1,
  "round": 0,
  "created_at": "ISO timestamp",
  "source_path": "optional path",
  "validation_status": "valid|invalid|warning",
  "validation_errors": [],
  "payload": {}
}
```

### Answer Kind

Each benchmark record should resolve to an answer kind before the first candidate artifact is requested.

The source of truth is the benchmark runner. `ChemQARunner` derives an immutable `answer_kind` from `BenchmarkRecord` and grading metadata before launch, writes it into the ChemQA launch context, and passes it to Artifact Flow as run metadata. Artifact Flow may validate that `answer_kind` is present, but it must not infer a different answer kind from model text or reviewer feedback.

Initial answer kinds:

- `numeric_short_answer`
- `short_text_answer`
- `multi_part_research_answer`
- `multiple_choice`
- `structure_answer`
- `generic_semantic_answer`

The answer kind controls candidate and final answer validation.

This avoids forcing all benchmark families through a single `direct_answer must be concise` rule.

If no narrower kind can be derived, the runner uses `generic_semantic_answer`. This keeps the six-kind surface explicit and prevents open-ended research records from falling back to numeric/short-answer validation.

### Artifact Kind Compatibility

The new typed names should map onto current artifact names during migration:

| Artifact Flow kind | Current / legacy name | Notes |
| --- | --- | --- |
| `candidate` | `candidate_submission`, `proposal` | Produced by `proposer-1`; stored as DebateClaw proposal artifact |
| `review` | `formal_review`, `transport_review` | Only formal reviewer-lane reviews count for ChemQA acceptance |
| `rebuttal` | `rebuttal` | Adds explicit `mode` while preserving current concession flag |
| `candidate_view` | none | New derived artifact; never directly authored by an agent |
| `final_answer` | `coordinator_protocol.final_answer`, `qa_result.final_answer` | New canonical success artifact projected into legacy shape |
| `failure` | `terminal_failure`, failure protocol payload | New canonical failure artifact projected into runner failure/recovery axes |

The compatibility table is part of the migration contract. Existing names may remain in file names or prompts while the Artifact Flow API uses the new typed names internally.

### Candidate Artifact

`CandidateArtifact` records the candidate owner's proposal.

It should include:

```json
{
  "answer_kind": "numeric_short_answer",
  "evaluator_answer": "7.59",
  "display_answer": "7.59 micrograms",
  "reasoning_summary": "...",
  "submission_trace": [],
  "claim_anchors": [],
  "evidence_limits": []
}
```

For `multi_part_research_answer`, `evaluator_answer` may be a structured markdown or text answer rather than a scalar number. Concision should be enforced only for answer kinds that need scalar extraction.

### Review Artifact

`ReviewArtifact` records a formal reviewer critique.

It should include:

```json
{
  "target_artifact_id": "...",
  "verdict": "blocking|non_blocking|insufficient_evidence",
  "review_items": [
    {
      "item_id": "trace-1",
      "severity": "high",
      "finding": "...",
      "requested_change": "...",
      "target_field": "evaluator_answer|reasoning_summary|claim_anchors|submission_trace"
    }
  ],
  "counts_for_acceptance": true
}
```

Reviewers still review only `proposer-1`'s candidate. They do not become alternate final-answer owners in this phase of the architecture.

### Rebuttal Artifact

`RebuttalArtifact` must distinguish response-only rebuttals from actual answer revisions.

It should include:

```json
{
  "mode": "response_only|answer_revision|concession",
  "concede": false,
  "response_summary": "...",
  "addressed_review_items": [],
  "updated_answer": null,
  "updated_trace": null,
  "remaining_open_items": []
}
```

If `mode = answer_revision`, Artifact Flow applies the update to the current candidate view after validation.

If `mode = response_only`, Artifact Flow records the response but does not change answer fields.

If `mode = concession`, Protocol Flow may advance to a new epoch or fail according to existing round limits.

### Review Item Identity and Closure

Review item state must be deterministic enough for multiple rounds.

Every blocking review item receives a stable key:

```text
<epoch>:<review_round>:<reviewer_lane>:<item_id-or-content-hash>
```

Reviewer-provided `item_id` is used when present. If it is missing or duplicated, Artifact Flow generates a content hash from reviewer lane, target field, severity, finding, and requested change.

Closure authority is:

- `open`: created by a valid blocking `ReviewArtifact`.
- `addressed_by_revision`: set by Artifact Flow when a valid `answer_revision` changes the targeted answer/view field and references the review item.
- `addressed_by_response`: set by Artifact Flow when a `response_only` rebuttal references the item and gives a validation-acceptable reason no answer change is needed.
- `waived_by_reviewer`: reserved for a later explicit reviewer waiver artifact; not required in the first phase.
- `unresolved_at_terminal`: set by Artifact Flow during finalization for any still-open item.

Later reviewer confirmation is not required for `addressed_by_revision` in the first phase. A later review round may reopen the issue by creating a new item if the revision is still insufficient. This keeps the current one-proposer/four-reviewer loop usable without adding a second reviewer-approval protocol.

### Current Candidate View

Artifact Flow maintains a derived `CurrentCandidateView`.

It is not simply the first proposal. It is:

```text
initial CandidateArtifact
+ accepted answer_revision rebuttals
+ validation normalization
+ open/closed review item accounting
```

The final answer is projected from `CurrentCandidateView`, not from raw proposal text.

This directly fixes the case where reviewers identify a correctable issue and the proposer responds, but the final protocol output still reads from the stale proposal.

### Final Answer Artifact

`FinalAnswerArtifact` is the only success artifact consumed by benchmark scoring.

It should include:

```json
{
  "terminal_state": "completed",
  "answer_kind": "numeric_short_answer",
  "evaluator_answer": "7.59",
  "display_answer": "7.59 micrograms",
  "full_answer": "...",
  "source_candidate_view_id": "...",
  "acceptance_status": "accepted|rejected",
  "review_summary": {},
  "confidence": {},
  "degraded_execution": false,
  "warnings": []
}
```

`acceptance_status = rejected` may still be evaluable if the answer kind and evaluator projection are valid and benchmark policy allows scoring rejected-but-answerful runs. That policy should remain explicit in benchmark result axes.

### Failure Artifact

`FailureArtifact` is the required terminal output when no valid final answer can be produced.

It should include:

```json
{
  "terminal_state": "failed",
  "failure_code": "candidate_validation_failed|artifact_finalization_failed|protocol_stalled|missing_required_artifact",
  "failure_message": "...",
  "last_valid_candidate_view": {},
  "answer_projection": null,
  "recovery_eligibility": {
    "evaluable": false,
    "scored": false,
    "reliability": "none",
    "recovery_mode": "none",
    "reason": "no_valid_answer_projection"
  },
  "missing_artifacts": [],
  "validation_errors": [],
  "open_review_items": [],
  "diagnostic_paths": []
}
```

A failed protocol should still leave enough structured data for benchmark reporting to distinguish:

- no answer was ever produced
- an answer existed but was invalid for the answer kind
- protocol ended before finalization
- final artifact persistence failed

A `FailureArtifact` may carry `answer_projection` only when Artifact Flow can derive a policy-approved answer from a valid `CurrentCandidateView` or last valid candidate artifact. That projection is not a success artifact. It exists to preserve the prior recovery design's evaluability semantics without forcing the runner to scrape transcripts.

When `answer_projection` is present:

```json
{
  "answer_kind": "multiple_choice",
  "evaluator_answer": "B",
  "display_answer": "B",
  "full_answer": "Recovered from last valid proposer-1 candidate view.",
  "source_candidate_view_id": "..."
}
```

Artifact Flow must also set `recovery_eligibility` so the runner can map the failure to `RunStatus.RECOVERED` without guessing.

When `answer_projection` is absent or `recovery_eligibility.evaluable = false`, the runner maps the record to `RunStatus.FAILED`.

## Finalization Semantics

Finalization must become an atomic Artifact Flow operation.

Current failure mode:

```text
protocol status says done
runner looks for qa_result.json
artifact may not exist or may have been cleaned
record becomes non-evaluable
```

Target behavior:

```text
Artifact Flow validates final candidate view
Artifact Flow writes final artifact and qa_result.json to a temp path
Artifact Flow fsyncs / renames into final location
Artifact Flow writes manifest
Artifact Flow updates run status with qa_result path and artifact manifest path
ChemQA run-status publishes benchmark terminal_state only after finalization succeeds
```

If finalization fails, Artifact Flow writes a `FailureArtifact` and terminal status points to that failure artifact.

The runner should wait for:

- `FinalAnswerArtifact`, or
- `FailureArtifact`

It should not rebuild success artifacts after the protocol is terminal except as a temporary compatibility fallback during migration.

Finalization caller:

- In Phase 1, `chemqa_review_openclaw_driver.py` should call Artifact Flow finalization after DebateClaw reports protocol `done`.
- A dedicated artifact-controller command may be introduced later, but it should use the same Artifact Flow APIs and must not create a second terminal-state policy.
- `debate_state.py` does not need to stop writing its internal `done`; the driver must translate that into `protocol_terminal_state` and keep benchmark `terminal_state` non-terminal until finalization completes.

## Legacy `qa_result.json` Projection

Phase 1 must keep the current benchmark runner and reporting code working. Artifact Flow therefore writes both canonical artifacts and a legacy-compatible `qa_result.json`.

For `FinalAnswerArtifact`, `qa_result.json` must include:

```json
{
  "terminal_state": "completed",
  "acceptance_status": "accepted|rejected",
  "answer_kind": "numeric_short_answer",
  "final_answer": {
    "direct_answer": "7.59",
    "answer": "7.59",
    "value": "7.59",
    "display_answer": "7.59 micrograms",
    "full_answer": "..."
  },
  "artifact_paths": {
    "final_answer_artifact": ".../final_answer_artifact.json",
    "artifact_manifest": ".../artifact_manifest.json",
    "candidate_view": ".../candidate_view.json"
  }
}
```

Mapping rules:

- `final_answer.direct_answer`, `final_answer.answer`, and `final_answer.value` mirror `FinalAnswerArtifact.evaluator_answer` for compatibility with existing extraction code.
- `final_answer.display_answer` mirrors `display_answer`.
- `final_answer.full_answer` mirrors `full_answer`.
- `artifact_paths` contains canonical final artifact paths, not only legacy collector outputs.
- New canonical fields may be duplicated at top level, but evaluators should continue reading through the existing runner contract until the migration removes compatibility reconstruction.

For `FailureArtifact`, `qa_result.json` should still be written when possible and include:

```json
{
  "terminal_state": "failed",
  "failure_code": "protocol_stalled",
  "answer_projection": null,
  "recovery_eligibility": {
    "evaluable": false,
    "scored": false,
    "reliability": "none",
    "recovery_mode": "none",
    "reason": "protocol_stalled"
  },
  "artifact_paths": {
    "failure_artifact": ".../failure_artifact.json",
    "artifact_manifest": ".../artifact_manifest.json"
  }
}
```

If `answer_projection` is present, `qa_result.json` must include the same projection fields as a final answer under `answer_projection`, not under `final_answer`. This prevents failed-but-recovered runs from being misclassified as native completed runs.

## Proposed Module Responsibilities

### New or Refactored Artifact Flow Module

Suggested location:

`workspace/skills/chemqa-review/scripts/chemqa_artifact_flow.py`

Responsibilities:

- read immutable answer kind from benchmark metadata / goal payload
- validate candidate, review, rebuttal, final, and failure artifacts
- maintain current candidate view
- track review item open / closed state
- generate final answer artifact
- generate failure artifact
- write `qa_result.json` and manifest atomically

### Existing Protocol Flow Modules

`debate_state.py` should remain responsible for:

- phase
- epoch
- review round
- rebuttal round
- required phase participants
- phase completion
- advance decisions

`chemqa_review_openclaw_driver.py` should become thinner over time:

- request role artifacts
- pass submitted artifacts to Artifact Flow
- ask Artifact Flow for phase artifact status
- ask Artifact Flow to finalize when Protocol Flow reaches terminal conditions
- publish ChemQA benchmark-visible run status from Artifact Flow output, not directly from DebateClaw internal `done`

### Runner

`benchmarking/runners/chemqa.py` should eventually:

- launch ChemQA
- wait for ChemQA benchmark-visible terminal status
- read `qa_result.json`
- archive artifact directory
- report result

It should not need to know how to rebuild ChemQA protocol artifacts from slot workspaces in the normal success path.

Normal runner behavior after Phase 2:

1. Poll `control/run-status/<run_id>.json`.
2. Treat `terminal_state=completed` as success only when `qa_result_path`, `artifact_manifest_path`, and `final_answer_artifact_path` are readable.
3. Treat `terminal_state=failed` with readable `failure_artifact_path` as a structured failure; if recovery eligibility is present and scoreable, return `RunStatus.RECOVERED`.
4. Treat `protocol_terminal_state=completed` with non-terminal benchmark state as still running until the artifact timeout expires.
5. Prefer `benchmark_terminal_state` and canonical paths over legacy `status` when deciding terminality.
6. Use legacy reconstruction only when run status marks `artifact_flow_state` as absent or compatibility mode for old runs.

## Data Flow

### Proposed Normal Success Path

```text
BenchmarkRecord
  -> ChemQARunner
  -> AnswerKindResolver
  -> Protocol Flow initializes propose phase
  -> proposer-1 writes CandidateArtifact
  -> Artifact Flow validates CandidateArtifact and CurrentCandidateView
  -> Protocol Flow advances to review
  -> four reviewers write ReviewArtifacts
  -> Artifact Flow records blocking/non-blocking items
  -> Protocol Flow advances to rebuttal if needed
  -> proposer-1 writes RebuttalArtifact
  -> Artifact Flow applies answer_revision or records response_only
  -> Protocol Flow continues or reaches terminal condition
  -> Artifact Flow finalizes FinalAnswerArtifact or FailureArtifact
  -> Runner reads qa_result.json and scores
```

### Proposed Failed But Diagnosable Path

```text
Candidate validation fails for answer kind
  -> Artifact Flow records validation errors
  -> Protocol Flow can retry according to existing retry budget
  -> retry budget exhausted
  -> Artifact Flow writes FailureArtifact
  -> terminal status points to FailureArtifact
  -> Runner reports non-evaluable with precise failure_code
```

## Compatibility Strategy

This design should be rolled out in phases.

### Phase 1: Add Artifact Flow Beside Existing Protocol

- Keep one proposer and four reviewers.
- Keep `debate_state.py` phase logic.
- Add artifact-flow validation and current-candidate-view logic.
- Continue producing current `qa_result.json` shape, but make it originate from Artifact Flow.
- Add compatibility fields so existing reports still work.
- Keep DebateClaw internal `done` behavior, but add ChemQA run-status overlay fields for `protocol_terminal_state`, `artifact_flow_state`, and `benchmark_terminal_state`.
- Enforce the benchmark-visible terminal invariant immediately: ChemQA run-status legacy `status` remains `running` during `artifact_flow_state=finalizing` and becomes `done` only after final artifacts are readable.

### Phase 2: Move Finalization Ownership

- Make runner normal path prefer canonical Artifact Flow paths and benchmark terminal fields.
- Make run status include canonical paths:
  - `final_answer_artifact_path`
  - `failure_artifact_path`
  - `qa_result_path`
  - `artifact_manifest_path`
- Make normal runner path read these paths instead of searching candidate source directories.
- Keep post-terminal reconstruction only for old runs whose status lacks Artifact Flow fields.

### Phase 3: Tighten Cleanup and Diagnostics

- Ensure cleanroom cleanup only removes generated run state after final artifacts and compact diagnostics are archived.
- Preserve:
  - final answer / failure artifact
  - artifact manifest
  - compact protocol summary
  - validation error summary
  - minimal per-role artifact metadata
- Full transcripts may still be cleaned, but final diagnostic evidence must survive.

### Phase 4: Retire Compatibility Reconstruction

- Remove or demote post-terminal artifact reconstruction from the normal path.
- Keep it only as an explicit recovery tool for old runs.

## Validation Rules by Answer Kind

### `numeric_short_answer`

Required:

- scalar `evaluator_answer`
- parseable numeric value or formula accepted by evaluator
- optional unit in `display_answer`

Rejected:

- multi-paragraph direct answer
- missing numeric value
- placeholder text

### `short_text_answer`

Required:

- concise text or entity answer
- no multiple conflicting answers unless evaluator supports it

### `multi_part_research_answer`

Required:

- structured or prose answer covering requested parts
- full answer may be long
- evaluator projection may be the full response, not a short scalar

Rejected:

- empty answer
- unrelated response
- ungrounded placeholder

Not rejected merely because:

- answer has multiple sentences
- answer contains step-by-step explanation
- answer is too long for numeric short-answer style

### `multiple_choice`

Required:

- extracted option label in `evaluator_answer`
- optional rationale in `full_answer`

### `structure_answer`

Required:

- structure payload or string accepted by the relevant evaluator
- normalized representation when deterministic tooling is available

### `generic_semantic_answer`

Required:

- non-empty `evaluator_answer` or `full_answer`
- answer text suitable for the generic semantic evaluator
- no unresolved placeholder language

## Review-to-Revision Semantics

Blocking review items should not merely cause more debate turns. Artifact Flow should track whether each item has been addressed.

Each blocking item can be:

- `open`
- `addressed_by_revision`
- `addressed_by_response`
- `waived_by_reviewer`
- `unresolved_at_terminal`

For the current one-proposer topology:

- `answer_revision` rebuttals update `CurrentCandidateView`
- `response_only` rebuttals can close items only when they explicitly justify why no answer change is needed
- unresolved high-severity items can prevent acceptance, but do not necessarily prevent benchmark evaluability if a typed final answer exists

## Runner Result Semantics

The runner should eventually map Artifact Flow terminal outputs to existing benchmark axes:

- `run_lifecycle_status`
- `protocol_completion_status`
- `answer_availability`
- `answer_reliability`
- `evaluable`
- `scored`
- `recovery_mode`
- `degraded_execution`

Expected mappings:

| Artifact terminal state | RunnerResult | Result axes |
| --- | --- | --- |
| valid `FinalAnswerArtifact`, accepted | `RunStatus.COMPLETED`, `should_score=True` | `run_lifecycle_status=completed`, `protocol_completion_status=completed`, `answer_availability=native_final`, `answer_reliability=native`, `evaluable=True`, `scored=True`, `recovery_mode=none`, `degraded_execution=False` |
| valid `FinalAnswerArtifact`, rejected but scoreable | `RunStatus.COMPLETED`, `should_score=True` | `run_lifecycle_status=completed`, `protocol_completion_status=completed`, `answer_availability=native_final`, `answer_reliability=native`, `evaluable=True`, `scored=True`, `recovery_mode=none`, `degraded_execution` follows rejection/acceptance policy |
| valid `FinalAnswerArtifact`, forced quorum | `RunStatus.COMPLETED`, `should_score=True` | `answer_availability=native_final`, `answer_reliability=native`, `degraded_execution=True`, `recovery_mode=none` |
| `FailureArtifact` with `answer_projection` and `recovery_eligibility.scored=true` | `RunStatus.RECOVERED`, `should_score=True` | `run_lifecycle_status=failed`, `protocol_completion_status` from failure artifact, `answer_availability=recovered_candidate`, `answer_reliability` from `recovery_eligibility.reliability`, `evaluable=True`, `scored=True`, `recovery_mode` from `recovery_eligibility.recovery_mode`, `degraded_execution=True` |
| `FailureArtifact` with no scoreable projection | `RunStatus.FAILED`, `should_score=False` | `run_lifecycle_status=failed`, `answer_availability=missing`, `answer_reliability=none`, `evaluable=False`, `scored=False`, `recovery_mode=none`, `degraded_execution=True` |

Result axes should keep the current schema values:

- `answer_availability`: `native_final`, `recovered_candidate`, `preview_only`, `missing`
- `answer_reliability`: `native`, `high_confidence_recovered`, `low_confidence_recovered`, `none`

Policy details such as rejection, forced quorum, or finalization warnings should be carried by `degraded_execution`, `terminal_reason_code`, `execution_error_kind`, or runner metadata, not by inventing new axis values.

`FailureArtifact.answer_projection` is the only normal-path source for recovered evaluable answers after Phase 2. The current proposal-scraping fallback remains a legacy compatibility path for old runs and should be reported with a distinct recovery mode.

## Cleanup and Diagnostic Contract

Cleanup must not be able to erase the only copy of a terminal answer or terminal failure reason.

Required ordering:

1. Artifact Flow writes canonical final artifacts to the generated artifact directory.
2. Artifact Flow reopens the required files and writes `artifact_manifest.json` with file paths, sizes, and content hashes.
3. ChemQA run status publishes terminal state and canonical paths.
4. `ChemQARunner` archives the canonical final artifact directory into the per-record benchmark archive.
5. Cleanroom cleanup may remove disposable runtime state only after the archive attempt has completed.

The cleanup manifest should treat these as preserve/archive roots, not disposable roots:

- generated final artifact directory for the run
- `qa_result.json`
- `artifact_manifest.json`
- `final_answer_artifact.json` or `failure_artifact.json`
- `candidate_view.json`
- compact validation and protocol summaries

If archiving fails but canonical final artifacts still exist, the runner should return the result using canonical paths and attach `artifact_archive_status=error`. If canonical final artifacts are missing before archive, the runner should return a finalization or persistence failure rather than attempting to score from transcripts.

Required compact diagnostics:

- `failure_code` and `failure_message` when failed
- validation error list grouped by artifact id
- last valid candidate view id/path if any
- open review item count and item keys
- canonical path and hash for every final artifact file

## Testing Strategy

Tests should focus on boundary behavior rather than model quality.

### Unit Tests

- answer kind resolution from benchmark records
- candidate validation by answer kind
- research answer is not rejected for narrative length
- numeric answer is rejected when no numeric evaluator projection exists
- rebuttal `answer_revision` updates `CurrentCandidateView`
- rebuttal `response_only` does not update answer fields
- blocking item state transitions
- finalization writes success artifact and manifest together
- failure finalization writes structured failure artifact

### Integration Tests

- one full successful ChemQA protocol produces `qa_result.json` before terminal success
- completed protocol without final artifact is treated as finalization failure, not silent success
- runner consumes canonical `qa_result_path` from terminal status
- DebateClaw internal `done` with `artifact_flow_state=finalizing` remains non-terminal for the ChemQA runner
- `FailureArtifact` with scoreable `answer_projection` maps to `RunStatus.RECOVERED`
- legacy `qa_result.json` projection preserves `final_answer.direct_answer`, `answer`, and `value`
- cleanroom preserves final artifacts and compact diagnostics
- legacy artifact reconstruction remains available only for compatibility paths

### Regression Fixtures

Use fixtures modeled on the temporary benchmark failures:

- `numeric_short_answer`: proposal has correct scalar answer and reviewers request format correction
- `multi_part_research_answer`: long answer should validate
- `multiple_choice`: candidate answer recovered from proposal after protocol failure
- `completed_missing_qa_result`: status cannot become completed until final artifact exists
- `protocol_done_finalizing`: DebateClaw done is not benchmark terminal until final artifacts exist
- `failure_with_answer_projection`: failed protocol can still be recovered evaluable through structured projection

## Acceptance Criteria

The first implementation phase is acceptable when:

1. The collaboration topology remains one proposer plus four reviewers.
2. Protocol Flow no longer owns final benchmark answer projection.
3. Artifact Flow can validate candidate/review/rebuttal artifacts independently.
4. Rebuttal can explicitly act as `response_only`, `answer_revision`, or `concession`.
5. Current candidate view is derived from proposal plus valid answer revisions.
6. `qa_result.json` is written by Artifact Flow finalization.
7. ChemQA benchmark terminal success is impossible without a readable final artifact, manifest, and `qa_result.json`.
8. Terminal failure always writes a structured failure artifact when the filesystem is available.
9. Research-style answers are not rejected by numeric/short-answer concision rules.
10. Runner can score from the final artifact without transcript inspection in the normal path.
11. DebateClaw internal `done` and ChemQA benchmark `terminal_state` are explicitly separated in run status.
12. Recovered evaluable failures are represented by `FailureArtifact.answer_projection`, not by transcript scraping in the normal path.

## Risks and Mitigations

### Risk: Large refactor destabilizes the current benchmark runner

Mitigation:

- add Artifact Flow beside existing protocol first
- keep existing runner fallback during migration
- compare old and new `qa_result.json` outputs on small fixtures

### Risk: Schema proliferation creates more complexity

Mitigation:

- start with six answer kinds only
- require every answer kind to project to a common final artifact surface
- keep evaluator-specific fields under `details`

### Risk: Coordinator remains overloaded

Mitigation:

- move validation, current-candidate-view, and finalization into Artifact Flow
- coordinator consumes summaries only
- do not let coordinator parse artifact bodies for benchmark semantics

### Risk: Cleanup removes evidence needed for diagnosis

Mitigation:

- finalizer writes compact diagnostics before cleanup
- cleanup manifest treats final artifacts as archive roots, not disposable generated state

## Open Implementation Questions

These should be decided in the implementation plan:

1. How much of `chemqa_review_artifacts.py` should be preserved versus split into smaller modules.
2. Exact answer-kind derivation table from benchmark/evaluator metadata to the six initial answer kinds.
3. Exact timeout for the runner while `protocol_terminal_state=completed` but `artifact_flow_state=finalizing`.
4. How long compatibility reconstruction should remain enabled for old runs.
5. Whether a later dedicated artifact-controller command is worth introducing after the coordinator path is stable.

Resolved architecture decisions before implementation planning:

- Artifact Flow consumes existing DebateClaw SQLite submissions and artifact metadata; it does not create a second protocol ledger.
- ChemQA benchmark-visible terminal state is an overlay published in run status after Artifact Flow finalization, not the raw DebateClaw internal `done`.
- `answer_kind` is derived by `ChemQARunner` from `BenchmarkRecord` and grading metadata before launch.
- Phase 1 finalization is called by the ChemQA coordinator/driver after protocol terminal conditions.
- `FailureArtifact.answer_projection` plus `recovery_eligibility` is the normal-path source for recovered evaluable failures.

## Recommended Implementation Order

1. Add answer-kind resolution and typed validation helpers.
2. Introduce Artifact Flow data structures and current-candidate-view builder.
3. Adapt existing proposal/review/rebuttal validation to call Artifact Flow.
4. Make rebuttal mode explicit and support answer revisions.
5. Add finalizer that writes `FinalAnswerArtifact`, `FailureArtifact`, manifest, and `qa_result.json`.
6. Update coordinator run-status updates to include final artifact paths.
7. Update runner to prefer final artifact paths and keep old reconstruction as fallback.
8. Update cleanup to preserve final artifacts and compact diagnostics.
9. Add regression tests based on the temporary benchmark failure modes.

## Summary

The first-stage design keeps ChemQA's current one-proposer and four-reviewer workflow intact. The architectural change is to stop using the protocol layer as the implicit source of benchmark truth.

Protocol Flow should control phase progression. Artifact Flow should control artifact validity, candidate revision state, final answer projection, persistence, and failure diagnostics.

This separation directly targets the observed instability: completed runs without durable outputs, research answers rejected by short-answer validation, rebuttal feedback not affecting final answers, and runner dependence on post-terminal artifact reconstruction.
