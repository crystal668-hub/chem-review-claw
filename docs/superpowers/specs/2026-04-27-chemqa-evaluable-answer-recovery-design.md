# ChemQA Evaluable Answer Recovery Design

## Goal

Change ChemQA benchmark execution so that recovery prioritizes preserving a scoreable answer whenever one can be reconstructed with acceptable confidence, rather than treating non-`completed` workflow terminal states as automatically non-evaluable.

This design covers both:

- runtime / runner behavior for recovering and classifying evaluable answers
- benchmark result schema, aggregation, and reporting semantics

The target semantic model is Level 3. The recommended implementation rollout is Level 2 first, then promote to full Level 3 once downstream consumers are migrated.

## Background

Current behavior conflates three different concerns:

- whether the workflow engine reached a successful terminal state
- whether the ChemQA protocol reached internal acceptance
- whether the benchmark still has enough information to evaluate an answer

This creates systematic false negatives in benchmark accounting.

Observed failure modes:

1. A run reaches `terminal_state=failed`, but archived artifacts still contain a valid `proposer-1` candidate answer.
2. A run reaches `terminal_state=completed` with `acceptance_status=rejected`, but still contains a scoreable short answer that should be evaluated by the benchmark.
3. Recovery already reconstructs candidate answers in some non-success cases, but explicitly marks them `scored=False`, preventing evaluation.
4. Aggregate reporting only highlights `pass_count` and `avg_normalized_score`, so operational instability and answer quality are mixed together.

This is the wrong optimization target for benchmark operation. The primary benchmark obligation is:

- preserve evaluability if a faithful, reconstructable answer exists
- classify the provenance and quality of that answer explicitly
- only mark a record as non-evaluable when no trustworthy scoreable answer can be recovered

## Non-Goals

- redesign the whole ChemQA multi-agent workflow
- change ConformaBench or other benchmark evaluators themselves
- redefine internal ChemQA acceptance criteria
- remove existing diagnostics about workflow failure or reviewer disagreement

## Design Principles

1. Separate workflow completion from answer evaluability.
2. Preserve provenance for every recovered answer.
3. Prefer deterministic reconstruction over narrative salvage.
4. Keep benchmark scoring strict; only widen eligibility to be scored.
5. Make degraded-but-evaluable results first-class in schemas and reports.
6. Roll out without breaking all existing consumers at once.

## Current-State Problems

### Problem 1: Success semantics are workflow-centric

`is_chemqa_success_status()` currently treats success as:

- normalized status is `done`
- terminal state is `completed`

This ignores whether a scoreable answer exists in archived artifacts.

Impact:

- any non-`completed` terminal state is treated as benchmark failure first
- answer recovery becomes secondary and non-authoritative

### Problem 2: Recovery exists but is intentionally non-scoreable

`ChemQARunner._build_candidate_submission_fallback()` can recover a short answer and full response from:

- archived `proposer-1` proposal artifacts
- `final_answer_preview` in run status

But the resulting `RunnerResult` uses:

- `status=RECOVERED`
- `RecoveryInfo.scored=False`

Given `RunnerResult.should_score()`, this makes recovered answers non-evaluable by construction.

Impact:

- the system can already save some answers
- the benchmark then discards them

### Problem 3: Per-record schema lacks explicit evaluability semantics

Current `GroupRecordResult` does not distinguish:

- execution failed and no answer exists
- execution failed but scoreable answer recovered
- execution completed but protocol rejected
- execution completed and accepted

Impact:

- `error` becomes overloaded
- downstream logic cannot reason about degraded success

### Problem 4: Aggregate reporting hides operational quality

Current summary output emphasizes:

- `pass_count`
- `avg_normalized_score`

It does not surface:

- how many runs completed operationally
- how many records remained evaluable
- how many were salvaged by recovery

Impact:

- architecture instability is not separately measurable
- improvements to recovery cannot be observed in the main reports

## Target Semantic Model

Each benchmark record should be described along independent status axes.

### Axis 1: Workflow Execution

- `run_lifecycle_status`: `completed` | `failed` | `cancelled`
- Meaning: whether the runner finished the execution attempt and has a final operational classification

### Axis 2: Protocol Completion

- `protocol_completion_status`: `completed` | `failed` | `missing` | `not_applicable`
- Meaning: whether ChemQA protocol finalization produced a usable protocol artifact

### Axis 3: Answer Availability

- `answer_availability`: `native_final` | `recovered_candidate` | `preview_only` | `missing`
- Meaning: where the answer came from

### Axis 4: Evaluability

- `evaluable`: `true` | `false`
- Meaning: whether the benchmark has a sufficiently trustworthy scoreable answer track

### Axis 5: Scoring

- `scored`: `true` | `false`
- Meaning: whether the evaluator actually ran

### Axis 6: Benchmark Outcome

- `passed`: `true` | `false`
- Meaning: benchmark task score outcome, independent of workflow success

### Axis 7: Recovery Provenance

- `recovery_mode`: `none` | `candidate_submission` | `run_status_preview` | `archived_final_answer` | `protocol_reconstruction`
- Meaning: which fallback source won

### Axis 8: Reliability Tier

- `answer_reliability`: `native` | `high_confidence_recovered` | `low_confidence_recovered` | `none`
- Meaning: whether the recovered answer is trustworthy enough for evaluation

## Core Policy Change

### New rule

If a run does not end in native success, the runner must still attempt to produce an evaluable answer.

A record becomes evaluable when all of the following hold:

1. A short answer can be deterministically extracted.
2. The answer source is attributable to a ranked fallback source.
3. The source is not known to be placeholder-only, rejection-only, or malformed.
4. The reconstructed answer can be passed to the benchmark evaluator without violating evaluator contract.

Only when these conditions fail should the record be classified as non-evaluable execution failure.

### Explicit anti-rule

Do not equate:

- `terminal_state=failed`

with:

- `not evaluable`

That implication is the specific behavior this design removes.

## Answer Recovery Design

### Recovery Source Priority

For ChemQA records, answer recovery should search sources in this order:

1. archived normalized `qa_result.json` with valid scalar `final_answer`
2. archived `final_submission.json` or equivalent candidate payload
3. archived `candidate_submission.json`
4. archived `proposer-1` proposal artifact
5. protocol `final_answer` block if scalar and non-placeholder
6. run-status `final_answer_preview`

The first source that satisfies validation wins.

### Source Trust Tiers

#### Tier A: Native final answer

Examples:

- archived `qa_result.json`
- archived `final_submission.json`
- archived `candidate_submission.json` produced by collector

Properties:

- structured
- normalized
- already part of artifact pipeline

Result:

- `answer_reliability = native` or `high_confidence_recovered`
- scoreable

#### Tier B: Archived proposal recovery

Examples:

- latest `proposer-1.md`
- reconstructed proposal payload from clawteam artifacts

Properties:

- structured, but may bypass final coordinator protocol
- usually faithful to candidate answer

Result:

- `answer_reliability = high_confidence_recovered`
- scoreable when validation passes

#### Tier C: Preview-only fallback

Examples:

- `final_answer_preview`

Properties:

- minimal
- may omit rationale
- may be ambiguous or placeholder-like

Result:

- scoreable only if scalar answer extraction passes strict validation
- otherwise non-evaluable

### Recovery Validation Rules

Recovered answer sources must pass:

1. non-empty short answer extraction
2. no obvious placeholder markers
3. no serialized rejection blob as answer
4. no structured protocol object emitted as final answer string
5. format compatibility with evaluator expectations

Additional benchmark-specific validation may run before declaring evaluable:

- for ConformaBench, final answer line extraction remains required
- for strict scalar tasks, the short answer must be scalar-like
- for chemistry structure tasks, answer format must at least be syntactically eligible for the evaluator

### Placeholder Rejection Rules

The following should never count as evaluable answers:

- empty strings
- JSON/YAML rejection summaries
- coordinator failure text
- strings that only restate inability to answer
- transport placeholders
- text known to come from synthetic recovery review bodies

## Runtime / Runner Design Changes

### Change 1: Expand RunnerResult semantics

Current model:

- `COMPLETED`
- `RECOVERED`
- `FAILED`

Recommended interpretation:

- `COMPLETED`: native successful answer path
- `RECOVERED`: degraded execution, but evaluable answer preserved
- `FAILED`: no evaluable answer

`RECOVERED` should no longer imply `scored=False`.

### Change 2: Promote recovered evaluability into `should_score()`

New rule:

- `COMPLETED` always scoreable
- `RECOVERED` scoreable when recovery classification says answer is evaluable
- `FAILED` never scoreable

This is the minimum behavioral change needed to stop discarding valid recovered answers.

### Change 3: Replace binary fallback outcome with classified recovery outcome

Instead of `_build_candidate_submission_fallback()` returning only:

- short answer
- full response
- raw fallback metadata

it should return a structured recovery assessment:

- `source`
- `reliability`
- `evaluable`
- `reason_if_not_evaluable`
- `short_answer_text`
- `full_response_text`
- `artifact_path`

### Change 4: Introduce runner-level answer classification

Add a helper responsible for turning recovered artifacts into one of:

- `native_final`
- `recovered_evaluable`
- `recovered_non_evaluable`
- `missing_answer`

This keeps scoring eligibility logic out of ad hoc exception branches.

### Change 5: Preserve failure diagnostics even when answer is evaluable

Recovered scoreable results must still retain:

- original terminal failure
- terminal reason code
- run status snapshot
- protocol acceptance state
- archive status

This avoids papering over workflow instability.

The desired behavior is:

- evaluate the answer
- separately classify the run as degraded

### Change 6: Distinguish protocol rejection from execution failure

Cases like:

- `terminal_state=completed`
- `acceptance_status=rejected`

should remain scoreable if a valid short answer exists.

This matters because benchmark outcome is evaluator-owned, not protocol-owned.

## Schema Design

### Per-Record Schema Changes

Add the following fields to `GroupRecordResult`:

- `schema_version: int`
- `run_lifecycle_status: str`
- `protocol_completion_status: str`
- `protocol_acceptance_status: str | None`
- `answer_availability: str`
- `answer_reliability: str`
- `evaluable: bool`
- `scored: bool`
- `recovery_mode: str`
- `degraded_execution: bool`
- `execution_error_kind: str | None`

Existing fields to retain:

- `group_id`
- `record_id`
- `evaluation`
- `runner_meta`
- `raw`
- `error`
- `answer_text`
- `short_answer_text`
- `full_response_text`

### Per-Record Semantics

`error` should mean:

- execution or recovery encountered a problem worth surfacing

It should not mean:

- the record was necessarily unscored

Examples:

- degraded but scored run: `error` may be non-empty and `scored=true`
- unrecoverable failure: `error` non-empty and `scored=false`

### Results Manifest Changes

Add to top-level `results.json`:

- `schema_version`
- `status_axes_description`

Each result entry should carry the new status fields above.

### Summary Changes

For each group and subset, add:

- `run_completed_count`
- `run_failed_count`
- `protocol_completed_count`
- `protocol_failed_count`
- `evaluable_count`
- `scored_count`
- `recovered_evaluable_count`
- `native_evaluable_count`
- `non_evaluable_count`
- `degraded_execution_count`

Retain:

- `pass_count`
- `avg_normalized_score`

But treat them as answer-quality indicators, not operational health indicators.

## CSV Reporting Design

### Short-Term Level 2 Reporting

Keep the existing CSV files:

- `summary_by_group.csv`
- `summary_by_group_and_subset.csv`

Add new columns without removing old ones:

- `run_completed_count`
- `protocol_completed_count`
- `evaluable_count`
- `scored_count`
- `recovered_evaluable_count`
- `degraded_execution_count`
- `non_evaluable_count`

This keeps most downstream parsing intact while exposing the new semantics.

### Long-Term Level 3 Reporting

Once downstream consumers are migrated, make the primary reading order:

1. `evaluable_count`
2. `scored_count`
3. `pass_count`
4. `avg_normalized_score`

This reorients interpretation toward:

- how many records remained benchmark-usable
- then how many passed

## Aggregation Design

### New Aggregate Buckets

Aggregation must become multi-dimensional.

Current bucket model:

- count
- pass_count
- avg_score
- avg_normalized_score

New bucket model should additionally compute:

- operational lifecycle counts
- protocol lifecycle counts
- evaluability counts
- recovery provenance counts

### Recommended Group Summary Shape

Each group summary should include:

- score metrics
- operational metrics
- answer provenance metrics

By evaluation kind and by subset, the same fields should be produced.

## Compatibility Strategy

### Phase 1: Dual semantics

Introduce new fields while preserving:

- old dataclass fields
- old CSV columns
- old result top-level structure

This is the Level 2 rollout stage.

### Phase 2: Consumer migration

Update:

- tests
- docs
- local result readers
- any analysis notebooks or scripts

to consume new status fields.

### Phase 3: Promote Level 3 semantics

After migration:

- document `schema_version`
- treat new status axes as required
- update developer guidance to stop interpreting `error != null` as automatically unscored

## Detailed Implementation Design

### Part A: Contracts

Modify runner contracts so `RecoveryInfo` carries:

- `scored`
- `evaluable`
- `reliability`
- `recovery_mode`
- `reason`

Optional simplification:

- collapse some duplicated metadata into a single recovery assessment object

### Part B: ChemQA Runner

Refactor non-success terminal branch into:

1. archive artifacts
2. run recovery assessment
3. if `evaluable=true`, return `RECOVERED` with `scored=true`
4. else return `FAILED`

Do not leave scoreability encoded as a hardcoded constant in the fallback path.

### Part C: GroupRecordResult construction

When `run_result.should_score()` is true:

- evaluate normally
- record recovery / degraded metadata alongside evaluation

When `run_result.should_score()` is false:

- build execution-error evaluation
- mark `evaluable=false`, `scored=false`

### Part D: Error result construction

`build_error_group_record_result()` must be extended to populate the new required status fields for unrecoverable cases.

### Part E: Aggregation

Update `aggregate_bucket()` to compute the new counts without changing score math definitions.

### Part F: Results export

Update:

- `results.json`
- `summary_by_group.csv`
- `summary_by_group_and_subset.csv`

to include new fields.

## Edge Cases

### Edge Case 1: Completed protocol, empty answer

If the workflow reaches `completed` but the extracted short answer is empty or obviously invalid:

- mark `evaluable=false`
- keep lifecycle as completed
- do not score

### Edge Case 2: Failed workflow, valid archived candidate

If the workflow fails but a valid `proposer-1` candidate is archived:

- mark `degraded_execution=true`
- `recovery_mode=candidate_submission`
- `evaluable=true`
- score it

### Edge Case 3: Rejected protocol, valid answer

If protocol acceptance is rejected but the candidate contains a usable answer:

- `protocol_acceptance_status=rejected`
- `evaluable=true`
- score it

### Edge Case 4: Preview-only answer

If only `final_answer_preview` exists:

- score only if extraction and validator heuristics say it is trustworthy enough
- otherwise record `recovered_non_evaluable`

### Edge Case 5: Placeholder or rejection blob

If recovered content is a failure blob:

- `answer_availability=missing`
- `evaluable=false`

## Testing Plan

### Unit Tests

Add tests for:

- recovered archived candidate becomes scoreable
- rejected completed run with real answer becomes scoreable
- failed run with only rejection blob remains non-evaluable
- preview-only scalar answer is scoreable when valid
- preview-only placeholder answer is non-evaluable

### Aggregation Tests

Add tests that verify:

- `evaluable_count`
- `scored_count`
- `recovered_evaluable_count`
- `degraded_execution_count`

are computed correctly.

### CSV Tests

Update CSV tests to verify:

- existing columns remain
- new columns appear with expected values

### Regression Tests

Use realistic ChemQA fixtures for:

- failed-but-archived-answer
- completed-but-rejected-answer
- failed-and-empty

## Rollout Plan

### Stage 1: Runtime correctness

Implement evaluable recovery and scoreability in runner/runtime first.

Success criterion:

- recovered answers actually reach evaluators

### Stage 2: Schema extension

Extend per-record and aggregate result schemas while preserving old fields.

Success criterion:

- no local callers break
- new metrics appear in JSON and CSV

### Stage 3: Consumer migration

Update tests, docs, and scripts to consume:

- `evaluable`
- `scored`
- `recovery_mode`

Success criterion:

- operational and scoring metrics are interpreted separately

### Stage 4: Promote Level 3 semantics

Document new schema version and treat status axes as canonical.

Success criterion:

- internal analysis stops using `pass_count` as a proxy for system stability

## Risks

### Risk 1: Scoring low-confidence recovered answers

Mitigation:

- strict source ranking
- strict placeholder rejection
- reliability tier tracking

### Risk 2: Downstream schema breakage

Mitigation:

- additive Level 2 rollout first
- explicit `schema_version`
- preserve old columns during transition

### Risk 3: Hidden semantic drift in `error`

Mitigation:

- document `error` as diagnostic, not scoreability gate
- add explicit `evaluable` and `scored`

## Recommended Implementation Order

1. update runner contracts to represent evaluable recovery explicitly
2. refactor ChemQA non-success branch to return scoreable recovered answers
3. extend per-record schema with status axes
4. extend aggregation and CSV output
5. update tests
6. update `GLOBAL_DEV_SPEC.md`

## Acceptance Criteria

This design is considered successfully implemented when all of the following are true:

1. A ChemQA run with failed workflow but valid archived candidate answer is scored.
2. A ChemQA run with rejected protocol but valid answer is scored.
3. A ChemQA run with only rejection/failure blobs is not scored.
4. Per-record outputs explicitly indicate whether a record was evaluable and whether it was scored.
5. Aggregate reports distinguish operational completion from benchmark pass rate.
6. Existing score metrics remain available during the migration window.
