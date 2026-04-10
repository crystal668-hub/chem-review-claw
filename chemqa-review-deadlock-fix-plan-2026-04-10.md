# ChemQA Review Deadlock Fix Patch Plan (2026-04-10)

## Goal

Eliminate the recurring `propose`-phase deadlock pattern in `chemqa-review` runs where:

- reviewer lanes stop after planning instead of actually registering a placeholder proposal,
- the coordinator says it will wait and then exits,
- the debate remains `running` but stuck at `propose N/5`,
- there is no watchdog / deterministic recovery path.

This plan intentionally does **not** focus on rescuing one bad run. It changes the runtime so future runs do not rely on a model to remember transport mechanics or to keep itself alive while waiting.

---

## Root Cause Summary

The current architecture has three brittle assumptions:

1. **Reviewer lanes are asked to perform transport actions themselves**
   - write `proposal.md`
   - call `debate_state.py submit-proposal`
   - verify submission
   
   In practice, some models stop after saying they will do it.

2. **Coordinator waiting is delegated to the model**
   The coordinator frequently produces a natural-language "I will wait" answer, then the one-shot OpenClaw wrapper exits.

3. **No deterministic liveness layer exists above model turns**
   Missing transport submissions and a dead coordinator do not trigger a structured recovery path.

So the fix must be architectural, not just prompt-level.

---

## Patch Strategy

### High-level changes

1. **Move transport registration out of the model and into a deterministic runtime driver.**
2. **Move waiting / polling / advancement loops out of the model and into a deterministic runtime driver.**
3. **Treat model turns as artifact-generation steps, not protocol-execution steps.**
4. **Add explicit liveness / fail-fast checks so a stuck run becomes diagnosable failure instead of silent limbo.**

---

## File-Level Patch Plan

## 1) Add a ChemQA-specific OpenClaw driver wrapper

### New file

- `/home/dministrator/.openclaw/skills/chemqa-review/scripts/chemqa_review_openclaw_driver.py`

### Purpose

Replace the current fragile behavior of directly launching `openclaw_debate_agent.py` for ChemQA roles.

This driver becomes the role runtime entrypoint for `chemqa-review` runs.

### Responsibilities

#### Common

- load env and slot workspace similar to `openclaw_debate_agent.py`
- preserve explicit `--session-id`
- call OpenClaw only when model reasoning is actually needed
- perform deterministic state checks before and after each model turn
- fail fast with machine-readable error when required postconditions are not met

#### Reviewer lanes (`proposer-2`..`proposer-5`)

**During `propose`:**
- if no placeholder is registered for this lane:
  - generate a deterministic placeholder body in Python from a built-in template
  - write `proposal.md`
  - call `debate_state.py submit-proposal ...`
  - force-verify state now includes this proposal
- do **not** ask the model to do placeholder transport work

**During `review`:**
- invoke the model to draft the formal review artifact only
- after the model turn:
  - verify a review file exists
  - parse the required metadata shape minimally
  - call `debate_state.py submit-review ...`
  - force-verify review registration
- if the model returns without a file, retry with a corrective prompt once or twice, then fail explicitly

**During `wait`:**
- do not ask the model to wait
- the driver sleeps / polls itself at bounded intervals
- exit only when role work is actually complete or a hard blocker is detected

#### Main proposer (`proposer-1`)

**During `propose`:**
- invoke the model to draft `proposal.md`
- driver submits with `submit-proposal`
- force-verify proposal registration
- retry once with a corrective prompt if no candidate file was produced

**During `rebuttal`:**
- invoke the model to draft rebuttal / concession artifact
- driver submits with `submit-rebuttal`
- force-verify registration

#### Coordinator

- maintain a real runtime loop in Python:
  - get compact snapshot / next action
  - if `advance`, run `debate_state.py advance`
  - if `wait`, sleep and re-check
  - if `done`, invoke model only for the terminal summary / protocol artifact
- do **not** rely on the model to keep the process alive during waiting
- if idle for too long with missing proposer submissions or missing required reviewer actions, emit a structured blocker and exit nonzero

### Why this is the core fix

This removes the two failure modes seen in the incident:

- models that stop after planning transport,
- coordinator sessions that terminate after saying "I will wait".

---

## 2) Route ChemQA command maps to the new driver

### File to modify

- `/home/dministrator/.openclaw/skills/chemqa-review/scripts/materialize_runplan.py`

### Current behavior

`build_command_map()` points every role to the generic DebateClaw wrapper:

- `.../openclaw_debate_agent.py --slot ... --session-id ...`

### Patch

Change command-map generation so ChemQA runs use the new driver instead.

### Proposed command shape

For each role, generate something like:

```json
[
  "python3",
  "/home/dministrator/.openclaw/skills/chemqa-review/scripts/chemqa_review_openclaw_driver.py",
  "--skill-root", "<chemqa-skill-root>",
  "--team", "<run-id>",
  "--role", "proposer-2",
  "--slot", "debate-2",
  "--session-id", "<session-id>",
  "--env-file", "~/.openclaw/.env",
  "--thinking", "medium"
]
```

### Required arguments to pass through

- `--skill-root`
- `--team`
- `--role`
- `--slot`
- `--session-id`
- `--env-file`
- optional `--thinking`

### Note

Do **not** change generic `debateclaw-v1` command-map behavior globally. Keep this override local to `chemqa-review`.

---

## 3) Add deterministic placeholder templates in code

### New file

- `/home/dministrator/.openclaw/skills/chemqa-review/scripts/chemqa_review_transport.py`

### Purpose

Centralize artifact-generation helpers that must be deterministic and should not depend on model creativity.

### Helpers to add

- `render_placeholder_proposal(role: str, semantic_role: str) -> str`
- `expected_review_filename(role: str, target: str) -> str`
- `expected_rebuttal_filename(role: str) -> str`
- `validate_formal_review_shape(text: str, role: str, target: str) -> list[str]`
- `validate_candidate_submission_shape(text: str) -> list[str]`
- `current_submission_state(team, role, ...)` convenience helpers

### Key design choice

Reviewer placeholder proposals should be deterministic boilerplate, not LLM-authored.

That removes an entire class of deadlocks without losing useful semantics.

---

## 4) Tighten prompt contracts so models only generate artifacts, not transport or waiting behavior

### Files to modify

- `/home/dministrator/.openclaw/skills/chemqa-review/prompts/contracts/coordinator.md`
- `/home/dministrator/.openclaw/skills/chemqa-review/prompts/contracts/proposer-main.md`
- `/home/dministrator/.openclaw/skills/chemqa-review/prompts/contracts/reviewer-search-coverage.md`
- `/home/dministrator/.openclaw/skills/chemqa-review/prompts/contracts/reviewer-evidence-trace.md`
- `/home/dministrator/.openclaw/skills/chemqa-review/prompts/contracts/reviewer-reasoning-consistency.md`
- `/home/dministrator/.openclaw/skills/chemqa-review/prompts/contracts/reviewer-counterevidence.md`
- `/home/dministrator/.openclaw/skills/chemqa-review/prompts/modules/policies/review-loop-bridge.md`
- `/home/dministrator/.openclaw/skills/chemqa-review/prompts/modules/policies/state-query-discipline.md`

### Prompt changes

#### Reviewer contracts

Replace wording like:

- "draft placeholder then explicitly register it with DebateClaw transport"

with wording like:

- "during `propose`, the runtime will register your placeholder transport artifact automatically"
- "your responsibility is to produce substantive review content only when the engine opens `review`"
- "if the runtime asks for a corrective rewrite, revise the review file; do not spend turns on waiting or transport bookkeeping"

#### Proposer contract

Replace wording like:

- "draft candidate, then explicitly register it"

with:

- "draft the candidate artifact file; the runtime wrapper will register it"
- "if the runtime reports missing structure, revise the file"

#### Coordinator contract

Replace model-owned waiting behavior with:

- "the runtime wrapper handles waiting / polling / advancing"
- "your role is only to produce terminal coordinator artifacts and diagnose blockers when the wrapper asks"

#### State-query discipline

Add a hard line:

- models should not implement their own sleep/poll loops in ChemQA runs
- the wrapper owns waiting and state refresh

### Why this matters

Once the wrapper owns protocol mechanics, leaving old wording in prompts would keep models wasting turns on the wrong responsibilities.

---

## 5) Add liveness / fail-fast policy in the driver

### New behavior

The driver should distinguish:

- temporary wait
- protocol blocker
- terminal completion

### Suggested thresholds

#### Reviewer / proposer lanes

- if expected state transition does not happen after `N` consecutive checks and this role has already done its part, keep sleeping but emit periodic diagnostics
- if the role is expected to submit and did not manage to produce a valid file after `max_attempts` (e.g. 2), exit nonzero with an explicit machine-readable blocker

#### Coordinator

If any of these are true for too long, exit with explicit failure:

- `propose` unchanged for > 10 min and some proposer lanes are missing submissions
- `review` unchanged for > 10 min and required reviewer lanes are missing
- `rebuttal` unchanged for > 10 min and active candidate owners are missing rebuttals

### Failure payload should include

- team
- phase
- expected actions missing
- missing lanes
- whether coordinator advanced at least once
- whether runtime saw any state movement during the window

### Why

A dead run should become a visible failed run, not an indefinitely `running` run.

---

## 6) Add post-turn verification and corrective re-prompting

### In the new driver

After each model turn, enforce a role-specific postcondition.

#### proposer-1 during propose

Expected postcondition:
- `proposal.md` exists and is non-empty

If missing:
- send one corrective follow-up in the same OpenClaw session:
  - explain exactly what file is missing
  - ask for only that file
- if still missing after retry budget, fail nonzero

#### reviewer lanes during review

Expected postcondition:
- formal review file exists
- metadata shape minimally valid

If missing:
- same corrective follow-up pattern

#### coordinator during done

Expected postcondition:
- `chemqa_review_protocol.json` exists and parses as JSON

If missing:
- corrective follow-up once or twice
- then fail explicitly

### Why

This directly addresses the observed failure mode:

- model says "Let me create the file first"
- then exits without a file

---

## 7) Make task status updates deterministic in the wrapper

### New runtime behavior

The driver should own task state transitions instead of relying on the model to remember them.

### For every role

At startup:
- find the task owned by this role
- set it to `in_progress`

At terminal success:
- set task to `completed`

At hard blocker / explicit failure:
- either leave `in_progress` and exit nonzero, or set `blocked` if your ClawTeam workflow expects that

### Why

This removes another source of divergence:
- model exits early and task resets to pending
- protocol state and task board drift apart

---

## 8) Preserve generated-state snapshots only as optimization, not as control plane

### File to adjust lightly

- `/home/dministrator/.openclaw/skills/chemqa-review/scripts/chemqa_review_state_snapshot.py`

### Patch

Keep the snapshot helper, but treat it as a read optimization only.

Add optional fields such as:

- `missing_proposer_submissions`
- `missing_required_reviewer_lanes`
- `active_candidate_owner`
- `qualifying_formal_reviews_count`

These fields will help the wrapper make deterministic decisions without having to parse full raw state repeatedly.

### Important

Do **not** use snapshot caching to hide state changes after wrapper-owned submissions.
After any state-changing action, the wrapper should always force-refresh once.

---

## 9) Optional but recommended: add a supervisor-level watchdog script

### New file

- `/home/dministrator/.openclaw/skills/chemqa-review/scripts/check_run_liveness.py`

### Purpose

Allow operators or future automation to inspect a running ChemQA team and answer:

- Is it healthy?
- Which phase is stuck?
- Which lanes are missing?
- Is the coordinator alive or absent?

### Output

JSON like:

```json
{
  "team": "chemqa-review-...",
  "healthy": false,
  "phase": "propose",
  "progress": {"actual": 3, "expected": 5},
  "missing_roles": ["proposer-2", "proposer-3"],
  "coordinator_task_status": "pending",
  "recommendation": "restart-missing-agents-or-fail-run"
}
```

This is not the primary fix, but it gives you a clean operational surface.

---

## Suggested Implementation Order

### Patch set A (must-have)

1. Add `chemqa_review_openclaw_driver.py`
2. Change `materialize_runplan.py` command-map generation to use it
3. Add deterministic placeholder generation helpers
4. Add post-turn verification + corrective retry logic
5. Move coordinator waiting/advance loop into the wrapper

### Patch set B (strongly recommended)

6. Rewrite prompt contracts to match the new runtime responsibilities
7. Add liveness timeout / fail-fast behavior
8. Make task status updates wrapper-owned

### Patch set C (ops polish)

9. Extend `chemqa_review_state_snapshot.py` with missing-lane summaries
10. Add `check_run_liveness.py`

---

## Acceptance Tests

## Test 1: reviewer propose no longer depends on model transport behavior

### Setup
Launch a ChemQA run where reviewer lanes are weak models.

### Expected
Within a short bounded time after launch:
- `propose` progress reaches at least `5/5` if proposer-1 also succeeds
- each reviewer lane has a registered placeholder proposal even if the model never explicitly called `submit-proposal`

## Test 2: coordinator does not exit just because it says "waiting"

### Setup
Launch a run and intentionally leave it idle between phases.

### Expected
- coordinator process remains alive
- task stays `in_progress`
- no `Agent 'debate-coordinator' exited unexpectedly` message appears during normal waiting

## Test 3: model that only says "I will create the file" gets corrected or failed fast

### Setup
Force a reviewer lane model prone to stopping after planning.

### Expected
- wrapper detects missing review artifact
- sends a corrective follow-up in the same session
- either obtains the artifact and submits it, or exits nonzero with explicit blocker
- run does not sit forever in silent limbo

## Test 4: stuck run becomes explicit failure, not silent running deadlock

### Setup
Simulate a missing agent submission and no recovery.

### Expected
- liveness timeout triggers
- blocker payload identifies missing lanes and phase
- run is diagnosable without manual transcript archaeology

## Test 5: task board and protocol state remain aligned

### Expected
- active roles have `in_progress`
- completed roles become `completed`
- tasks do not revert to `pending` during ordinary waiting loops

---

## Concrete Non-Goals

This patch should **not**:

- globally change generic `debateclaw-v1` behavior for non-ChemQA runs
- rely only on stronger prompt wording
- depend on synthetic reviews to bypass missing required reviewers
- treat placeholder proposals as substantive review completion
- solve this only by changing models

---

## Minimal Viable Diff Summary

If you want the smallest patch with the biggest effect, do exactly this:

1. Add `chemqa_review_openclaw_driver.py`
2. Make ChemQA command maps use it instead of `openclaw_debate_agent.py`
3. In that driver:
   - auto-submit reviewer placeholders deterministically
   - wrap coordinator in a real loop that polls / advances / sleeps itself
   - verify postconditions after model turns
4. Update prompt text so models stop owning transport and waiting

That is the shortest path from the current deadlock-prone architecture to a stable one.
