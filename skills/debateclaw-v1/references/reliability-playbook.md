# DebateClaw Reliability Playbook

Use this playbook for DebateClaw launches that depend on local runtime behavior.
It is intentionally machine-agnostic: record machine-specific facts in workspace
notes, not in the shared skill body.

## Local Runtime Notes Policy

Before making machine-specific decisions, check local notes in this order:

1. A user-specified path or explicit project convention
2. Any stronger workspace mechanism already in use, such as `.learnings/`,
   `TOOLS.md`, or a dedicated memory/self-improvement skill
3. Weak fallback: a workspace-local `debateclaw/` directory

The fallback is intentionally weak. Use it only when nothing stronger exists.
Do not move machine-specific facts back into the shared DebateClaw skill.

## Launch Preconditions

A DebateClaw launch is ready only when all of these are true:

1. The motion is concrete, unless the user explicitly asked for a smoke test
2. `clawteam` syntax has been checked on the current machine with `--help`
3. The assigned providers passed a real smoke test, not just env-name discovery
4. Slot reuse and session isolation are both defined
5. The user-facing progress instructions are ready to send back immediately

## Slot And Session Policy

- Reuse fixed DebateClaw slots whenever possible
- Reuse slots, **not** run instances: default every new debate launch to a
  fresh `run_id` and a new debate round
- Do not create fresh slots just to clear history
- Clear history by starting fresh session ids, or by explicitly clearing the
  previous session state if the runtime requires it
- Only inspect, re-render, or recover an existing run when the user explicitly
  asks for that exact old run or names the `run_id`
- Canonical implementation: keep slot provisioning reusable, then inject
  per-team session ids while preparing the final template or per-role commands
- If the final template still lacks session ids, patch the generated command
  map or wrapper before launch rather than compensating with new slots

A good per-role session-id pattern is:

```text
 debate:<team-name>:coordinator
 debate:<team-name>:proposer-1
 debate:<team-name>:proposer-2
 ...
```

## Common Failure Modes

### 1. CLI Syntax Drift

Symptoms:
- `clawteam launch` rejects `--team`
- `board show` or `task wait` syntax differs from template examples

Response:
- Read local `clawteam ... --help`
- Adapt launch and monitoring commands to the installed version
- Do not trust generated helper output blindly when the local CLI disagrees

### 2. Env Discovery Passes But Runtime Auth Fails

Symptoms:
- Env inspection says a family looks complete
- Real agent execution still fails with auth or endpoint errors

Response:
- Treat env-name inspection as necessary but not sufficient
- Run one minimal real call per assigned family before launch
- Record the machine-specific result in local notes

### 3. Stale Session Contamination

Symptoms:
- New proposers mention old debates or old motions
- A worker appears to continue an earlier run instead of the current one

Response:
- Keep the slot, replace the session
- Abort stale waiters or wrapper processes before retrying
- Only retry after you can describe the new session isolation plan

### 4. Team Exists But Debate Is Not Healthy

Symptoms:
- `clawteam launch` succeeds but no proposals enter state
- Some tasks stay pending or idle without meaningful progress

Response:
- Distinguish team creation from debate health
- Do not tell the user the debate is truly running until at least one proposal
  is accepted into the state ledger
- If workers are idle, inspect task board, inbox, and per-role runtime state
  before retrying

### 5. Layered Retries

Symptoms:
- Multiple `task wait` or wrapper processes stack up
- Old runs keep consuming tokens while a new run is being debugged

Response:
- Stop old waiters and stale worker processes first
- Summarize the failure mode plainly
- Retry only once the next plan is materially different

### 6. Self-Review Confusion

Symptoms:
- A proposer tries to submit a review for its own proposal
- The runtime rejects the submission with `A proposer cannot review its own proposal.`

Response:
- Treat `next-action.targets` / `target_proposals` as the source of truth
- Cross-review means review only the listed targets, never yourself
- If a round stalls, inspect the missing reviewer -> target pairs instead of assuming the
  proposal with blocking objections is the blocker

### 7. Failed Proposal, Active Reviewer

Symptoms:
- A proposer concedes or fails its own proposal and then marks its task completed early
- The protocol still expects that proposer to continue submitting cross-reviews

Response:
- In the current policy, a failed proposer is still an active reviewer until `next-action`
  returns `stop`
- Proposal failure removes candidate status, not review obligations
- Do not let the task board override `debate_state.py next-action`

### 8. Task / Protocol Divergence

Symptoms:
- The task board says `completed` but `next-action` is still `review`, `rebuttal`, or `wait`
- UI progress looks finished while the protocol still has missing actions

Response:
- Treat protocol state as the source of truth for debate progress
- Use the task board as an execution hint, not a completion oracle
- Surface and log divergence explicitly; if a worker is completed early, intervene instead of
  waiting for self-recovery
- If you must unblock with a synthetic/manual review, mark it explicitly and record who submitted it and why

## User Update Contract

Immediately after creating a debate, report:

- run ID: the `team-name`
- template name
- role -> slot -> model mapping
- how to watch progress:
  - `clawteam board show <team-name>`
  - `clawteam board attach <team-name>`
  - `clawteam inbox log <team-name>`
  - `~/.clawteam/debateclaw/bin/debate_state.py summary --team <team-name> --json --include-bodies`

Use precise language:
- `team created` means ClawTeam accepted the launch
- `debate healthy` means workers are responding and state is advancing
- `results ready` means the outer judge can read completed proposals from state

## Abort Rules

Abort and report cleanly instead of improvising more retries when any of these
are true:

- the motion is still ambiguous
- the same provider fails the same smoke test twice
- the session isolation plan is still unclear
- retries would only add more partial runs without new information
