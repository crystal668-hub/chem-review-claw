# DebateClaw Slot Workspace

This directory is a sterile DebateClaw slot workspace.

## Rules

- Treat this workspace as debate-only scratch space.
- Do not edit or rely on `USER.md`, `TOOLS.md`, `SOUL.md`, `MEMORY.md`, `BOOTSTRAP.md`, `IDENTITY.md`, `HEARTBEAT.md`, or any similar personal-context files.
- Do not create those files here.
- Do not modify this `AGENTS.md` file.
- Top-level non-hidden files and directories other than `AGENTS.md` are disposable run outputs and may be reset automatically when a new run starts.
- Prefer the current DebateClaw prompt and runtime state over filesystem exploration.
- Do not broad-scan the workspace just to orient yourself. Avoid `find`, recursive `ls`, or large grep sweeps unless the current prompt explicitly requires them.
- Use the exact artifact filename requested by the current runtime prompt. Do not infer the current contract from leftovers or examples from older runs.

## Allowed working set

Use only:

- the current task prompt
- DebateClaw runtime state (`debate_state.py`, `next-action`, proposal/review/rebuttal history surfaced there)
- the exact top-level run-scoped scratch file(s) named by the current prompt/runtime, for example:
  - `proposal.yaml` or `proposal.md`
  - `review-*.yaml` or `review-*.md`
  - `rebuttal.yaml` or `rebuttal.md`
  - `coordinator-summary.md`
  - `chemqa_review_protocol.yaml`

If the prompt tells you to write one exact file, write that file directly and stop instead of exploring the workspace.

## Debate norms

- Use blocking objections only when they are substantive.
- Review only the targets listed by `debate_state.py next-action`; never review your own proposal.
- Rebut by attack theme, not by reviewer name.
- If your own proposal is no longer defensible, concede explicitly instead of bluffing.
- Conceding or failing your own proposal does **not** end your participation; keep following `next-action` until it returns `stop`.
- If the same tool call fails twice with the same validation or schema error, stop retrying and report the blocker to `debate-coordinator`.
