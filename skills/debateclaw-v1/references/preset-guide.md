# DebateClaw V1 Preset Guide

Assume `.` means the DebateClaw V1 skill root containing `SKILL.md`.

## Presets

### `parallel@1`

Use when:

- the user wants multiple independent proposals
- no internal review/rebuttal loop is needed
- the outer entry agent should pick the final answer after the internal team finishes

Best for:

- solution ideation
- alternative plans or designs
- multiple candidate answers to the same question

Example:

```bash
python3 ./scripts/launch_from_preset.py \
  --root . \
  --preset parallel@1 \
  --goal "Motion or prompt: propose the best persistence design for DebateClaw shared state. Context: compare implementation complexity, observability, reliability, and migration cost. Evidence boundaries: provided context plus any inherited external file-access capability. Decision criteria: the outer entry agent should prefer the best reliability-to-complexity tradeoff for an MVP. Deliverable: a final recommendation with decisive reasons and residual risks." \
  --launch-mode print
```

To pass extra file context for one run:

```bash
python3 ./scripts/launch_from_preset.py \
  --root . \
  --preset parallel@1 \
  --goal "..." \
  --additional-file-workspace "minio://bucket/path/to/run-inputs/" \
  --launch-mode print
```

### `review-loop@1`

Use when:

- the user wants proposal, full cross-review, and rebuttal rounds
- candidate detection should be based on review rounds producing no meaningful new blocking objection
- a stronger internal stress-test loop is desired before the outer entry agent decides

Best for:

- architecture debates
- execution strategy debates
- plan stress tests with multiple surviving possibilities
- evidence-first debates where repeated rounds matter

Example:

```bash
python3 ./scripts/launch_from_preset.py \
  --root . \
  --preset review-loop@1 \
  --goal "Motion or prompt: determine the best DebateClaw runtime packaging strategy. Context: the bootstrap path must support macOS and Linux, and Python environment management should use uv. Evidence boundaries: provided context plus any inherited external file-access capability. Decision criteria: the outer entry agent should choose or fuse the strongest surviving proposal. Deliverable: final recommendation, decisive reasons, strongest unresolved risk, and next actions." \
  --launch-mode print
```

## Final Judgment

The internal DebateClaw team does not produce the final winner by itself.
After the internal team completes, the outer entry agent should read the
resulting debate state and synthesize or select the final answer.

## Additional File Boundary

Use `--additional-file-workspace <opaque-value>` only when the debate run should receive extra task files or reference material from the surrounding runtime.

Important:

- the value is opaque to DebateClaw
- it may be a path, object-store URI, or another backend-specific locator
- DebateClaw passes it through into run plans, runtime context, prompt bundles, and rendered templates
- DebateClaw does not interpret or materialize it

## Backend Note

The current V1 OpenClaw path uses `subprocess`, not tmux.

## Missing Template Recovery

If launch fails because the template/runtime assets are missing or stale, use the V1 bootstrap helpers to inspect status and redeploy assets before retrying:

```bash
python3 ./scripts/bootstrap_status.py --root .
python3 ./scripts/deploy_templates.py --dry-run
```
