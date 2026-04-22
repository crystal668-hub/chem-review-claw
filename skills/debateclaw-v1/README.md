# DebateClaw V1 Skill Bundle

This directory is the **copyable install surface** for DebateClaw V1.

If you cloned the full repo but only want the agent-facing skill bundle, copy
this directory to your OpenClaw skills location and rename it as needed, for
example:

```bash
cp -R debateclaw-v1 ~/.openclaw/skills/debateclaw-v1
```

What is inside:

- `SKILL.md` — the installable V1 skill entrypoint
- `scripts/` — self-contained runtime/bootstrap/operator scripts (including `launch_from_config.py` for external reusable `config_id` consumption)
- `references/` — agent-facing guidance docs
- `control/`, `workflows/`, `presets/`, `prompts/` — V1 metadata and prompt assets
- `generated/` — runtime-derived outputs, shipped empty

What is intentionally **not** inside:

- developer handoff docs
- implementation planning notes
- engineering examples

Those stay at the repo root beside this bundle so agents using the skill do not
need to see them.
