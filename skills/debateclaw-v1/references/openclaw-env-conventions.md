## OpenClaw Env Name Discovery

When DebateClaw is preparing an OpenClaw-based runtime, inspect only env
variable names. Do not print or quote any secret values.

Assume `.` means the DebateClaw V1 skill root containing `SKILL.md`.

Use:

```bash
uv run --script ./scripts/inspect_openclaw_env.py \
  --env-file ~/.openclaw/.env \
  --family qwen \
  --family minimax \
  --family kimi \
  --family glm \
  --json
```

The helper accepts user-specific naming. It searches for provider/model-related
names rather than assuming a fixed scheme:

- `qwen`: match names containing `QWEN` or `DASHSCOPE`
- `minimax`: match names containing `MINIMAX`
- `kimi`: match names containing `KIMI` or `MOONSHOT`
- `glm`: match names containing `GLM` or `BIGMODEL`

## Decision Rule

- If exactly one API key name and one base URL name are discoverable for a
  family, use them.
- If multiple candidate names exist for a family, present only the variable
  names to the user and ask which mapping should be used.
- If no suitable names exist, stop and ask whether the user wants to create the
  standard names below.

## Standard Naming Fallback

Recommend this naming convention when the user wants DebateClaw-compatible,
predictable names:

- `QWEN_API_KEY`
- `QWEN_ANTHROPIC_BASE_URL`
- `MINIMAX_API_KEY`
- `MINIMAX_ANTHROPIC_BASE_URL`
- `KIMI_API_KEY`
- `KIMI_ANTHROPIC_BASE_URL`
- `GLM_API_KEY`
- `GLM_ANTHROPIC_BASE_URL`

The standard assumes Anthropic-compatible base URLs for the default DebateClaw
OpenClaw path.
