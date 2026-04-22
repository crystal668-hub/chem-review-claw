# DebateClaw Runtime Matrix

## Status

- macOS path: locally validated on 2026-03-20 in this workspace
- Linux path: not locally validated; based on official docs and upstream README

## Core Runtime Requirements

DebateClaw relies on ClawTeam. Upstream ClawTeam documents these baseline
requirements:

- Python 3.10+
- `tmux`
- a CLI agent such as `claude`, `codex`, or `openclaw`

Official upstream reference:

- https://github.com/HKUDS/ClawTeam/blob/main/README.md

ClawTeam can then launch DebateClaw templates with:

```bash
clawteam launch <template> --goal "..."
```

## Recommended Install Shape

Prefer this order:

1. install `uv`
2. install `tmux`
3. install exactly one agent CLI to use as the ClawTeam worker runtime
4. install `clawteam` with `uv`
5. deploy DebateClaw templates
6. deploy DebateClaw runtime helpers

For end users, prefer:

```bash
uv tool install clawteam
```

That keeps ClawTeam as a standalone CLI and lets `uv` manage Python for the
tool environment.

## uv

Official docs:

- https://docs.astral.sh/uv/getting-started/installation/

Typical official install paths:

- macOS: `brew install uv`
- macOS or Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## tmux

ClawTeam requires `tmux` for the default interactive backend.

Typical install paths:

- macOS: `brew install tmux`
- Debian/Ubuntu: `sudo apt-get install tmux`
- Fedora/RHEL: `sudo dnf install tmux`
- Arch: `sudo pacman -S tmux`

## Entry CLI Options

### Codex

Official docs:

- https://help.openai.com/en/articles/11096431-openai-codex-ci-getting-started

Official install command:

```bash
npm install -g @openai/codex
```

Validated locally in this workspace:

- `codex-cli 0.116.0`
- binary path: `/opt/homebrew/bin/codex`

### Claude Code

Official docs:

- https://docs.anthropic.com/en/docs/claude-code/getting-started

Official install command:

```bash
npm install -g @anthropic-ai/claude-code
```

The official docs describe support for macOS and Linux.

### OpenClaw

Official docs:

- https://docs.openclaw.ai/install

Official install command:

```bash
curl -fsSL https://openclaw.ai/install.sh | bash
```

The official docs also describe verifying the install afterward.

## Shared State Runtime

DebateClaw uses a local SQLite database for protocol state:

- path: `${CLAWTEAM_DATA_DIR:-~/.clawteam}/teams/<team>/debate/state.db`
- implementation: Python standard library `sqlite3`
- extra service required: none

This is intentional. The debate protocol now needs durable, queryable state for:

- proposer history across epochs
- full cross-review matrices
- rebuttal rounds
- candidate detection
- restart logic after all proposals fail

Do not introduce a separate database server for the MVP.

## macOS Path We Actually Tested

Validated locally:

- `uv 0.10.12`
- `Python 3.11.15`
- `tmux 3.6a`
- `codex-cli 0.116.0`
- project-local editable ClawTeam install with:

  ```bash
  uv venv --python /opt/homebrew/bin/python3.11 .venv
  uv pip install --python .venv/bin/python -e /Users/losfyrid/ref-source-code/ClawTeam
  .venv/bin/clawteam --help
  ```

- isolated `uv tool install clawteam` validation using local `UV_TOOL_DIR` and
  `UV_TOOL_BIN_DIR`
- DebateClaw runtime helpers can be deployed into `~/.clawteam/debateclaw/bin`
  and default templates can be generated into `~/.clawteam/templates`

Observed nuance:

- the PyPI tool install succeeded, but its built-in template set was smaller
  than the current source checkout. DebateClaw should not rely on any newly
  added upstream builtin templates; it should always deploy its own templates.

## Linux Guidance

Linux bootstrap is doc-based in this skill bundle. The intended flow is:

1. install `uv` from the official uv docs
2. install `tmux` with the distro package manager
3. install the chosen agent CLI from its official docs
4. run the bundled DebateClaw install and deploy scripts

Use the same DebateClaw scripts on Linux, but treat the result as unverified
until it is tested on a real Linux machine.
