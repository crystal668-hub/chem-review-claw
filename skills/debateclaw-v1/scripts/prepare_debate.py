#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///

"""Prepare a DebateClaw launch by rendering a concrete template and initializing state."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path

from debate_state import DebateConfig, init_debate_state
from debate_templates import (
    DEFAULT_RUNTIME_ROOT,
    build_parallel_judge_template,
    build_review_loop_template,
    template_name_for,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a DebateClaw template and SQLite state.")
    parser.add_argument("--workflow", required=True, choices=("parallel-judge", "review-loop", "chemqa-review"))
    parser.add_argument("--team", required=True, help="ClawTeam team name.")
    parser.add_argument("--goal", required=True, help="Debate goal or motion.")
    parser.add_argument(
        "--evidence-policy",
        default=(
            "Evidence first. Use only sources explicitly allowed for this launch. "
            "Label unsupported claims as hypotheses or open questions."
        ),
        help="Evidence policy written into the debate state.",
    )
    parser.add_argument("--proposer-count", type=int, default=4)
    parser.add_argument("--max-review-rounds", type=int, default=5)
    parser.add_argument("--max-rebuttal-rounds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--command", default="codex", help="Worker CLI command for ClawTeam.")
    parser.add_argument("--backend", default="tmux", choices=("tmux", "subprocess"))
    parser.add_argument("--runtime-root", default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--template-name", help="Override the generated template name.")
    parser.add_argument(
        "--agent-command-map-file",
        help="Optional JSON file containing per-role ClawTeam command overrides.",
    )
    parser.add_argument(
        "--prompt-bundle-file",
        help="Optional JSON file containing fully materialized per-role prompt text overrides.",
    )
    parser.add_argument(
        "--template-dir",
        default=str(Path.home() / ".clawteam" / "templates"),
        help="Directory for the generated ClawTeam template file.",
    )
    parser.add_argument("--reset-state", action="store_true", help="Reset any existing state.db for this team.")
    return parser.parse_args()


def _expected_roles(*, proposer_count: int) -> set[str]:
    return {"debate-coordinator"} | {f"proposer-{index}" for index in range(1, proposer_count + 1)}


def load_agent_command_map(path: Path, *, proposer_count: int) -> dict[str, list[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Agent command map file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Agent command map file is not valid JSON: {path} ({exc})") from None

    if not isinstance(data, dict):
        raise SystemExit(f"Agent command map file must contain a JSON object: {path}")

    expected_roles = _expected_roles(proposer_count=proposer_count)
    actual_roles = set()
    normalized: dict[str, list[str]] = {}

    for role, command in data.items():
        if not isinstance(role, str) or not role:
            raise SystemExit(f"Agent command map has a non-string or empty role key in {path}")
        actual_roles.add(role)
        if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
            raise SystemExit(
                f"Agent command map role `{role}` must map to a non-empty JSON array of non-empty strings in {path}"
            )
        normalized[role] = list(command)

    missing_roles = sorted(expected_roles - actual_roles)
    extra_roles = sorted(actual_roles - expected_roles)
    if missing_roles:
        raise SystemExit(
            "Agent command map is missing required roles: " + ", ".join(missing_roles)
        )
    if extra_roles:
        raise SystemExit(
            "Agent command map contains unexpected roles: " + ", ".join(extra_roles)
        )

    return normalized


def load_prompt_bundle(path: Path, *, proposer_count: int) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Prompt bundle file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Prompt bundle file is not valid JSON: {path} ({exc})") from None

    if not isinstance(data, dict):
        raise SystemExit(f"Prompt bundle file must contain a JSON object: {path}")

    expected_roles = _expected_roles(proposer_count=proposer_count)
    actual_roles = set(data.keys())
    missing_roles = sorted(expected_roles - actual_roles)
    extra_roles = sorted(actual_roles - expected_roles)
    if missing_roles:
        raise SystemExit("Prompt bundle is missing required roles: " + ", ".join(missing_roles))
    if extra_roles:
        raise SystemExit("Prompt bundle contains unexpected roles: " + ", ".join(extra_roles))

    normalized: dict[str, str] = {}
    for role, prompt in data.items():
        if not isinstance(role, str) or not role:
            raise SystemExit(f"Prompt bundle has a non-string or empty role key in {path}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise SystemExit(f"Prompt bundle role `{role}` must map to a non-empty string in {path}")
        normalized[role] = prompt.strip()
    return normalized


def safe_session_id(*parts: str) -> str:
    raw = "-".join(part for part in parts if part)
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", raw)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    if not normalized:
        raise SystemExit("Could not build a valid OpenClaw session id.")
    return normalized


def role_session_id(*, team: str, role: str) -> str:
    if role == "debate-coordinator":
        role_suffix = "coordinator"
    elif role.startswith("proposer-"):
        role_suffix = role
    else:
        raise SystemExit(f"Unsupported DebateClaw role for session-id injection: {role}")
    return safe_session_id("debate", team, role_suffix)


def command_uses_openclaw_wrapper(command: list[str]) -> bool:
    if not command:
        return False
    candidates = command[:2]
    return any(Path(token).name in {"openclaw_debate_agent.py", "openclaw_debate_agent_session.py"} for token in candidates)


def ensure_openclaw_session_ids(agent_commands: dict[str, list[str]], *, team: str) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for role, command in agent_commands.items():
        updated = list(command)
        if command_uses_openclaw_wrapper(updated):
            desired_session_id = role_session_id(team=team, role=role)
            if "--session-id" in updated:
                index = updated.index("--session-id")
                if index + 1 >= len(updated):
                    raise SystemExit(f"Missing value after --session-id for role `{role}`.")
                updated[index + 1] = desired_session_id
            else:
                updated.extend(["--session-id", desired_session_id])
        normalized[role] = updated
    return normalized


def detect_clawteam_team_flag() -> str:
    try:
        result = subprocess.run(
            ["clawteam", "launch", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return "--team"
    help_text = (result.stdout or "") + "\n" + (result.stderr or "")
    if "--team-name" in help_text:
        return "--team-name"
    return "--team"


def main() -> int:
    args = parse_args()
    if args.proposer_count < 1:
        raise SystemExit("--proposer-count must be at least 1.")
    if args.max_review_rounds < 0 or args.max_rebuttal_rounds < 0 or args.max_epochs < 1:
        raise SystemExit("Round limits must be zero or greater, and --max-epochs must be at least 1.")

    if args.proposer_count > 6:
        print(
            "Warning: proposer counts above 6 make full cross-review much more expensive. "
            "Proceed only if the user explicitly wants the larger debate.",
            flush=True,
        )

    agent_commands = None
    if args.agent_command_map_file:
        agent_commands = ensure_openclaw_session_ids(
            load_agent_command_map(
                Path(args.agent_command_map_file).expanduser().resolve(),
                proposer_count=args.proposer_count,
            ),
            team=args.team,
        )

    prompt_bundle = None
    if args.prompt_bundle_file:
        prompt_bundle = load_prompt_bundle(
            Path(args.prompt_bundle_file).expanduser().resolve(),
            proposer_count=args.proposer_count,
        )

    template_name = args.template_name or template_name_for(args.workflow, args.team)
    template_dir = Path(args.template_dir).expanduser().resolve()
    template_dir.mkdir(parents=True, exist_ok=True)
    template_path = template_dir / f"{template_name}.toml"

    if args.workflow == "parallel-judge":
        template_text = build_parallel_judge_template(
            name=template_name,
            proposer_count=args.proposer_count,
            command=args.command,
            backend=args.backend,
            runtime_root=args.runtime_root,
            agent_commands=agent_commands,
            task_overrides=prompt_bundle,
        )
    else:
        template_text = build_review_loop_template(
            name=template_name,
            proposer_count=args.proposer_count,
            max_review_rounds=args.max_review_rounds,
            max_rebuttal_rounds=args.max_rebuttal_rounds,
            command=args.command,
            backend=args.backend,
            runtime_root=args.runtime_root,
            agent_commands=agent_commands,
            task_overrides=prompt_bundle,
        )

    template_path.write_text(template_text, encoding="utf-8")

    config = DebateConfig(
        team_name=args.team,
        workflow=args.workflow,
        goal=args.goal,
        evidence_policy=args.evidence_policy,
        proposer_count=args.proposer_count,
        max_review_rounds=args.max_review_rounds,
        max_rebuttal_rounds=args.max_rebuttal_rounds,
        max_epochs=args.max_epochs,
    )
    state_path = init_debate_state(config, reset=args.reset_state)

    print(f"template_name={template_name}")
    print(f"template_path={template_path}")
    print(f"state_path={state_path}")
    team_flag = detect_clawteam_team_flag()
    launch_command = [
        "clawteam",
        "launch",
        template_name,
        team_flag,
        args.team,
        "--goal",
        args.goal,
        "--backend",
        args.backend,
    ]
    if not agent_commands:
        launch_command.extend(["--command", args.command])
    if args.agent_command_map_file:
        print(f"agent_command_map_file={Path(args.agent_command_map_file).expanduser().resolve()}")
    print("prepare_output_version=2")
    print(f"launch_command_json={json.dumps(launch_command, ensure_ascii=False)}")
    print(f"launch_command={shlex.join(launch_command)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
