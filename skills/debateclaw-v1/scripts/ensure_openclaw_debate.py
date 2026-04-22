#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///

"""Provision DebateClaw OpenClaw slots and per-role command mappings."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openclaw_debate_common import (
    DEFAULT_FAMILY_SEQUENCE,
    FAMILY_SPECS,
    build_provider_config,
    choose_variable_name,
    classify_names,
    default_family_assignment,
    dump_json_file,
    load_json_file,
    model_ref_for,
    parse_env_entries,
    parse_env_names,
    provider_id_for,
)


DEFAULT_RUNTIME_ROOT = Path.home() / ".clawteam" / "debateclaw" / "bin"
SLOT_SENTINEL_FILENAME = ".debateclaw-slot.json"
SLOT_SENTINEL_KIND = "debateclaw-slot-workspace"
SLOT_SENTINEL_VERSION = 1
SLOT_AGENTS_TEMPLATE_NAME = "debate-slot-AGENTS.md"
TRASH_ROOT = Path.home() / ".Trash" / "openclaw-debateclaw-slot-trash"


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slot_agents_template_path() -> Path:
    return Path(__file__).resolve().parent / "templates" / SLOT_AGENTS_TEMPLATE_NAME


def load_slot_agents_template() -> str:
    path = slot_agents_template_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SystemExit(f"DebateClaw slot AGENTS template not found: {path}") from exc
    if not text.strip():
        raise SystemExit(f"DebateClaw slot AGENTS template is empty: {path}")
    return text.rstrip() + "\n"


def validate_slot_workspace_path(workspace: Path, *, slot_id: str, workspace_root: Path) -> Path:
    resolved_workspace = workspace.expanduser().resolve()
    resolved_root = workspace_root.expanduser().resolve()
    if resolved_workspace.name != slot_id:
        raise SystemExit(
            f"Refusing DebateClaw slot workspace operation: workspace name `{resolved_workspace.name}` does not match slot `{slot_id}`."
        )
    if resolved_workspace.parent != resolved_root:
        raise SystemExit(
            "Refusing DebateClaw slot workspace operation: workspace is not a direct child of the configured workspace root. "
            f"workspace={resolved_workspace} root={resolved_root}"
        )
    return resolved_workspace


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def move_top_level_markdown_to_trash(workspace: Path, *, slot_id: str, reason: str) -> list[str]:
    trashed: list[str] = []
    trash_dir = TRASH_ROOT / slot_id / f"{now_stamp()}-{reason}"
    for entry in sorted(workspace.iterdir(), key=lambda item: item.name):
        if not entry.is_file() or entry.suffix.lower() != ".md":
            continue
        trash_dir.mkdir(parents=True, exist_ok=True)
        destination = unique_destination(trash_dir / entry.name)
        shutil.move(str(entry), str(destination))
        trashed.append(entry.name)
    return trashed


def sentinel_path_for_workspace(workspace: Path) -> Path:
    return workspace / SLOT_SENTINEL_FILENAME


def read_existing_last_session_id(workspace: Path) -> str:
    path = sentinel_path_for_workspace(workspace)
    if not path.is_file():
        return ""
    data = load_json_file(path)
    last_session_id = data.get("last_session_id", "")
    return str(last_session_id) if isinstance(last_session_id, str) else ""


def write_slot_sentinel(workspace: Path, *, slot_id: str, workspace_root: Path, last_session_id: str) -> None:
    payload = {
        "kind": SLOT_SENTINEL_KIND,
        "version": SLOT_SENTINEL_VERSION,
        "slot": slot_id,
        "workspace": str(workspace),
        "workspace_root": str(workspace_root.expanduser().resolve()),
        "last_session_id": last_session_id,
        "managed_by": "debateclaw",
    }
    dump_json_file(sentinel_path_for_workspace(workspace), payload)


def prepare_slot_workspace(
    workspace: Path,
    *,
    slot_id: str,
    workspace_root: Path,
    reset_markdown: bool,
    last_session_id: str | None = None,
) -> dict[str, Any]:
    resolved_workspace = validate_slot_workspace_path(workspace, slot_id=slot_id, workspace_root=workspace_root)
    resolved_workspace.mkdir(parents=True, exist_ok=True)
    preserved_last_session_id = read_existing_last_session_id(resolved_workspace)
    effective_last_session_id = last_session_id if last_session_id is not None else preserved_last_session_id
    trashed_files = move_top_level_markdown_to_trash(
        resolved_workspace,
        slot_id=slot_id,
        reason="slot-reset",
    ) if reset_markdown else []
    (resolved_workspace / "AGENTS.md").write_text(load_slot_agents_template(), encoding="utf-8")
    write_slot_sentinel(
        resolved_workspace,
        slot_id=slot_id,
        workspace_root=workspace_root,
        last_session_id=effective_last_session_id,
    )
    return {
        "workspace": str(resolved_workspace),
        "trashed_files": trashed_files,
        "last_session_id": effective_last_session_id,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensure OpenClaw slots and model mappings for DebateClaw.")
    parser.add_argument("--proposer-count", type=int, required=True, help="Number of proposer agents in the debate.")
    parser.add_argument(
        "--family",
        action="append",
        choices=tuple(FAMILY_SPECS.keys()),
        help=(
            "Model family for proposer slots. Repeat to define a sequence; if fewer than proposer-count, "
            "the sequence repeats."
        ),
    )
    parser.add_argument(
        "--coordinator-family",
        choices=tuple(FAMILY_SPECS.keys()),
        default=DEFAULT_FAMILY_SEQUENCE[0],
        help="Model family for the internal DebateClaw coordinator.",
    )
    parser.add_argument(
        "--env-file",
        default=str(Path.home() / ".openclaw" / ".env"),
        help="OpenClaw .env file. Values are loaded internally but never printed.",
    )
    parser.add_argument(
        "--config-file",
        default=str(Path.home() / ".openclaw" / "openclaw.json"),
        help="OpenClaw config file to update.",
    )
    parser.add_argument(
        "--workspace-root",
        default=str(Path.home() / ".openclaw" / "debateclaw" / "workspaces"),
        help="Workspace root for DebateClaw-managed isolated agents.",
    )
    parser.add_argument("--slot-prefix", default="debate", help="Prefix for proposer slot ids.")
    parser.add_argument(
        "--coordinator-slot",
        default="debate-coordinator",
        help="OpenClaw isolated agent id for the internal coordinator.",
    )
    parser.add_argument(
        "--wrapper-path",
        default=str(DEFAULT_RUNTIME_ROOT / "openclaw_debate_agent.py"),
        help="Path to the DebateClaw OpenClaw wrapper command.",
    )
    parser.add_argument(
        "--command-map-file",
        help="Optional path to write the per-role base command map JSON for prepare_debate.py.",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON.")
    return parser.parse_args()


def expand_family_sequence(values: list[str] | None, proposer_count: int) -> list[str]:
    if proposer_count < 1:
        raise SystemExit("--proposer-count must be at least 1.")
    if not values:
        return default_family_assignment(proposer_count)
    sequence = list(values)
    return [sequence[index % len(sequence)] for index in range(proposer_count)]


def ensure_base_config(config_file: Path) -> dict[str, Any]:
    if not config_file.is_file():
        raise SystemExit(
            f"OpenClaw config file not found: {config_file}. Run `openclaw configure` or `openclaw onboard` first."
        )
    data = load_json_file(config_file)
    if not data:
        raise SystemExit(f"OpenClaw config file is empty or invalid JSON: {config_file}")

    models_cfg = data.setdefault("models", {})
    if not isinstance(models_cfg, dict):
        raise SystemExit(f"`models` is not an object in {config_file}")
    models_cfg.setdefault("mode", "merge")
    providers = models_cfg.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise SystemExit(f"`models.providers` is not an object in {config_file}")

    agents_cfg = data.setdefault("agents", {})
    if not isinstance(agents_cfg, dict):
        raise SystemExit(f"`agents` is not an object in {config_file}")
    defaults_cfg = agents_cfg.setdefault("defaults", {})
    if not isinstance(defaults_cfg, dict):
        raise SystemExit(f"`agents.defaults` is not an object in {config_file}")
    model_defaults = defaults_cfg.setdefault("model", {})
    if not isinstance(model_defaults, dict):
        raise SystemExit(f"`agents.defaults.model` is not an object in {config_file}")
    agent_list = agents_cfg.setdefault("list", [])
    if not isinstance(agent_list, list):
        raise SystemExit(f"`agents.list` is not an array in {config_file}")
    return data


def resolve_family_bindings(env_file: Path, required_families: list[str]) -> dict[str, dict[str, str]]:
    env_entries = parse_env_entries(env_file)
    env_names = parse_env_names(env_file)
    if not env_entries:
        raise SystemExit(
            f"OpenClaw .env file not found or has no parseable variable names: {env_file}. "
            "Create the required provider env vars first."
        )

    resolved: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    for family in required_families:
        spec = FAMILY_SPECS[family]
        report = classify_names(env_names, spec)
        try:
            api_key_name = choose_variable_name(report, key="api_key")
            base_url_name = choose_variable_name(report, key="base_url")
        except ValueError as exc:
            errors.append(str(exc))
            continue

        base_url = env_entries.get(base_url_name, "").strip()
        if not base_url:
            errors.append(
                f"{family} resolved base URL variable `{base_url_name}` but its value is empty in {env_file}."
            )
            continue

        resolved[family] = {
            "api_key_name": api_key_name,
            "base_url_name": base_url_name,
            "base_url": base_url,
        }

    if errors:
        raise SystemExit("\n".join(errors))
    return resolved


def upsert_provider_configs(config: dict[str, Any], family_bindings: dict[str, dict[str, str]]) -> None:
    providers: dict[str, Any] = config["models"]["providers"]
    for family, binding in family_bindings.items():
        spec = FAMILY_SPECS[family]
        providers[provider_id_for(spec)] = build_provider_config(
            spec,
            api_key_name=binding["api_key_name"],
            base_url=binding["base_url"],
        )


def agent_entries_by_id(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = config["agents"]["list"]
    return {
        str(item.get("id", "")): item
        for item in entries
        if isinstance(item, dict) and item.get("id")
    }


def run_openclaw(command: list[str]) -> None:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode == 0:
        return
    output = (result.stderr or result.stdout or "").strip()
    raise SystemExit(output or f"OpenClaw command failed: {' '.join(command)}")


def ensure_slot_exists(slot_id: str, *, workspace: Path, model_ref: str) -> None:
    command = [
        "openclaw",
        "agents",
        "add",
        slot_id,
        "--non-interactive",
        "--workspace",
        str(workspace),
        "--model",
        model_ref,
        "--json",
    ]
    run_openclaw(command)


def ensure_slots(
    config_file: Path,
    *,
    env_file: Path,
    workspace_root: Path,
    proposer_families: list[str],
    coordinator_family: str,
    slot_prefix: str,
    coordinator_slot: str,
) -> dict[str, Any]:
    config = ensure_base_config(config_file)
    required_families = sorted(set(proposer_families + [coordinator_family]))
    family_bindings = resolve_family_bindings(env_file, required_families)
    upsert_provider_configs(config, family_bindings)
    if not config["agents"]["defaults"]["model"].get("primary"):
        config["agents"]["defaults"]["model"]["primary"] = model_ref_for(FAMILY_SPECS[coordinator_family])
    dump_json_file(config_file, config)

    slots = [
        {
            "role": "debate-coordinator",
            "slot": coordinator_slot,
            "family": coordinator_family,
            "workspace": str((workspace_root / coordinator_slot).expanduser().resolve()),
        }
    ]
    for index, family in enumerate(proposer_families, start=1):
        slot_id = f"{slot_prefix}-{index}"
        slots.append(
            {
                "role": f"proposer-{index}",
                "slot": slot_id,
                "family": family,
                "workspace": str((workspace_root / slot_id).expanduser().resolve()),
            }
        )

    current = agent_entries_by_id(ensure_base_config(config_file))
    created_slots: list[str] = []
    for slot in slots:
        workspace = Path(slot["workspace"])
        workspace.mkdir(parents=True, exist_ok=True)
        model_ref = model_ref_for(FAMILY_SPECS[slot["family"]])
        slot["model"] = model_ref
        slot["provider_id"] = provider_id_for(FAMILY_SPECS[slot["family"]])
        binding = family_bindings[slot["family"]]
        slot["api_key_name"] = binding["api_key_name"]
        slot["base_url_name"] = binding["base_url_name"]
        if slot["slot"] not in current:
            ensure_slot_exists(slot["slot"], workspace=workspace, model_ref=model_ref)
            created_slots.append(slot["slot"])
        slot["workspace_prep"] = prepare_slot_workspace(
            workspace,
            slot_id=str(slot["slot"]),
            workspace_root=workspace_root,
            reset_markdown=False,
        )

    config = ensure_base_config(config_file)
    current = agent_entries_by_id(config)
    for slot in slots:
        entry = current.get(slot["slot"])
        if not entry:
            continue
        entry["model"] = slot["model"]
        entry["workspace"] = slot["workspace"]
        entry.setdefault("name", slot["slot"])
    dump_json_file(config_file, config)

    return {
        "created_slots": created_slots,
        "family_bindings": {
            family: {
                "api_key_name": binding["api_key_name"],
                "base_url_name": binding["base_url_name"],
                "provider_id": provider_id_for(FAMILY_SPECS[family]),
                "model_ref": model_ref_for(FAMILY_SPECS[family]),
            }
            for family, binding in family_bindings.items()
        },
        "slots": slots,
    }


def build_command_map(
    slots: list[dict[str, Any]],
    *,
    wrapper_path: Path,
    env_file: Path,
) -> dict[str, list[str]]:
    return {
        str(slot["role"]): [
            str(wrapper_path),
            "--slot",
            str(slot["slot"]),
            "--env-file",
            str(env_file),
        ]
        for slot in slots
    }


def render_text(payload: dict[str, Any]) -> str:
    lines = [
        f"Config file: {payload['config_file']}",
        f"Env file: {payload['env_file']}",
        f"Workspace root: {payload['workspace_root']}",
        f"Wrapper path: {payload['wrapper_path']}",
        "",
        "Resolved families:",
    ]
    for family, binding in sorted(payload["family_bindings"].items()):
        lines.append(
            f"- {family}: model={binding['model_ref']} | api key name={binding['api_key_name']} | "
            f"base url name={binding['base_url_name']}"
        )
    lines.append("")
    lines.append("Role assignments:")
    for slot in payload["slots"]:
        lines.append(
            f"- {slot['role']} -> {slot['slot']} | family={slot['family']} | model={slot['model']}"
        )
    if payload["created_slots"]:
        lines.append("")
        lines.append("Created slots: " + ", ".join(payload["created_slots"]))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    proposer_families = expand_family_sequence(args.family, args.proposer_count)
    env_file = Path(args.env_file).expanduser().resolve()
    config_file = Path(args.config_file).expanduser().resolve()
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    wrapper_path = Path(args.wrapper_path).expanduser().resolve()

    slot_payload = ensure_slots(
        config_file,
        env_file=env_file,
        workspace_root=workspace_root,
        proposer_families=proposer_families,
        coordinator_family=args.coordinator_family,
        slot_prefix=args.slot_prefix,
        coordinator_slot=args.coordinator_slot,
    )
    command_map = build_command_map(
        slot_payload["slots"],
        wrapper_path=wrapper_path,
        env_file=env_file,
    )

    payload = {
        "config_file": str(config_file),
        "env_file": str(env_file),
        "workspace_root": str(workspace_root),
        "wrapper_path": str(wrapper_path),
        "proposer_count": args.proposer_count,
        "proposer_families": proposer_families,
        "coordinator_family": args.coordinator_family,
        "created_slots": slot_payload["created_slots"],
        "family_bindings": slot_payload["family_bindings"],
        "slots": slot_payload["slots"],
        "command_map": command_map,
    }

    if args.command_map_file:
        command_map_file = Path(args.command_map_file).expanduser().resolve()
        dump_json_file(command_map_file, command_map)
        payload["command_map_file"] = str(command_map_file)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
