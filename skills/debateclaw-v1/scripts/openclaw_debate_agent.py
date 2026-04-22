#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///

"""Compatibility wrapper between ClawTeam and OpenClaw's single-turn agent command.

When a fixed DebateClaw slot is rebound to a different explicit session id, this
wrapper drops the slot's `agent:<slot>:main` session-store entry before invoking
OpenClaw so the next turn is created as a fresh session instead of inheriting
stale main-session metadata from the prior run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openclaw_debate_common import parse_env_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one DebateClaw turn through an isolated OpenClaw agent.")
    parser.add_argument("--slot", required=True, help="OpenClaw isolated agent id.")
    parser.add_argument("--session-id", help="Explicit OpenClaw session id for per-run isolation.")
    parser.add_argument("--config-file", help="Explicit OpenClaw config path for this run.")
    parser.add_argument(
        "--env-file",
        default=str(Path.home() / ".openclaw" / ".env"),
        help="OpenClaw .env file loaded into the child process environment.",
    )
    parser.add_argument("-p", "--prompt", help="ClawTeam-compatible prompt argument.")
    parser.add_argument("-m", "--message", help="OpenClaw-compatible message argument.")
    parser.add_argument("--thinking", choices=("off", "minimal", "low", "medium", "high", "xhigh"))
    parser.add_argument("--timeout", type=int, help="Forward OpenClaw agent timeout override (seconds).")
    parser.add_argument("--json", action="store_true", help="Forward OpenClaw JSON output.")
    return parser.parse_args()


def session_store_path_for_slot(slot: str) -> Path:
    return Path.home() / ".openclaw" / "agents" / slot / "sessions" / "sessions.json"


def main_session_key_for_slot(slot: str) -> str:
    return f"agent:{slot}:main"


SLOT_SENTINEL_FILENAME = ".debateclaw-slot.json"
SLOT_SENTINEL_KIND = "debateclaw-slot-workspace"
SLOT_SENTINEL_VERSION = 1
SLOT_AGENTS_TEMPLATE_NAME = "debate-slot-AGENTS.md"
TRASH_ROOT = Path.home() / ".Trash" / "openclaw-debateclaw-slot-trash"
SLOT_WORKSPACE_PRESERVE_NAMES = {"AGENTS.md"}
DEFAULT_TRUSTED_PLUGIN_IDS = ("duckduckgo",)


def cleanroom_runtime_lease_module():
    root = Path(__file__).resolve().parents[2].parent / "benchmark-cleanroom" / "scripts" / "runtime_lease.py"
    if not root.is_file():
        return None
    spec = importlib.util.spec_from_file_location("benchmark_cleanroom_runtime_lease", root)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def write_cleanroom_lease(*, slot: str, session_id: str, status: str) -> tuple[Path, object] | tuple[None, None]:
    run_id = os.environ.get("BENCHMARK_CLEANROOM_RUN_ID", "").strip()
    lease_dir = os.environ.get("BENCHMARK_CLEANROOM_LEASE_DIR", "").strip()
    role = os.environ.get("BENCHMARK_CLEANROOM_ROLE", "").strip() or slot
    if not run_id or not lease_dir or not session_id:
        return (None, None)
    module = cleanroom_runtime_lease_module()
    if module is None:
        return (None, None)
    handle = module.open_lease(lease_dir, run_id=run_id, role=role, slot=slot, session_id=session_id)
    payload = handle.write(
        run_id=run_id,
        role=role,
        slot=slot,
        session_id=session_id,
        status=status,
        cwd=os.getcwd(),
        home=os.environ.get("HOME", ""),
        extra={"component": "openclaw_debate_agent"},
    )
    return (handle.path, handle)


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def openclaw_config_path() -> Path:
    return Path.home() / ".openclaw" / "openclaw.json"


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


def load_openclaw_config(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SystemExit(f"OpenClaw config not found while resolving DebateClaw slot workspace: {path}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"OpenClaw config is not valid JSON: {path} ({exc})") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"OpenClaw config must contain a JSON object: {path}")
    return data


def resolve_slot_workspace(slot: str, *, config_path: Path | None = None) -> Path:
    config = load_openclaw_config(config_path or openclaw_config_path())
    agents = config.get("agents", {})
    agent_list = agents.get("list", []) if isinstance(agents, dict) else []
    for entry in agent_list:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("id", "")) != slot:
            continue
        workspace = str(entry.get("workspace", "")).strip()
        if not workspace:
            raise SystemExit(f"DebateClaw slot `{slot}` is missing a workspace path in OpenClaw config.")
        return Path(workspace).expanduser().resolve()
    raise SystemExit(f"Could not find DebateClaw slot `{slot}` in OpenClaw config.")


def sentinel_path_for_workspace(workspace: Path) -> Path:
    return workspace / SLOT_SENTINEL_FILENAME


def validate_slot_sentinel(workspace: Path, *, slot: str) -> tuple[dict[str, Any], Path]:
    sentinel_path = sentinel_path_for_workspace(workspace)
    try:
        data = json.loads(sentinel_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Refusing to reset DebateClaw slot workspace without a sentinel file: {sentinel_path}. "
            "Run ensure_openclaw_debate.py before launching the slot."
        ) from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"DebateClaw slot sentinel is not valid JSON: {sentinel_path} ({exc})") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"DebateClaw slot sentinel must contain a JSON object: {sentinel_path}")
    if str(data.get("kind", "")) != SLOT_SENTINEL_KIND or int(data.get("version", 0)) != SLOT_SENTINEL_VERSION:
        raise SystemExit(f"DebateClaw slot sentinel has an unexpected kind/version: {sentinel_path}")
    if str(data.get("slot", "")) != slot:
        raise SystemExit(f"DebateClaw slot sentinel slot mismatch for `{slot}`: {sentinel_path}")
    recorded_workspace = Path(str(data.get("workspace", ""))).expanduser().resolve()
    if recorded_workspace != workspace:
        raise SystemExit(
            f"DebateClaw slot sentinel workspace mismatch for `{slot}`: recorded={recorded_workspace} actual={workspace}"
        )
    workspace_root = Path(str(data.get("workspace_root", ""))).expanduser().resolve()
    if workspace.parent != workspace_root or workspace.name != slot:
        raise SystemExit(
            "Refusing DebateClaw slot workspace reset because the workspace is not a direct child of the recorded root. "
            f"workspace={workspace} root={workspace_root} slot={slot}"
        )
    return data, workspace_root


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


def move_top_level_run_outputs_to_trash(workspace: Path, *, slot: str, requested_session_id: str) -> list[str]:
    trashed: list[str] = []
    safe_reason = requested_session_id.replace("/", "-")
    trash_dir = TRASH_ROOT / slot / f"{now_stamp()}-{safe_reason}"
    for entry in sorted(workspace.iterdir(), key=lambda item: item.name):
        if entry.name.startswith(".") or entry.name in SLOT_WORKSPACE_PRESERVE_NAMES:
            continue
        trash_dir.mkdir(parents=True, exist_ok=True)
        destination = unique_destination(trash_dir / entry.name)
        shutil.move(str(entry), str(destination))
        trashed.append(entry.name + ("/" if entry.is_dir() else ""))
    return trashed


def write_slot_sentinel(
    workspace: Path,
    *,
    slot: str,
    workspace_root: Path,
    last_session_id: str,
) -> None:
    payload = {
        "kind": SLOT_SENTINEL_KIND,
        "version": SLOT_SENTINEL_VERSION,
        "slot": slot,
        "workspace": str(workspace),
        "workspace_root": str(workspace_root),
        "last_session_id": last_session_id,
        "managed_by": "debateclaw",
    }
    atomic_write_json(sentinel_path_for_workspace(workspace), payload)


def reset_slot_workspace_if_session_id_changed(
    slot: str,
    requested_session_id: str | None,
    *,
    config_path: Path | None = None,
) -> None:
    if not requested_session_id:
        return
    workspace = resolve_slot_workspace(slot, config_path=config_path)
    sentinel, workspace_root = validate_slot_sentinel(workspace, slot=slot)
    current_session_id = str(sentinel.get("last_session_id", ""))
    if current_session_id == requested_session_id:
        return
    move_top_level_run_outputs_to_trash(workspace, slot=slot, requested_session_id=requested_session_id)
    (workspace / "AGENTS.md").write_text(load_slot_agents_template(), encoding="utf-8")
    write_slot_sentinel(
        workspace,
        slot=slot,
        workspace_root=workspace_root,
        last_session_id=requested_session_id,
    )


def load_session_store(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Session store is not valid JSON: {path} ({exc})") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Session store must contain a JSON object: {path}")
    return data


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def requested_model_for_slot(slot: str, *, config_path: Path | None = None) -> tuple[str | None, str | None]:
    config = load_openclaw_config(config_path or openclaw_config_path())
    agents = config.get("agents", {})
    agent_list = agents.get("list", []) if isinstance(agents, dict) else []
    slot_norm = slot.strip().lower()
    for entry in agent_list:
        if not isinstance(entry, dict):
            continue
        entry_id = str(entry.get("id", "")).strip()
        if not entry_id:
            continue
        if entry_id != slot and entry_id.lower() != slot_norm:
            continue
        provider = entry.get("provider")
        model = entry.get("model")
        provider_text = str(provider).strip() if provider is not None else ""
        model_text = str(model).strip() if model is not None else ""
        if not provider_text and model_text and "/" in model_text:
            provider_text, model_text = model_text.split("/", 1)
        return (provider_text or None, model_text or None)
    return (None, None)


def reset_slot_main_session_if_session_id_changed(
    slot: str,
    requested_session_id: str | None,
    *,
    config_path: Path | None = None,
) -> None:
    if not requested_session_id:
        return
    store_path = session_store_path_for_slot(slot)
    store = load_session_store(store_path)
    if not store:
        return
    session_key = main_session_key_for_slot(slot)
    current_entry = store.get(session_key)
    if not isinstance(current_entry, dict):
        return
    current_session_id = current_entry.get("sessionId")
    session_file = current_entry.get("sessionFile")
    session_file_matches_requested = False
    if isinstance(session_file, str) and session_file.strip():
        session_file_matches_requested = Path(session_file).name == f"{requested_session_id}.jsonl"
    requested_provider, requested_model = requested_model_for_slot(slot, config_path=config_path)
    current_provider = current_entry.get("modelProvider")
    current_model = current_entry.get("model")
    provider_matches_requested = requested_provider is None or current_provider == requested_provider
    model_matches_requested = requested_model is None or current_model == requested_model
    if (
        isinstance(current_session_id, str)
        and current_session_id == requested_session_id
        and session_file_matches_requested
        and provider_matches_requested
        and model_matches_requested
    ):
        return
    updated_store = dict(store)
    updated_store.pop(session_key, None)
    atomic_write_json(store_path, updated_store)


def normalize_trusted_plugin_ids(raw: str | None) -> list[str]:
    if raw is None:
        return list(DEFAULT_TRUSTED_PLUGIN_IDS)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or list(DEFAULT_TRUSTED_PLUGIN_IDS)


def create_debate_runtime_config(base_config_path: Path, *, env: dict[str, str], slot: str) -> Path | None:
    trusted_plugin_ids = normalize_trusted_plugin_ids(env.get("OPENCLAW_DEBATE_TRUSTED_PLUGINS"))
    if not trusted_plugin_ids:
        return None
    config = load_openclaw_config(base_config_path)
    plugins = dict(config.get("plugins") or {})
    entries = plugins.get("entries")
    if isinstance(entries, dict):
        plugins["entries"] = {
            str(key): value
            for key, value in entries.items()
            if str(key) in trusted_plugin_ids
        }
    plugins["allow"] = trusted_plugin_ids
    config["plugins"] = plugins
    fd, tmp_name = tempfile.mkstemp(prefix=f"openclaw-debate-{slot}-", suffix=".json")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return tmp_path


def main() -> int:
    args = parse_args()
    message = args.message or args.prompt
    if not message:
        raise SystemExit("Missing prompt/message. ClawTeam should pass the task prompt with `-p`.")

    env_file = Path(args.env_file).expanduser().resolve()
    env = os.environ.copy()
    if env_file.is_file():
        env.update(parse_env_entries(env_file))

    base_config_path = Path(args.config_file or env.get("OPENCLAW_CONFIG_PATH") or openclaw_config_path()).expanduser().resolve()

    reset_slot_workspace_if_session_id_changed(args.slot, args.session_id, config_path=base_config_path)
    reset_slot_main_session_if_session_id_changed(args.slot, args.session_id, config_path=base_config_path)

    temp_config_path = create_debate_runtime_config(base_config_path, env=env, slot=args.slot)
    if temp_config_path is not None:
        env["OPENCLAW_CONFIG_PATH"] = str(temp_config_path)

    command = [
        "openclaw",
        "agent",
        "--local",
        "--agent",
        args.slot,
    ]
    if args.session_id:
        command.extend(["--session-id", args.session_id])
    command.extend([
        "--message",
        message,
    ])
    if args.thinking:
        command.extend(["--thinking", args.thinking])
    if args.timeout is not None:
        command.extend(["--timeout", str(max(1, int(args.timeout)))])
    if args.json:
        command.append("--json")

    lease_handle = None
    _, lease_handle = write_cleanroom_lease(slot=args.slot, session_id=str(args.session_id or ""), status="starting")

    try:
        result = subprocess.run(command, env=env, check=False)
        if lease_handle is not None and args.session_id:
            lease_handle.write(
                run_id=os.environ.get("BENCHMARK_CLEANROOM_RUN_ID", ""),
                role=os.environ.get("BENCHMARK_CLEANROOM_ROLE", "").strip() or args.slot,
                slot=args.slot,
                session_id=str(args.session_id or ""),
                status="exited",
                cwd=os.getcwd(),
                home=os.environ.get("HOME", ""),
                extra={"returncode": int(result.returncode), "component": "openclaw_debate_agent"},
            )
        return int(result.returncode)
    finally:
        if lease_handle is not None:
            lease_handle.remove()
        if temp_config_path is not None:
            temp_config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
