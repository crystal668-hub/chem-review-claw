#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

from bundle_common import (
    default_runtime_dir,
    dump_json,
    engine_skill_root,
    engine_script_path,
    load_json,
    openclaw_env_file,
    read_text,
    resolve_python_interpreter,
    resolve_skill_root,
)
from control_store import FileControlStore


def openclaw_config_path() -> Path:
    return (Path.home() / ".openclaw" / "openclaw.json").resolve()


def model_ref_from_definition(model_def: dict) -> str:
    return f"{model_def['provider_ref']}/{model_def['remote_model_id']}"


def apply_slot_models_to_openclaw_config(
    skill_root: Path,
    slot_assignments: dict[str, dict],
    *,
    config_path: Path,
    dry_run: bool,
) -> dict[str, object]:
    config = load_json(config_path)
    agents = config.get("agents", {}).get("list", [])
    by_id = {
        str(item.get("id", "")): item
        for item in agents
        if isinstance(item, dict) and item.get("id")
    }

    models_root = engine_skill_root(skill_root) / "control" / "models"
    changes: list[dict[str, str]] = []
    missing_slots: list[str] = []

    for slot_id, payload in slot_assignments.items():
        payload = dict(payload or {})
        model_ref = str(payload.get("model_ref") or "").strip()
        if not model_ref:
            continue
        entry = by_id.get(slot_id)
        if not entry:
            missing_slots.append(slot_id)
            continue
        model_def_path = models_root / f"{model_ref}.json"
        if not model_def_path.is_file():
            raise SystemExit(f"Missing model definition for slot {slot_id}: {model_def_path}")
        expected = model_ref_from_definition(load_json(model_def_path))
        before = str(entry.get("model") or "")
        changes.append(
            {
                "slot_id": slot_id,
                "before": before,
                "after": expected,
                "status": "unchanged" if before == expected else "changed",
            }
        )
        if not dry_run and before != expected:
            entry["model"] = expected

    if missing_slots:
        raise SystemExit("Missing fixed debate slots in openclaw.json: " + ", ".join(sorted(missing_slots)))

    changed = any(item["status"] == "changed" for item in changes)
    if changed and not dry_run:
        dump_json(config_path, config)

    return {
        "config_file": str(config_path),
        "changed": changed,
        "changes": changes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a chemqa-review run plan.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="chemqa-review skill root")
    parser.add_argument("--run-id", required=True, help="Persisted run plan id")
    parser.add_argument("--template-dir", help="Output template directory")
    parser.add_argument("--command-map-dir", help="Output command-map directory")
    parser.add_argument("--runtime-dir", help="Path to deployed DebateClaw runtime helpers")
    parser.add_argument("--openclaw-config", help="Path to the run-scoped OpenClaw config file")
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_prepare_output(stdout: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key] = value
    return parsed


def current_python() -> str:
    return resolve_python_interpreter()


def build_command_map(
    run_plan: dict,
    *,
    wrapper_path: Path,
    env_file: str,
    skill_root: Path,
    clawteam_data_dir: str,
    openclaw_config: Path,
    lease_dir: str = "",
) -> dict[str, list[str]]:
    command_map: dict[str, list[str]] = {}
    slot_assignments = dict(run_plan.get("slot_assignments") or {})
    role_slots = dict(run_plan.get("launch_spec", {}).get("role_slots") or {})
    stop_loss = dict((((run_plan.get("runtime_context") or {}).get("chemqa_review") or {}).get("stop_loss") or {}))
    driver_path = skill_root / "scripts" / "chemqa_review_openclaw_driver.py"
    python = current_python()
    for role_name, slot_id in role_slots.items():
        session_id = run_plan["session_assignments"][slot_id]
        command = [
            python,
            str(driver_path),
            "--skill-root",
            str(skill_root),
            "--team",
            str(run_plan["run_id"]),
            "--role",
            role_name,
            "--slot",
            slot_id,
            "--env-file",
            env_file,
            "--config-file",
            str(openclaw_config),
            "--session-id",
            session_id,
            "--runtime-dir",
            str(wrapper_path.parent),
            "--data-dir",
            clawteam_data_dir,
        ]
        option_map = (
            ("stale_timeout_seconds", "--stale-timeout-seconds"),
            ("respawn_cooldown_seconds", "--respawn-cooldown-seconds"),
            ("max_model_attempts", "--max-model-attempts"),
            ("lane_retry_budget", "--lane-retry-budget"),
            ("phase_repair_budget", "--phase-repair-budget"),
            ("max_respawns_per_role_phase_signature", "--max-respawns-per-role-phase-signature"),
        )
        for key, flag in option_map:
            value = stop_loss.get(key)
            if value is None:
                continue
            command.extend([flag, str(value)])
        thinking = dict(slot_assignments.get(slot_id) or {}).get("thinking")
        if thinking:
            command.extend(["--thinking", str(thinking)])
        if lease_dir:
            command.extend(["--lease-dir", lease_dir])
        command_map[role_name] = command
    return command_map


def render_role_prompt(
    root: Path,
    run_plan: dict,
    *,
    role_name: str,
    runtime_root: str,
) -> str:
    assembly = dict(run_plan["prompt_assembly"][role_name])
    semantic_role = str(assembly["semantic_role"])
    additional_workspace = run_plan.get("runtime_context", {}).get("additional_file_workspace") or "none"
    python = current_python()
    state_snapshot_cmd = (
        f"{python} {root / 'scripts' / 'chemqa_review_state_snapshot.py'} "
        f"--skill-root {root} --runtime-dir {runtime_root} --team {{team_name}} --agent {{agent_name}}"
    )
    role_intro = [
        f"You are chemqa-review role `{role_name}`.",
        f"Semantic role: `{semantic_role}`.",
        "This run uses the ChemQA fixed-lane review protocol: only `proposer-1` is the candidate owner.",
        "The other proposer slots are fixed reviewer lanes and must not invent alternate final answers.",
        "",
        "Preferred state command (compact + cached; use this instead of polling raw JSON):",
        f"- `{state_snapshot_cmd}`",
        "",
        "Fallback runtime commands (use only when the compact snapshot is insufficient):",
        f"- `{python} {runtime_root}/debate_state.py status --team {{team_name}} --agent {{agent_name}} --json`",
        f"- `{python} {runtime_root}/debate_state.py next-action --team {{team_name}} --agent {{agent_name}} --json`",
        f"- `{python} {runtime_root}/debate_state.py advance --team {{team_name}} --agent {{agent_name}}`",
        "",
        "Sibling skill roots are available under the same skills directory as this bundle.",
        f"Additional file workspace: {additional_workspace}",
    ]
    parts = ["\n".join(role_intro).strip()]
    for rel_path in assembly.get("contracts", []):
        parts.append(read_text(root / rel_path).strip())
    for rel_path in assembly.get("modules", []):
        parts.append(read_text(root / rel_path).strip())
    return "\n\n---\n\n".join(part for part in parts if part.strip())


def main() -> int:
    args = parse_args()
    root = resolve_skill_root(args.root)
    store = FileControlStore(root)
    run_plan = store.get_run_plan(args.run_id)

    runtime_root = Path(args.runtime_dir).expanduser().resolve() if args.runtime_dir else default_runtime_dir()
    wrapper_path = runtime_root / "openclaw_debate_agent.py"
    debate_state_path = runtime_root / "debate_state.py"
    if not wrapper_path.is_file() or not debate_state_path.is_file():
        raise SystemExit(
            f"Deployed runtime helpers are missing under {runtime_root}. Expected openclaw_debate_agent.py and debate_state.py."
        )

    openclaw_config = Path(
        args.openclaw_config or os.environ.get("OPENCLAW_CONFIG_PATH") or openclaw_config_path()
    ).expanduser().resolve()
    slot_apply_result = apply_slot_models_to_openclaw_config(
        root,
        dict(run_plan.get("slot_assignments") or {}),
        config_path=openclaw_config,
        dry_run=args.dry_run,
    )

    command_map_dir = Path(args.command_map_dir).resolve() if args.command_map_dir else (root / "generated" / "command-maps")
    template_dir = Path(args.template_dir).resolve() if args.template_dir else (root / "generated" / "templates")
    prompt_bundle_dir = root / "generated" / "prompt-bundles"
    runtime_context_dir = root / "generated" / "runtime-context"
    command_map_dir.mkdir(parents=True, exist_ok=True)
    template_dir.mkdir(parents=True, exist_ok=True)
    prompt_bundle_dir.mkdir(parents=True, exist_ok=True)
    runtime_context_dir.mkdir(parents=True, exist_ok=True)

    clawteam_data_dir = str(root / "generated" / "clawteam-data" / "runs" / args.run_id)
    lease_dir = str(Path(os.environ.get("BENCHMARK_CLEANROOM_LEASE_DIR") or "").expanduser().resolve()) if os.environ.get("BENCHMARK_CLEANROOM_LEASE_DIR") else ""
    command_map = build_command_map(
        run_plan,
        wrapper_path=wrapper_path,
        env_file=openclaw_env_file(),
        skill_root=root,
        clawteam_data_dir=clawteam_data_dir,
        openclaw_config=openclaw_config,
        lease_dir=lease_dir,
    )
    command_map_path = dump_json(command_map_dir / f"{args.run_id}-command-map.json", command_map)

    prompt_bundle = {
        role_name: render_role_prompt(root, run_plan, role_name=role_name, runtime_root=str(runtime_root))
        for role_name in dict(run_plan.get("launch_spec", {}).get("role_slots") or {})
    }
    prompt_bundle_path = dump_json(prompt_bundle_dir / f"{args.run_id}-prompts.json", prompt_bundle)

    runtime_context_payload = {
        "run_id": run_plan["run_id"],
        "workflow_ref": run_plan["workflow_ref"],
        "engine_workflow_ref": run_plan["engine_workflow_ref"],
        "preset_ref": run_plan["preset_ref"],
        "runtime_context": run_plan.get("runtime_context", {}),
        "resolved_model_profile": run_plan.get("resolved_model_profile", {}),
    }
    runtime_context_path = dump_json(runtime_context_dir / f"{args.run_id}-context.json", runtime_context_payload)

    prepare_script = engine_script_path(root, "prepare_debate.py")
    prepare_cmd = [
        current_python(),
        str(prepare_script),
        "--workflow",
        str(run_plan["launch_spec"].get("engine_workflow_name") or "chemqa-review"),
        "--team",
        run_plan["run_id"],
        "--goal",
        run_plan["request_snapshot"]["goal"],
        "--proposer-count",
        str(run_plan["protocol_defaults"]["proposer_count"]),
        "--backend",
        run_plan["launch_spec"]["backend"],
        "--command",
        "openclaw",
        "--runtime-root",
        str(runtime_root),
        "--template-dir",
        str(template_dir),
        "--agent-command-map-file",
        str(command_map_path),
        "--prompt-bundle-file",
        str(prompt_bundle_path),
    ]
    if run_plan["protocol_defaults"].get("review_rounds") is not None:
        prepare_cmd.extend(["--max-review-rounds", str(run_plan["protocol_defaults"]["review_rounds"])])
    if run_plan["protocol_defaults"].get("rebuttal_rounds") is not None:
        prepare_cmd.extend(["--max-rebuttal-rounds", str(run_plan["protocol_defaults"]["rebuttal_rounds"])])
    if run_plan["protocol_defaults"].get("max_epochs") is not None:
        prepare_cmd.extend(["--max-epochs", str(run_plan["protocol_defaults"]["max_epochs"])])
    if args.reset_state:
        prepare_cmd.append("--reset-state")

    payload = {
        "run_id": run_plan["run_id"],
        "workflow_ref": run_plan["engine_workflow_ref"],
        "workflow_name": "review-loop",
        "command_map_path": str(command_map_path),
        "prompt_bundle_path": str(prompt_bundle_path),
        "runtime_context_path": str(runtime_context_path),
        "clawteam_data_dir": clawteam_data_dir,
        "template_dir": str(template_dir),
        "prepare_command": prepare_cmd,
        "slot_apply": slot_apply_result,
        "openclaw_config_path": str(openclaw_config),
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(prepare_script.parent)
    env.setdefault("CLAWTEAM_DATA_DIR", clawteam_data_dir)
    env["OPENCLAW_CONFIG_PATH"] = str(openclaw_config)
    result = subprocess.run(prepare_cmd, env=env, check=False, capture_output=True, text=True)
    payload["prepare_stdout"] = result.stdout
    payload["prepare_stderr"] = result.stderr
    payload["returncode"] = result.returncode
    parsed_prepare_output = parse_prepare_output(result.stdout)
    payload.update(parsed_prepare_output)
    if parsed_prepare_output.get("launch_command_json"):
        try:
            launch_command = json.loads(parsed_prepare_output["launch_command_json"])
        except json.JSONDecodeError as exc:
            raise SystemExit(f"prepare_debate.py returned invalid launch_command_json: {exc}") from exc
        if not isinstance(launch_command, list) or not launch_command or not all(isinstance(item, str) and item for item in launch_command):
            raise SystemExit(
                "prepare_debate.py returned malformed launch_command_json; expected a non-empty JSON array of non-empty strings."
            )
        payload["launch_command"] = launch_command
    elif payload.get("launch_command"):
        payload["launch_command"] = shlex.split(payload["launch_command"])
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
