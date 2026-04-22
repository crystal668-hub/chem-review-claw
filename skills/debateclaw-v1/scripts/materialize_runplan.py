#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from control_store import FileControlStore
from debate_templates import default_task_bundle
from openclaw_debate_common import resolve_python_interpreter


WORKFLOW_MAP = {
    "parallel@1": "parallel-judge",
    "review-loop@1": "review-loop",
}


def parse_prepare_output(stdout: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key] = value
    return parsed


def escape_template_literal(value: object) -> str:
    text = str(value)
    return text.replace("{", "{{").replace("}", "}}")


def render_additional_file_workspace_context(run_plan: dict) -> str:
    additional = run_plan.get("runtime_context", {}).get("additional_file_workspace")
    if not additional:
        return (
            "Additional file workspace context:\n\n"
            "- additional_file_workspace: none\n"
            "- No extra file workspace was attached to this run.\n"
            "- Use only the normal debate runtime state, prompts, and any other explicitly provided context."
        )

    return "\n".join(
        [
            "Additional file workspace context:",
            "",
            f"- additional_file_workspace: {escape_template_literal(additional)}",
            "- Treat this value as an opaque operator-provided locator for extra file context.",
            "- DebateClaw itself does not interpret or materialize it; use the inherited file-access capability provided by the surrounding agent/runtime environment.",
        ]
    )


def render_run_brief(run_plan: dict) -> str:
    request = run_plan["request_snapshot"]
    runtime_context = run_plan.get("runtime_context", {})
    lines = [
        "Run brief:",
        "",
        f"Goal: {escape_template_literal(request['goal'])}",
        f"Priority: {escape_template_literal(request.get('metadata', {}).get('priority', 'normal'))}",
    ]
    if runtime_context.get("evidence_mode"):
        lines.append(f"Evidence mode: {escape_template_literal(runtime_context['evidence_mode'])}")
    if runtime_context.get("final_decider"):
        lines.append(f"Final decider: {escape_template_literal(runtime_context['final_decider'])}")
    return "\n".join(lines)


def merge_task_text(base_task: str, supplemental_sections: list[str]) -> str:
    extras = [section.strip() for section in supplemental_sections if section and section.strip()]
    if not extras:
        return base_task.strip()
    return "\n\n---\n\n".join(
        [
            base_task.strip(),
            (
                "Supplemental DebateClaw V1 guidance:\n\n"
                "Treat the following as additional run-specific constraints and context layered "
                "on top of the runtime commands and operating loop above."
            ),
            *extras,
        ]
    )


def assemble_prompt_bundle(root: Path, run_plan: dict, *, workflow_name: str, runtime_root: str) -> dict[str, str]:
    proposer_count = int(run_plan["protocol_defaults"]["proposer_count"])
    base_tasks = default_task_bundle(
        workflow=workflow_name,
        proposer_count=proposer_count,
        runtime_root=runtime_root,
    )

    bundle: dict[str, str] = {}
    additional_context = render_additional_file_workspace_context(run_plan)
    run_brief = render_run_brief(run_plan)
    for role_name, assembly in run_plan.get("prompt_assembly", {}).items():
        base_task = base_tasks.get(role_name)
        if not base_task:
            raise SystemExit(f"No default DebateClaw task text is available for role `{role_name}`.")

        parts: list[str] = []
        for rel in assembly.get("contracts", []):
            parts.append((root / rel).read_text(encoding="utf-8").strip())
        for rel in assembly.get("modules", []):
            parts.append((root / rel).read_text(encoding="utf-8").strip())
        parts.append(additional_context)
        parts.append(run_brief)
        bundle[role_name] = merge_task_text(base_task, parts)
    return bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a DebateClaw V1 run plan into old-runtime assets.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="DebateClaw V1 root (repo root or installed skill root)")
    parser.add_argument("--run-id", required=True, help="Persisted run plan id")
    parser.add_argument("--template-dir", help="Output template directory")
    parser.add_argument("--command-map-dir", help="Output command-map directory")
    parser.add_argument("--runtime-dir", help="Path to deployed DebateClaw runtime helpers used by generated templates and command maps")
    parser.add_argument("--reset-state", action="store_true", help="Pass --reset-state to the copied prepare_debate.py")
    parser.add_argument("--dry-run", action="store_true", help="Render command map and print the prepare command without executing it")
    return parser.parse_args()


def role_name_for_slot(slot_id: str, proposer_slots: list[str]) -> str:
    if slot_id == "debate-coordinator":
        return "debate-coordinator"
    try:
        index = proposer_slots.index(slot_id) + 1
    except ValueError as exc:
        raise SystemExit(f"Slot `{slot_id}` is not part of proposer slot list {proposer_slots}.") from exc
    return f"proposer-{index}"


def build_command_map(run_plan: dict, *, wrapper_path: Path, env_file: str) -> dict[str, list[str]]:
    proposer_slots = run_plan["launch_spec"]["proposer_slots"]
    command_map: dict[str, list[str]] = {}
    python = resolve_python_interpreter()
    for slot_id, session_id in run_plan["session_assignments"].items():
        role_name = role_name_for_slot(slot_id, proposer_slots)
        command = [
            python,
            str(wrapper_path),
            "--slot",
            slot_id,
            "--env-file",
            env_file,
            "--session-id",
            session_id,
        ]
        thinking = run_plan["slot_assignments"].get(slot_id, {}).get("thinking")
        if thinking:
            command.extend(["--thinking", thinking])
        command_map[role_name] = command
    return command_map


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    store = FileControlStore(root)
    run_plan = store.get_run_plan(args.run_id)

    workflow_ref = run_plan["workflow_ref"]
    workflow_name = WORKFLOW_MAP.get(workflow_ref)
    if not workflow_name:
        raise SystemExit(f"Unsupported workflow ref for old runtime bridge: {workflow_ref}")

    scripts_dir = root / "scripts"
    prepare_script = scripts_dir / "prepare_debate.py"
    runtime_root = (
        Path(args.runtime_dir).expanduser().resolve()
        if args.runtime_dir
        else (Path.home() / ".clawteam" / "debateclaw" / "bin")
    )
    wrapper_path = runtime_root / "openclaw_debate_agent.py"
    debate_state_path = runtime_root / "debate_state.py"
    if not wrapper_path.is_file() or not debate_state_path.is_file():
        raise SystemExit(
            "Deployed DebateClaw runtime helpers were not found under "
            f"{runtime_root}. Run deploy_templates.py first or pass --runtime-dir explicitly."
        )

    env_file = os.environ.get("OPENCLAW_ENV_FILE", str(Path.home() / ".openclaw" / ".env"))
    command_map_dir = Path(args.command_map_dir).resolve() if args.command_map_dir else (root / "generated" / "command-maps")
    template_dir = Path(args.template_dir).resolve() if args.template_dir else (root / "generated" / "templates")
    runtime_context_dir = root / "generated" / "runtime-context"
    prompt_bundle_dir = root / "generated" / "prompt-bundles"
    command_map_dir.mkdir(parents=True, exist_ok=True)
    template_dir.mkdir(parents=True, exist_ok=True)
    runtime_context_dir.mkdir(parents=True, exist_ok=True)
    prompt_bundle_dir.mkdir(parents=True, exist_ok=True)

    command_map = build_command_map(run_plan, wrapper_path=wrapper_path, env_file=env_file)
    command_map_path = command_map_dir / f"{args.run_id}-command-map.json"
    command_map_path.write_text(json.dumps(command_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    prompt_bundle = assemble_prompt_bundle(
        root,
        run_plan,
        workflow_name=workflow_name,
        runtime_root=str(runtime_root),
    )
    prompt_bundle_path = prompt_bundle_dir / f"{args.run_id}-prompts.json"
    prompt_bundle_path.write_text(json.dumps(prompt_bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    runtime_context_payload = {
        "run_id": run_plan["run_id"],
        "workflow_ref": run_plan["workflow_ref"],
        "preset_ref": run_plan["preset_ref"],
        "runtime_context": run_plan.get("runtime_context", {}),
        "resolved_model_profile": run_plan.get("resolved_model_profile", {}),
    }
    runtime_context_path = runtime_context_dir / f"{args.run_id}-context.json"
    runtime_context_path.write_text(json.dumps(runtime_context_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    request = run_plan["request_snapshot"]
    defaults = run_plan["protocol_defaults"]
    prepare_cmd = [
        resolve_python_interpreter(),
        str(prepare_script),
        "--workflow",
        workflow_name,
        "--team",
        run_plan["run_id"],
        "--goal",
        request["goal"],
        "--proposer-count",
        str(defaults["proposer_count"]),
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
    if defaults.get("review_rounds") is not None:
        prepare_cmd.extend(["--max-review-rounds", str(defaults["review_rounds"])])
    if defaults.get("rebuttal_rounds") is not None:
        prepare_cmd.extend(["--max-rebuttal-rounds", str(defaults["rebuttal_rounds"])])
    if args.reset_state:
        prepare_cmd.append("--reset-state")

    payload = {
        "run_id": run_plan["run_id"],
        "workflow_ref": workflow_ref,
        "workflow_name": workflow_name,
        "command_map_path": str(command_map_path),
        "prompt_bundle_path": str(prompt_bundle_path),
        "runtime_context_path": str(runtime_context_path),
        "template_dir": str(template_dir),
        "prepare_command": prepare_cmd,
    }

    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(scripts_dir)
    result = subprocess.run(prepare_cmd, env=env, check=False, capture_output=True, text=True)
    payload["prepare_stdout"] = result.stdout
    payload["prepare_stderr"] = result.stderr
    payload["returncode"] = result.returncode
    payload.update(parse_prepare_output(result.stdout))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
