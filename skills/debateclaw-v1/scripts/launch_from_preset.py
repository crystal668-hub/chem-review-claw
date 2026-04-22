#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from openclaw_debate_common import resolve_python_interpreter


DEFAULT_CLAWTEAM_TEMPLATE_DIR = Path.home() / ".clawteam" / "templates"


def run_json(command: list[str], *, cwd: Path) -> dict[str, Any]:
    result = subprocess.run(command, cwd=str(cwd), check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"Command failed ({result.returncode}): {shlex.join(command)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Command did not return JSON: {shlex.join(command)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        ) from exc


def detect_clawteam_team_flag(*, cwd: Path) -> str:
    result = subprocess.run(
        ["clawteam", "launch", "--help"],
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    help_text = (result.stdout or "") + "\n" + (result.stderr or "")
    if "--team-name" in help_text:
        return "--team-name"
    return "--team"


def effective_template_dir(*, explicit: str | None, launch_mode: str) -> str | None:
    if explicit:
        return explicit
    if launch_mode == "run":
        return str(DEFAULT_CLAWTEAM_TEMPLATE_DIR)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified DebateClaw V1 entrypoint from preset to launch-ready assets.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="DebateClaw V1 root (repo root or installed skill root)")
    parser.add_argument("--preset", required=True, help="Preset ref, e.g. parallel@1 or review-loop@1")
    parser.add_argument("--goal", required=True, help="Run goal or motion")
    parser.add_argument("--run-id", help="Optional explicit run id")
    parser.add_argument("--additional-file-workspace", help="Optional run-scoped opaque string for extra file context")
    parser.add_argument("--model-profile", help="Override model profile")
    parser.add_argument("--proposer-count", type=int)
    parser.add_argument("--review-rounds", type=int)
    parser.add_argument("--rebuttal-rounds", type=int)
    parser.add_argument("--priority", default="normal")
    parser.add_argument("--reset-state", action="store_true", help="Reset protocol state while materializing old-runtime assets")
    parser.add_argument(
        "--launch-mode",
        choices=("none", "print", "run"),
        default="print",
        help="none=compile+materialize only, print=also print launch command, run=invoke clawteam launch",
    )
    parser.add_argument("--template-dir", help="Optional template output directory for materialization")
    parser.add_argument("--command-map-dir", help="Optional command-map output directory for materialization")
    parser.add_argument("--runtime-dir", help="Optional deployed DebateClaw runtime helper directory")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary (default behavior)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    scripts_dir = root / "scripts"

    compile_cmd = [
        resolve_python_interpreter(),
        str(scripts_dir / "compile_runplan.py"),
        "--root",
        str(root),
        "--preset",
        args.preset,
        "--goal",
        args.goal,
        "--priority",
        args.priority,
    ]
    if args.run_id:
        compile_cmd.extend(["--run-id", args.run_id])
    if args.additional_file_workspace:
        compile_cmd.extend(["--additional-file-workspace", args.additional_file_workspace])
    if args.model_profile:
        compile_cmd.extend(["--model-profile", args.model_profile])
    if args.proposer_count is not None:
        compile_cmd.extend(["--proposer-count", str(args.proposer_count)])
    if args.review_rounds is not None:
        compile_cmd.extend(["--review-rounds", str(args.review_rounds)])
    if args.rebuttal_rounds is not None:
        compile_cmd.extend(["--rebuttal-rounds", str(args.rebuttal_rounds)])

    compiled = run_json(compile_cmd, cwd=root)
    run_id = compiled["run_id"]

    materialize_cmd = [
        resolve_python_interpreter(),
        str(scripts_dir / "materialize_runplan.py"),
        "--root",
        str(root),
        "--run-id",
        run_id,
    ]
    resolved_template_dir = effective_template_dir(explicit=args.template_dir, launch_mode=args.launch_mode)
    if resolved_template_dir:
        materialize_cmd.extend(["--template-dir", resolved_template_dir])
    if args.command_map_dir:
        materialize_cmd.extend(["--command-map-dir", args.command_map_dir])
    if args.runtime_dir:
        materialize_cmd.extend(["--runtime-dir", args.runtime_dir])
    if args.reset_state:
        materialize_cmd.append("--reset-state")

    materialized = run_json(materialize_cmd, cwd=root)

    team_flag = detect_clawteam_team_flag(cwd=root)
    template_name = materialized.get("template_name")
    template_path = materialized.get("template_path")
    launch_command = None

    if materialized.get("launch_command"):
        launch_command = shlex.split(materialized["launch_command"])
    elif template_name:
        launch_command = [
            "clawteam",
            "launch",
            template_name,
            team_flag,
            run_id,
            "--goal",
            args.goal,
            "--backend",
            compiled["launch_spec"]["backend"],
        ]

    launched = None
    if args.launch_mode == "run":
        if not launch_command:
            raise SystemExit("No launch command is available after materialization.")
        launch_result = subprocess.run(launch_command, cwd=str(root), check=False, capture_output=True, text=True)
        launched = {
            "command": launch_command,
            "returncode": launch_result.returncode,
            "stdout": launch_result.stdout,
            "stderr": launch_result.stderr,
        }
        if launch_result.returncode != 0:
            raise SystemExit(
                f"Launch failed ({launch_result.returncode}): {shlex.join(launch_command)}\n\nSTDOUT:\n{launch_result.stdout}\n\nSTDERR:\n{launch_result.stderr}"
            )

    payload = {
        "preset": args.preset,
        "goal": args.goal,
        "run_id": run_id,
        "compile": compiled,
        "materialize": materialized,
        "template_name": template_name,
        "template_path": template_path,
        "resolved_template_dir": resolved_template_dir,
        "launch_mode": args.launch_mode,
        "launch_command": launch_command,
        "launched": launched,
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
