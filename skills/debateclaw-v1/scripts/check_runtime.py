#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


SUPPORTED_AGENTS = ("codex", "claude", "openclaw")
DEFAULT_TEMPLATES = ("debate-parallel-judge", "debate-review-loop")
RUNTIME_HELPERS = (
    "prepare_debate.py",
    "debate_state.py",
    "debate_templates.py",
    "inspect_openclaw_env.py",
    "openclaw_debate_common.py",
    "ensure_openclaw_debate.py",
    "openclaw_debate_agent.py",
)


def run_capture(command: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return 1, str(exc)

    output = (result.stdout or result.stderr or "").strip()
    return result.returncode, output


def probe_binary(name: str, version_command: list[str]) -> dict[str, object]:
    resolved = shutil.which(name)
    if not resolved:
        return {
            "name": name,
            "found": False,
            "path": "",
            "version": "",
        }

    code, output = run_capture(version_command)
    return {
        "name": name,
        "found": code == 0,
        "path": resolved,
        "version": output,
    }


def detect_agent(choice: str) -> str:
    if choice != "auto":
        return choice

    for candidate in SUPPORTED_AGENTS:
        if shutil.which(candidate):
            return candidate
    return "none"


def collect_debate_templates(template_dir: Path) -> list[str]:
    if not template_dir.is_dir():
        return []
    return sorted(path.stem for path in template_dir.glob("debate-*.toml"))


def collect_runtime_helpers(runtime_dir: Path) -> dict[str, bool]:
    return {
        helper: (runtime_dir / helper).is_file()
        for helper in RUNTIME_HELPERS
    }


def resolve_backend(agent_choice: str, backend_choice: str) -> str:
    if backend_choice != "auto":
        return backend_choice
    selected_agent = detect_agent(agent_choice)
    return "subprocess" if selected_agent == "openclaw" else "tmux"


def build_report(
    agent_choice: str,
    backend_choice: str,
    require_clawteam: bool,
    require_runtime_assets: bool,
) -> dict[str, object]:
    selected_agent = detect_agent(agent_choice)
    selected_backend = resolve_backend(agent_choice, backend_choice)
    template_dir = (Path.home() / ".clawteam" / "templates").resolve()
    runtime_dir = (Path.home() / ".clawteam" / "debateclaw" / "bin").resolve()
    local_bin = str(Path.home() / ".local" / "bin")
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    deployed_templates = collect_debate_templates(template_dir)
    runtime_helpers = collect_runtime_helpers(runtime_dir)

    report: dict[str, object] = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "path_contains_local_bin": local_bin in path_entries,
        "templates_dir": str(template_dir),
        "runtime_dir": str(runtime_dir),
        "deployed_debate_templates": deployed_templates,
        "default_templates_present": [name for name in DEFAULT_TEMPLATES if name in deployed_templates],
        "runtime_helpers": runtime_helpers,
        "selected_agent": selected_agent,
        "selected_backend": selected_backend,
        "checks": {
            "uv": probe_binary("uv", ["uv", "--version"]),
            "tmux": probe_binary("tmux", ["tmux", "-V"]),
            "clawteam": probe_binary("clawteam", ["clawteam", "--version"]),
            "codex": probe_binary("codex", ["codex", "--version"]),
            "claude": probe_binary("claude", ["claude", "--version"]),
            "openclaw": probe_binary("openclaw", ["openclaw", "--version"]),
        },
    }

    required = ["uv"]
    if selected_backend == "tmux":
        required.append("tmux")
    if require_clawteam:
        required.append("clawteam")
    if selected_agent != "none":
        required.append(selected_agent)

    missing = [name for name in required if not report["checks"][name]["found"]]
    if require_runtime_assets:
        if any(name not in deployed_templates for name in DEFAULT_TEMPLATES):
            missing.append("debateclaw-templates")
        if not all(runtime_helpers.values()):
            missing.append("debateclaw-runtime")
    report["required_components"] = required
    report["missing_required_components"] = missing
    report["ready"] = not missing
    return report


def render_text(report: dict[str, object]) -> str:
    platform_info = report["platform"]
    lines = [
        f"Platform: {platform_info['system']} {platform_info['release']} ({platform_info['machine']})",
        f"Selected agent: {report['selected_agent']}",
        f"Selected backend: {report['selected_backend']}",
        f"PATH contains ~/.local/bin: {'yes' if report['path_contains_local_bin'] else 'no'}",
        f"Templates dir: {report['templates_dir']}",
    ]

    deployed = report["deployed_debate_templates"]
    if deployed:
        lines.append("DebateClaw templates: " + ", ".join(deployed))
    else:
        lines.append("DebateClaw templates: none")
    helpers = report["runtime_helpers"]
    helper_text = ", ".join(f"{name}={'yes' if present else 'no'}" for name, present in helpers.items())
    lines.append("Runtime helpers: " + helper_text)

    lines.append("")
    lines.append("Checks:")

    checks = report["checks"]
    for name in ("uv", "tmux", "clawteam", "codex", "claude", "openclaw"):
        item = checks[name]
        status = "ok" if item["found"] else "missing"
        detail = item["version"] or item["path"] or "not found"
        lines.append(f"- {name}: {status} | {detail}")

    lines.append("")
    if report["ready"]:
        lines.append("Result: ready")
    else:
        missing = ", ".join(report["missing_required_components"])
        lines.append(f"Result: missing required components -> {missing}")

    if not report["path_contains_local_bin"]:
        lines.append("Note: uv tool installs usually expose binaries via ~/.local/bin.")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether DebateClaw prerequisites are ready.")
    parser.add_argument(
        "--agent",
        choices=("auto", "codex", "claude", "openclaw", "none"),
        default="auto",
        help="Agent CLI that ClawTeam should use.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "tmux", "subprocess"),
        default="auto",
        help="ClawTeam backend to use. Auto selects subprocess for OpenClaw and tmux otherwise.",
    )
    parser.add_argument(
        "--require-clawteam",
        action="store_true",
        help="Treat clawteam as a required component.",
    )
    parser.add_argument(
        "--require-runtime-assets",
        action="store_true",
        help="Require the deployed DebateClaw runtime helpers and default templates.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.agent, args.backend, args.require_clawteam, args.require_runtime_assets)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))

    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
