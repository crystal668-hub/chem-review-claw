#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


GIT_SOURCE = "git+https://github.com/HKUDS/ClawTeam"


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def resolve_package_spec(source: str, source_path: str | None) -> tuple[str, bool]:
    if source == "pypi":
        return "clawteam", False
    if source == "git":
        return GIT_SOURCE, False
    if source == "path":
        if not source_path:
            raise SystemExit("--source-path is required when --source path is used.")
        resolved = Path(source_path).expanduser().resolve()
        if not resolved.exists():
            raise SystemExit(f"Source path does not exist: {resolved}")
        return str(resolved), True
    raise SystemExit(f"Unsupported source: {source}")


def resolve_uv_env(tool_dir: str | None, bin_dir: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if tool_dir:
        env["UV_TOOL_DIR"] = str(Path(tool_dir).expanduser().resolve())
    if bin_dir:
        env["UV_TOOL_BIN_DIR"] = str(Path(bin_dir).expanduser().resolve())
    return env


def resolve_venv_python(venv_path: Path) -> Path:
    unix_python = venv_path / "bin" / "python"
    if unix_python.exists():
        return unix_python
    windows_python = venv_path / "Scripts" / "python.exe"
    if windows_python.exists():
        return windows_python
    raise SystemExit(f"Could not find Python inside virtual environment: {venv_path}")


def install_tool(
    package_spec: str,
    *,
    editable: bool,
    python: str | None,
    tool_dir: str | None,
    bin_dir: str | None,
) -> None:
    env = resolve_uv_env(tool_dir, bin_dir)
    command = ["uv", "tool", "install", "--reinstall"]
    if python:
        command.extend(["--python", python])
    if editable:
        command.append("--editable")
    command.append(package_spec)
    run(command, env=env)

    candidate = None
    if bin_dir:
        candidate = Path(bin_dir).expanduser().resolve() / "clawteam"
    else:
        resolved = shutil.which("clawteam", path=env.get("PATH"))
        if resolved:
            candidate = Path(resolved)

    if candidate and candidate.exists():
        run([str(candidate), "--version"], env=env)
    else:
        print("Installed clawteam, but no executable was found on PATH in this shell.")
        print("If needed, run `uv tool update-shell` or add ~/.local/bin to PATH.")


def install_venv(
    package_spec: str,
    *,
    editable: bool,
    python: str | None,
    venv_path: str,
) -> None:
    resolved_venv = Path(venv_path).expanduser().resolve()
    command = ["uv", "venv"]
    if python:
        command.extend(["--python", python])
    command.append(str(resolved_venv))
    run(command)

    venv_python = resolve_venv_python(resolved_venv)
    install_command = ["uv", "pip", "install", "--python", str(venv_python), "--reinstall"]
    if editable:
        install_command.extend(["-e", package_spec])
    else:
        install_command.append(package_spec)
    run(install_command)

    clawteam_bin = resolved_venv / "bin" / "clawteam"
    if not clawteam_bin.exists():
        clawteam_bin = resolved_venv / "Scripts" / "clawteam.exe"
    run([str(clawteam_bin), "--version"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install or refresh ClawTeam with uv.")
    parser.add_argument("--mode", choices=("tool", "venv"), default="tool")
    parser.add_argument("--source", choices=("pypi", "git", "path"), default="pypi")
    parser.add_argument("--source-path", help="Local ClawTeam checkout when --source path is used.")
    parser.add_argument("--python", help="Python version or interpreter path to hand to uv.")
    parser.add_argument("--venv-path", default=".venv", help="Target virtual environment path.")
    parser.add_argument("--tool-dir", help="Override UV_TOOL_DIR.")
    parser.add_argument("--bin-dir", help="Override UV_TOOL_BIN_DIR.")
    return parser.parse_args()


def main() -> int:
    if not shutil.which("uv"):
        raise SystemExit("uv is required but was not found on PATH.")

    args = parse_args()
    package_spec, editable = resolve_package_spec(args.source, args.source_path)

    if args.mode == "tool":
        install_tool(
            package_spec,
            editable=editable,
            python=args.python,
            tool_dir=args.tool_dir,
            bin_dir=args.bin_dir,
        )
    else:
        install_venv(
            package_spec,
            editable=editable,
            python=args.python,
            venv_path=args.venv_path,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
