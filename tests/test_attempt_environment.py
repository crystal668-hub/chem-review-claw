from __future__ import annotations

import subprocess
from pathlib import Path

from benchmarking.runtime.attempt_environment import (
    cleanup_attempt_environment,
    create_attempt_environment,
)


def test_create_attempt_environment_uses_uv_seed_no_project(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        if command[1] == "venv":
            python = tmp_path / "scratch" / "venv" / "bin" / "python"
            pip = python.parent / "pip"
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("", encoding="utf-8")
            pip.write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    env = create_attempt_environment(
        tmp_path / "scratch",
        bootstrap_python="/bootstrap/python",
        pypi_cutoff="2026-09-03T00:00:00Z",
        uv_executable="/usr/bin/uv",
        run_subprocess=fake_run,
    )
    assert commands[0] == [
        "/usr/bin/uv",
        "venv",
        "--seed",
        "--no-project",
        "--no-python-downloads",
        "--python",
        "/bootstrap/python",
        str(tmp_path / "scratch" / "venv"),
    ]
    assert env.to_env()["UV_EXCLUDE_NEWER"] == "2026-09-03T00:00:00Z"
    assert env.to_env()["UV_PYTHON"] == str(env.python)


def test_cleanup_attempt_environment_removes_venv_and_cache(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        python = tmp_path / "scratch" / "venv" / "bin" / "python"
        pip = python.parent / "pip"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("", encoding="utf-8")
        pip.write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    env = create_attempt_environment(tmp_path / "scratch", uv_executable="/usr/bin/uv", run_subprocess=fake_run)
    marker = env.cache_dir / "marker"
    marker.write_text("x", encoding="utf-8")
    report = cleanup_attempt_environment(env)
    assert report == {"venv_removed": True, "cache_removed": True, "tool_bin_removed": True}
    assert not env.venv_dir.exists()
    assert not env.cache_dir.exists()
