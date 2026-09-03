from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from benchmarking.skills.runtime import (
    WorkspaceUvSkillRunner,
    classify_skill_process_failure,
)


def test_builds_workspace_uv_python_command() -> None:
    runner = WorkspaceUvSkillRunner(workspace_root=Path("/repo"), uv_executable="/usr/bin/uv")

    command = runner.build_command(Path("/repo/skills/chem-calculator/scripts/ksp_solver.py"), ["--json"])

    assert command == [
        "/usr/bin/uv",
        "run",
        "--project",
        "/repo",
        "python",
        "/repo/skills/chem-calculator/scripts/ksp_solver.py",
        "--json",
    ]


def test_paper_parse_command_includes_optional_extra() -> None:
    runner = WorkspaceUvSkillRunner(workspace_root=Path("/repo"), uv_executable="/usr/bin/uv")

    command = runner.build_command(Path("/repo/skills/paper-parse/scripts/paper_parse.py"), ["--help"])

    assert command[:6] == ["/usr/bin/uv", "run", "--project", "/repo", "--extra", "paper-parse"]
    assert command[6:] == ["python", "/repo/skills/paper-parse/scripts/paper_parse.py", "--help"]


def test_builds_attempt_python_command() -> None:
    runner = WorkspaceUvSkillRunner(workspace_root=Path("/repo"), uv_executable="/usr/bin/uv")
    command = runner.build_command(
        Path("/repo/skills/rdkit/scripts/run.py"),
        ["--json"],
        attempt_python="/attempt/venv/bin/python",
    )
    assert command == ["/attempt/venv/bin/python", "/repo/skills/rdkit/scripts/run.py", "--json"]


def test_missing_module_stderr_is_structured_missing_dependency() -> None:
    payload = classify_skill_process_failure(
        returncode=1,
        stdout="",
        stderr="ModuleNotFoundError: No module named 'bs4'",
        command=["uv", "run", "python", "script.py"],
    )

    assert payload["available"] is False
    assert payload["error_kind"] == "missing_dependency"
    assert payload["reason"] == "missing Python module: bs4"


def test_http_provider_error_is_structured_provider_failure() -> None:
    payload = classify_skill_process_failure(
        returncode=1,
        stdout="",
        stderr="requests.exceptions.HTTPError: 403 Client Error: Forbidden",
        command=["uv", "run", "python", "script.py"],
    )

    assert payload["available"] is False
    assert payload["error_kind"] == "provider_failure"
    assert "403" in payload["reason"]


def test_runner_returns_structured_invalid_output_payload() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        execution_cwd = root / "agent-workspace"
        execution_cwd.mkdir()
        script = root / "script.py"
        script.write_text("print('not json')\n", encoding="utf-8")
        observed: dict[str, object] = {}

        def fake_run(command, *, cwd=None, env=None, text=None, capture_output=None, check=None, timeout=None):
            observed["cwd"] = cwd
            return subprocess.CompletedProcess(command, 0, stdout="not json\n", stderr="")

        runner = WorkspaceUvSkillRunner(
            workspace_root=root,
            execution_cwd=execution_cwd,
            uv_executable="uv",
            run_subprocess=fake_run,
        )
        payload = runner.run_script(script, [])

    assert payload["available"] is False
    assert payload["error_kind"] == "invalid_output"
    assert payload["command"][:4] == ["uv", "run", "--project", str(root)]
    assert observed["cwd"] == str(execution_cwd)


def test_run_skill_cli_prints_structured_missing_dependency_payload() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "pyproject.toml").write_text("[project]\nname = 'demo-root'\nversion = '0.0.0'\n", encoding="utf-8")
        (root / "uv.lock").write_text("version = 1\nrevision = 1\nrequires-python = '>=3.11'\n", encoding="utf-8")
        (root / "benchmarking").mkdir()
        script = root / "skills" / "demo" / "scripts" / "needs_missing.py"
        script.parent.mkdir(parents=True)
        script.write_text("raise ModuleNotFoundError(\"No module named 'missing_demo'\")\n", encoding="utf-8")

        runner_script = Path(__file__).resolve().parents[1] / "scripts" / "run_skill.py"
        execution_cwd = root / "agent-workspace"
        execution_cwd.mkdir()
        completed = subprocess.run(
            [
                "uv",
                "run",
                "python",
                str(runner_script),
                "--workspace-root",
                str(root),
                "--execution-cwd",
                str(execution_cwd),
                "--script",
                str(script),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    payload = json.loads(completed.stdout)
    assert completed.returncode == 2
    assert payload["available"] is False
    assert payload["error_kind"] == "missing_dependency"
    assert payload["reason"] == "missing Python module: missing_demo"


def test_run_skill_cli_rejects_non_project_workspace_root() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        script = root / "skills" / "demo" / "scripts" / "noop.py"
        script.parent.mkdir(parents=True)
        script.write_text("print('{}')\n", encoding="utf-8")

        runner_script = Path(__file__).resolve().parents[1] / "scripts" / "run_skill.py"
        completed = subprocess.run(
            ["uv", "run", "python", str(runner_script), "--workspace-root", str(root), "--script", str(script)],
            text=True,
            capture_output=True,
            check=False,
        )

    payload = json.loads(completed.stdout)
    assert completed.returncode == 2
    assert payload["available"] is False
    assert payload["error_kind"] == "invalid_workspace_root"
    assert "pyproject.toml" in payload["reason"]
    assert "uv.lock" in payload["reason"]
