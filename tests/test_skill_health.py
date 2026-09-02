from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from benchmarking.runtime import paths as runtime_paths
from benchmarking.skills.health import (
    HealthRequirement,
    check_skill_health,
    health_requirements_for_allowlist,
    summarize_skill_health,
)
from benchmarking.skills.tree import benchmark_skill_allowlist


def test_health_requirements_cover_every_allowlisted_skill() -> None:
    requirements = health_requirements_for_allowlist(benchmark_skill_allowlist())

    assert set(requirements) == set(benchmark_skill_allowlist())


def test_missing_python_module_marks_skill_unavailable() -> None:
    def fake_run(command, *, cwd=None, env=None, text=None, capture_output=None, check=None, timeout=None):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="ModuleNotFoundError: No module named 'bs4'")

    requirement = HealthRequirement(skill="paper-access", python_modules=("bs4",))
    report = check_skill_health(requirement, workspace_root=Path("/repo"), run_subprocess=fake_run)

    assert report["skill"] == "paper-access"
    assert report["available"] is False
    assert report["checks"]["python_modules"]["bs4"]["ok"] is False
    assert report["unavailable_reasons"][0]["kind"] == "missing_dependency"


def test_missing_api_key_marks_skill_unavailable(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runtime_paths, "openclaw_env", tmp_path / "missing.env")
    requirement = HealthRequirement(skill="materials-project", api_keys=("MP_API_KEY",))
    report = check_skill_health(requirement, workspace_root=Path("/repo"), env={})

    assert report["available"] is False
    assert report["checks"]["api_keys"]["MP_API_KEY"]["ok"] is False
    assert report["unavailable_reasons"][0]["kind"] == "missing_api_key"


def test_api_key_can_be_read_from_openclaw_env_file(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SU8_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setattr(runtime_paths, "openclaw_env", env_file)

    requirement = HealthRequirement(skill="test-skill", api_keys=("SU8_API_KEY",))
    report = check_skill_health(requirement, workspace_root=tmp_path, env={})

    assert report["available"] is True
    assert report["checks"]["api_keys"]["SU8_API_KEY"]["ok"] is True


def test_paper_parse_health_uses_pdf_backend_without_pdfinfo() -> None:
    requirements = health_requirements_for_allowlist(["paper-parse"])
    requirement = requirements["paper-parse"]

    assert requirement.pdf_backend_modules == (("pymupdf", "fitz"),)
    assert "pdfinfo" not in requirement.executables


def test_xtb_cli_health_requires_skill_doc_and_xtb_executable() -> None:
    requirements = health_requirements_for_allowlist(["xtb-cli"])
    requirement = requirements["xtb-cli"]

    assert requirement.data_files == ("skills/xtb-cli/SKILL.md",)
    assert requirement.executables == ("xtb",)


def test_network_timeout_is_passed_to_probe_command() -> None:
    commands: list[list[str]] = []
    timeouts: list[int] = []

    def fake_run(command, *, cwd=None, env=None, text=None, capture_output=None, check=None, timeout=None):
        commands.append(command)
        timeouts.append(timeout)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    requirement = HealthRequirement(skill="network-demo", network_urls=("https://example.test/status",))
    report = check_skill_health(
        requirement,
        workspace_root=Path("/repo"),
        run_subprocess=fake_run,
        network_timeout_seconds=10,
    )

    assert report["available"] is True
    assert commands[0][-2:] == ["https://example.test/status", "10"]
    assert timeouts == [12]
    assert report["checks"]["network"]["https://example.test/status"]["timeout_seconds"] == 10


def test_chembl_database_uses_longer_network_timeout() -> None:
    requirement = health_requirements_for_allowlist(["chembl-database"])["chembl-database"]

    assert requirement.network_probe_timeout_seconds == 10


def test_missing_data_file_marks_skill_unavailable() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        requirement = HealthRequirement(skill="demo", data_files=("skills/demo/SKILL.md",))
        report = check_skill_health(requirement, workspace_root=root)

    assert report["available"] is False
    assert report["checks"]["data_files"]["skills/demo/SKILL.md"]["ok"] is False
    assert report["unavailable_reasons"][0]["kind"] == "missing_data_file"


def test_summary_lists_available_and_unavailable_skills() -> None:
    reports = {
        "ok-skill": {"skill": "ok-skill", "available": True, "unavailable_reasons": []},
        "bad-skill": {"skill": "bad-skill", "available": False, "unavailable_reasons": [{"kind": "missing_dependency"}]},
    }

    summary = summarize_skill_health(reports)

    assert summary["available_skill_count"] == 1
    assert summary["unavailable_skill_count"] == 1
    assert summary["available_skills"] == ["ok-skill"]
    assert summary["unavailable_skills"] == ["bad-skill"]
