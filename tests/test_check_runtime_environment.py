from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills" / "chemqa-review" / "scripts" / "check_runtime.py"
SPEC = importlib.util.spec_from_file_location("chemqa_check_runtime", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPT.parent))
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_process_environment_report_reads_process_and_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MINERU_AGENT_API_URL=https://from-file.test/agent\n"
        "MINERU_API_TOKEN=file-token\n",
        encoding="utf-8",
    )

    effective, report = module.build_process_environment_report(
        process_environment={"MINERU_API_TOKEN_ENV": "MINERU_API_TOKEN"},
        env_file=env_file,
    )

    assert effective["MINERU_AGENT_API_URL"] == "https://from-file.test/agent"
    assert effective["MINERU_API_TOKEN"] == "file-token"
    assert report["variables"]["MINERU_AGENT_API_URL"] == {
        "process": False,
        "env_file": True,
        "effective": True,
        "source": "env_file",
    }
    assert report["variables"]["MINERU_API_TOKEN"]["effective"] is True
    assert "file-token" not in str(report)


def test_process_environment_takes_precedence_over_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MINERU_AGENT_API_URL=https://from-file.test/agent\n", encoding="utf-8")

    effective, report = module.build_process_environment_report(
        process_environment={"MINERU_AGENT_API_URL": "https://from-process.test/agent"},
        env_file=env_file,
    )

    assert effective["MINERU_AGENT_API_URL"] == "https://from-process.test/agent"
    assert report["variables"]["MINERU_AGENT_API_URL"]["source"] == "process"


def test_build_report_uses_effective_process_environment(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MINERU_API_TOKEN=from-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENCLAW_ENV_FILE", str(env_file))
    monkeypatch.delenv("MINERU_API_TOKEN", raising=False)

    report = module.build_report(skill_root=ROOT / "skills" / "chemqa-review", agent_choice="none", backend_choice="subprocess")

    paper_parse = report["paper_skill_runtime"]["paper-parse"]
    assert paper_parse["precision_token_configured"] is True
    assert paper_parse["pdf_backends"]["mineru_precision_api"] is True
    assert report["process_environment"]["variables"]["MINERU_API_TOKEN"]["source"] == "env_file"
