from __future__ import annotations

from pathlib import Path

from benchmarking.runtime.openclaw_env import (
    build_openclaw_subprocess_env,
    parse_scutil_proxy_output,
    proxy_environment_report,
)


def test_build_openclaw_subprocess_env_prefixes_workspace_venv_bin() -> None:
    workspace_root = Path(__file__).resolve().parents[1]
    env = build_openclaw_subprocess_env(
        base_env={"PATH": "/usr/bin:/bin"},
        system_proxy_text="",
    )

    venv_bin = str(workspace_root / ".venv" / "bin")
    assert env["PATH"].split(":")[:1] == [venv_bin]
    assert env["VIRTUAL_ENV"] == str(workspace_root / ".venv")
    assert env["PYTHONNOUSERSITE"] == "1"


def test_build_openclaw_subprocess_env_uses_attempt_python_and_clears_overrides(tmp_path) -> None:
    venv_dir = tmp_path / "venv"
    bin_dir = venv_dir / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text("", encoding="utf-8")
    env = build_openclaw_subprocess_env(
        base_env={
            "PATH": "/usr/bin",
            "VIRTUAL_ENV": "/old/venv",
            "PYTHONPATH": "/old/pythonpath",
            "UV_CACHE_DIR": "/old/cache",
            "API_TOKEN": "keep",
        },
        attempt_python=python,
    )
    assert env["PATH"].split(":")[0] == str(bin_dir)
    assert env["VIRTUAL_ENV"] == str(venv_dir)
    assert env["BENCHMARK_ATTEMPT_PYTHON"] == str(python)
    assert "PYTHONPATH" not in env
    assert "UV_CACHE_DIR" not in env
    assert env["API_TOKEN"] == "keep"


def test_parse_scutil_proxy_output_extracts_http_and_https_proxy() -> None:
    payload = """
<dictionary> {
  HTTPEnable : 1
  HTTPPort : 7892
  HTTPProxy : 127.0.0.1
  HTTPSEnable : 1
  HTTPSPort : 7892
  HTTPSProxy : 127.0.0.1
}
"""

    proxies = parse_scutil_proxy_output(payload)

    assert proxies["HTTP_PROXY"] == "http://127.0.0.1:7892"
    assert proxies["HTTPS_PROXY"] == "http://127.0.0.1:7892"


def test_build_openclaw_subprocess_env_enables_node_proxy_from_system_proxy() -> None:
    env = build_openclaw_subprocess_env(
        base_env={},
        config_path=Path("/tmp/openclaw.json"),
        system_proxy_text="""
HTTPEnable : 1
HTTPProxy : 127.0.0.1
HTTPPort : 7892
HTTPSEnable : 1
HTTPSProxy : 127.0.0.1
HTTPSPort : 7892
""",
    )

    assert env["OPENCLAW_CONFIG_PATH"] == "/tmp/openclaw.json"
    assert env["NODE_USE_ENV_PROXY"] == "1"
    assert env["HTTP_PROXY"] == "http://127.0.0.1:7892"
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:7892"
    assert "127.0.0.1" in env["NO_PROXY"]


def test_build_openclaw_subprocess_env_does_not_override_explicit_proxy() -> None:
    env = build_openclaw_subprocess_env(
        base_env={
            "NODE_USE_ENV_PROXY": "0",
            "HTTP_PROXY": "http://proxy.example:8080",
            "HTTPS_PROXY": "http://proxy.example:8080",
        },
        system_proxy_text="""
HTTPEnable : 1
HTTPProxy : 127.0.0.1
HTTPPort : 7892
HTTPSEnable : 1
HTTPSProxy : 127.0.0.1
HTTPSPort : 7892
""",
    )

    assert env["NODE_USE_ENV_PROXY"] == "0"
    assert env["HTTP_PROXY"] == "http://proxy.example:8080"
    assert env["HTTPS_PROXY"] == "http://proxy.example:8080"


def test_proxy_environment_report_redacts_proxy_credentials() -> None:
    report = proxy_environment_report(
        {
            "NODE_USE_ENV_PROXY": "1",
            "HTTPS_PROXY": "http://user:secret@proxy.example:8080",
        }
    )

    assert report["NODE_USE_ENV_PROXY"] == "1"
    assert report["HTTPS_PROXY"] == "http://***@proxy.example:8080"
