from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

PROXY_KEYS = (
    "NODE_USE_ENV_PROXY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)
DEFAULT_NO_PROXY_ENTRIES = ("localhost", "127.0.0.1", "::1")
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_VENV = WORKSPACE_ROOT / ".venv"
WORKSPACE_VENV_BIN = WORKSPACE_VENV / "bin"


def parse_scutil_proxy_output(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in str(text or "").splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        values[key.strip()] = value.strip()

    proxies: dict[str, str] = {}
    if values.get("HTTPEnable") == "1" and values.get("HTTPProxy") and values.get("HTTPPort"):
        proxies["HTTP_PROXY"] = f"http://{values['HTTPProxy']}:{values['HTTPPort']}"
    if values.get("HTTPSEnable") == "1" and values.get("HTTPSProxy") and values.get("HTTPSPort"):
        proxies["HTTPS_PROXY"] = f"http://{values['HTTPSProxy']}:{values['HTTPSPort']}"
    return proxies


def macos_system_proxy_env() -> dict[str, str]:
    try:
        completed = subprocess.run(
            ["scutil", "--proxy"],
            text=True,
            capture_output=True,
            check=False,
            timeout=3,
        )
    except Exception:
        return {}
    if completed.returncode != 0:
        return {}
    return parse_scutil_proxy_output(completed.stdout)


def _has_explicit_proxy(environment: Mapping[str, str], key: str) -> bool:
    return bool(str(environment.get(key) or environment.get(key.lower()) or "").strip())


def _merge_no_proxy(existing: str) -> str:
    seen: set[str] = set()
    entries: list[str] = []
    for value in [*(str(existing or "").split(",")), *DEFAULT_NO_PROXY_ENTRIES]:
        item = value.strip()
        if item and item not in seen:
            seen.add(item)
            entries.append(item)
    return ",".join(entries)


def _prefix_path(path: str, prefix: Path) -> str:
    prefix_text = str(prefix)
    entries = [item for item in str(path or "").split(os.pathsep) if item]
    filtered = [item for item in entries if item != prefix_text]
    return os.pathsep.join([prefix_text, *filtered])


def _inject_workspace_python_env(environment: dict[str, str]) -> None:
    if not WORKSPACE_VENV_BIN.is_dir():
        return
    environment["PATH"] = _prefix_path(environment.get("PATH", ""), WORKSPACE_VENV_BIN)
    environment.setdefault("VIRTUAL_ENV", str(WORKSPACE_VENV))
    environment["PYTHONNOUSERSITE"] = "1"


def _clear_attempt_python_overrides(environment: dict[str, str]) -> None:
    for key in tuple(environment):
        upper = key.upper()
        if (
            upper in {"VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE"}
            or upper.startswith("PIP_")
            or upper.startswith("UV_")
        ):
            environment.pop(key, None)


def _inject_attempt_python_env(environment: dict[str, str], attempt_python: Path) -> None:
    attempt_cache = environment.get("BENCHMARK_ATTEMPT_UV_CACHE", "")
    pypi_cutoff = environment.get("BENCHMARK_PYPI_CUTOFF", "")
    attempt_python = Path(attempt_python).expanduser().resolve()
    if not attempt_python.is_file():
        raise FileNotFoundError(f"attempt Python executable not found: {attempt_python}")
    attempt_bin = attempt_python.parent
    _clear_attempt_python_overrides(environment)
    environment["PATH"] = _prefix_path(environment.get("PATH", ""), attempt_bin)
    environment["VIRTUAL_ENV"] = str(attempt_bin.parent)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["BENCHMARK_ATTEMPT_PYTHON"] = str(attempt_python)
    environment["UV_PYTHON"] = str(attempt_python)
    environment["UV_DEFAULT_INDEX"] = "https://pypi.org/simple"
    if attempt_cache:
        environment["UV_CACHE_DIR"] = attempt_cache
    if pypi_cutoff:
        environment["UV_EXCLUDE_NEWER"] = pypi_cutoff


def build_openclaw_subprocess_env(
    *,
    base_env: Mapping[str, str] | None = None,
    config_path: Path | str | None = None,
    attempt_python: Path | str | None = None,
    system_proxy_text: str | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if base_env is None else base_env)
    if config_path is not None:
        environment["OPENCLAW_CONFIG_PATH"] = str(Path(config_path).expanduser())
    effective_attempt_python = attempt_python or environment.get("BENCHMARK_ATTEMPT_PYTHON")
    if effective_attempt_python is not None:
        _inject_attempt_python_env(environment, Path(effective_attempt_python))
    else:
        _inject_workspace_python_env(environment)

    system_proxies = parse_scutil_proxy_output(system_proxy_text) if system_proxy_text is not None else macos_system_proxy_env()
    for key in ("HTTP_PROXY", "HTTPS_PROXY"):
        if not _has_explicit_proxy(environment, key):
            value = system_proxies.get(key)
            if value:
                environment[key] = value

    has_any_proxy = any(_has_explicit_proxy(environment, key) for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"))
    if has_any_proxy and "NODE_USE_ENV_PROXY" not in environment:
        environment["NODE_USE_ENV_PROXY"] = "1"
    if has_any_proxy and "NO_PROXY" not in environment and "no_proxy" not in environment:
        environment["NO_PROXY"] = _merge_no_proxy("")
    elif has_any_proxy and "NO_PROXY" in environment:
        environment["NO_PROXY"] = _merge_no_proxy(environment["NO_PROXY"])
    return environment


def _redact_proxy_value(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
    except Exception:
        return re.sub(r"//[^/@]+@", "//***@", raw)
    if "@" not in parts.netloc:
        return raw
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit((parts.scheme, f"***@{host}", parts.path, parts.query, parts.fragment))


def proxy_environment_report(env: Mapping[str, str]) -> dict[str, str]:
    report: dict[str, str] = {}
    for key in PROXY_KEYS:
        if key in env:
            value = str(env.get(key) or "")
            report[key] = _redact_proxy_value(value) if "PROXY" in key.upper() and key != "NODE_USE_ENV_PROXY" else value
    return report
