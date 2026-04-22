#!/usr/bin/env python3

"""Shared utilities for DebateClaw's OpenClaw integration."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
API_KEY_HINTS = ("API_KEY", "ACCESS_TOKEN", "TOKEN", "KEY")
BASE_URL_HINTS = ("ANTHROPIC_BASE_URL", "OPENAI_RESPONSE_API_BASE_URL", "API_BASE_URL", "BASE_URL", "ENDPOINT", "URL")
DEFAULT_FAMILY_SEQUENCE = ("minimax", "kimi", "glm")


@dataclass(frozen=True)
class FamilySpec:
    family: str
    aliases: tuple[str, ...]
    standard_api_key: str
    standard_base_url: str
    provider_id: str
    model_id: str
    api: str = "anthropic-messages"
    auth_header: bool = False
    reasoning: bool = True
    compat: dict[str, Any] | None = None
    context_window: int = 131072
    max_tokens: int = 32768


FAMILY_SPECS: dict[str, FamilySpec] = {
    "qwen": FamilySpec(
        family="qwen",
        aliases=("QWEN", "DASHSCOPE"),
        standard_api_key="DASHSCOPE_API_KEY",
        standard_base_url="DASHSCOPE_OPENAI_RESPONSES_BASE_URL",
        provider_id="dashscope-responses",
        model_id="qwen3.5-plus",
        api="openai-responses",
        compat={"thinkingFormat": "qwen"},
        context_window=1000000,
        max_tokens=65536,
    ),
    "minimax": FamilySpec(
        family="minimax",
        aliases=("MINIMAX",),
        standard_api_key="MINIMAX_API_KEY",
        standard_base_url="MINIMAX_ANTHROPIC_BASE_URL",
        provider_id="minimax",
        model_id="MiniMax-M2.7-highspeed",
        auth_header=True,
        context_window=204800,
        max_tokens=8192,
    ),
    "kimi": FamilySpec(
        family="kimi",
        aliases=("KIMI", "MOONSHOT"),
        standard_api_key="KIMI_API_KEY",
        standard_base_url="KIMI_ANTHROPIC_BASE_URL",
        provider_id="kimi",
        model_id="kimi-k2.5",
        context_window=256000,
        max_tokens=32768,
    ),
    "glm": FamilySpec(
        family="glm",
        aliases=("GLM", "BIGMODEL"),
        standard_api_key="GLM_API_KEY",
        standard_base_url="GLM_ANTHROPIC_BASE_URL",
        provider_id="glmprovider",
        model_id="GLM-5-Turbo",
        context_window=200000,
        max_tokens=128000,
    ),
}


def parse_env_entries(env_file: Path) -> dict[str, str]:
    if not env_file.is_file():
        return {}

    entries: dict[str, str] = {}
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not NAME_PATTERN.match(name):
            continue
        parsed_value = value.strip()
        if parsed_value and parsed_value[0] == parsed_value[-1] and parsed_value[0] in ("'", '"'):
            parsed_value = parsed_value[1:-1]
        else:
            hash_index = parsed_value.find(" #")
            if hash_index >= 0:
                parsed_value = parsed_value[:hash_index].rstrip()
        entries[name] = parsed_value
    return entries


def parse_env_names(env_file: Path) -> list[str]:
    return sorted(parse_env_entries(env_file).keys())


def matches_family(name: str, spec: FamilySpec) -> bool:
    upper = name.upper()
    return any(alias in upper for alias in spec.aliases)


def classify_names(names: list[str], spec: FamilySpec) -> dict[str, object]:
    api_key_names: list[str] = []
    base_url_names: list[str] = []
    related_names: list[str] = []

    for name in names:
        upper = name.upper()
        if not matches_family(name, spec):
            continue
        related_names.append(name)
        if any(hint in upper for hint in BASE_URL_HINTS):
            base_url_names.append(name)
            continue
        if any(hint in upper for hint in API_KEY_HINTS):
            api_key_names.append(name)

    exact_standard_names = {
        "api_key": spec.standard_api_key in names,
        "base_url": spec.standard_base_url in names,
    }
    missing_standard_names = []
    if not exact_standard_names["api_key"]:
        missing_standard_names.append(spec.standard_api_key)
    if not exact_standard_names["base_url"]:
        missing_standard_names.append(spec.standard_base_url)

    if api_key_names and base_url_names:
        status = "complete"
    elif api_key_names or base_url_names:
        status = "partial"
    else:
        status = "missing"

    return {
        "family": spec.family,
        "aliases": list(spec.aliases),
        "status": status,
        "api_key_names": sorted(api_key_names),
        "base_url_names": sorted(base_url_names),
        "related_names": sorted(set(related_names)),
        "standard_names": {
            "api_key": spec.standard_api_key,
            "base_url": spec.standard_base_url,
        },
        "exact_standard_names": exact_standard_names,
        "missing_standard_names": missing_standard_names,
    }


def choose_variable_name(family_report: dict[str, Any], *, key: str) -> str:
    standard_name = str(family_report["standard_names"][key])
    exact = bool(family_report["exact_standard_names"][key])
    if exact:
        return standard_name

    candidate_key = "api_key_names" if key == "api_key" else "base_url_names"
    candidates = list(family_report[candidate_key])
    if len(candidates) == 1:
        return str(candidates[0])
    if not candidates:
        raise ValueError(
            f"{family_report['family']} is missing a discoverable {key.replace('_', ' ')} variable name. "
            f"Suggested name: {standard_name}"
        )
    raise ValueError(
        f"{family_report['family']} has multiple candidate {key.replace('_', ' ')} names: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def default_family_assignment(proposer_count: int) -> list[str]:
    sequence = list(DEFAULT_FAMILY_SEQUENCE)
    return [sequence[index % len(sequence)] for index in range(proposer_count)]


def provider_id_for(spec: FamilySpec) -> str:
    return spec.provider_id


def model_ref_for(spec: FamilySpec) -> str:
    return f"{provider_id_for(spec)}/{spec.model_id}"


def build_provider_config(
    spec: FamilySpec,
    *,
    api_key_name: str,
    base_url: str,
) -> dict[str, Any]:
    model_payload: dict[str, Any] = {
        "id": spec.model_id,
        "name": spec.model_id,
        "reasoning": spec.reasoning,
        "input": ["text"],
        "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
        },
        "contextWindow": spec.context_window,
        "maxTokens": spec.max_tokens,
    }
    if spec.api != "anthropic-messages":
        model_payload["api"] = spec.api
    if spec.compat:
        model_payload["compat"] = spec.compat

    payload: dict[str, Any] = {
        "baseUrl": base_url,
        "apiKey": {
            "source": "env",
            "provider": "default",
            "id": api_key_name,
        },
        "api": spec.api,
        "models": [model_payload],
    }
    if spec.auth_header:
        payload["authHeader"] = True
    return payload


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def dump_json_file(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_python_interpreter(*, fallback: str | Path | None = None) -> str:
    venv = os.environ.get("VIRTUAL_ENV", "").strip()
    if venv:
        venv_root = Path(venv).expanduser()
        for candidate in (venv_root / "bin" / "python", venv_root / "Scripts" / "python.exe"):
            if candidate.is_file():
                return str(candidate)
    if fallback:
        return str(Path(fallback).expanduser())
    return str(Path(sys.executable).expanduser())
