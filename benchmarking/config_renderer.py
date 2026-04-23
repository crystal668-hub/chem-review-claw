from __future__ import annotations

import json
from typing import Any

from .experiments import ExperimentSpec
from .provisioning import ProvisionedAgent, ProvisionedExperiment


class ConfigRenderError(ValueError):
    pass


def _deep_copy_jsonish(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _upsert_agent_entry(
    payload: dict[str, Any],
    *,
    provisioned_agent: ProvisionedAgent,
    model: str,
) -> None:
    agents = payload.setdefault("agents", {})
    entries = agents.setdefault("list", [])
    if not isinstance(entries, list):
        raise ConfigRenderError("OpenClaw config agents.list is not a list")
    normalized_workspace = str(provisioned_agent.workspace.resolve())
    normalized_agent_dir = str(provisioned_agent.agent_dir.resolve())
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("id", "")) == provisioned_agent.agent_id:
            entry["name"] = provisioned_agent.agent_id
            entry["workspace"] = normalized_workspace
            entry["agentDir"] = normalized_agent_dir
            entry["model"] = model
            entry.pop("thinking", None)
            return
    entries.append(
        {
            "id": provisioned_agent.agent_id,
            "name": provisioned_agent.agent_id,
            "workspace": normalized_workspace,
            "agentDir": normalized_agent_dir,
            "model": model,
        }
    )


def render_run_config(
    *,
    base_payload: dict[str, Any],
    spec: ExperimentSpec,
    provisioned: ProvisionedExperiment,
    judge_model: str,
    runner_model: str,
) -> dict[str, Any]:
    payload = _deep_copy_jsonish(base_payload)
    tools = payload.setdefault("tools", {})
    web = tools.setdefault("web", {})
    search = web.setdefault("search", {})
    search["enabled"] = spec.websearch_enabled

    plugins = payload.setdefault("plugins", {})
    entries = plugins.setdefault("entries", {})
    duckduckgo = entries.setdefault("duckduckgo", {})
    duckduckgo["enabled"] = spec.websearch_enabled
    duckduckgo.setdefault("config", {})

    _upsert_agent_entry(payload, provisioned_agent=provisioned.judge, model=judge_model)
    for runner_agent in provisioned.runner_agents:
        _upsert_agent_entry(payload, provisioned_agent=runner_agent, model=runner_model)
    return payload
