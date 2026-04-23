from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SLOT_SENTINEL_FILENAME = ".debateclaw-slot.json"
SLOT_SENTINEL_KIND = "debateclaw-slot-workspace"
SLOT_SENTINEL_VERSION = 1


@dataclass(frozen=True)
class ProvisionedAgent:
    agent_id: str
    workspace: Path
    agent_dir: Path


@dataclass(frozen=True)
class ProvisionedExperiment:
    judge: ProvisionedAgent
    runner_agents: tuple[ProvisionedAgent, ...]


def ensure_basic_agent_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def provision_slot_workspace(
    *,
    workspace: Path,
    workspace_root: Path,
    slot_id: str,
    agents_template_text: str,
    last_session_id: str = "",
) -> None:
    workspace_root.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "AGENTS.md").write_text(agents_template_text, encoding="utf-8")
    payload = {
        "kind": SLOT_SENTINEL_KIND,
        "version": SLOT_SENTINEL_VERSION,
        "slot": slot_id,
        "workspace": str(workspace.resolve()),
        "workspace_root": str(workspace_root.resolve()),
        "last_session_id": last_session_id,
        "managed_by": "debateclaw",
    }
    (workspace / SLOT_SENTINEL_FILENAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
