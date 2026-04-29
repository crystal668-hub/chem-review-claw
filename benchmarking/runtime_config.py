from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config_renderer import ConfigRenderError, render_run_config
from .experiments import ExperimentSpec
from .provisioning import (
    ProvisionedAgent,
    ProvisionedExperiment,
    ensure_basic_agent_dirs,
    provision_slot_workspace,
)


class ExperimentGroupLike(Protocol):
    id: str
    label: str
    runner: str
    websearch: bool


class RuntimeConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeConfigContext:
    baseline_workspace_root: Path
    chemqa_workspace_roots: Mapping[str, Path]
    agents_root: Path
    judge_agent_id: str
    chemqa_slot_sets: Mapping[str, str]
    experiment_specs: Mapping[str, ExperimentSpec]
    load_slot_agents_template: Callable[[], str]


def logical_slot_ids() -> tuple[str, ...]:
    return ("debate-coordinator", "debate-1", "debate-2", "debate-3", "debate-4", "debate-5")


def actual_slot_ids(slot_set: str) -> dict[str, str]:
    normalized = str(slot_set).strip()
    prefix = f"debate{normalized}"
    return {
        "debate-coordinator": f"{prefix}-coordinator",
        "debate-1": f"{prefix}-1",
        "debate-2": f"{prefix}-2",
        "debate-3": f"{prefix}-3",
        "debate-4": f"{prefix}-4",
        "debate-5": f"{prefix}-5",
    }


def slot_role_map(slot_set: str) -> dict[str, str]:
    slots = actual_slot_ids(slot_set)
    return {
        "debate-coordinator": slots["debate-coordinator"],
        "proposer-1": slots["debate-1"],
        "proposer-2": slots["debate-2"],
        "proposer-3": slots["debate-3"],
        "proposer-4": slots["debate-4"],
        "proposer-5": slots["debate-5"],
    }


def _render_run_config_or_raise(
    *,
    base_payload: dict[str, Any],
    spec: ExperimentSpec,
    provisioned: ProvisionedExperiment,
    judge_model: str,
    runner_model: str,
) -> dict[str, Any]:
    try:
        return render_run_config(
            base_payload=base_payload,
            spec=spec,
            provisioned=provisioned,
            judge_model=judge_model,
            runner_model=runner_model,
        )
    except ConfigRenderError as exc:
        raise RuntimeConfigError(str(exc)) from exc


def build_run_scoped_config_payload(
    base_payload: dict[str, Any],
    *,
    context: RuntimeConfigContext,
    group: ExperimentGroupLike,
    single_agent_model: str,
    judge_model: str,
    single_agent_id_override: str | None = None,
) -> dict[str, Any]:
    judge_workspace = context.baseline_workspace_root / context.judge_agent_id
    judge_agent_dir = context.agents_root / context.judge_agent_id / "agent"
    ensure_basic_agent_dirs(judge_workspace, judge_agent_dir)
    judge = ProvisionedAgent(
        agent_id=context.judge_agent_id,
        workspace=judge_workspace,
        agent_dir=judge_agent_dir,
    )
    runner_agents: list[ProvisionedAgent] = []
    spec = context.experiment_specs.get(
        group.id,
        ExperimentSpec(
            id=group.id,
            label=group.label,
            runner_kind=group.runner,
            websearch_enabled=group.websearch,
        ),
    )

    if group.id == "benchmark-judge-runtime":
        return _render_run_config_or_raise(
            base_payload=base_payload,
            spec=spec,
            provisioned=ProvisionedExperiment(judge=judge, runner_agents=()),
            judge_model=judge_model,
            runner_model=single_agent_model,
        )

    if group.runner == "single_llm":
        agent_id = spec.resolve_single_agent_id(single_agent_id_override)
        if not agent_id:
            raise RuntimeConfigError(f"Experiment group `{group.id}` missing single-agent id in experiment spec.")
        workspace = context.baseline_workspace_root / agent_id
        agent_dir = context.agents_root / agent_id / "agent"
        ensure_basic_agent_dirs(workspace, agent_dir)
        runner_agents.append(
            ProvisionedAgent(
                agent_id=agent_id,
                workspace=workspace,
                agent_dir=agent_dir,
            )
        )
        single_spec = ExperimentSpec(
            id=spec.id,
            label=spec.label,
            runner_kind=spec.runner_kind,
            websearch_enabled=spec.websearch_enabled,
            single_agent_id=agent_id,
            slot_set=spec.slot_set,
        )
        return _render_run_config_or_raise(
            base_payload=base_payload,
            spec=single_spec,
            provisioned=ProvisionedExperiment(judge=judge, runner_agents=tuple(runner_agents)),
            judge_model=judge_model,
            runner_model=single_agent_model,
        )

    slot_set = context.chemqa_slot_sets[group.id]
    workspace_root = context.chemqa_workspace_roots[slot_set]
    slot_map = actual_slot_ids(slot_set)
    agents_template_text = context.load_slot_agents_template()
    for actual_slot_id in slot_map.values():
        workspace = workspace_root / actual_slot_id
        agent_dir = context.agents_root / actual_slot_id / "agent"
        ensure_basic_agent_dirs(agent_dir)
        provision_slot_workspace(
            workspace=workspace,
            workspace_root=workspace_root,
            slot_id=actual_slot_id,
            agents_template_text=agents_template_text,
        )
        runner_agents.append(
            ProvisionedAgent(
                agent_id=actual_slot_id,
                workspace=workspace,
                agent_dir=agent_dir,
            )
        )
    chemqa_spec = ExperimentSpec(
        id=group.id,
        label=group.label,
        runner_kind=group.runner,
        websearch_enabled=group.websearch,
        slot_set=slot_set,
    )
    return _render_run_config_or_raise(
        base_payload=base_payload,
        spec=chemqa_spec,
        provisioned=ProvisionedExperiment(judge=judge, runner_agents=tuple(runner_agents)),
        judge_model=judge_model,
        runner_model=single_agent_model,
    )


class ConfigPool:
    def __init__(
        self,
        *,
        base_config_path: Path,
        output_root: Path,
        context: RuntimeConfigContext,
        single_agent_model: str | None = None,
        judge_model: str | None = None,
        single_agent_id_override: str | None = None,
    ) -> None:
        self.base_config_path = base_config_path
        self.output_root = output_root
        self.context = context
        self._payload = json.loads(base_config_path.read_text(encoding="utf-8"))
        self._config_dir = output_root / "runtime-config"
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._group_paths: dict[str, Path] = {}
        self._judge_path: Path | None = None
        discovered_single = self._discover_agent_model("debate-1") or "su8/gpt-5.4"
        discovered_judge = self._discover_agent_model("debate-coordinator") or "su8/gpt-5.4"
        self._single_agent_model = str(single_agent_model or discovered_single).strip() or discovered_single
        self._judge_model = str(judge_model or discovered_judge).strip() or discovered_judge
        self._single_agent_id_override = single_agent_id_override

    def _discover_agent_model(self, agent_id: str) -> str | None:
        agents = ((self._payload.get("agents") or {}).get("list") or [])
        for entry in agents:
            if isinstance(entry, dict) and str(entry.get("id", "")) == agent_id:
                model = str(entry.get("model") or "").strip()
                if model:
                    return model
        return None

    def config_for_group(self, group: ExperimentGroupLike) -> Path:
        existing = self._group_paths.get(group.id)
        if existing is not None:
            return existing
        payload = build_run_scoped_config_payload(
            self._payload,
            context=self.context,
            group=group,
            single_agent_model=self._single_agent_model,
            judge_model=self._judge_model,
            single_agent_id_override=self._single_agent_id_override,
        )
        path = self._config_dir / f"{group.id}-openclaw.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._group_paths[group.id] = path
        return path

    def judge_config_path(self) -> Path:
        if self._judge_path is not None:
            return self._judge_path
        judge_group = _JudgeExperimentGroup()
        payload = build_run_scoped_config_payload(
            self._payload,
            context=self.context,
            group=judge_group,
            single_agent_model=self._single_agent_model,
            judge_model=self._judge_model,
        )
        path = self._config_dir / "benchmark-judge-openclaw.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._judge_path = path
        return path

    def cleanup(self) -> None:
        return


@dataclass(frozen=True)
class _JudgeExperimentGroup:
    id: str = "benchmark-judge-runtime"
    label: str = "benchmark judge runtime"
    runner: str = "single_llm"
    websearch: bool = False
