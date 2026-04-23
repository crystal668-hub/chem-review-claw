from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    label: str
    runner_kind: str
    websearch_enabled: bool
    single_agent_id: str | None = None
    slot_set: str | None = None

    def resolve_single_agent_id(self, override: str | None) -> str | None:
        if override is not None:
            stripped = override.strip()
            if stripped:
                return stripped
        return self.single_agent_id
