#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FileControlStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.control = self.root / 'control'
        self.workflows = self.root / 'workflows'
        self.presets = self.root / 'presets'
        self.generated = self.root / 'generated'

    def _dir(self, *parts: str) -> Path:
        path = self.control.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding='utf-8'))

    def _dump_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    def _list(self, directory: Path) -> list[dict[str, Any]]:
        return [self._load_json(path) for path in sorted(directory.glob('*.json'))]

    def _delete(self, path: Path) -> bool:
        if not path.exists():
            return False
        path.unlink()
        return True

    def list_provider_definitions(self) -> list[dict[str, Any]]:
        return self._list(self._dir('providers'))

    def get_provider_definition_path(self, provider_id: str) -> Path:
        return self._dir('providers') / f'{provider_id}.json'

    def get_provider_definition(self, provider_id: str) -> dict[str, Any]:
        return self._load_json(self.get_provider_definition_path(provider_id))

    def put_provider_definition(self, payload: dict[str, Any]) -> None:
        self._dump_json(self.get_provider_definition_path(payload['id']), payload)

    def delete_provider_definition(self, provider_id: str) -> bool:
        return self._delete(self.get_provider_definition_path(provider_id))

    def list_model_definitions(self) -> list[dict[str, Any]]:
        return self._list(self._dir('models'))

    def get_model_definition_path(self, model_id: str) -> Path:
        return self._dir('models') / f'{model_id}.json'

    def get_model_definition(self, model_id: str) -> dict[str, Any]:
        return self._load_json(self.get_model_definition_path(model_id))

    def put_model_definition(self, payload: dict[str, Any]) -> None:
        self._dump_json(self.get_model_definition_path(payload['id']), payload)

    def delete_model_definition(self, model_id: str) -> bool:
        return self._delete(self.get_model_definition_path(model_id))

    def list_model_profiles(self) -> list[dict[str, Any]]:
        return self._list(self._dir('model-profiles'))

    def get_model_profile_path(self, profile_id: str) -> Path:
        return self._dir('model-profiles') / f'{profile_id}.json'

    def get_model_profile(self, profile_id: str) -> dict[str, Any]:
        return self._load_json(self.get_model_profile_path(profile_id))

    def put_model_profile(self, payload: dict[str, Any]) -> None:
        self._dump_json(self.get_model_profile_path(payload['id']), payload)

    def delete_model_profile(self, profile_id: str) -> bool:
        return self._delete(self.get_model_profile_path(profile_id))

    def get_bootstrap_manifest(self) -> dict[str, Any]:
        return self._load_json(self._dir('bootstrap') / 'manifest-latest.json')

    def save_bootstrap_manifest(self, payload: dict[str, Any]) -> None:
        self._dump_json(self._dir('bootstrap') / 'manifest-latest.json', payload)

    def list_workflows(self) -> list[dict[str, Any]]:
        return self._list(self.workflows)

    def get_workflow(self, workflow_ref: str) -> dict[str, Any]:
        return self._load_json(self.workflows / f'{workflow_ref}.json')

    def list_presets(self) -> list[dict[str, Any]]:
        return self._list(self.presets)

    def get_preset(self, preset_ref: str) -> dict[str, Any]:
        return self._load_json(self.presets / f'{preset_ref}.json')

    def list_run_plans(self) -> list[dict[str, Any]]:
        return self._list(self._dir('runplans'))

    def get_run_plan_path(self, run_id: str) -> Path:
        return self._dir('runplans') / f'{run_id}.json'

    def save_run_plan(self, payload: dict[str, Any]) -> None:
        self._dump_json(self.get_run_plan_path(payload['run_id']), payload)

    def get_run_plan(self, run_id: str) -> dict[str, Any]:
        return self._load_json(self.get_run_plan_path(run_id))

    def delete_run_plan(self, run_id: str) -> bool:
        return self._delete(self.get_run_plan_path(run_id))

    def list_run_status(self) -> list[dict[str, Any]]:
        return self._list(self._dir('run-status'))

    def get_run_status_path(self, run_id: str) -> Path:
        return self._dir('run-status') / f'{run_id}.json'

    def get_run_status(self, run_id: str) -> dict[str, Any]:
        return self._load_json(self.get_run_status_path(run_id))

    def update_run_status(self, run_id: str, payload: dict[str, Any]) -> None:
        self._dump_json(self.get_run_status_path(run_id), payload)

    def delete_run_status(self, run_id: str) -> bool:
        return self._delete(self.get_run_status_path(run_id))

    def generated_paths_for_run(self, run_id: str) -> dict[str, Path]:
        return {
            'command_map': self.generated / 'command-maps' / f'{run_id}-command-map.json',
            'prompt_bundle': self.generated / 'prompt-bundles' / f'{run_id}-prompts.json',
            'runtime_context': self.generated / 'runtime-context' / f'{run_id}-context.json',
        }
