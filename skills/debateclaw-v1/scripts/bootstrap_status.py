#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from control_store import FileControlStore
from openclaw_debate_common import parse_env_names


FIXED_DEBATE_SLOTS = ['debate-coordinator', 'debate-1', 'debate-2', 'debate-3', 'debate-4']


def model_ref_for(model_def: dict[str, Any]) -> str:
    return f"{model_def['provider_ref']}/{model_def['remote_model_id']}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Show the current DebateClaw V1 bootstrap/status overview.')
    parser.add_argument('--root', default=str(ROOT), help='DebateClaw V1 root (repo root or installed skill root)')
    parser.add_argument('--config-file', default=str(Path.home() / '.openclaw' / 'openclaw.json'), help='OpenClaw config file')
    parser.add_argument('--env-file', default=str(Path.home() / '.openclaw' / '.env'), help='OpenClaw env file (names only; values are not printed)')
    parser.add_argument('--json', action='store_true', help='Emit JSON')
    return parser.parse_args()


def indexed_agents(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    agents = config.get('agents', {}).get('list', [])
    return {
        item.get('id'): item
        for item in agents
        if isinstance(item, dict) and item.get('id')
    }


def render_text(payload: dict[str, Any]) -> str:
    lines = [
        f"Service: {payload['manifest'].get('service', {}).get('id', '<unknown>')}",
        f"Config file: {payload['config_file']}",
        f"Env file: {payload['env_file']}",
        '',
        'Fixed debate slots:',
    ]
    for item in payload['debate_slots']:
        lines.append(f"- {item['slot_id']}: {item.get('model')} (workspace={item.get('workspace')})")

    lines.append('')
    lines.append('Model profile matches:')
    for item in payload['model_profile_matches']:
        status = 'match' if item['matches'] else 'diff'
        lines.append(f"- {item['profile_id']}: {status}")
        if item['diffs']:
            for diff in item['diffs']:
                lines.append(f"  - {diff['slot_id']}: current={diff['current']} expected={diff['expected']}")

    lines.append('')
    lines.append('Providers:')
    for item in payload['providers']:
        lines.append(
            f"- {item['id']}: present_in_openclaw={item['present_in_openclaw']} api_key_env={item['api_key_env']} api_key_present={item['api_key_present']} base_url_env={item['base_url_env']} base_url_present={item['base_url_present']}"
        )

    run_inputs = payload['manifest'].get('run_scoped_inputs', {})
    if run_inputs:
        lines.append('')
        lines.append('Run-scoped inputs:')
        for key, value in run_inputs.items():
            lines.append(f"- {key}: {value}")

    return '\n'.join(lines)


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    store = FileControlStore(root)
    manifest = store.get_bootstrap_manifest()
    config_path = Path(args.config_file).expanduser().resolve()
    env_path = Path(args.env_file).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    env_names = set(parse_env_names(env_path))

    by_id = indexed_agents(config)
    debate_slots = []
    for slot_id in FIXED_DEBATE_SLOTS:
        entry = by_id.get(slot_id)
        debate_slots.append(
            {
                'slot_id': slot_id,
                'exists': bool(entry),
                'model': entry.get('model') if entry else None,
                'workspace': entry.get('workspace') if entry else None,
                'agentDir': entry.get('agentDir') if entry else None,
            }
        )

    providers_in_openclaw = set((config.get('models') or {}).get('providers', {}).keys())
    providers = []
    for provider in store.list_provider_definitions():
        providers.append(
            {
                'id': provider['id'],
                'present_in_openclaw': provider['id'] in providers_in_openclaw,
                'compat': provider.get('compat'),
                'api_key_env': provider.get('api_key_env'),
                'api_key_present': provider.get('api_key_env') in env_names if provider.get('api_key_env') else False,
                'base_url_env': provider.get('base_url_env'),
                'base_url_present': provider.get('base_url_env') in env_names if provider.get('base_url_env') else False,
            }
        )

    model_profile_matches = []
    for profile in store.list_model_profiles():
        diffs = []
        for slot_id, payload in profile.get('slot_models', {}).items():
            current = by_id.get(slot_id, {}).get('model')
            expected = model_ref_for(store.get_model_definition(payload['model_ref']))
            if current != expected:
                diffs.append({'slot_id': slot_id, 'current': current, 'expected': expected})
        model_profile_matches.append(
            {
                'profile_id': profile['id'],
                'matches': not diffs,
                'diffs': diffs,
            }
        )

    payload = {
        'config_file': str(config_path),
        'env_file': str(env_path),
        'manifest': manifest,
        'debate_slots': debate_slots,
        'providers': providers,
        'model_profile_matches': model_profile_matches,
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(render_text(payload))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
