#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from control_store import FileControlStore


def backup_path(path: Path, *, label: str = 'bak') -> Path:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%SZ')
    return path.with_suffix(path.suffix + f'.{label}.{stamp}')


def model_ref_for(model_def: dict[str, Any]) -> str:
    return f"{model_def['provider_ref']}/{model_def['remote_model_id']}"


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def save_config(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def indexed_agents(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    agents = config.get('agents', {}).get('list', [])
    return {
        item.get('id'): item
        for item in agents
        if isinstance(item, dict) and item.get('id')
    }


def build_plan(store: FileControlStore, *, profile_id: str, config: dict[str, Any]) -> dict[str, Any]:
    profile = store.get_model_profile(profile_id)
    by_id = indexed_agents(config)
    changes = []
    missing_slots = []

    for slot_id, payload in profile.get('slot_models', {}).items():
        model_def = store.get_model_definition(payload['model_ref'])
        expected = model_ref_for(model_def)
        entry = by_id.get(slot_id)
        if not entry:
            missing_slots.append(slot_id)
            continue
        current = entry.get('model')
        changes.append(
            {
                'slot_id': slot_id,
                'before': current,
                'after': expected,
                'thinking': payload.get('thinking'),
                'status': 'unchanged' if current == expected else 'changed',
            }
        )

    return {
        'profile_id': profile_id,
        'missing_slots': missing_slots,
        'changes': changes,
        'all_match': not missing_slots and all(item['status'] == 'unchanged' for item in changes),
    }


def latest_apply_backup_for(config_path: Path) -> Path | None:
    candidates = sorted(
        config_path.parent.glob(config_path.name + '.bak.*'),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Apply, verify, or roll back DebateClaw V1 model profiles against fixed OpenClaw debate slots.')
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]), help='DebateClaw V1 root (repo root or installed skill root)')
    parser.add_argument('profile_id', nargs='?', help='Model profile id to apply or verify')
    parser.add_argument('--config-file', default=str(Path.home() / '.openclaw' / 'openclaw.json'), help='OpenClaw config file to update')
    parser.add_argument('--dry-run', action='store_true', help='Print the model binding plan without writing config')
    parser.add_argument('--verify', action='store_true', help='Verify that the current config matches the named model profile')
    parser.add_argument('--rollback-from', help='Restore config from an explicit backup path')
    parser.add_argument('--rollback-latest', action='store_true', help='Restore config from the latest apply backup created by this tool')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rollback_mode = bool(args.rollback_from or args.rollback_latest)
    if rollback_mode and args.verify:
        raise SystemExit('Choose either verify or rollback, not both.')
    if not rollback_mode and not args.profile_id:
        raise SystemExit('profile_id is required unless using rollback.')

    config_path = Path(args.config_file).expanduser().resolve()
    store = FileControlStore(args.root)

    if args.rollback_from or args.rollback_latest:
        source = Path(args.rollback_from).expanduser().resolve() if args.rollback_from else latest_apply_backup_for(config_path)
        if not source or not source.exists():
            raise SystemExit('No rollback backup was found.')
        current_backup = backup_path(config_path, label='pre-rollback.bak')
        shutil.copy2(config_path, current_backup)
        shutil.copy2(source, config_path)
        payload = {
            'config_file': str(config_path),
            'restored_from': str(source),
            'current_backup': str(current_backup),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    config = load_config(config_path)
    plan = build_plan(store, profile_id=args.profile_id, config=config)
    plan['config_file'] = str(config_path)

    if args.verify:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0 if plan['all_match'] else 1

    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0 if not plan['missing_slots'] else 1

    if plan['missing_slots']:
        raise SystemExit('Cannot apply model profile because some slots are missing: ' + ', '.join(plan['missing_slots']))

    if plan['all_match']:
        print(json.dumps({**plan, 'changed': False}, indent=2, ensure_ascii=False))
        return 0

    by_id = indexed_agents(config)
    for change in plan['changes']:
        if change['status'] == 'changed':
            by_id[change['slot_id']]['model'] = change['after']

    backup = backup_path(config_path, label='bak')
    shutil.copy2(config_path, backup)
    save_config(config_path, config)
    plan['backup'] = str(backup)
    plan['changed'] = True
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
