#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from control_store import FileControlStore
from openclaw_debate_common import resolve_python_interpreter

CONTROL_UI_HOME_ENV = 'DEBATECLAW_CONTROL_UI_HOME'


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def default_control_ui_home() -> Path:
    override = os.environ.get(CONTROL_UI_HOME_ENV, '').strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / '.clawteam' / 'debateclaw' / 'control-ui').resolve()


def config_store_dir(base: Path) -> Path:
    path = base / 'run-configs'
    path.mkdir(parents=True, exist_ok=True)
    return path


def metadata_store_dir(base: Path) -> Path:
    path = base / 'run-metadata'
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_config_path(base: Path, config_id: str) -> Path:
    return config_store_dir(base) / f'{config_id}.json'


def run_metadata_path(base: Path, run_id: str) -> Path:
    return metadata_store_dir(base) / f'{run_id}.json'


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def backup_path(path: Path, *, label: str = 'bak') -> Path:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%SZ')
    return path.with_suffix(path.suffix + f'.{label}.{stamp}')


def indexed_agents(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get('id')): item
        for item in config.get('agents', {}).get('list', [])
        if isinstance(item, dict) and item.get('id')
    }


def openclaw_config_path() -> Path:
    return Path.home() / '.openclaw' / 'openclaw.json'


def model_ref_for(store: FileControlStore, model_id: str) -> str:
    model_def = store.get_model_definition(model_id)
    provider_ref = str(model_def.get('provider_ref', ''))
    remote_model_id = str(model_def.get('remote_model_id', ''))
    if not provider_ref or not remote_model_id:
        raise SystemExit(f'Model definition is incomplete: {model_id}')
    return f'{provider_ref}/{remote_model_id}'


def load_model_catalog(ui_home: Path) -> dict[str, Any] | None:
    path = ui_home / 'model-catalog.json'
    if not path.exists():
        return None
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f'Model catalog must be a JSON object: {path}')
    allowed = payload.get('allowed_model_ids')
    if allowed is not None:
        if not isinstance(allowed, list) or not all(isinstance(item, str) and item for item in allowed):
            raise SystemExit(f'`allowed_model_ids` must be a JSON array of non-empty strings: {path}')
    return payload


def validate_models_against_catalog(config_payload: dict[str, Any], *, ui_home: Path) -> dict[str, Any] | None:
    catalog = load_model_catalog(ui_home)
    if not catalog:
        return None
    allowed = catalog.get('allowed_model_ids') or []
    if not allowed:
        return catalog
    allowed_set = set(allowed)
    requested = [str(config_payload.get('coordinator_model', ''))] + [str(item) for item in config_payload.get('proposer_models', [])]
    invalid = sorted({item for item in requested if item and item not in allowed_set})
    if invalid:
        raise SystemExit(
            'Config references model ids outside model-catalog.json allowlist: ' + ', '.join(invalid)
        )
    return catalog


def apply_inline_slot_model_mapping(
    store: FileControlStore,
    slot_models: dict[str, dict[str, Any]],
    *,
    config_file: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_json(config_file)
    by_id = indexed_agents(config)
    changes = []
    missing_slots = []

    for slot_id, slot_payload in slot_models.items():
        expected = model_ref_for(store, str(slot_payload.get('model_ref', '')))
        current = by_id.get(slot_id)
        if not current:
            missing_slots.append(slot_id)
            continue
        before = current.get('model')
        changes.append(
            {
                'slot_id': slot_id,
                'before': before,
                'after': expected,
                'status': 'unchanged' if before == expected else 'changed',
            }
        )

    if missing_slots:
        raise SystemExit('Missing fixed debate slots in openclaw.json: ' + ', '.join(missing_slots))

    if dry_run:
        return {
            'config_file': str(config_file),
            'changed': any(item['status'] == 'changed' for item in changes),
            'changes': changes,
            'backup': None,
        }

    backup = None
    if any(item['status'] == 'changed' for item in changes):
        for item in changes:
            if item['status'] == 'changed':
                by_id[item['slot_id']]['model'] = item['after']
        backup = backup_path(config_file)
        shutil.copy2(config_file, backup)
        dump_json(config_file, config)

    return {
        'config_file': str(config_file),
        'changed': any(item['status'] == 'changed' for item in changes),
        'changes': changes,
        'backup': str(backup) if backup else None,
    }


def run_json_command(command: list[str], *, cwd: Path) -> dict[str, Any]:
    result = subprocess.run(command, cwd=str(cwd), check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f'Command failed ({result.returncode}): {shlex.join(command)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}'
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f'Command did not return JSON: {shlex.join(command)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}'
        ) from exc


def detect_clawteam_team_flag(*, cwd: Path) -> str:
    result = subprocess.run(['clawteam', 'launch', '--help'], cwd=str(cwd), check=False, capture_output=True, text=True)
    help_text = (result.stdout or '') + '\n' + (result.stderr or '')
    if '--team-name' in help_text:
        return '--team-name'
    return '--team'


def effective_template_dir(*, explicit: str | None, launch_mode: str) -> str | None:
    if explicit:
        return explicit
    if launch_mode == 'run':
        return str(Path.home() / '.clawteam' / 'templates')
    return None


def patch_run_plan_with_inline_config(store: FileControlStore, run_id: str, *, config_id: str, config_payload: dict[str, Any]) -> dict[str, Any]:
    run_plan = store.get_run_plan(run_id)
    run_plan['resolved_model_profile'] = {
        'id': f'inline:{config_id}',
        'description': f'Inline slot model mapping derived from reusable config {config_id}',
        'slot_models': config_payload['slot_models'],
    }
    run_plan['slot_assignments'] = config_payload['slot_models']
    request_snapshot = run_plan.setdefault('request_snapshot', {})
    metadata = request_snapshot.setdefault('metadata', {})
    metadata['config_id'] = config_id
    request_snapshot['config_snapshot'] = {
        'id': config_payload['id'],
        'mode': config_payload['mode'],
        'coordinator_model': config_payload['coordinator_model'],
        'proposer_models': config_payload['proposer_models'],
    }
    store.save_run_plan(run_plan)
    return run_plan


def attach_run_metadata(
    *,
    store_root: Path,
    run_id: str,
    config_id: str,
    entry_session_key: str | None,
    ui_home: Path,
) -> dict[str, Any]:
    metadata_path = run_metadata_path(ui_home, run_id)
    if metadata_path.exists():
        payload = load_json(metadata_path)
    else:
        payload = {
            'run_id': run_id,
            'created_at': iso_now(),
        }

    payload['config_id'] = config_id
    if entry_session_key:
        payload['entry_session_key'] = entry_session_key

    store = FileControlStore(store_root)
    try:
        run_plan = store.get_run_plan(run_id)
        payload['team_name'] = run_plan.get('run_id', run_id)
        payload['workflow_ref'] = run_plan.get('workflow_ref')
        payload['preset_ref'] = run_plan.get('preset_ref')
    except FileNotFoundError:
        payload.setdefault('team_name', run_id)

    payload['updated_at'] = iso_now()
    dump_json(metadata_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Launch DebateClaw from an external reusable config_id store.')
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]), help='skill root or debateclaw-v1 root')
    parser.add_argument('--control-ui-home', default=str(default_control_ui_home()), help='shared runtime directory for model catalog, run-configs, and run-metadata')
    parser.add_argument('--config-id', required=True)
    parser.add_argument('--goal', required=True)
    parser.add_argument('--entry-session-key')
    parser.add_argument('--run-id')
    parser.add_argument('--priority', default='normal')
    parser.add_argument('--launch-mode', choices=('none', 'print', 'run'), default='print')
    parser.add_argument('--reset-state', action='store_true')
    parser.add_argument('--template-dir')
    parser.add_argument('--command-map-dir')
    parser.add_argument('--runtime-dir')
    parser.add_argument('--config-file', default=str(openclaw_config_path()), help='OpenClaw config file to update with inline slot model mapping')
    parser.add_argument('--dry-run-slot-apply', action='store_true', help='Validate slot mapping without mutating openclaw.json')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    ui_home = Path(args.control_ui_home).expanduser().resolve()
    store = FileControlStore(root)
    config_path = run_config_path(ui_home, args.config_id)
    if not config_path.exists():
        raise SystemExit(f'Unknown config id: {args.config_id} (expected {config_path})')
    config_payload = load_json(config_path)
    catalog = validate_models_against_catalog(config_payload, ui_home=ui_home)

    apply_result = apply_inline_slot_model_mapping(
        store,
        config_payload['slot_models'],
        config_file=Path(args.config_file).expanduser().resolve(),
        dry_run=args.dry_run_slot_apply,
    )
    if args.dry_run_slot_apply:
        print(json.dumps({'config_id': args.config_id, 'apply_slots': apply_result}, indent=2, ensure_ascii=False))
        return 0

    scripts_dir = root / 'scripts'
    compile_cmd = [
        resolve_python_interpreter(),
        str(scripts_dir / 'compile_runplan.py'),
        '--root',
        str(root),
        '--preset',
        str(config_payload['preset_ref']),
        '--goal',
        args.goal,
        '--priority',
        args.priority,
        '--proposer-count',
        str(config_payload['proposer_count']),
    ]
    if args.run_id:
        compile_cmd.extend(['--run-id', args.run_id])
    max_rounds = config_payload.get('max_rounds')
    if max_rounds is not None:
        compile_cmd.extend(['--review-rounds', str(max_rounds), '--rebuttal-rounds', str(max_rounds)])

    compiled = run_json_command(compile_cmd, cwd=root)
    run_id_value = str(compiled['run_id'])
    patched_run_plan = patch_run_plan_with_inline_config(store, run_id_value, config_id=args.config_id, config_payload=config_payload)

    materialize_cmd = [
        resolve_python_interpreter(),
        str(scripts_dir / 'materialize_runplan.py'),
        '--root',
        str(root),
        '--run-id',
        run_id_value,
    ]
    resolved_template_dir = effective_template_dir(explicit=args.template_dir, launch_mode=args.launch_mode)
    if resolved_template_dir:
        materialize_cmd.extend(['--template-dir', resolved_template_dir])
    if args.command_map_dir:
        materialize_cmd.extend(['--command-map-dir', args.command_map_dir])
    if args.runtime_dir:
        materialize_cmd.extend(['--runtime-dir', args.runtime_dir])
    if args.reset_state:
        materialize_cmd.append('--reset-state')

    materialized = run_json_command(materialize_cmd, cwd=root)

    launch_command = None
    team_flag = detect_clawteam_team_flag(cwd=root)
    if materialized.get('launch_command'):
        launch_command = shlex.split(str(materialized['launch_command']))
    elif materialized.get('template_name'):
        launch_command = [
            'clawteam',
            'launch',
            str(materialized['template_name']),
            team_flag,
            run_id_value,
            '--goal',
            args.goal,
            '--backend',
            str(compiled.get('launch_spec', {}).get('backend', 'subprocess')),
        ]

    launched = None
    if args.launch_mode == 'run':
        if not launch_command:
            raise SystemExit('No launch command is available after materialization.')
        result = subprocess.run(launch_command, cwd=str(root), check=False, capture_output=True, text=True)
        launched = {
            'command': launch_command,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }
        if result.returncode != 0:
            raise SystemExit(
                f'Launch failed ({result.returncode}): {shlex.join(launch_command)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}'
            )

    metadata = attach_run_metadata(
        store_root=root,
        run_id=run_id_value,
        config_id=args.config_id,
        entry_session_key=args.entry_session_key,
        ui_home=ui_home,
    )

    payload = {
        'config_id': args.config_id,
        'goal': args.goal,
        'control_ui_home': str(ui_home),
        'model_catalog_path': str(ui_home / 'model-catalog.json'),
        'model_catalog': catalog,
        'run_id': run_id_value,
        'apply_slots': apply_result,
        'compile': compiled,
        'patched_run_plan': patched_run_plan,
        'materialize': materialized,
        'launch_mode': args.launch_mode,
        'launch_command': launch_command,
        'launched': launched,
        'metadata': metadata,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
