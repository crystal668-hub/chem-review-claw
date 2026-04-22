#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from control_store import FileControlStore


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def default_run_id(preset_ref: str) -> str:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    preset_id = preset_ref.split('@', 1)[0]
    return f'{preset_id}-{stamp}'


def safe_session_id(*parts: str) -> str:
    raw = "-".join(part for part in parts if part)
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", raw)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    if not normalized:
        raise SystemExit("Could not build a valid OpenClaw session id.")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compile a DebateClaw V1 run plan from file-backed control metadata.')
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]), help='DebateClaw V1 root (repo root or installed skill root)')
    parser.add_argument('--preset', required=True, help='Preset ref, e.g. parallel@1 or review-loop@1')
    parser.add_argument('--goal', required=True, help='Run goal or motion')
    parser.add_argument('--run-id', help='Optional explicit run id')
    parser.add_argument('--additional-file-workspace', help='Optional run-scoped opaque string for extra file context (path, URI, or backend-specific locator)')
    parser.add_argument('--model-profile', help='Override model profile name')
    parser.add_argument('--proposer-count', type=int)
    parser.add_argument('--review-rounds', type=int)
    parser.add_argument('--rebuttal-rounds', type=int)
    parser.add_argument('--priority', default='normal')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--json', action='store_true')
    return parser.parse_args()


def apply_override(*, resolved: dict[str, Any], preset: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    rule = preset.get('overrides', {}).get(key)
    if not rule or rule.get('exposure') != 'tunable':
        raise SystemExit(
            f"Preset `{preset['id']}@{preset['version']}` does not allow override for `{key}`."
        )
    if 'min' in rule and value < rule['min']:
        raise SystemExit(f"Override `{key}`={value} is below minimum {rule['min']}.")
    if 'max' in rule and value > rule['max']:
        raise SystemExit(f"Override `{key}`={value} is above maximum {rule['max']}.")
    resolved[key] = value


def proposer_slots_from_profile(profile: dict[str, Any], count: int) -> list[str]:
    candidates = [slot for slot in profile.get('slot_models', {}) if slot.startswith('debate-') and slot != 'debate-coordinator']

    def key(slot: str) -> int:
        try:
            return int(slot.split('-', 1)[1])
        except Exception as exc:  # pragma: no cover - defensive
            raise SystemExit(f'Invalid proposer slot id in model profile: {slot}') from exc

    ordered = sorted(candidates, key=key)
    if len(ordered) < count:
        raise SystemExit(f'Model profile only has {len(ordered)} proposer slots, but proposer_count={count}.')
    return ordered[:count]


def build_prompt_assembly(preset: dict[str, Any], proposer_slots: list[str], *, has_additional_file_workspace: bool) -> dict[str, Any]:
    pack = preset['prompt_pack']
    modules = list(pack.get('modules', []))
    if has_additional_file_workspace:
        modules.append('prompts/modules/context/additional-file-workspace.md')

    assembly: dict[str, Any] = {
        'debate-coordinator': {
            'contracts': [pack['role_contracts']['coordinator']],
            'modules': modules,
        }
    }
    for index, _slot in enumerate(proposer_slots, start=1):
        assembly[f'proposer-{index}'] = {
            'contracts': [pack['role_contracts']['proposer']],
            'modules': modules,
        }
    return assembly


def main() -> int:
    args = parse_args()
    store = FileControlStore(args.root)

    preset = store.get_preset(args.preset)
    workflow = store.get_workflow(preset['workflow_ref'])
    _bootstrap = store.get_bootstrap_manifest()

    resolved = dict(preset.get('defaults', {}))
    apply_override(resolved=resolved, preset=preset, key='model_profile', value=args.model_profile)
    apply_override(resolved=resolved, preset=preset, key='proposer_count', value=args.proposer_count)
    apply_override(resolved=resolved, preset=preset, key='review_rounds', value=args.review_rounds)
    apply_override(resolved=resolved, preset=preset, key='rebuttal_rounds', value=args.rebuttal_rounds)

    model_profile_id = resolved['model_profile']
    if model_profile_id not in preset.get('allowed_model_profiles', []):
        raise SystemExit(f'Model profile `{model_profile_id}` is not allowed by preset `{args.preset}`.')
    model_profile = store.get_model_profile(model_profile_id)

    proposer_count = int(resolved['proposer_count'])
    proposer_slots = proposer_slots_from_profile(model_profile, proposer_count)
    coordinator_slot = 'debate-coordinator'
    if coordinator_slot not in model_profile.get('slot_models', {}):
        raise SystemExit('Model profile is missing debate-coordinator slot mapping.')

    slot_assignments = {
        coordinator_slot: model_profile['slot_models'][coordinator_slot],
    }
    slot_assignments.update({slot: model_profile['slot_models'][slot] for slot in proposer_slots})

    run_id = args.run_id or default_run_id(args.preset)
    session_assignments = {
        coordinator_slot: safe_session_id('debate', run_id, 'coordinator'),
    }
    for index, slot in enumerate(proposer_slots, start=1):
        session_assignments[slot] = safe_session_id('debate', run_id, f'proposer-{index}')

    additional_file_workspace = (args.additional_file_workspace or '').strip() or None

    run_plan = {
        'run_id': run_id,
        'created_at': iso_now(),
        'request_snapshot': {
            'preset_ref': args.preset,
            'goal': args.goal,
            'inputs': {
                'additional_file_workspace': additional_file_workspace,
            },
            'metadata': {'priority': args.priority},
            'overrides': resolved,
        },
        'bootstrap_manifest_ref': 'manifest-latest',
        'workflow_ref': f"{workflow['id']}@{workflow['version']}",
        'preset_ref': f"{preset['id']}@{preset['version']}",
        'resolved_model_profile': model_profile,
        'slot_assignments': slot_assignments,
        'session_assignments': session_assignments,
        'prompt_assembly': build_prompt_assembly(
            preset,
            proposer_slots,
            has_additional_file_workspace=bool(additional_file_workspace),
        ),
        'launch_spec': {
            'backend': resolved.get('backend', 'subprocess'),
            'final_decider': resolved.get('final_decider', 'outer-entry-agent'),
            'proposer_slots': proposer_slots,
            'coordinator_slot': coordinator_slot,
        },
        'protocol_defaults': {
            'proposer_count': proposer_count,
            'review_rounds': resolved.get('review_rounds'),
            'rebuttal_rounds': resolved.get('rebuttal_rounds'),
            'evidence_mode': resolved.get('evidence_mode'),
        },
        'runtime_context': {
            'additional_file_workspace': additional_file_workspace,
            'final_decider': resolved.get('final_decider'),
            'backend': resolved.get('backend', 'subprocess'),
            'evidence_mode': resolved.get('evidence_mode'),
        },
        'artifacts_root': f'<debate-run-root>/{run_id}/artifacts',
        'protocol_state_path': f'<debate-run-root>/{run_id}/state.db',
        'status': 'planned',
    }

    if not args.dry_run:
        store.save_run_plan(run_plan)
        store.update_run_status(run_id, {'run_id': run_id, 'status': 'planned', 'updated_at': iso_now()})

    print(json.dumps(run_plan, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
