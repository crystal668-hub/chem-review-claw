#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from control_store import FileControlStore


FIXED_DEBATE_SLOTS = [
    'debate-coordinator',
    'debate-1',
    'debate-2',
    'debate-3',
    'debate-4',
]


def validate_profile(store: FileControlStore, profile: dict[str, Any]) -> dict[str, Any]:
    missing = []
    unknown_slots = []
    slot_models = profile.get('slot_models', {})
    for slot_id, payload in slot_models.items():
        if slot_id not in FIXED_DEBATE_SLOTS:
            unknown_slots.append(slot_id)
        model_ref = payload.get('model_ref')
        try:
            store.get_model_definition(model_ref)
        except FileNotFoundError:
            missing.append({'slot_id': slot_id, 'model_ref': model_ref})
    return {
        'profile_id': profile.get('id'),
        'valid': not missing and not unknown_slots,
        'missing_models': missing,
        'unknown_slots': unknown_slots,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Manage DebateClaw V1 file-backed model profiles.')
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]), help='DebateClaw V1 root (repo root or installed skill root)')
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('list-models', help='List model definitions.')
    sub.add_parser('list-profiles', help='List model profiles.')

    show = sub.add_parser('show-profile', help='Show one model profile.')
    show.add_argument('profile_id')

    create = sub.add_parser('create-profile', help='Create an empty model profile.')
    create.add_argument('profile_id')
    create.add_argument('--description', default='')

    clone = sub.add_parser('clone-profile', help='Clone one model profile into another id.')
    clone.add_argument('source_profile_id')
    clone.add_argument('target_profile_id')
    clone.add_argument('--description')

    delete = sub.add_parser('delete-profile', help='Delete one model profile.')
    delete.add_argument('profile_id')
    delete.add_argument('--yes', action='store_true')

    validate = sub.add_parser('validate-profile', help='Validate that all slot model refs resolve.')
    validate.add_argument('profile_id')

    diff = sub.add_parser('diff-profiles', help='Diff two model profiles by slot assignment.')
    diff.add_argument('left_profile_id')
    diff.add_argument('right_profile_id')

    set_slot = sub.add_parser('set-slot-model', help='Bind one slot to one model definition in a profile.')
    set_slot.add_argument('profile_id')
    set_slot.add_argument('slot_id')
    set_slot.add_argument('model_id')
    set_slot.add_argument('--thinking')

    unset_slot = sub.add_parser('unset-slot-model', help='Remove one slot binding from a profile.')
    unset_slot.add_argument('profile_id')
    unset_slot.add_argument('slot_id')

    return parser


def main() -> int:
    args = build_parser().parse_args()
    store = FileControlStore(args.root)

    if args.command == 'list-models':
        print(json.dumps(store.list_model_definitions(), indent=2, ensure_ascii=False))
        return 0

    if args.command == 'list-profiles':
        print(json.dumps(store.list_model_profiles(), indent=2, ensure_ascii=False))
        return 0

    if args.command == 'show-profile':
        print(json.dumps(store.get_model_profile(args.profile_id), indent=2, ensure_ascii=False))
        return 0

    if args.command == 'create-profile':
        payload = {
            'id': args.profile_id,
            'description': args.description,
            'slot_models': {},
        }
        store.put_model_profile(payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == 'clone-profile':
        source = copy.deepcopy(store.get_model_profile(args.source_profile_id))
        source['id'] = args.target_profile_id
        if args.description is not None:
            source['description'] = args.description
        store.put_model_profile(source)
        print(json.dumps(source, indent=2, ensure_ascii=False))
        return 0

    if args.command == 'delete-profile':
        if not args.yes:
            raise SystemExit('Refusing to delete without --yes.')
        deleted = store.delete_model_profile(args.profile_id)
        print(json.dumps({'profile_id': args.profile_id, 'deleted': deleted}, indent=2, ensure_ascii=False))
        return 0

    if args.command == 'validate-profile':
        profile = store.get_model_profile(args.profile_id)
        report = validate_profile(store, profile)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report['valid'] else 1

    if args.command == 'diff-profiles':
        left = store.get_model_profile(args.left_profile_id)
        right = store.get_model_profile(args.right_profile_id)
        left_slots = left.get('slot_models', {})
        right_slots = right.get('slot_models', {})
        all_slots = sorted(set(left_slots) | set(right_slots))
        changes = []
        for slot_id in all_slots:
            before = left_slots.get(slot_id)
            after = right_slots.get(slot_id)
            if before != after:
                changes.append({'slot_id': slot_id, 'left': before, 'right': after})
        print(json.dumps({
            'left_profile_id': args.left_profile_id,
            'right_profile_id': args.right_profile_id,
            'changes': changes,
        }, indent=2, ensure_ascii=False))
        return 0

    if args.command == 'set-slot-model':
        if args.slot_id not in FIXED_DEBATE_SLOTS:
            raise SystemExit(f'Unsupported fixed debate slot: {args.slot_id}')
        profile = store.get_model_profile(args.profile_id)
        store.get_model_definition(args.model_id)
        slot_models = profile.setdefault('slot_models', {})
        slot_models[args.slot_id] = {'model_ref': args.model_id}
        if args.thinking:
            slot_models[args.slot_id]['thinking'] = args.thinking
        store.put_model_profile(profile)
        print(json.dumps(profile, indent=2, ensure_ascii=False))
        return 0

    if args.command == 'unset-slot-model':
        profile = store.get_model_profile(args.profile_id)
        removed = profile.setdefault('slot_models', {}).pop(args.slot_id, None)
        store.put_model_profile(profile)
        print(json.dumps({'profile_id': args.profile_id, 'slot_id': args.slot_id, 'removed': removed}, indent=2, ensure_ascii=False))
        return 0

    raise SystemExit(f'Unhandled command: {args.command}')


if __name__ == '__main__':
    raise SystemExit(main())
