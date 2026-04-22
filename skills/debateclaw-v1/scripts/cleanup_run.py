#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from control_store import FileControlStore


WORKFLOW_TEMPLATE_PREFIX = {
    'parallel@1': 'debate-parallel-judge',
    'review-loop@1': 'debate-review-loop',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Clean generated DebateClaw V1 artifacts for one run id.')
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]), help='DebateClaw V1 root (repo root or installed skill root)')
    parser.add_argument('run_id')
    parser.add_argument('--remove-control', action='store_true', help='Also delete control/runplans and control/run-status entries for this run')
    parser.add_argument('--yes', action='store_true', help='Required when using destructive flags such as --remove-control')
    return parser.parse_args()


def delete_if_exists(path: Path) -> bool:
    if not path.exists():
        return False
    path.unlink()
    return True


def main() -> int:
    args = parse_args()
    if args.remove_control and not args.yes:
        raise SystemExit('Refusing to remove control metadata without --yes.')

    root = Path(args.root).resolve()
    store = FileControlStore(root)

    try:
        run_plan = store.get_run_plan(args.run_id)
        workflow_ref = run_plan.get('workflow_ref')
    except FileNotFoundError:
        run_plan = None
        workflow_ref = None

    deleted = {}
    for name, path in store.generated_paths_for_run(args.run_id).items():
        deleted[name] = {'path': str(path), 'deleted': delete_if_exists(path)}

    template_deleted = None
    if workflow_ref in WORKFLOW_TEMPLATE_PREFIX:
        template_path = root / 'generated' / 'templates' / f"{WORKFLOW_TEMPLATE_PREFIX[workflow_ref]}-{args.run_id}.toml"
        template_deleted = {'path': str(template_path), 'deleted': delete_if_exists(template_path)}

    control_deleted = None
    if args.remove_control:
        control_deleted = {
            'run_plan': store.delete_run_plan(args.run_id),
            'run_status': store.delete_run_status(args.run_id),
        }

    payload = {
        'run_id': args.run_id,
        'generated_deleted': deleted,
        'template_deleted': template_deleted,
        'control_deleted': control_deleted,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
