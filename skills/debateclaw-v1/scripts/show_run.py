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
    parser = argparse.ArgumentParser(description='Show one DebateClaw V1 run plan and related generated artifacts.')
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]), help='DebateClaw V1 root (repo root or installed skill root)')
    parser.add_argument('run_id')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    store = FileControlStore(root)
    run_plan = store.get_run_plan(args.run_id)
    try:
        run_status = store.get_run_status(args.run_id)
    except FileNotFoundError:
        run_status = None

    generated_paths = {
        name: {'path': str(path), 'exists': path.exists()}
        for name, path in store.generated_paths_for_run(args.run_id).items()
    }

    template_payload = None
    prefix = WORKFLOW_TEMPLATE_PREFIX.get(run_plan.get('workflow_ref'))
    if prefix:
        template_path = root / 'generated' / 'templates' / f'{prefix}-{args.run_id}.toml'
        template_payload = {
            'path': str(template_path),
            'exists': template_path.exists(),
        }

    payload = {
        'run_id': args.run_id,
        'run_plan': run_plan,
        'run_status': run_status,
        'generated_paths': generated_paths,
        'template': template_payload,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
