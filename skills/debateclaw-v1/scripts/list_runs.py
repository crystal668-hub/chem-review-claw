#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from control_store import FileControlStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='List DebateClaw V1 runs known to the control store.')
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]), help='DebateClaw V1 root (repo root or installed skill root)')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = FileControlStore(args.root)
    statuses = {item.get('run_id'): item for item in store.list_run_status() if isinstance(item, dict) and item.get('run_id')}
    payload = []
    for run_plan in store.list_run_plans():
        run_id = run_plan.get('run_id')
        payload.append({
            'run_id': run_id,
            'preset_ref': run_plan.get('preset_ref'),
            'workflow_ref': run_plan.get('workflow_ref'),
            'created_at': run_plan.get('created_at'),
            'goal': run_plan.get('request_snapshot', {}).get('goal'),
            'status': statuses.get(run_id, {}).get('status', run_plan.get('status')),
        })
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
