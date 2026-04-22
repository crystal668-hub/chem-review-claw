#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from openclaw_debate_common import resolve_python_interpreter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Re-materialize one persisted DebateClaw V1 run.')
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]), help='DebateClaw V1 root (repo root or installed skill root)')
    parser.add_argument('run_id')
    parser.add_argument('--template-dir')
    parser.add_argument('--command-map-dir')
    parser.add_argument('--reset-state', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    command = [
        resolve_python_interpreter(),
        str(root / 'scripts' / 'materialize_runplan.py'),
        '--root',
        str(root),
        '--run-id',
        args.run_id,
    ]
    if args.template_dir:
        command.extend(['--template-dir', args.template_dir])
    if args.command_map_dir:
        command.extend(['--command-map-dir', args.command_map_dir])
    if args.reset_state:
        command.append('--reset-state')
    if args.dry_run:
        command.append('--dry-run')

    result = subprocess.run(command, cwd=str(root), check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout or f'Command failed: {command}')
    print(json.dumps(json.loads(result.stdout), indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
