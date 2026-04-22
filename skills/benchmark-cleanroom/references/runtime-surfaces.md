# Runtime Surfaces

`benchmark-cleanroom` operates on run-scoped runtime metadata emitted by
benchmark launchers and runtime wrappers.

## Cleanup Manifest

Expected top-level keys:

- `kind`: `benchmark-cleanroom-manifest`
- `version`
- `run_id`
- `benchmark_kind`
- `group_id`
- `output_root`
- `launch_home`
- `clawteam_data_dir`
- `session_assignments`
- `control_roots`
- `generated_roots`
- `artifact_roots`
- `lease_dir`
- `created_at`
- `updated_at`

Optional fields may include launch outputs, command map locations, template
paths, and session file hints.

## Runtime Lease

Expected top-level keys:

- `kind`: `benchmark-cleanroom-lease`
- `version`
- `run_id`
- `role`
- `slot`
- `session_id`
- `pid`
- `pgid`
- `ppid`
- `cwd`
- `home`
- `status`
- `updated_at`

Leases are written by run-scoped driver and wrapper processes. They let the
cleaner terminate only the current run's processes instead of scanning all
OpenClaw activity heuristically.

## Cleanup Targets

The cleaner removes run-scoped state only:

- processes referenced by leases or explicit run/session ids
- `clawteam-data/teams/<run_id>`
- `clawteam-data/tasks/<run_id>`
- run session jsonl/checkpoint/lock files
- matching `agents/*/sessions/sessions.json` entries
- run control and generated artifacts
- run-scoped artifact directories
- manifest / lease / cleanup report files under the output root

It must not delete:

- shared `agents/*/agent` configs
- shared model/profile definitions
- unrelated slot workspaces
