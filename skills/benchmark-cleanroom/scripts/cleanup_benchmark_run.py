#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime_lease import (
    LEASE_KIND,
    MANIFEST_KIND,
    cleanup_report_filename_for_run,
    iso_now,
    lease_dir_from_manifest,
    manifest_path,
    maybe_int,
    read_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean benchmark run-scoped processes, sessions, and state.")
    parser.add_argument("--manifest", help="Cleanup manifest path")
    parser.add_argument("--run-id", help="Manual run id fallback")
    parser.add_argument("--kind", choices=("chemqa", "review-loop"), help="Manual benchmark kind")
    parser.add_argument("--output-root", help="Manual output root")
    parser.add_argument("--grace-seconds", type=float, default=5.0)
    parser.add_argument("--kill-after-seconds", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


@dataclass
class CleanupContext:
    manifest: dict[str, Any]
    manifest_path: Path | None


def load_context(args: argparse.Namespace) -> CleanupContext:
    if args.manifest:
        path = Path(args.manifest).expanduser().resolve()
        payload = read_json(path)
        if str(payload.get("kind") or "") != MANIFEST_KIND:
            raise SystemExit(f"Manifest has unexpected kind: {path}")
        return CleanupContext(manifest=payload, manifest_path=path)
    if not args.run_id or not args.kind or not args.output_root:
        raise SystemExit("Manual mode requires --run-id, --kind, and --output-root.")
    output_root = Path(args.output_root).expanduser().resolve()
    payload = {
        "kind": MANIFEST_KIND,
        "version": 1,
        "run_id": args.run_id,
        "benchmark_kind": args.kind,
        "group_id": "",
        "output_root": str(output_root),
        "launch_home": "",
        "clawteam_data_dir": "",
        "session_assignments": {},
        "control_roots": [],
        "generated_roots": [],
        "artifact_roots": [],
        "lease_dir": str(output_root / "cleanroom" / "leases"),
        "created_at": iso_now(),
        "updated_at": iso_now(),
    }
    return CleanupContext(manifest=payload, manifest_path=None)


def safe_read_cmdline(pid: int) -> str:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def normalize_path_list(items: list[Any] | None) -> list[Path]:
    normalized: list[Path] = []
    seen: set[str] = set()
    for item in items or []:
        text = str(item or "").strip()
        if not text:
            continue
        path = Path(text).expanduser().resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(path)
    return normalized


def iter_lease_payloads(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    lease_dir = lease_dir_from_manifest(manifest)
    if not lease_dir.is_dir():
        return []
    payloads: list[dict[str, Any]] = []
    run_id = str(manifest.get("run_id") or "")
    for path in sorted(lease_dir.glob("*.lease.json")):
        try:
            payload = read_json(path)
        except Exception:
            continue
        if str(payload.get("kind") or "") != LEASE_KIND:
            continue
        if str(payload.get("run_id") or "") != run_id:
            continue
        payload["_lease_path"] = str(path)
        payloads.append(payload)
    return payloads


def session_paths_from_manifest(manifest: dict[str, Any]) -> list[Path]:
    launch_home = str(manifest.get("launch_home") or "").strip()
    if not launch_home:
        return []
    launch_root = Path(launch_home).expanduser().resolve()
    session_assignments = dict(manifest.get("session_assignments") or {})
    paths: list[Path] = []
    seen: set[str] = set()
    for slot_id, session_id in session_assignments.items():
        slot = str(slot_id or "").strip()
        session = str(session_id or "").strip()
        if not slot or not session:
            continue
        session_dir = launch_root / ".openclaw" / "agents" / slot / "sessions"
        candidates = [
            session_dir / f"{session}.jsonl",
            session_dir / f"{session}.jsonl.lock",
            *sorted(session_dir.glob(f"{session}.checkpoint.*.jsonl")),
            *sorted(session_dir.glob(f"{session}.checkpoint.*.jsonl.lock")),
        ]
        for path in candidates:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return paths


def candidate_session_stores(manifest: dict[str, Any]) -> list[Path]:
    stores: list[Path] = []
    session_assignments = dict(manifest.get("session_assignments") or {})
    seen: set[str] = set()
    for slot_id in session_assignments:
        path = Path.home() / ".openclaw" / "agents" / str(slot_id) / "sessions" / "sessions.json"
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        stores.append(path)
    launch_home = str(manifest.get("launch_home") or "").strip()
    if launch_home:
        launch_root = Path(launch_home).expanduser().resolve()
        for slot_id in session_assignments:
            path = launch_root / ".openclaw" / "agents" / str(slot_id) / "sessions" / "sessions.json"
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            stores.append(path)
    return stores


def session_store_entry_matches(entry: dict[str, Any], *, run_id: str, session_ids: set[str]) -> bool:
    session_id = str(entry.get("sessionId") or "").strip()
    session_file = str(entry.get("sessionFile") or "").strip()
    if session_id and session_id in session_ids:
        return True
    if run_id and run_id in session_file:
        return True
    return False


def scrub_session_store(path: Path, *, run_id: str, session_ids: set[str], dry_run: bool) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "changed": False, "removed_keys": []}
    payload = read_json(path)
    if not isinstance(payload, dict):
        return {"path": str(path), "exists": True, "changed": False, "removed_keys": [], "warning": "not-object"}
    removed_keys: list[str] = []
    updated = dict(payload)
    for key, value in list(payload.items()):
        if not isinstance(value, dict):
            continue
        if session_store_entry_matches(value, run_id=run_id, session_ids=session_ids):
            updated.pop(key, None)
            removed_keys.append(str(key))
    changed = bool(removed_keys)
    if changed and not dry_run:
        path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"path": str(path), "exists": True, "changed": changed, "removed_keys": removed_keys}


def remove_path(path: Path, *, dry_run: bool) -> bool:
    if not path.exists():
        return False
    if dry_run:
        return True
    if path.is_dir():
        for child in sorted(path.iterdir(), key=lambda item: item.name, reverse=True):
            remove_path(child, dry_run=False)
        path.rmdir()
        return True
    path.unlink()
    return True


def process_targets(manifest: dict[str, Any], lease_payloads: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_id = str(manifest.get("run_id") or "")
    session_ids = {str(value).strip() for value in dict(manifest.get("session_assignments") or {}).values() if str(value).strip()}
    pid_targets: dict[int, dict[str, Any]] = {}
    pgid_targets: dict[int, dict[str, Any]] = {}
    current_pid = os.getpid()
    current_pgid = 0
    try:
        current_pgid = os.getpgid(0)
    except Exception:
        current_pgid = 0

    for lease in lease_payloads:
        pid = maybe_int(lease.get("pid"))
        pgid = maybe_int(lease.get("pgid"))
        role = str(lease.get("role") or "")
        slot = str(lease.get("slot") or "")
        session_id = str(lease.get("session_id") or "")
        if pid > 0 and pid != current_pid:
            pid_targets.setdefault(
                pid,
                {
                    "pid": pid,
                    "pgid": pgid,
                    "role": role,
                    "slot": slot,
                    "session_id": session_id,
                    "source": "lease",
                    "cmdline": safe_read_cmdline(pid),
                },
            )
        if pgid > 0 and pgid != current_pgid and pgid != current_pid:
            pgid_targets.setdefault(
                pgid,
                {
                    "pgid": pgid,
                    "source": "lease",
                    "role": role,
                    "slot": slot,
                    "session_id": session_id,
                },
            )

    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        pid = int(pid_dir.name)
        if pid == current_pid:
            continue
        cmdline = safe_read_cmdline(pid)
        if not cmdline:
            continue
        matches_run = run_id and run_id in cmdline
        matches_session = any(session_id in cmdline for session_id in session_ids)
        if not matches_run and not matches_session:
            continue
        try:
            pgid = os.getpgid(pid)
        except OSError:
            pgid = 0
        pid_targets.setdefault(
            pid,
            {
                "pid": pid,
                "pgid": pgid,
                "role": "",
                "slot": "",
                "session_id": "",
                "source": "proc-scan",
                "cmdline": cmdline,
            },
        )
        if pgid > 0 and pgid != current_pgid and pgid != current_pid:
            pgid_targets.setdefault(pgid, {"pgid": pgid, "source": "proc-scan"})

    return list(pgid_targets.values()), list(pid_targets.values())


def terminate_process_groups(
    groups: list[dict[str, Any]],
    *,
    sig: int,
    dry_run: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in groups:
        pgid = maybe_int(item.get("pgid"))
        if pgid <= 0:
            continue
        payload = {"pgid": pgid, "signal": sig, "sent": False}
        if dry_run:
            payload["sent"] = True
            results.append(payload)
            continue
        try:
            os.killpg(pgid, sig)
            payload["sent"] = True
        except ProcessLookupError:
            payload["sent"] = False
            payload["missing"] = True
        except Exception as exc:
            payload["error"] = str(exc)
        results.append(payload)
    return results


def terminate_pids(
    targets: list[dict[str, Any]],
    *,
    sig: int,
    dry_run: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in targets:
        pid = maybe_int(item.get("pid"))
        if pid <= 0:
            continue
        payload = {"pid": pid, "signal": sig, "sent": False}
        if dry_run:
            payload["sent"] = True
            results.append(payload)
            continue
        try:
            os.kill(pid, sig)
            payload["sent"] = True
        except ProcessLookupError:
            payload["sent"] = False
            payload["missing"] = True
        except Exception as exc:
            payload["error"] = str(exc)
        results.append(payload)
    return results


def wait_for_exit(targets: list[dict[str, Any]], *, timeout_seconds: float) -> list[int]:
    deadline = time.time() + max(0.0, timeout_seconds)
    pids = sorted({maybe_int(item.get("pid")) for item in targets if maybe_int(item.get("pid")) > 0})
    remaining = set(pids)
    while remaining and time.time() < deadline:
        finished: list[int] = []
        for pid in list(remaining):
            if not pid_exists(pid):
                finished.append(pid)
        for pid in finished:
            remaining.discard(pid)
        if remaining:
            time.sleep(0.2)
    return sorted(remaining)


def postcheck_session_stores(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    run_id = str(manifest.get("run_id") or "")
    session_ids = {str(value).strip() for value in dict(manifest.get("session_assignments") or {}).values() if str(value).strip()}
    leftovers: list[dict[str, Any]] = []
    for path in candidate_session_stores(manifest):
        if not path.is_file():
            continue
        try:
            payload = read_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            if session_store_entry_matches(value, run_id=run_id, session_ids=session_ids):
                leftovers.append({"path": str(path), "key": str(key), "sessionId": value.get("sessionId")})
    return leftovers


def cleanup(context: CleanupContext, *, grace_seconds: float, kill_after_seconds: float, dry_run: bool) -> dict[str, Any]:
    manifest = context.manifest
    run_id = str(manifest.get("run_id") or "")
    lease_payloads = iter_lease_payloads(manifest)
    process_groups, pid_targets = process_targets(manifest, lease_payloads)

    report: dict[str, Any] = {
        "run_id": run_id,
        "manifest_path": str(context.manifest_path) if context.manifest_path else "",
        "dry_run": dry_run,
        "started_at": iso_now(),
        "lease_count": len(lease_payloads),
        "process_groups": process_groups,
        "pid_targets": pid_targets,
        "termination": {},
        "session_store_scrub": [],
        "removed_paths": [],
        "warnings": [],
        "errors": [],
    }

    report["termination"]["term_groups"] = terminate_process_groups(process_groups, sig=signal.SIGTERM, dry_run=dry_run)
    report["termination"]["term_pids"] = terminate_pids(pid_targets, sig=signal.SIGTERM, dry_run=dry_run)
    remaining_after_term = [] if dry_run else wait_for_exit(pid_targets, timeout_seconds=grace_seconds)
    report["termination"]["remaining_after_term"] = remaining_after_term

    if remaining_after_term:
        kill_targets = [item for item in pid_targets if maybe_int(item.get("pid")) in set(remaining_after_term)]
        report["termination"]["kill_groups"] = terminate_process_groups(process_groups, sig=signal.SIGKILL, dry_run=dry_run)
        report["termination"]["kill_pids"] = terminate_pids(kill_targets, sig=signal.SIGKILL, dry_run=dry_run)
        remaining_after_kill = [] if dry_run else wait_for_exit(kill_targets, timeout_seconds=max(0.0, kill_after_seconds))
    else:
        remaining_after_kill = []
        report["termination"]["kill_groups"] = []
        report["termination"]["kill_pids"] = []
    report["termination"]["remaining_after_kill"] = remaining_after_kill

    session_ids = {str(value).strip() for value in dict(manifest.get("session_assignments") or {}).values() if str(value).strip()}
    for store_path in candidate_session_stores(manifest):
        result = scrub_session_store(store_path, run_id=run_id, session_ids=session_ids, dry_run=dry_run)
        report["session_store_scrub"].append(result)

    removable_paths: list[Path] = []
    clawteam_dir = str(manifest.get("clawteam_data_dir") or "").strip()
    if clawteam_dir:
        clawteam_root = Path(clawteam_dir).expanduser().resolve()
        removable_paths.extend([clawteam_root / "teams" / run_id, clawteam_root / "tasks" / run_id])
    removable_paths.extend(session_paths_from_manifest(manifest))
    removable_paths.extend(normalize_path_list(list(manifest.get("control_roots") or [])))
    removable_paths.extend(normalize_path_list(list(manifest.get("generated_roots") or [])))
    removable_paths.extend(normalize_path_list(list(manifest.get("artifact_roots") or [])))
    removable_paths.extend([Path(item["_lease_path"]).expanduser().resolve() for item in lease_payloads if str(item.get("_lease_path") or "").strip()])
    if context.manifest_path is not None:
        removable_paths.append(context.manifest_path)
    output_root = Path(str(manifest.get("output_root") or "")).expanduser().resolve() if str(manifest.get("output_root") or "").strip() else None
    if output_root is not None:
        removable_paths.append(output_root / "cleanroom" / "reports" / cleanup_report_filename_for_run(run_id))

    seen_paths: set[str] = set()
    for path in removable_paths:
        key = str(path)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        try:
            removed = remove_path(path, dry_run=dry_run)
            report["removed_paths"].append({"path": str(path), "removed": removed})
        except FileNotFoundError:
            report["warnings"].append(f"Missing path during cleanup: {path}")
        except Exception as exc:
            report["errors"].append(f"{path}: {exc}")

    remaining_processes = []
    if not dry_run:
        for item in pid_targets:
            pid = maybe_int(item.get("pid"))
            if pid_exists(pid):
                remaining_processes.append({"pid": pid, "cmdline": safe_read_cmdline(pid)})
    remaining_session_entries = [] if dry_run else postcheck_session_stores(manifest)
    report["postcheck"] = {
        "remaining_processes": remaining_processes,
        "remaining_session_entries": remaining_session_entries,
    }
    report["completed_at"] = iso_now()
    report["success"] = not remaining_processes and not remaining_session_entries and not report["errors"]
    return report


def write_report(manifest: dict[str, Any], report: dict[str, Any]) -> Path | None:
    output_root = str(manifest.get("output_root") or "").strip()
    if not output_root:
        return None
    path = Path(output_root).expanduser().resolve() / "cleanroom" / "reports" / cleanup_report_filename_for_run(str(manifest.get("run_id") or "run"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    context = load_context(args)
    report = cleanup(
        context,
        grace_seconds=max(0.0, float(args.grace_seconds)),
        kill_after_seconds=max(0.0, float(args.kill_after_seconds)),
        dry_run=args.dry_run,
    )
    report_path = write_report(context.manifest, report)
    if report_path is not None:
        report["report_path"] = str(report_path)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"cleanup success={report['success']} run_id={report['run_id']}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
