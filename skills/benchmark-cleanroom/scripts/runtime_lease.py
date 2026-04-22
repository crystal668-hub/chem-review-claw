#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LEASE_KIND = "benchmark-cleanroom-lease"
LEASE_VERSION = 1
MANIFEST_KIND = "benchmark-cleanroom-manifest"
MANIFEST_VERSION = 1


def iso_now(epoch: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(epoch or time.time()))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def current_pgid() -> int:
    try:
        return os.getpgid(0)
    except Exception:
        return 0


def current_ppid() -> int:
    try:
        return os.getppid()
    except Exception:
        return 0


def normalize_session_assignments(payload: dict[str, Any] | None) -> dict[str, str]:
    raw = dict(payload or {})
    normalized: dict[str, str] = {}
    for key, value in raw.items():
        slot = str(key or "").strip()
        session_id = str(value or "").strip()
        if slot and session_id:
            normalized[slot] = session_id
    return normalized


def manifest_filename_for_run(run_id: str) -> str:
    return f"{run_id}.manifest.json"


def cleanup_report_filename_for_run(run_id: str) -> str:
    return f"{run_id}.cleanup-report.json"


def lease_filename_for_identity(*, run_id: str, role: str, slot: str, session_id: str, pid: int) -> str:
    safe_role = role.strip() or "unknown-role"
    safe_slot = slot.strip() or "unknown-slot"
    safe_session = session_id.strip() or "unknown-session"
    return f"{run_id}--{safe_role}--{safe_slot}--{safe_session}--{pid}.lease.json"


def build_manifest_payload(
    *,
    run_id: str,
    benchmark_kind: str,
    group_id: str,
    output_root: str | Path,
    launch_home: str | Path = "",
    clawteam_data_dir: str | Path = "",
    session_assignments: dict[str, str] | None = None,
    control_roots: list[str] | None = None,
    generated_roots: list[str] | None = None,
    artifact_roots: list[str] | None = None,
    lease_dir: str | Path = "",
    created_at: str | None = None,
    updated_at: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "kind": MANIFEST_KIND,
        "version": MANIFEST_VERSION,
        "run_id": run_id,
        "benchmark_kind": benchmark_kind,
        "group_id": group_id,
        "output_root": str(Path(output_root).expanduser().resolve()),
        "launch_home": str(Path(launch_home).expanduser().resolve()) if str(launch_home).strip() else "",
        "clawteam_data_dir": str(Path(clawteam_data_dir).expanduser().resolve()) if str(clawteam_data_dir).strip() else "",
        "session_assignments": normalize_session_assignments(session_assignments),
        "control_roots": [str(Path(item).expanduser().resolve()) for item in (control_roots or []) if str(item).strip()],
        "generated_roots": [str(Path(item).expanduser().resolve()) for item in (generated_roots or []) if str(item).strip()],
        "artifact_roots": [str(Path(item).expanduser().resolve()) for item in (artifact_roots or []) if str(item).strip()],
        "lease_dir": str(Path(lease_dir).expanduser().resolve()) if str(lease_dir).strip() else "",
        "created_at": created_at or iso_now(),
        "updated_at": updated_at or iso_now(),
    }
    if extra:
        payload.update(extra)
    return payload


def manifest_path(output_root: str | Path, run_id: str) -> Path:
    root = Path(output_root).expanduser().resolve()
    return root / "cleanroom" / "manifests" / manifest_filename_for_run(run_id)


def lease_dir_from_manifest(manifest: dict[str, Any]) -> Path:
    explicit = str(manifest.get("lease_dir") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    output_root = Path(str(manifest["output_root"])).expanduser().resolve()
    return output_root / "cleanroom" / "leases"


def lease_path(lease_dir: str | Path, *, run_id: str, role: str, slot: str, session_id: str, pid: int) -> Path:
    root = Path(lease_dir).expanduser().resolve()
    return root / lease_filename_for_identity(run_id=run_id, role=role, slot=slot, session_id=session_id, pid=pid)


def build_lease_payload(
    *,
    run_id: str,
    role: str,
    slot: str,
    session_id: str,
    status: str,
    cwd: str | Path = "",
    home: str | Path = "",
    pid: int | None = None,
    pgid: int | None = None,
    ppid: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "kind": LEASE_KIND,
        "version": LEASE_VERSION,
        "run_id": run_id,
        "role": role,
        "slot": slot,
        "session_id": session_id,
        "pid": int(pid if pid is not None else os.getpid()),
        "pgid": int(pgid if pgid is not None else current_pgid()),
        "ppid": int(ppid if ppid is not None else current_ppid()),
        "cwd": str(Path(cwd).expanduser().resolve()) if str(cwd).strip() else "",
        "home": str(Path(home).expanduser().resolve()) if str(home).strip() else "",
        "status": status,
        "updated_at": iso_now(),
    }
    if extra:
        payload.update(extra)
    return payload


def write_manifest(path: Path, payload: dict[str, Any]) -> Path:
    atomic_write_json(path, payload)
    return path


def update_manifest(path: Path, patch: dict[str, Any]) -> dict[str, Any]:
    payload = read_json(path) if path.exists() else {}
    payload.update(patch)
    payload.setdefault("kind", MANIFEST_KIND)
    payload.setdefault("version", MANIFEST_VERSION)
    payload["updated_at"] = iso_now()
    atomic_write_json(path, payload)
    return payload


@dataclass
class LeaseHandle:
    path: Path

    def write(
        self,
        *,
        run_id: str,
        role: str,
        slot: str,
        session_id: str,
        status: str,
        cwd: str | Path = "",
        home: str | Path = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = build_lease_payload(
            run_id=run_id,
            role=role,
            slot=slot,
            session_id=session_id,
            status=status,
            cwd=cwd,
            home=home,
            extra=extra,
        )
        atomic_write_json(self.path, payload)
        return payload

    def remove(self) -> None:
        self.path.unlink(missing_ok=True)


def open_lease(
    lease_dir: str | Path,
    *,
    run_id: str,
    role: str,
    slot: str,
    session_id: str,
    pid: int | None = None,
) -> LeaseHandle:
    path = lease_path(
        lease_dir,
        run_id=run_id,
        role=role,
        slot=slot,
        session_id=session_id,
        pid=int(pid if pid is not None else os.getpid()),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return LeaseHandle(path=path)
