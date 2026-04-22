#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///

"""Deploy DebateClaw runtime helpers and default templates."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from debate_templates import (
    DEFAULT_RUNTIME_ROOT,
    build_parallel_judge_template,
    build_review_loop_template,
    template_name_for,
)


RUNTIME_HELPERS = (
    "prepare_debate.py",
    "debate_state.py",
    "debate_templates.py",
    "inspect_openclaw_env.py",
    "openclaw_debate_common.py",
    "ensure_openclaw_debate.py",
    "openclaw_debate_agent.py",
    "check_runtime.py",
    "install_clawteam.py",
    "bootstrap_status.py",
    "compile_runplan.py",
    "materialize_runplan.py",
    "launch_from_preset.py",
    "list_runs.py",
    "show_run.py",
    "rerender_run.py",
    "cleanup_run.py",
    "model_profiles.py",
    "apply_model_profile.py",
    "control_store.py",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy DebateClaw runtime helpers and default templates.")
    parser.add_argument(
        "--target-dir",
        default=str(Path.home() / ".clawteam" / "templates"),
        help="Target ClawTeam template directory.",
    )
    parser.add_argument(
        "--runtime-dir",
        default=DEFAULT_RUNTIME_ROOT,
        help="Target directory for DebateClaw runtime helper scripts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite changed files after creating timestamped backups.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without copying files.",
    )
    return parser.parse_args()


def backup_path(path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%SZ")
    return path.with_suffix(path.suffix + f".bak.{timestamp}")


def deploy_bytes(data: bytes, target: Path, *, force: bool, dry_run: bool, executable: bool = False) -> str:
    if not target.exists():
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            if executable:
                target.chmod(target.stat().st_mode | 0o755)
        return f"created {target}"

    if sha256_bytes(data) == sha256_file(target):
        return f"unchanged {target}"

    if not force:
        return f"conflict {target}"

    backup = backup_path(target)
    if not dry_run:
        shutil.copy2(target, backup)
        target.write_bytes(data)
        if executable:
            target.chmod(target.stat().st_mode | 0o755)
    return f"updated {target} (backup: {backup.name})"


def main() -> int:
    args = parse_args()
    template_dir = Path(args.target_dir).expanduser().resolve()
    runtime_dir = Path(args.runtime_dir).expanduser().resolve()
    template_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    results = []
    conflicts = 0

    for helper_name in RUNTIME_HELPERS:
        source = SCRIPT_DIR / helper_name
        if not source.is_file():
            raise SystemExit(f"Runtime helper is missing: {source}")
        result = deploy_bytes(
            source.read_bytes(),
            runtime_dir / helper_name,
            force=args.force,
            dry_run=args.dry_run,
            executable=helper_name.endswith(".py"),
        )
        if result.startswith("conflict "):
            conflicts += 1
        results.append(result)

    default_parallel = build_parallel_judge_template(
        name=template_name_for("parallel-judge"),
        proposer_count=4,
        command="codex",
        backend="tmux",
        runtime_root=os.path.expanduser(args.runtime_dir),
    )
    default_review = build_review_loop_template(
        name=template_name_for("review-loop"),
        proposer_count=4,
        max_review_rounds=5,
        max_rebuttal_rounds=5,
        command="codex",
        backend="tmux",
        runtime_root=os.path.expanduser(args.runtime_dir),
    )

    for template_name, template_text in (
        (template_name_for("parallel-judge"), default_parallel),
        (template_name_for("review-loop"), default_review),
    ):
        result = deploy_bytes(
            template_text.encode("utf-8"),
            template_dir / f"{template_name}.toml",
            force=args.force,
            dry_run=args.dry_run,
        )
        if result.startswith("conflict "):
            conflicts += 1
        results.append(result)

    print(f"Template dir: {template_dir}")
    print(f"Runtime dir: {runtime_dir}")
    for result in results:
        print(f"- {result}")

    if conflicts:
        print("Some DebateClaw runtime assets already existed with different contents. Re-run with --force to overwrite them.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
