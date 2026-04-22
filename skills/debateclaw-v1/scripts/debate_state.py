#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///

"""SQLite-backed shared state runtime for DebateClaw."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


VALID_WORKFLOWS = ("parallel-judge", "review-loop", "chemqa-review")
VALID_PHASES = ("propose", "review", "rebuttal", "done")
CHEMQA_CANDIDATE_OWNER = "proposer-1"
CHEMQA_REVIEWER_LANES = ("proposer-2", "proposer-3", "proposer-4", "proposer-5")


@dataclass(frozen=True)
class DebateConfig:
    team_name: str
    workflow: str
    goal: str
    evidence_policy: str
    proposer_count: int
    max_review_rounds: int
    max_rebuttal_rounds: int
    max_epochs: int


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_data_dir() -> Path:
    custom = Path.cwd()
    _ = custom
    env_value = Path.home()
    _ = env_value
    from_env = None
    if "CLAWTEAM_DATA_DIR" in os_environ():
        from_env = os_environ()["CLAWTEAM_DATA_DIR"].strip()
    if from_env:
        path = Path(from_env).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    config_path = Path.home() / ".clawteam" / "config.json"
    if config_path.is_file():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            cfg_path = str(data.get("data_dir", "")).strip()
            if cfg_path:
                path = Path(cfg_path).expanduser()
                path.mkdir(parents=True, exist_ok=True)
                return path
        except Exception:
            pass

    path = Path.home() / ".clawteam"
    path.mkdir(parents=True, exist_ok=True)
    return path


def os_environ() -> dict[str, str]:
    import os

    return os.environ


def team_dir(team_name: str) -> Path:
    path = resolve_data_dir() / "teams" / team_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def db_path(team_name: str) -> Path:
    path = team_dir(team_name) / "debate" / "state.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def debate_runtime_dir(team_name: str) -> Path:
    path = team_dir(team_name) / "debate"
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_root(team_name: str) -> Path:
    path = debate_runtime_dir(team_name) / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def connect(team_name: str) -> sqlite3.Connection:
    path = db_path(team_name)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _ensure_columns(conn: sqlite3.Connection, *, table: str, columns: dict[str, str]) -> None:
    existing = {
        str(row["name"])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for name, ddl in columns.items():
        if name in existing:
            continue
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS agents (
            name TEXT PRIMARY KEY,
            idx INTEGER NOT NULL,
            role TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS proposals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            epoch INTEGER NOT NULL,
            proposer TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            status TEXT NOT NULL,
            failure_reason TEXT NOT NULL DEFAULT '',
            fingerprint TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(epoch, proposer)
        );

        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            epoch INTEGER NOT NULL,
            review_round INTEGER NOT NULL,
            reviewer TEXT NOT NULL,
            target_proposer TEXT NOT NULL,
            target_proposal_id INTEGER NOT NULL,
            blocking INTEGER NOT NULL,
            body TEXT NOT NULL,
            attack_points_json TEXT NOT NULL,
            novel_blocking_points_json TEXT NOT NULL,
            synthetic INTEGER NOT NULL DEFAULT 0,
            submitted_by TEXT NOT NULL DEFAULT '',
            synthetic_reason TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            UNIQUE(epoch, review_round, reviewer, target_proposer)
        );

        CREATE TABLE IF NOT EXISTS rebuttals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            epoch INTEGER NOT NULL,
            rebuttal_round INTEGER NOT NULL,
            proposer TEXT NOT NULL,
            proposal_id INTEGER NOT NULL,
            conceded_failure INTEGER NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(epoch, rebuttal_round, proposer)
        );

        CREATE TABLE IF NOT EXISTS attack_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_proposal_id INTEGER NOT NULL,
            attack_key TEXT NOT NULL,
            attack_text TEXT NOT NULL,
            first_epoch INTEGER NOT NULL,
            first_review_round INTEGER NOT NULL,
            first_reviewer TEXT NOT NULL,
            UNIQUE(target_proposal_id, attack_key)
        );

        CREATE TABLE IF NOT EXISTS artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_type TEXT NOT NULL,
            record_id INTEGER NOT NULL,
            epoch INTEGER NOT NULL,
            round_index INTEGER NOT NULL DEFAULT 0,
            agent TEXT NOT NULL,
            target_proposer TEXT NOT NULL DEFAULT '',
            source_path TEXT NOT NULL,
            archive_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(record_type, record_id)
        );
        """
    )
    _ensure_columns(
        conn,
        table="reviews",
        columns={
            "synthetic": "INTEGER NOT NULL DEFAULT 0",
            "submitted_by": "TEXT NOT NULL DEFAULT ''",
            "synthetic_reason": "TEXT NOT NULL DEFAULT ''",
        },
    )


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO meta(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (key, value),
    )


def get_meta(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return str(row["value"]) if row else default


def load_meta(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT key, value FROM meta ORDER BY key").fetchall()
    return {str(row["key"]): str(row["value"]) for row in rows}


def get_phase_targets(conn: sqlite3.Connection) -> list[str]:
    raw = get_meta(conn, "phase_targets_json", "[]")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed]


def set_phase_targets(conn: sqlite3.Connection, targets: list[str]) -> None:
    set_meta(conn, "phase_targets_json", json.dumps(targets))


def agent_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT name FROM agents ORDER BY idx").fetchall()
    return [str(row["name"]) for row in rows]


def normalize_text(value: str) -> str:
    lowered = value.lower()
    collapsed = "".join(char if char.isalnum() else " " for char in lowered)
    return " ".join(collapsed.split())


def proposal_fingerprint(body: str) -> str:
    return hashlib.sha256(normalize_text(body).encode("utf-8")).hexdigest()


def parse_title_and_body(path: Path) -> tuple[str, str]:
    body = path.read_text(encoding="utf-8").strip()
    if not body:
        raise SystemExit(f"File is empty: {path}")

    title = ""
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("title:"):
            title = stripped.split(":", 1)[1].strip()
            break
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            break
        title = stripped
        break

    if not title:
        raise SystemExit(f"Could not determine a title from: {path}")
    return title, body


def parse_attack_points(body: str) -> list[str]:
    points = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            points.append(stripped[2:].strip())
        elif stripped.startswith("* "):
            points.append(stripped[2:].strip())
    if points:
        return [point for point in points if point]

    fallback = [line.strip() for line in body.splitlines() if line.strip()]
    if not fallback:
        return []
    return fallback[:3]


def active_proposals(conn: sqlite3.Connection, epoch: int | None = None) -> list[sqlite3.Row]:
    target_epoch = epoch if epoch is not None else current_epoch(conn)
    return conn.execute(
        """
        SELECT *
        FROM proposals
        WHERE epoch = ? AND status = 'active'
        ORDER BY proposer
        """,
        (target_epoch,),
    ).fetchall()


def current_epoch(conn: sqlite3.Connection) -> int:
    return int(get_meta(conn, "epoch", "1"))


def current_phase(conn: sqlite3.Connection) -> str:
    return get_meta(conn, "phase", "propose")


def status_value(conn: sqlite3.Connection) -> str:
    return get_meta(conn, "status", "running")


def review_round_value(conn: sqlite3.Connection) -> int:
    return int(get_meta(conn, "review_round", "0"))


def rebuttal_round_value(conn: sqlite3.Connection) -> int:
    return int(get_meta(conn, "rebuttal_round", "0"))


def max_epochs_value(conn: sqlite3.Connection) -> int:
    return int(get_meta(conn, "max_epochs", "1"))


def current_proposal_for(conn: sqlite3.Connection, proposer: str, epoch: int | None = None) -> sqlite3.Row | None:
    target_epoch = epoch if epoch is not None else current_epoch(conn)
    return conn.execute(
        """
        SELECT *
        FROM proposals
        WHERE epoch = ? AND proposer = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (target_epoch, proposer),
    ).fetchone()


def reviews_for_round(conn: sqlite3.Connection, target_proposer: str, epoch: int, review_round: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT *
        FROM reviews
        WHERE epoch = ? AND review_round = ? AND target_proposer = ?
        ORDER BY reviewer
        """,
        (epoch, review_round, target_proposer),
    ).fetchall()


def prior_proposals_for_agent(conn: sqlite3.Connection, proposer: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id, epoch, proposer, title, body, status, failure_reason, created_at, updated_at
        FROM proposals
        WHERE proposer = ?
        ORDER BY epoch, id
        """,
        (proposer,),
    ).fetchall()


def json_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def serialize_proposal_row(
    row: sqlite3.Row,
    *,
    include_body: bool,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "epoch": int(row["epoch"]),
        "proposer": str(row["proposer"]),
        "title": str(row["title"]),
        "status": str(row["status"]),
        "failure_reason": str(row["failure_reason"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }
    if include_body:
        payload["body"] = str(row["body"])
    if artifact:
        payload["artifact"] = artifact
    return payload


def serialize_review_row(
    row: sqlite3.Row,
    *,
    include_body: bool,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "epoch": int(row["epoch"]),
        "review_round": int(row["review_round"]),
        "reviewer": str(row["reviewer"]),
        "target_proposer": str(row["target_proposer"]),
        "blocking": bool(row["blocking"]),
        "attack_points": json_list(row["attack_points_json"]),
        "novel_blocking_points": json_list(row["novel_blocking_points_json"]),
        "synthetic": bool(row["synthetic"]) if "synthetic" in row.keys() else False,
        "submitted_by": str(row["submitted_by"]) if "submitted_by" in row.keys() else "",
        "synthetic_reason": str(row["synthetic_reason"]) if "synthetic_reason" in row.keys() else "",
    }
    if "created_at" in row.keys():
        payload["created_at"] = str(row["created_at"])
    if include_body:
        payload["body"] = str(row["body"])
    if artifact:
        payload["artifact"] = artifact
    return payload


def serialize_rebuttal_row(
    row: sqlite3.Row,
    *,
    include_body: bool,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "epoch": int(row["epoch"]),
        "rebuttal_round": int(row["rebuttal_round"]),
        "proposer": str(row["proposer"]),
        "conceded_failure": bool(row["conceded_failure"]),
    }
    if "created_at" in row.keys():
        payload["created_at"] = str(row["created_at"])
    if include_body:
        payload["body"] = str(row["body"])
    if artifact:
        payload["artifact"] = artifact
    return payload


def attack_registry_payload(conn: sqlite3.Connection, proposal_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT attack_text, first_epoch, first_review_round, first_reviewer
        FROM attack_registry
        WHERE target_proposal_id = ?
        ORDER BY first_epoch, first_review_round, first_reviewer, id
        """,
        (proposal_id,),
    ).fetchall()
    return [
        {
            "attack_text": str(row["attack_text"]),
            "first_epoch": int(row["first_epoch"]),
            "first_review_round": int(row["first_review_round"]),
            "first_reviewer": str(row["first_reviewer"]),
        }
        for row in rows
    ]


def proposal_context_payload(
    conn: sqlite3.Connection,
    proposal: sqlite3.Row,
    *,
    review_round_limit: int | None = None,
    rebuttal_round_limit: int | None = None,
) -> dict[str, Any]:
    proposal_id = int(proposal["id"])

    review_query = """
        SELECT id, epoch, review_round, reviewer, target_proposer, blocking, body, attack_points_json,
               novel_blocking_points_json, synthetic, submitted_by, synthetic_reason, created_at
        FROM reviews
        WHERE target_proposal_id = ?
    """
    review_params: list[Any] = [proposal_id]
    if review_round_limit is not None:
        review_query += " AND review_round <= ?"
        review_params.append(review_round_limit)
    review_query += " ORDER BY review_round, reviewer"
    review_rows = conn.execute(review_query, tuple(review_params)).fetchall()

    rebuttal_query = """
        SELECT id, epoch, rebuttal_round, proposer, conceded_failure, body, created_at
        FROM rebuttals
        WHERE proposal_id = ?
    """
    rebuttal_params: list[Any] = [proposal_id]
    if rebuttal_round_limit is not None:
        rebuttal_query += " AND rebuttal_round <= ?"
        rebuttal_params.append(rebuttal_round_limit)
    rebuttal_query += " ORDER BY rebuttal_round, proposer"
    rebuttal_rows = conn.execute(rebuttal_query, tuple(rebuttal_params)).fetchall()

    return {
        "proposal": serialize_proposal_row(
            proposal,
            include_body=True,
            artifact=artifact_metadata_for(conn, record_type="proposal", record_id=int(proposal["id"])),
        ),
        "review_history": [
            serialize_review_row(
                row,
                include_body=True,
                artifact=artifact_metadata_for(conn, record_type="review", record_id=int(row["id"])),
            )
            for row in review_rows
        ],
        "rebuttal_history": [
            serialize_rebuttal_row(
                row,
                include_body=True,
                artifact=artifact_metadata_for(conn, record_type="rebuttal", record_id=int(row["id"])),
            )
            for row in rebuttal_rows
        ],
        "attack_registry": attack_registry_payload(conn, proposal_id),
    }


def _yaml_mapping_from_body(body: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(body)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_revision_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    normalized: dict[str, Any] = {}
    for key in ("item_id", "severity", "finding", "requested_change"):
        value = item.get(key)
        if value not in (None, ""):
            normalized[key] = str(value)
    return normalized or None


def _revision_item_key(item: dict[str, Any]) -> str:
    if item.get("item_id"):
        return f"item:{item['item_id']}"
    return f"finding:{item.get('finding', '')}|change:{item.get('requested_change', '')}"


def latest_proposal_before_epoch(conn: sqlite3.Connection, proposer: str, epoch: int) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT *
        FROM proposals
        WHERE proposer = ? AND epoch < ?
        ORDER BY epoch DESC, id DESC
        LIMIT 1
        """,
        (proposer, epoch),
    ).fetchone()


def chemqa_revision_context(conn: sqlite3.Connection, *, epoch: int) -> dict[str, Any] | None:
    if epoch <= 1:
        return None
    previous = current_proposal_for(conn, CHEMQA_CANDIDATE_OWNER, epoch - 1)
    if not previous:
        previous = latest_proposal_before_epoch(conn, CHEMQA_CANDIDATE_OWNER, epoch)
    if not previous:
        return None

    proposal_ctx = proposal_context_payload(conn, previous)
    proposal_payload = _yaml_mapping_from_body(str(proposal_ctx["proposal"].get("body") or ""))

    required_revision_items: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    review_feedback: list[dict[str, Any]] = []
    for review in proposal_ctx["review_history"]:
        review_payload = _yaml_mapping_from_body(str(review.get("body") or ""))
        review_items: list[dict[str, Any]] = []
        for raw_item in review_payload.get("review_items") or []:
            item = _normalize_revision_item(raw_item)
            if not item:
                continue
            review_items.append(item)
            dedupe_key = _revision_item_key(item)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            required_revision_items.append({
                **item,
                "from_reviewer": str(review.get("reviewer") or ""),
                "blocking": bool(review.get("blocking")),
                "review_round": int(review.get("review_round") or 0),
            })
        review_feedback.append({
            "reviewer": str(review.get("reviewer") or ""),
            "verdict": str(review_payload.get("verdict") or ""),
            "summary": str(review_payload.get("summary") or ""),
            "blocking": bool(review.get("blocking")),
            "review_round": int(review.get("review_round") or 0),
            "review_items": review_items,
        })

    latest_rebuttal = proposal_ctx["rebuttal_history"][-1] if proposal_ctx["rebuttal_history"] else None
    rebuttal_context = None
    if latest_rebuttal:
        rebuttal_payload = _yaml_mapping_from_body(str(latest_rebuttal.get("body") or ""))
        response_items: list[dict[str, Any]] = []
        for raw_item in rebuttal_payload.get("response_items") or []:
            item = _normalize_revision_item(raw_item)
            if item:
                response_items.append(item)
        rebuttal_context = {
            "conceded_failure": bool(latest_rebuttal.get("conceded_failure")),
            "rebuttal_round": int(latest_rebuttal.get("rebuttal_round") or 0),
            "response_summary": str(rebuttal_payload.get("response_summary") or ""),
            "updated_direct_answer": str(rebuttal_payload.get("updated_direct_answer") or ""),
            "response_items": response_items,
        }

    guard = "Do not resubmit the prior failed candidate unchanged; address the required revision items explicitly."
    if rebuttal_context and rebuttal_context.get("conceded_failure"):
        guard = "The prior candidate was conceded as failed. Do not repeat the conceded reasoning without an explicit corrected derivation and clear item-by-item fixes."

    return {
        "source_epoch": int(previous["epoch"]),
        "previous_candidate_status": str(previous["status"]),
        "previous_failure_reason": str(previous["failure_reason"]),
        "previous_direct_answer": str(proposal_payload.get("direct_answer") or ""),
        "previous_summary": str(proposal_payload.get("summary") or ""),
        "review_feedback": review_feedback,
        "required_revision_items": required_revision_items,
        "attack_registry": proposal_ctx["attack_registry"],
        "latest_rebuttal": rebuttal_context,
        "repeat_failure_guard": guard,
    }


def file_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def archive_path_for_record(
    team_name: str,
    *,
    record_type: str,
    epoch: int,
    agent: str,
    round_index: int = 0,
    target_proposer: str = "",
) -> Path:
    root = artifact_root(team_name)
    if record_type == "proposal":
        return root / "proposals" / f"epoch-{epoch:03d}" / f"{agent}.md"
    if record_type == "review":
        return (
            root
            / "reviews"
            / f"epoch-{epoch:03d}"
            / f"round-{round_index:02d}"
            / target_proposer
            / f"{agent}.md"
        )
    if record_type == "rebuttal":
        return root / "rebuttals" / f"epoch-{epoch:03d}" / f"round-{round_index:02d}" / f"{agent}.md"
    raise SystemExit(f"Unsupported artifact type: {record_type}")


def store_artifact(
    conn: sqlite3.Connection,
    *,
    team_name: str,
    record_type: str,
    record_id: int,
    epoch: int,
    agent: str,
    source_path: Path,
    body: str,
    round_index: int = 0,
    target_proposer: str = "",
    created_at: str,
) -> dict[str, Any]:
    archive_path = archive_path_for_record(
        team_name,
        record_type=record_type,
        epoch=epoch,
        agent=agent,
        round_index=round_index,
        target_proposer=target_proposer,
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(body, encoding="utf-8")

    payload = {
        "source_path": str(source_path.expanduser().resolve()),
        "archive_path": str(archive_path),
        "sha256": file_sha256(body),
        "size_bytes": len(body.encode("utf-8")),
        "created_at": created_at,
    }
    conn.execute(
        """
        INSERT INTO artifacts(
            record_type,
            record_id,
            epoch,
            round_index,
            agent,
            target_proposer,
            source_path,
            archive_path,
            sha256,
            size_bytes,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record_type,
            record_id,
            epoch,
            round_index,
            agent,
            target_proposer,
            payload["source_path"],
            payload["archive_path"],
            payload["sha256"],
            payload["size_bytes"],
            created_at,
        ),
    )
    return payload


def artifact_metadata_for(conn: sqlite3.Connection, *, record_type: str, record_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT source_path, archive_path, sha256, size_bytes, created_at
        FROM artifacts
        WHERE record_type = ? AND record_id = ?
        """,
        (record_type, record_id),
    ).fetchone()
    if not row:
        return None
    return {
        "source_path": str(row["source_path"]),
        "archive_path": str(row["archive_path"]),
        "sha256": str(row["sha256"]),
        "size_bytes": int(row["size_bytes"]),
        "created_at": str(row["created_at"]),
    }


def init_debate_state(config: DebateConfig, *, reset: bool) -> Path:
    path = db_path(config.team_name)
    if reset and path.exists():
        path.unlink()

    with connect(config.team_name) as conn:
        ensure_schema(conn)
        conn.execute("DELETE FROM meta")
        conn.execute("DELETE FROM agents")
        conn.execute("DELETE FROM proposals")
        conn.execute("DELETE FROM reviews")
        conn.execute("DELETE FROM rebuttals")
        conn.execute("DELETE FROM attack_registry")
        conn.execute("DELETE FROM artifacts")

        set_meta(conn, "team_name", config.team_name)
        set_meta(conn, "workflow", config.workflow)
        set_meta(conn, "goal", config.goal)
        set_meta(conn, "evidence_policy", config.evidence_policy)
        set_meta(conn, "proposer_count", str(config.proposer_count))
        set_meta(conn, "max_review_rounds", str(config.max_review_rounds))
        set_meta(conn, "max_rebuttal_rounds", str(config.max_rebuttal_rounds))
        set_meta(conn, "max_epochs", str(config.max_epochs))
        set_meta(conn, "terminal_state", "")
        set_meta(conn, "failure_reason", "")
        set_meta(conn, "status", "running")
        set_meta(conn, "phase", "propose")
        set_meta(conn, "epoch", "1")
        set_meta(conn, "review_round", "0")
        set_meta(conn, "rebuttal_round", "0")
        set_meta(conn, "final_candidates_json", "[]")

        proposers = [f"proposer-{index}" for index in range(1, config.proposer_count + 1)]
        if config.workflow == "chemqa-review":
            set_phase_targets(conn, [CHEMQA_CANDIDATE_OWNER])
        else:
            set_phase_targets(conn, proposers)
        for index, proposer in enumerate(proposers, start=1):
            conn.execute(
                "INSERT INTO agents(name, idx, role) VALUES(?, ?, ?)",
                (proposer, index, "proposer"),
            )

        conn.commit()
    return path


@contextmanager
def transaction(conn: sqlite3.Connection):
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


def _require_initialized(conn: sqlite3.Connection) -> None:
    if not get_meta(conn, "workflow"):
        raise SystemExit("Debate state is not initialized for this team.")


def submit_proposal(conn: sqlite3.Connection, *, agent: str, file_path: Path) -> dict[str, Any]:
    _require_initialized(conn)
    if current_phase(conn) != "propose":
        raise SystemExit("submit-proposal is only allowed during the propose phase.")
    if status_value(conn) != "running":
        raise SystemExit("Debate is already done.")
    if get_meta(conn, "workflow") == "chemqa-review" and agent != CHEMQA_CANDIDATE_OWNER:
        raise SystemExit("In chemqa-review, only proposer-1 may submit a proposal.")

    title, body = parse_title_and_body(file_path)
    epoch = current_epoch(conn)
    fingerprint = proposal_fingerprint(body)

    duplicate = conn.execute(
        """
        SELECT epoch, title
        FROM proposals
        WHERE proposer = ? AND fingerprint = ?
        ORDER BY epoch DESC
        LIMIT 1
        """,
        (agent, fingerprint),
    ).fetchone()
    if duplicate:
        raise SystemExit(
            f"Proposal matches a prior submission from epoch {duplicate['epoch']}: {duplicate['title']}"
        )

    existing = current_proposal_for(conn, agent, epoch)
    if existing:
        raise SystemExit(f"{agent} already submitted a proposal for epoch {epoch}.")

    timestamp = now_iso()
    with transaction(conn):
        cursor = conn.execute(
            """
            INSERT INTO proposals(epoch, proposer, title, body, status, failure_reason, fingerprint, created_at, updated_at)
            VALUES(?, ?, ?, ?, 'active', '', ?, ?, ?)
            """,
            (epoch, agent, title, body, fingerprint, timestamp, timestamp),
        )
        artifact = store_artifact(
            conn,
            team_name=get_meta(conn, "team_name"),
            record_type="proposal",
            record_id=int(cursor.lastrowid),
            epoch=epoch,
            agent=agent,
            source_path=file_path,
            body=body,
            created_at=timestamp,
        )
    return {
        "team": get_meta(conn, "team_name"),
        "epoch": epoch,
        "agent": agent,
        "title": title,
        "status": "accepted",
        "artifact": artifact,
    }


def submit_review(
    conn: sqlite3.Connection,
    *,
    agent: str,
    target: str,
    blocking: bool,
    file_path: Path,
    synthetic: bool = False,
    submitted_by: str = "",
    synthetic_reason: str = "",
) -> dict[str, Any]:
    _require_initialized(conn)
    if current_phase(conn) != "review":
        raise SystemExit("submit-review is only allowed during the review phase.")
    if agent == target:
        raise SystemExit("A proposer cannot review its own proposal.")
    workflow = get_meta(conn, "workflow")
    if workflow == "chemqa-review":
        if agent not in CHEMQA_REVIEWER_LANES:
            raise SystemExit("In chemqa-review, only fixed reviewer lanes may submit reviews.")
        if target != CHEMQA_CANDIDATE_OWNER:
            raise SystemExit("In chemqa-review, reviews must target proposer-1.")

    epoch = current_epoch(conn)
    review_round = review_round_value(conn)
    targets = set(get_phase_targets(conn))
    if target not in targets:
        raise SystemExit(f"{target} is not an active review target in this round.")

    proposal = current_proposal_for(conn, target, epoch)
    if not proposal or proposal["status"] != "active":
        raise SystemExit(f"{target} does not currently have an active proposal.")

    body = file_path.read_text(encoding="utf-8").strip()
    if not body:
        raise SystemExit(f"Review file is empty: {file_path}")
    attack_points = parse_attack_points(body)
    submitted_by_value = submitted_by.strip()
    synthetic_reason_value = synthetic_reason.strip()
    if synthetic:
        if not submitted_by_value:
            raise SystemExit("Synthetic reviews require --submitted-by for auditability.")
        if not synthetic_reason_value:
            raise SystemExit("Synthetic reviews require --reason for auditability.")

    existing = conn.execute(
        """
        SELECT 1
        FROM reviews
        WHERE epoch = ? AND review_round = ? AND reviewer = ? AND target_proposer = ?
        """,
        (epoch, review_round, agent, target),
    ).fetchone()
    if existing:
        raise SystemExit(f"{agent} already submitted a review for {target} in round {review_round}.")

    novel_points: list[str] = []
    timestamp = now_iso()
    with transaction(conn):
        for point in attack_points:
            key = normalize_text(point)
            if not key:
                continue
            if not blocking:
                continue
            existing_attack = conn.execute(
                """
                SELECT 1
                FROM attack_registry
                WHERE target_proposal_id = ? AND attack_key = ?
                """,
                (int(proposal["id"]), key),
            ).fetchone()
            if existing_attack:
                continue
            conn.execute(
                """
                INSERT INTO attack_registry(target_proposal_id, attack_key, attack_text, first_epoch, first_review_round, first_reviewer)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (int(proposal["id"]), key, point, epoch, review_round, agent),
            )
            novel_points.append(point)

        cursor = conn.execute(
            """
            INSERT INTO reviews(
                epoch,
                review_round,
                reviewer,
                target_proposer,
                target_proposal_id,
                blocking,
                body,
                attack_points_json,
                novel_blocking_points_json,
                synthetic,
                submitted_by,
                synthetic_reason,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                epoch,
                review_round,
                agent,
                target,
                int(proposal["id"]),
                1 if blocking else 0,
                body,
                json.dumps(attack_points),
                json.dumps(novel_points),
                1 if synthetic else 0,
                submitted_by_value,
                synthetic_reason_value,
                timestamp,
            ),
        )
        artifact = store_artifact(
            conn,
            team_name=get_meta(conn, "team_name"),
            record_type="review",
            record_id=int(cursor.lastrowid),
            epoch=epoch,
            round_index=review_round,
            agent=agent,
            target_proposer=target,
            source_path=file_path,
            body=body,
            created_at=timestamp,
        )

    return {
        "team": get_meta(conn, "team_name"),
        "epoch": epoch,
        "review_round": review_round,
        "reviewer": agent,
        "target": target,
        "blocking": blocking,
        "attack_points": attack_points,
        "novel_blocking_points": novel_points,
        "synthetic": synthetic,
        "submitted_by": submitted_by_value,
        "synthetic_reason": synthetic_reason_value,
        "artifact": artifact,
    }


def submit_rebuttal(
    conn: sqlite3.Connection,
    *,
    agent: str,
    file_path: Path,
    concede: bool,
) -> dict[str, Any]:
    _require_initialized(conn)
    if current_phase(conn) != "rebuttal":
        raise SystemExit("submit-rebuttal is only allowed during the rebuttal phase.")
    if get_meta(conn, "workflow") == "chemqa-review" and agent != CHEMQA_CANDIDATE_OWNER:
        raise SystemExit("In chemqa-review, only proposer-1 may submit a rebuttal.")

    epoch = current_epoch(conn)
    rebuttal_round = rebuttal_round_value(conn)
    targets = set(get_phase_targets(conn))
    if agent not in targets:
        raise SystemExit(f"{agent} is not required to rebut in this round.")

    proposal = current_proposal_for(conn, agent, epoch)
    if not proposal:
        raise SystemExit(f"{agent} has no proposal for epoch {epoch}.")

    existing = conn.execute(
        """
        SELECT 1
        FROM rebuttals
        WHERE epoch = ? AND rebuttal_round = ? AND proposer = ?
        """,
        (epoch, rebuttal_round, agent),
    ).fetchone()
    if existing:
        raise SystemExit(f"{agent} already submitted a rebuttal in round {rebuttal_round}.")

    body = file_path.read_text(encoding="utf-8").strip()
    if not body:
        raise SystemExit(f"Rebuttal file is empty: {file_path}")

    timestamp = now_iso()
    with transaction(conn):
        cursor = conn.execute(
            """
            INSERT INTO rebuttals(epoch, rebuttal_round, proposer, proposal_id, conceded_failure, body, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (epoch, rebuttal_round, agent, int(proposal["id"]), 1 if concede else 0, body, timestamp),
        )
        artifact = store_artifact(
            conn,
            team_name=get_meta(conn, "team_name"),
            record_type="rebuttal",
            record_id=int(cursor.lastrowid),
            epoch=epoch,
            round_index=rebuttal_round,
            agent=agent,
            source_path=file_path,
            body=body,
            created_at=timestamp,
        )
        if concede:
            conn.execute(
                """
                UPDATE proposals
                SET status = 'failed', failure_reason = ?, updated_at = ?
                WHERE id = ?
                """,
                (body.splitlines()[0].strip()[:400], now_iso(), int(proposal["id"])),
            )

    return {
        "team": get_meta(conn, "team_name"),
        "epoch": epoch,
        "rebuttal_round": rebuttal_round,
        "agent": agent,
        "conceded_failure": concede,
        "artifact": artifact,
    }


def chemqa_candidate_proposal(conn: sqlite3.Connection, epoch: int | None = None) -> sqlite3.Row | None:
    return current_proposal_for(conn, CHEMQA_CANDIDATE_OWNER, epoch)


def chemqa_review_rows(conn: sqlite3.Connection, epoch: int, review_round: int) -> list[sqlite3.Row]:
    return [
        row for row in reviews_for_round(conn, CHEMQA_CANDIDATE_OWNER, epoch, review_round)
        if str(row["reviewer"]) in CHEMQA_REVIEWER_LANES
    ]


def chemqa_exited_reviewer_state(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    raw = get_meta(conn, "chemqa_exited_reviewers_json", "{}")
    try:
        payload = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    state: dict[str, dict[str, Any]] = {}
    for lane in CHEMQA_REVIEWER_LANES:
        item = payload.get(lane)
        if isinstance(item, dict):
            state[lane] = dict(item)
    return state



def chemqa_exited_reviewer_lanes(conn: sqlite3.Connection) -> list[str]:
    state = chemqa_exited_reviewer_state(conn)
    return [lane for lane in CHEMQA_REVIEWER_LANES if lane in state]



def chemqa_active_reviewer_lanes(conn: sqlite3.Connection) -> list[str]:
    exited = set(chemqa_exited_reviewer_lanes(conn))
    return [lane for lane in CHEMQA_REVIEWER_LANES if lane not in exited]



def chemqa_missing_reviewer_lanes(conn: sqlite3.Connection, epoch: int | None = None, review_round: int | None = None) -> list[str]:
    target_epoch = epoch if epoch is not None else current_epoch(conn)
    target_round = review_round if review_round is not None else review_round_value(conn)
    seen = {str(row["reviewer"]) for row in chemqa_review_rows(conn, target_epoch, target_round)}
    return [lane for lane in chemqa_active_reviewer_lanes(conn) if lane not in seen]


def unresolved_targets_from_reviews(conn: sqlite3.Connection) -> list[str]:
    epoch = current_epoch(conn)
    review_round = review_round_value(conn)
    if get_meta(conn, "workflow") == "chemqa-review":
        rows = chemqa_review_rows(conn, epoch, review_round)
        if any(int(row["blocking"]) == 1 for row in rows):
            return [CHEMQA_CANDIDATE_OWNER]
        return []
    unresolved = []
    for target in get_phase_targets(conn):
        proposal = current_proposal_for(conn, target, epoch)
        if not proposal or proposal["status"] != "active":
            continue
        rows = reviews_for_round(conn, target, epoch, review_round)
        if any(int(row["blocking"]) == 1 for row in rows):
            unresolved.append(target)
    return unresolved


def propose_phase_progress(conn: sqlite3.Connection) -> dict[str, Any]:
    epoch = current_epoch(conn)
    workflow = get_meta(conn, "workflow")
    if workflow == "chemqa-review":
        proposal = chemqa_candidate_proposal(conn, epoch)
        actual = 1 if proposal else 0
        return {
            "kind": "propose",
            "expected": 1,
            "actual": actual,
            "complete": actual == 1,
            "targets": [CHEMQA_CANDIDATE_OWNER],
        }
    expected = int(get_meta(conn, "proposer_count", "0"))
    actual_row = conn.execute(
        "SELECT COUNT(*) AS count FROM proposals WHERE epoch = ?",
        (epoch,),
    ).fetchone()
    actual = int(actual_row["count"])
    return {
        "kind": "propose",
        "expected": expected,
        "actual": actual,
        "complete": actual == expected,
    }


def review_phase_progress(conn: sqlite3.Connection) -> dict[str, Any]:
    epoch = current_epoch(conn)
    review_round = review_round_value(conn)
    targets = get_phase_targets(conn)
    workflow = get_meta(conn, "workflow")
    if workflow == "chemqa-review":
        rows = chemqa_review_rows(conn, epoch, review_round)
        actual = len(rows)
        blocking = sum(1 for row in rows if int(row["blocking"]) == 1)
        active_reviewer_lanes = chemqa_active_reviewer_lanes(conn)
        exited_reviewer_lanes = chemqa_exited_reviewer_lanes(conn)
        expected = len(active_reviewer_lanes)
        return {
            "kind": "review",
            "round": review_round,
            "expected": expected,
            "expected_original": len(CHEMQA_REVIEWER_LANES),
            "actual": actual,
            "complete": actual >= expected,
            "targets": [CHEMQA_CANDIDATE_OWNER],
            "missing_reviewer_lanes": chemqa_missing_reviewer_lanes(conn, epoch, review_round),
            "active_reviewer_lanes": active_reviewer_lanes,
            "exited_reviewer_lanes": exited_reviewer_lanes,
            "counts_by_target": [
                {
                    "target": CHEMQA_CANDIDATE_OWNER,
                    "submitted": actual,
                    "blocking": blocking,
                }
            ],
        }
    reviewers = agent_names(conn)
    expected = sum(1 for reviewer in reviewers for target in targets if reviewer != target)
    actual_row = conn.execute(
        "SELECT COUNT(*) AS count FROM reviews WHERE epoch = ? AND review_round = ?",
        (epoch, review_round),
    ).fetchone()
    actual = int(actual_row["count"])
    counts_by_target = [
        {
            "target": str(row["target_proposer"]),
            "submitted": int(row["count"]),
            "blocking": int(row["blocking_count"]),
        }
        for row in conn.execute(
            """
            SELECT target_proposer, COUNT(*) AS count, COALESCE(SUM(blocking), 0) AS blocking_count
            FROM reviews
            WHERE epoch = ? AND review_round = ?
            GROUP BY target_proposer
            ORDER BY target_proposer
            """,
            (epoch, review_round),
        ).fetchall()
    ]
    return {
        "kind": "review",
        "round": review_round,
        "expected": expected,
        "actual": actual,
        "complete": actual == expected,
        "targets": targets,
        "counts_by_target": counts_by_target,
    }


def rebuttal_phase_progress(conn: sqlite3.Connection) -> dict[str, Any]:
    epoch = current_epoch(conn)
    rebuttal_round = rebuttal_round_value(conn)
    targets = get_phase_targets(conn)
    actual_row = conn.execute(
        "SELECT COUNT(*) AS count FROM rebuttals WHERE epoch = ? AND rebuttal_round = ?",
        (epoch, rebuttal_round),
    ).fetchone()
    actual = int(actual_row["count"])
    if get_meta(conn, "workflow") == "chemqa-review":
        targets = [CHEMQA_CANDIDATE_OWNER]
    return {
        "kind": "rebuttal",
        "round": rebuttal_round,
        "expected": len(targets),
        "actual": actual,
        "complete": actual == len(targets),
        "targets": targets,
    }


def current_phase_progress(conn: sqlite3.Connection) -> dict[str, Any]:
    phase = current_phase(conn)
    if phase == "propose":
        return propose_phase_progress(conn)
    if phase == "review":
        return review_phase_progress(conn)
    if phase == "rebuttal":
        return rebuttal_phase_progress(conn)
    return {"kind": phase, "complete": phase == "done"}


def review_phase_complete(conn: sqlite3.Connection) -> bool:
    return bool(review_phase_progress(conn)["complete"])


def rebuttal_phase_complete(conn: sqlite3.Connection) -> bool:
    return bool(rebuttal_phase_progress(conn)["complete"])


def _set_done_with_candidates(conn: sqlite3.Connection, candidates: list[str]) -> dict[str, Any]:
    with transaction(conn):
        epoch = current_epoch(conn)
        timestamp = now_iso()
        if candidates:
            placeholders = ",".join("?" for _ in candidates)
            conn.execute(
                f"""
                UPDATE proposals
                SET status = 'candidate', updated_at = ?
                WHERE epoch = ? AND proposer IN ({placeholders})
                """,
                (timestamp, epoch, *candidates),
            )
            conn.execute(
                f"""
                UPDATE proposals
                SET status = 'superseded', updated_at = ?
                WHERE epoch = ? AND status = 'active' AND proposer NOT IN ({placeholders})
                """,
                (timestamp, epoch, *candidates),
            )
        else:
            conn.execute(
                """
                UPDATE proposals
                SET status = 'superseded', updated_at = ?
                WHERE epoch = ? AND status = 'active'
                """,
                (timestamp, epoch),
            )
        set_meta(conn, "phase", "done")
        set_meta(conn, "status", "done")
        set_meta(conn, "terminal_state", "completed")
        set_meta(conn, "failure_reason", "")
        set_meta(conn, "final_candidates_json", json.dumps(candidates))
    return {
        "team": get_meta(conn, "team_name"),
        "phase": "done",
        "status": "done",
        "terminal_state": "completed",
        "final_candidates": candidates,
    }


def _set_failed_done(conn: sqlite3.Connection, *, reason: str) -> dict[str, Any]:
    with transaction(conn):
        timestamp = now_iso()
        epoch = current_epoch(conn)
        conn.execute(
            """
            UPDATE proposals
            SET status = CASE WHEN status = 'active' THEN 'failed' ELSE status END,
                failure_reason = CASE WHEN status = 'active' AND failure_reason = '' THEN ? ELSE failure_reason END,
                updated_at = ?
            WHERE epoch = ?
            """,
            (reason[:400], timestamp, epoch),
        )
        set_meta(conn, "phase", "done")
        set_meta(conn, "status", "done")
        set_meta(conn, "terminal_state", "failed")
        set_meta(conn, "failure_reason", reason)
        set_meta(conn, "final_candidates_json", "[]")
    return {
        "team": get_meta(conn, "team_name"),
        "phase": "done",
        "status": "done",
        "terminal_state": "failed",
        "failure_reason": reason,
        "final_candidates": [],
    }


def advance_state(conn: sqlite3.Connection, *, agent: str) -> dict[str, Any]:
    _require_initialized(conn)
    if status_value(conn) == "done":
        return {
            "team": get_meta(conn, "team_name"),
            "status": "done",
            "phase": "done",
            "message": "Debate already completed.",
        }

    workflow = get_meta(conn, "workflow")
    phase = current_phase(conn)
    epoch = current_epoch(conn)
    proposer_count = int(get_meta(conn, "proposer_count", "0"))
    max_review_rounds = int(get_meta(conn, "max_review_rounds", "0"))
    max_rebuttal_rounds = int(get_meta(conn, "max_rebuttal_rounds", "0"))
    max_epochs = max_epochs_value(conn)

    if workflow == "parallel-judge":
        submitted = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM proposals
            WHERE epoch = ?
            """,
            (epoch,),
        ).fetchone()
        if phase == "propose" and int(submitted["count"]) == proposer_count:
            candidates = [row["proposer"] for row in conn.execute(
                "SELECT proposer FROM proposals WHERE epoch = ? ORDER BY proposer",
                (epoch,),
            ).fetchall()]
            return _set_done_with_candidates(conn, candidates)
        return {
            "team": get_meta(conn, "team_name"),
            "phase": phase,
            "status": status_value(conn),
            "message": "Parallel debate is still waiting for proposals.",
        }

    if workflow == "chemqa-review":
        if phase == "propose":
            proposal = chemqa_candidate_proposal(conn, epoch)
            if not proposal:
                return {
                    "team": get_meta(conn, "team_name"),
                    "phase": phase,
                    "message": "Still waiting for proposer-1 to submit the candidate proposal.",
                }
            with transaction(conn):
                set_meta(conn, "phase", "review")
                set_meta(conn, "review_round", "1")
                set_phase_targets(conn, [CHEMQA_CANDIDATE_OWNER])
            return {
                "team": get_meta(conn, "team_name"),
                "phase": "review",
                "epoch": epoch,
                "review_round": 1,
                "targets": [CHEMQA_CANDIDATE_OWNER],
                "message": f"{agent} advanced the ChemQA debate into review round 1.",
            }

        if phase == "review":
            if not review_phase_complete(conn):
                return {
                    "team": get_meta(conn, "team_name"),
                    "phase": phase,
                    "epoch": epoch,
                    "review_round": review_round_value(conn),
                    "message": "ChemQA review round is not complete yet.",
                }
            unresolved_targets = unresolved_targets_from_reviews(conn)
            if not unresolved_targets:
                result = _set_done_with_candidates(conn, [CHEMQA_CANDIDATE_OWNER])
                result["message"] = f"{agent} finished the ChemQA debate because all required reviewer lanes completed non-blocking reviews."
                return result
            if rebuttal_round_value(conn) >= max_rebuttal_rounds:
                result = _set_done_with_candidates(conn, [CHEMQA_CANDIDATE_OWNER])
                result["message"] = f"{agent} finished the ChemQA debate because the rebuttal round budget is exhausted."
                return result
            next_rebuttal_round = rebuttal_round_value(conn) + 1
            with transaction(conn):
                set_meta(conn, "phase", "rebuttal")
                set_meta(conn, "rebuttal_round", str(next_rebuttal_round))
                set_phase_targets(conn, [CHEMQA_CANDIDATE_OWNER])
            return {
                "team": get_meta(conn, "team_name"),
                "phase": "rebuttal",
                "epoch": epoch,
                "rebuttal_round": next_rebuttal_round,
                "targets": [CHEMQA_CANDIDATE_OWNER],
                "message": f"{agent} advanced the ChemQA debate into rebuttal round {next_rebuttal_round}.",
            }

        if phase == "rebuttal":
            if not rebuttal_phase_complete(conn):
                return {
                    "team": get_meta(conn, "team_name"),
                    "phase": phase,
                    "epoch": epoch,
                    "rebuttal_round": rebuttal_round_value(conn),
                    "message": "ChemQA rebuttal round is not complete yet.",
                }

            survivors = [row["proposer"] for row in active_proposals(conn, epoch)]
            if not survivors:
                if epoch >= max_epochs:
                    result = _set_failed_done(
                        conn,
                        reason=f"max_epochs_exhausted_after_candidate_failures (epoch={epoch}, max_epochs={max_epochs})",
                    )
                    result["message"] = f"{agent} finished the ChemQA debate as failed because proposer-1 exhausted the max epoch budget ({max_epochs})."
                    return result
                next_epoch = epoch + 1
                with transaction(conn):
                    set_meta(conn, "epoch", str(next_epoch))
                    set_meta(conn, "phase", "propose")
                    set_meta(conn, "review_round", "0")
                    set_meta(conn, "rebuttal_round", "0")
                    set_phase_targets(conn, [CHEMQA_CANDIDATE_OWNER])
                return {
                    "team": get_meta(conn, "team_name"),
                    "phase": "propose",
                    "epoch": next_epoch,
                    "targets": [CHEMQA_CANDIDATE_OWNER],
                    "message": f"{agent} started ChemQA epoch {next_epoch} because proposer-1 candidate failed in epoch {epoch}; proposer-1 must submit a revised candidate addressing prior review items.",
                }

            if review_round_value(conn) >= max_review_rounds:
                result = _set_done_with_candidates(conn, [CHEMQA_CANDIDATE_OWNER])
                result["message"] = f"{agent} finished the ChemQA debate because the review round budget is exhausted."
                return result
            next_review_round = review_round_value(conn) + 1
            with transaction(conn):
                set_meta(conn, "phase", "review")
                set_meta(conn, "review_round", str(next_review_round))
                set_phase_targets(conn, [CHEMQA_CANDIDATE_OWNER])
            return {
                "team": get_meta(conn, "team_name"),
                "phase": "review",
                "epoch": epoch,
                "review_round": next_review_round,
                "targets": [CHEMQA_CANDIDATE_OWNER],
                "message": f"{agent} advanced the ChemQA debate into review round {next_review_round}.",
            }

        raise SystemExit(f"Cannot advance from phase: {phase}")

    if workflow != "review-loop":
        raise SystemExit(f"Unsupported workflow: {workflow}")

    if phase == "propose":
        submitted = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM proposals
            WHERE epoch = ?
            """,
            (epoch,),
        ).fetchone()
        if int(submitted["count"]) < proposer_count:
            return {
                "team": get_meta(conn, "team_name"),
                "phase": phase,
                "message": "Still waiting for one or more proposers to submit.",
            }
        targets = [row["proposer"] for row in active_proposals(conn, epoch)]
        with transaction(conn):
            set_meta(conn, "phase", "review")
            set_meta(conn, "review_round", "1")
            set_phase_targets(conn, targets)
        return {
            "team": get_meta(conn, "team_name"),
            "phase": "review",
            "epoch": epoch,
            "review_round": 1,
            "targets": targets,
            "message": f"{agent} advanced the debate into review round 1.",
        }

    if phase == "review":
        if not review_phase_complete(conn):
            return {
                "team": get_meta(conn, "team_name"),
                "phase": phase,
                "epoch": epoch,
                "review_round": review_round_value(conn),
                "message": "Review round is not complete yet.",
            }

        unresolved_targets = unresolved_targets_from_reviews(conn)
        if not unresolved_targets:
            survivors = [row["proposer"] for row in active_proposals(conn, epoch)]
            result = _set_done_with_candidates(conn, survivors)
            result["message"] = f"{agent} finished the debate because every active proposal completed the review round without blocking objections."
            return result

        if rebuttal_round_value(conn) >= max_rebuttal_rounds:
            survivors = [row["proposer"] for row in active_proposals(conn, epoch)]
            result = _set_done_with_candidates(conn, survivors)
            result["message"] = f"{agent} finished the debate because the rebuttal round budget is exhausted."
            return result

        next_rebuttal_round = rebuttal_round_value(conn) + 1
        with transaction(conn):
            set_meta(conn, "phase", "rebuttal")
            set_meta(conn, "rebuttal_round", str(next_rebuttal_round))
            set_phase_targets(conn, unresolved_targets)
        return {
            "team": get_meta(conn, "team_name"),
            "phase": "rebuttal",
            "epoch": epoch,
            "rebuttal_round": next_rebuttal_round,
            "targets": unresolved_targets,
            "message": f"{agent} advanced the debate into rebuttal round {next_rebuttal_round}.",
        }

    if phase == "rebuttal":
        if not rebuttal_phase_complete(conn):
            return {
                "team": get_meta(conn, "team_name"),
                "phase": phase,
                "epoch": epoch,
                "rebuttal_round": rebuttal_round_value(conn),
                "message": "Rebuttal round is not complete yet.",
            }

        survivors = [row["proposer"] for row in active_proposals(conn, epoch)]
        if not survivors:
            next_epoch = epoch + 1
            targets = agent_names(conn)
            with transaction(conn):
                set_meta(conn, "epoch", str(next_epoch))
                set_meta(conn, "phase", "propose")
                set_meta(conn, "review_round", "0")
                set_meta(conn, "rebuttal_round", "0")
                set_phase_targets(conn, targets)
            return {
                "team": get_meta(conn, "team_name"),
                "phase": "propose",
                "epoch": next_epoch,
                "targets": targets,
                "message": f"{agent} started epoch {next_epoch} because every proposal in epoch {epoch} failed.",
            }

        if review_round_value(conn) >= max_review_rounds:
            result = _set_done_with_candidates(conn, survivors)
            result["message"] = f"{agent} finished the debate because the review round budget is exhausted."
            return result

        next_review_round = review_round_value(conn) + 1
        with transaction(conn):
            set_meta(conn, "phase", "review")
            set_meta(conn, "review_round", str(next_review_round))
            set_phase_targets(conn, survivors)
        return {
            "team": get_meta(conn, "team_name"),
            "phase": "review",
            "epoch": epoch,
            "review_round": next_review_round,
            "targets": survivors,
            "message": f"{agent} advanced the debate into review round {next_review_round}.",
        }

    raise SystemExit(f"Cannot advance from phase: {phase}")


def next_action_payload(conn: sqlite3.Connection, *, agent: str) -> dict[str, Any]:
    _require_initialized(conn)
    workflow = get_meta(conn, "workflow")
    phase = current_phase(conn)
    epoch = current_epoch(conn)
    phase_progress = current_phase_progress(conn)

    if status_value(conn) == "done":
        return {
            "team": get_meta(conn, "team_name"),
            "agent": agent,
            "action": "stop",
            "phase": "done",
            "epoch": epoch,
            "phase_progress": phase_progress,
            "advance_ready": False,
            "final_candidates": json.loads(get_meta(conn, "final_candidates_json", "[]")),
        }

    if agent == "debate-coordinator":
        if workflow == "parallel-judge":
            advance_ready = bool(propose_phase_progress(conn)["complete"])
            return {
                "team": get_meta(conn, "team_name"),
                "agent": agent,
                "action": "advance" if advance_ready else "wait",
                "phase": phase,
                "epoch": epoch,
                "phase_progress": phase_progress,
                "advance_ready": advance_ready,
                "message": "All proposals are in; run advance once." if advance_ready else "Waiting for all proposals before advancing.",
            }
        if workflow == "review-loop":
            if phase == "propose":
                advance_ready = bool(propose_phase_progress(conn)["complete"])
                return {
                    "team": get_meta(conn, "team_name"),
                    "agent": agent,
                    "action": "advance" if advance_ready else "wait",
                    "phase": phase,
                    "epoch": epoch,
                    "phase_progress": phase_progress,
                    "advance_ready": advance_ready,
                    "message": "All proposals are in; run advance once." if advance_ready else "Waiting for all proposals before advancing.",
                }
            if phase == "review":
                advance_ready = bool(review_phase_progress(conn)["complete"])
                return {
                    "team": get_meta(conn, "team_name"),
                    "agent": agent,
                    "action": "advance" if advance_ready else "wait",
                    "phase": phase,
                    "epoch": epoch,
                    "review_round": review_round_value(conn),
                    "phase_progress": phase_progress,
                    "advance_ready": advance_ready,
                    "message": "All required reviews are in; run advance once." if advance_ready else "Waiting for all cross-reviews before advancing.",
                }
            if phase == "rebuttal":
                advance_ready = bool(rebuttal_phase_progress(conn)["complete"])
                return {
                    "team": get_meta(conn, "team_name"),
                    "agent": agent,
                    "action": "advance" if advance_ready else "wait",
                    "phase": phase,
                    "epoch": epoch,
                    "rebuttal_round": rebuttal_round_value(conn),
                    "phase_progress": phase_progress,
                    "advance_ready": advance_ready,
                    "message": "All required rebuttals are in; run advance once." if advance_ready else "Waiting for all rebuttals before advancing.",
                }
        if workflow == "chemqa-review":
            advance_ready = bool(phase_progress.get("complete", False))
            payload = {
                "team": get_meta(conn, "team_name"),
                "agent": agent,
                "action": "advance" if advance_ready else "wait",
                "phase": phase,
                "epoch": epoch,
                "phase_progress": phase_progress,
                "advance_ready": advance_ready,
            }
            if phase == "review":
                payload["review_round"] = review_round_value(conn)
                proposal = chemqa_candidate_proposal(conn, epoch)
                if not proposal or str(proposal["status"]) != "active":
                    payload["state_issue"] = "review_phase_without_active_candidate"
                    payload["advance_ready"] = False
                    payload["action"] = "wait"
            if phase == "rebuttal":
                payload["rebuttal_round"] = rebuttal_round_value(conn)
            if phase == "propose":
                payload["message"] = "Candidate submission is ready; run advance once." if advance_ready else "Waiting for proposer-1 to submit the candidate before advancing."
            elif phase == "review":
                if payload.get("state_issue") == "review_phase_without_active_candidate":
                    payload["message"] = "Review phase is invalid because proposer-1 has no active candidate; waiting for epoch reset or failure termination."
                else:
                    active_reviewers = phase_progress.get("active_reviewer_lanes") or CHEMQA_REVIEWER_LANES
                    exited_reviewers = phase_progress.get("exited_reviewer_lanes") or []
                    if advance_ready:
                        payload["message"] = "All active reviewer lanes completed review; run advance once."
                    elif exited_reviewers:
                        payload["message"] = (
                            f"Waiting for active reviewer lanes to finish reviewing proposer-1. "
                            f"Exited lanes: {', '.join(exited_reviewers)}. Active lanes: {', '.join(active_reviewers)}."
                        )
                    else:
                        payload["message"] = "Waiting for the four fixed reviewer lanes to finish reviewing proposer-1."
            elif phase == "rebuttal":
                payload["message"] = "Required rebuttal is in; run advance once." if advance_ready else "Waiting for proposer-1 to submit the rebuttal."
            else:
                payload["message"] = "Waiting for workflow progression."
            return payload
        raise SystemExit(f"Unsupported workflow: {workflow}")

    if workflow == "parallel-judge":
        proposal = current_proposal_for(conn, agent, epoch)
        if proposal:
            return {
                "team": get_meta(conn, "team_name"),
                "agent": agent,
                "action": "wait",
                "phase": phase,
                "epoch": epoch,
                "phase_progress": phase_progress,
                "advance_ready": False,
                "message": "Proposal already submitted. Wait for the coordinator to finish.",
            }
        return {
            "team": get_meta(conn, "team_name"),
            "agent": agent,
            "action": "propose",
            "phase": phase,
            "epoch": epoch,
            "phase_progress": phase_progress,
            "advance_ready": False,
            "goal": get_meta(conn, "goal"),
            "evidence_policy": get_meta(conn, "evidence_policy"),
        }

    if workflow == "chemqa-review":
        if phase == "propose":
            if agent != CHEMQA_CANDIDATE_OWNER:
                return {
                    "team": get_meta(conn, "team_name"),
                    "agent": agent,
                    "action": "wait",
                    "phase": phase,
                    "epoch": epoch,
                    "phase_progress": phase_progress,
                    "advance_ready": False,
                    "message": "Waiting for proposer-1 to submit the candidate proposal.",
                }
            proposal = chemqa_candidate_proposal(conn, epoch)
            if proposal:
                return {
                    "team": get_meta(conn, "team_name"),
                    "agent": agent,
                    "action": "wait",
                    "phase": phase,
                    "epoch": epoch,
                    "phase_progress": phase_progress,
                    "advance_ready": False,
                    "message": "Candidate proposal already submitted.",
                }
            payload = {
                "team": get_meta(conn, "team_name"),
                "agent": agent,
                "action": "propose",
                "phase": phase,
                "epoch": epoch,
                "phase_progress": phase_progress,
                "advance_ready": False,
                "goal": get_meta(conn, "goal"),
                "evidence_policy": get_meta(conn, "evidence_policy"),
            }
            revision_context = chemqa_revision_context(conn, epoch=epoch)
            if revision_context:
                payload["revision_context"] = revision_context
                payload["message"] = "Submit a revised proposer-1 candidate that explicitly addresses the prior epoch's review items before advancing."
            return payload

        if phase == "review":
            proposal = chemqa_candidate_proposal(conn, epoch)
            exited_reviewers = set(chemqa_exited_reviewer_lanes(conn))
            if agent == CHEMQA_CANDIDATE_OWNER:
                payload = {
                    "team": get_meta(conn, "team_name"),
                    "agent": agent,
                    "action": "wait",
                    "phase": phase,
                    "epoch": epoch,
                    "review_round": review_round_value(conn),
                    "phase_progress": phase_progress,
                    "advance_ready": False,
                    "message": "Waiting for the fixed reviewer lanes to complete review.",
                }
                if not proposal or str(proposal["status"]) != "active":
                    payload["state_issue"] = "no_active_candidate_review_target"
                    payload["message"] = "No active candidate is registered for proposer-1 in this review phase."
                return payload
            if agent not in CHEMQA_REVIEWER_LANES:
                raise SystemExit(f"Unsupported ChemQA role: {agent}")
            review_round = review_round_value(conn)
            if agent in exited_reviewers:
                exit_payload = chemqa_exited_reviewer_state(conn).get(agent) or {}
                return {
                    "team": get_meta(conn, "team_name"),
                    "agent": agent,
                    "action": "stop",
                    "phase": phase,
                    "epoch": epoch,
                    "review_round": review_round,
                    "phase_progress": phase_progress,
                    "advance_ready": False,
                    "reviewer_exit": exit_payload,
                    "message": f"{agent} exited the ChemQA debate and is no longer required for active review quorum.",
                }
            existing = conn.execute(
                """
                SELECT 1
                FROM reviews
                WHERE epoch = ? AND review_round = ? AND reviewer = ? AND target_proposer = ?
                """,
                (epoch, review_round, agent, CHEMQA_CANDIDATE_OWNER),
            ).fetchone()
            if not proposal or str(proposal["status"]) != "active":
                return {
                    "team": get_meta(conn, "team_name"),
                    "agent": agent,
                    "action": "wait",
                    "phase": phase,
                    "epoch": epoch,
                    "review_round": review_round,
                    "phase_progress": phase_progress,
                    "advance_ready": False,
                    "state_issue": "no_active_candidate_review_target",
                    "message": "No active proposer-1 candidate exists for review; waiting for epoch reset / revised proposal.",
                }
            target_proposals = [
                proposal_context_payload(
                    conn,
                    proposal,
                    review_round_limit=review_round - 1 if review_round > 1 else 0,
                    rebuttal_round_limit=rebuttal_round_value(conn),
                )
            ]
            if existing:
                return {
                    "team": get_meta(conn, "team_name"),
                    "agent": agent,
                    "action": "wait",
                    "phase": phase,
                    "epoch": epoch,
                    "review_round": review_round,
                    "phase_progress": phase_progress,
                    "advance_ready": False,
                    "message": "Formal review for proposer-1 already submitted in this round.",
                }
            return {
                "team": get_meta(conn, "team_name"),
                "agent": agent,
                "action": "review",
                "phase": phase,
                "epoch": epoch,
                "review_round": review_round,
                "phase_progress": phase_progress,
                "advance_ready": False,
                "targets": [CHEMQA_CANDIDATE_OWNER],
                "target_proposals": target_proposals,
                "goal": get_meta(conn, "goal"),
                "evidence_policy": get_meta(conn, "evidence_policy"),
            }

        if phase == "rebuttal":
            rebuttal_round = rebuttal_round_value(conn)
            if agent != CHEMQA_CANDIDATE_OWNER:
                return {
                    "team": get_meta(conn, "team_name"),
                    "agent": agent,
                    "action": "wait",
                    "phase": phase,
                    "epoch": epoch,
                    "rebuttal_round": rebuttal_round,
                    "phase_progress": phase_progress,
                    "advance_ready": False,
                    "message": "Waiting for proposer-1 to submit the rebuttal.",
                }
            proposal = chemqa_candidate_proposal(conn, epoch)
            if not proposal:
                return {
                    "team": get_meta(conn, "team_name"),
                    "agent": agent,
                    "action": "wait",
                    "phase": phase,
                    "epoch": epoch,
                    "rebuttal_round": rebuttal_round,
                    "phase_progress": phase_progress,
                    "advance_ready": False,
                    "message": "No candidate proposal is recorded for proposer-1.",
                }
            existing = conn.execute(
                """
                SELECT 1
                FROM rebuttals
                WHERE epoch = ? AND rebuttal_round = ? AND proposer = ?
                """,
                (epoch, rebuttal_round, agent),
            ).fetchone()
            if existing:
                return {
                    "team": get_meta(conn, "team_name"),
                    "agent": agent,
                    "action": "wait",
                    "phase": phase,
                    "epoch": epoch,
                    "rebuttal_round": rebuttal_round,
                    "phase_progress": phase_progress,
                    "advance_ready": False,
                    "message": "Rebuttal already submitted for this round.",
                }
            review_rows = chemqa_review_rows(conn, epoch, review_round_value(conn))
            parsed_reviews = [
                serialize_review_row(
                    row,
                    include_body=True,
                    artifact=artifact_metadata_for(conn, record_type="review", record_id=int(row["id"])),
                )
                for row in review_rows
            ]
            proposal_context = proposal_context_payload(
                conn,
                proposal,
                review_round_limit=review_round_value(conn) - 1 if review_round_value(conn) > 1 else 0,
                rebuttal_round_limit=rebuttal_round - 1 if rebuttal_round > 1 else 0,
            )
            return {
                "team": get_meta(conn, "team_name"),
                "agent": agent,
                "action": "rebuttal",
                "phase": phase,
                "epoch": epoch,
                "rebuttal_round": rebuttal_round,
                "phase_progress": phase_progress,
                "advance_ready": False,
                "proposal": serialize_proposal_row(
                    proposal,
                    include_body=True,
                    artifact=artifact_metadata_for(conn, record_type="proposal", record_id=int(proposal["id"])),
                ),
                "reviews": parsed_reviews,
                "current_round_reviews": parsed_reviews,
                "review_history": proposal_context["review_history"],
                "rebuttal_history": proposal_context["rebuttal_history"],
                "attack_registry": proposal_context["attack_registry"],
                "goal": get_meta(conn, "goal"),
                "evidence_policy": get_meta(conn, "evidence_policy"),
            }

        return {
            "team": get_meta(conn, "team_name"),
            "agent": agent,
            "action": "wait",
            "phase": phase,
            "epoch": epoch,
            "phase_progress": phase_progress,
            "advance_ready": False,
            "message": "No action available.",
        }

    if workflow != "review-loop":
        raise SystemExit(f"Unsupported workflow: {workflow}")

    if phase == "propose":
        proposal = current_proposal_for(conn, agent, epoch)
        history_rows = prior_proposals_for_agent(conn, agent)
        history = [
            serialize_proposal_row(
                row,
                include_body=True,
                artifact=artifact_metadata_for(conn, record_type="proposal", record_id=int(row["id"])),
            )
            for row in history_rows
        ]
        if proposal:
            return {
                "team": get_meta(conn, "team_name"),
                "agent": agent,
                "action": "wait",
                "phase": phase,
                "epoch": epoch,
                "phase_progress": phase_progress,
                "advance_ready": False,
                "message": "Proposal for this epoch already submitted.",
                "history": history,
            }
        return {
            "team": get_meta(conn, "team_name"),
            "agent": agent,
            "action": "propose",
            "phase": phase,
            "epoch": epoch,
            "phase_progress": phase_progress,
            "advance_ready": False,
            "goal": get_meta(conn, "goal"),
            "evidence_policy": get_meta(conn, "evidence_policy"),
            "history": history,
        }

    if phase == "review":
        targets = [
            target
            for target in get_phase_targets(conn)
            if target != agent
        ]
        own_proposal = current_proposal_for(conn, agent, epoch)
        own_proposal_status = str(own_proposal["status"]) if own_proposal else "none"
        protocol_note_parts = [
            "Review only the listed targets from `next-action`; never review your own proposal.",
        ]
        if own_proposal_status not in {"active", "none"}:
            protocol_note_parts.append(
                f"Your own proposal status is `{own_proposal_status}`, but you must continue following the protocol until `next-action` returns `stop`."
            )
        protocol_note = " ".join(protocol_note_parts)
        pending = []
        target_proposals = []
        review_round = review_round_value(conn)
        for target in targets:
            row = conn.execute(
                """
                SELECT 1
                FROM reviews
                WHERE epoch = ? AND review_round = ? AND reviewer = ? AND target_proposer = ?
                """,
                (epoch, review_round, agent, target),
            ).fetchone()
            if not row:
                pending.append(target)
                proposal = current_proposal_for(conn, target, epoch)
                if proposal:
                    target_proposals.append(
                        proposal_context_payload(
                            conn,
                            proposal,
                            review_round_limit=review_round - 1 if review_round > 1 else 0,
                            rebuttal_round_limit=rebuttal_round_value(conn),
                        )
                    )
        if pending:
            return {
                "team": get_meta(conn, "team_name"),
                "agent": agent,
                "action": "review",
                "phase": phase,
                "epoch": epoch,
                "review_round": review_round,
                "phase_progress": phase_progress,
                "advance_ready": False,
                "targets": pending,
                "target_proposals": target_proposals,
                "goal": get_meta(conn, "goal"),
                "evidence_policy": get_meta(conn, "evidence_policy"),
                "own_proposal_status": own_proposal_status,
                "protocol_note": protocol_note,
            }
        return {
            "team": get_meta(conn, "team_name"),
            "agent": agent,
            "action": "wait",
            "phase": phase,
            "epoch": epoch,
            "review_round": review_round,
            "phase_progress": phase_progress,
            "advance_ready": False,
            "message": "No pending review targets for this round.",
            "own_proposal_status": own_proposal_status,
            "protocol_note": protocol_note,
        }

    if phase == "rebuttal":
        rebuttal_round = rebuttal_round_value(conn)
        if agent not in get_phase_targets(conn):
            return {
                "team": get_meta(conn, "team_name"),
                "agent": agent,
                "action": "wait",
                "phase": phase,
                "epoch": epoch,
                "rebuttal_round": rebuttal_round,
                "phase_progress": phase_progress,
                "advance_ready": False,
                "message": "Your proposal is not active in this rebuttal round.",
            }
        proposal = current_proposal_for(conn, agent, epoch)
        if not proposal:
            return {
                "team": get_meta(conn, "team_name"),
                "agent": agent,
                "action": "wait",
                "phase": phase,
                "epoch": epoch,
                "rebuttal_round": rebuttal_round,
                "phase_progress": phase_progress,
                "advance_ready": False,
                "message": "No proposal is recorded for your agent in this epoch.",
            }
        existing = conn.execute(
            """
            SELECT 1
            FROM rebuttals
            WHERE epoch = ? AND rebuttal_round = ? AND proposer = ?
            """,
            (epoch, rebuttal_round, agent),
        ).fetchone()
        if existing:
            return {
                "team": get_meta(conn, "team_name"),
                "agent": agent,
                "action": "wait",
                "phase": phase,
                "epoch": epoch,
                "rebuttal_round": rebuttal_round,
                "phase_progress": phase_progress,
                "advance_ready": False,
                "message": "Rebuttal already submitted for this round.",
            }
        review_rows = reviews_for_round(conn, agent, epoch, review_round_value(conn))
        parsed_reviews = [
            serialize_review_row(
                row,
                include_body=True,
                artifact=artifact_metadata_for(conn, record_type="review", record_id=int(row["id"])),
            )
            for row in review_rows
        ]
        proposal_context = proposal_context_payload(
            conn,
            proposal,
            review_round_limit=review_round_value(conn) - 1 if review_round_value(conn) > 1 else 0,
            rebuttal_round_limit=rebuttal_round - 1 if rebuttal_round > 1 else 0,
        )
        return {
            "team": get_meta(conn, "team_name"),
            "agent": agent,
            "action": "rebuttal",
            "phase": phase,
            "epoch": epoch,
            "rebuttal_round": rebuttal_round,
            "phase_progress": phase_progress,
            "advance_ready": False,
            "proposal": serialize_proposal_row(
                proposal,
                include_body=True,
                artifact=artifact_metadata_for(conn, record_type="proposal", record_id=int(proposal["id"])),
            ),
            "reviews": parsed_reviews,
            "current_round_reviews": parsed_reviews,
            "review_history": proposal_context["review_history"],
            "rebuttal_history": proposal_context["rebuttal_history"],
            "attack_registry": proposal_context["attack_registry"],
            "goal": get_meta(conn, "goal"),
            "evidence_policy": get_meta(conn, "evidence_policy"),
        }

    return {
        "team": get_meta(conn, "team_name"),
        "agent": agent,
        "action": "wait",
        "phase": phase,
        "epoch": epoch,
        "phase_progress": phase_progress,
        "advance_ready": False,
        "message": "No action available.",
    }


def summary_payload(conn: sqlite3.Connection, *, include_bodies: bool = False) -> dict[str, Any]:
    _require_initialized(conn)
    meta = load_meta(conn)
    epoch = current_epoch(conn)
    proposals = []
    proposal_rows = conn.execute(
        """
        SELECT id, epoch, proposer, title, body, status, failure_reason, created_at, updated_at
        FROM proposals
        ORDER BY epoch, proposer
        """
    ).fetchall()
    for row in proposal_rows:
        proposals.append(
            serialize_proposal_row(
                row,
                include_body=include_bodies,
                artifact=artifact_metadata_for(conn, record_type="proposal", record_id=int(row["id"])),
            )
        )

    reviews = []
    review_rows = conn.execute(
        """
        SELECT id, epoch, review_round, reviewer, target_proposer, blocking, body, attack_points_json,
               novel_blocking_points_json, synthetic, submitted_by, synthetic_reason, created_at
        FROM reviews
        ORDER BY epoch, review_round, reviewer, target_proposer
        """
    ).fetchall()
    for row in review_rows:
        reviews.append(
            serialize_review_row(
                row,
                include_body=include_bodies,
                artifact=artifact_metadata_for(conn, record_type="review", record_id=int(row["id"])),
            )
        )

    rebuttals = []
    rebuttal_rows = conn.execute(
        """
        SELECT id, epoch, rebuttal_round, proposer, conceded_failure, body, created_at
        FROM rebuttals
        ORDER BY epoch, rebuttal_round, proposer
        """
    ).fetchall()
    for row in rebuttal_rows:
        rebuttals.append(
            serialize_rebuttal_row(
                row,
                include_body=include_bodies,
                artifact=artifact_metadata_for(conn, record_type="rebuttal", record_id=int(row["id"])),
            )
        )

    final_candidates = json.loads(meta.get("final_candidates_json", "[]"))
    reviewer_exit_state = chemqa_exited_reviewer_state(conn) if meta.get("workflow", "") == "chemqa-review" else {}
    payload = {
        "team_name": meta.get("team_name", ""),
        "workflow": meta.get("workflow", ""),
        "goal": meta.get("goal", ""),
        "evidence_policy": meta.get("evidence_policy", ""),
        "status": meta.get("status", ""),
        "phase": meta.get("phase", ""),
        "epoch": epoch,
        "review_round": review_round_value(conn),
        "rebuttal_round": rebuttal_round_value(conn),
        "phase_targets": get_phase_targets(conn),
        "proposer_count": int(meta.get("proposer_count", "0")),
        "max_review_rounds": int(meta.get("max_review_rounds", "0")),
        "max_rebuttal_rounds": int(meta.get("max_rebuttal_rounds", "0")),
        "max_epochs": int(meta.get("max_epochs", "1")),
        "terminal_state": meta.get("terminal_state", ""),
        "failure_reason": meta.get("failure_reason", ""),
        "final_candidates": final_candidates,
        "agents": agent_names(conn),
        "proposals": proposals,
        "reviews": reviews,
        "rebuttals": rebuttals,
        "reviewer_exit_reasons": reviewer_exit_state,
        "exited_reviewer_lanes": [lane for lane in CHEMQA_REVIEWER_LANES if lane in reviewer_exit_state],
        "active_reviewer_lanes": [lane for lane in CHEMQA_REVIEWER_LANES if lane not in reviewer_exit_state],
    }
    payload["phase_progress"] = current_phase_progress(conn)
    payload["advance_ready"] = bool(payload["phase_progress"].get("complete", False)) and payload["status"] != "done"
    if include_bodies:
        attack_rows = conn.execute(
            """
            SELECT p.epoch, p.proposer AS target_proposer, p.title AS target_title, a.attack_text,
                   a.first_epoch, a.first_review_round, a.first_reviewer
            FROM attack_registry AS a
            JOIN proposals AS p ON p.id = a.target_proposal_id
            ORDER BY p.epoch, p.proposer, a.first_epoch, a.first_review_round, a.first_reviewer, a.id
            """
        ).fetchall()
        payload["attack_registry"] = [
            {
                "epoch": int(row["epoch"]),
                "target_proposer": str(row["target_proposer"]),
                "target_title": str(row["target_title"]),
                "attack_text": str(row["attack_text"]),
                "first_epoch": int(row["first_epoch"]),
                "first_review_round": int(row["first_review_round"]),
                "first_reviewer": str(row["first_reviewer"]),
            }
            for row in attack_rows
        ]
    return payload


def render_summary_text(payload: dict[str, Any]) -> str:
    phase_progress = payload.get("phase_progress", {})
    lines = [
        f"Team: {payload['team_name']}",
        f"Workflow: {payload['workflow']}",
        f"Status: {payload['status']} | Phase: {payload['phase']} | Epoch: {payload['epoch']}",
        f"Review round: {payload['review_round']} | Rebuttal round: {payload['rebuttal_round']}",
        f"Phase targets: {', '.join(payload['phase_targets']) if payload['phase_targets'] else 'none'}",
    ]
    if phase_progress.get("kind") in {"propose", "review", "rebuttal"}:
        lines.append(
            f"Phase progress: {phase_progress.get('actual', 0)}/{phase_progress.get('expected', 0)} | complete={'yes' if phase_progress.get('complete') else 'no'}"
        )
    if payload.get("advance_ready"):
        lines.append("Advance ready: yes")
    synthetic_count = sum(1 for review in payload.get("reviews", []) if review.get("synthetic"))
    if synthetic_count:
        lines.append(f"Synthetic reviews: {synthetic_count}")
    lines.extend([
        "",
        "Proposals:",
    ])
    for proposal in payload["proposals"]:
        line = f"- epoch {proposal['epoch']} | {proposal['proposer']} | {proposal['status']} | {proposal['title']}"
        if proposal["failure_reason"]:
            line += f" | failure: {proposal['failure_reason']}"
        lines.append(line)
    if not payload["proposals"]:
        lines.append("- none")
    lines.append("")
    lines.append(
        "Final candidates: "
        + (", ".join(payload["final_candidates"]) if payload["final_candidates"] else "none")
    )
    return "\n".join(lines)


def render_next_action_text(payload: dict[str, Any]) -> str:
    lines = [
        f"Team: {payload['team']}",
        f"Agent: {payload['agent']}",
        f"Action: {payload['action']} | Phase: {payload['phase']} | Epoch: {payload['epoch']}",
    ]
    if "review_round" in payload:
        lines.append(f"Review round: {payload['review_round']}")
    if "rebuttal_round" in payload:
        lines.append(f"Rebuttal round: {payload['rebuttal_round']}")
    if payload.get("message"):
        lines.append(f"Message: {payload['message']}")
    if payload.get("own_proposal_status"):
        lines.append(f"Own proposal status: {payload['own_proposal_status']}")
    if payload.get("protocol_note"):
        lines.append(f"Protocol note: {payload['protocol_note']}")
    phase_progress = payload.get("phase_progress", {})
    if phase_progress.get("kind") in {"propose", "review", "rebuttal"}:
        lines.append(
            f"Phase progress: {phase_progress.get('actual', 0)}/{phase_progress.get('expected', 0)} | complete={'yes' if phase_progress.get('complete') else 'no'}"
        )
    if payload.get("advance_ready"):
        lines.append("Advance ready: yes")
    if payload.get("targets"):
        lines.append("Targets: " + ", ".join(payload["targets"]))
    if payload.get("final_candidates"):
        lines.append("Final candidates: " + ", ".join(payload["final_candidates"]))
    history = payload.get("history", [])
    if history:
        lines.append("")
        lines.append("Prior proposals:")
        for proposal in history:
            line = f"- epoch {proposal['epoch']} | {proposal['status']} | {proposal['title']}"
            if proposal["failure_reason"]:
                line += f" | failure: {proposal['failure_reason']}"
            lines.append(line)
    current_reviews = payload.get("current_round_reviews", [])
    if current_reviews:
        lines.append("")
        lines.append("Current round reviews:")
        for review in current_reviews:
            marker = "blocking" if review["blocking"] else "non-blocking"
            attack_points = ", ".join(review["attack_points"]) or "no attack points"
            lines.append(f"- {review['reviewer']} | {marker} | {attack_points}")
    return "\n".join(lines)


def render_event_text(payload: dict[str, Any]) -> str:
    lines = [
        f"Team: {payload['team']}",
        f"Status: {payload.get('status', 'running')} | Phase: {payload['phase']}",
    ]
    if "epoch" in payload:
        lines.append(f"Epoch: {payload['epoch']}")
    if "review_round" in payload:
        lines.append(f"Review round: {payload['review_round']}")
    if "rebuttal_round" in payload:
        lines.append(f"Rebuttal round: {payload['rebuttal_round']}")
    if payload.get("targets"):
        lines.append("Targets: " + ", ".join(payload["targets"]))
    if payload.get("final_candidates"):
        lines.append("Final candidates: " + ", ".join(payload["final_candidates"]))
    if payload.get("message"):
        lines.append(f"Message: {payload['message']}")
    return "\n".join(lines)


def emit(payload: dict[str, Any], json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if "workflow" in payload and "proposals" in payload:
        print(render_summary_text(payload))
        return
    if "action" in payload and "agent" in payload:
        print(render_next_action_text(payload))
        return
    print(render_event_text(payload))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DebateClaw SQLite state runtime.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize debate state for a team.")
    init_parser.add_argument("--team", required=True)
    init_parser.add_argument("--workflow", required=True, choices=VALID_WORKFLOWS)
    init_parser.add_argument("--goal", required=True)
    init_parser.add_argument("--evidence-policy", required=True)
    init_parser.add_argument("--proposer-count", type=int, default=4)
    init_parser.add_argument("--max-review-rounds", type=int, default=5)
    init_parser.add_argument("--max-rebuttal-rounds", type=int, default=5)
    init_parser.add_argument("--max-epochs", type=int, default=3)
    init_parser.add_argument("--reset", action="store_true")

    status_parser = subparsers.add_parser("status", help="Show overall debate state.")
    status_parser.add_argument("--team", required=True)
    status_parser.add_argument("--agent")
    status_parser.add_argument("--json", action="store_true")

    next_parser = subparsers.add_parser("next-action", help="Show the next action for an agent.")
    next_parser.add_argument("--team", required=True)
    next_parser.add_argument("--agent", required=True)
    next_parser.add_argument("--json", action="store_true")

    proposal_parser = subparsers.add_parser("submit-proposal", help="Register a proposal for the current epoch.")
    proposal_parser.add_argument("--team", required=True)
    proposal_parser.add_argument("--agent", required=True)
    proposal_parser.add_argument("--file", required=True)

    review_parser = subparsers.add_parser("submit-review", help="Register a cross-review.")
    review_parser.add_argument("--team", required=True)
    review_parser.add_argument("--agent", required=True)
    review_parser.add_argument("--target", required=True)
    review_parser.add_argument("--blocking", required=True, choices=("yes", "no"))
    review_parser.add_argument("--file", required=True)
    review_parser.add_argument("--synthetic", action="store_true")
    review_parser.add_argument("--submitted-by", default="")
    review_parser.add_argument("--reason", default="")

    rebuttal_parser = subparsers.add_parser("submit-rebuttal", help="Register a rebuttal or concession.")
    rebuttal_parser.add_argument("--team", required=True)
    rebuttal_parser.add_argument("--agent", required=True)
    rebuttal_parser.add_argument("--file", required=True)
    rebuttal_parser.add_argument("--concede", action="store_true")

    advance_parser = subparsers.add_parser("advance", help="Advance the protocol when the current phase is complete.")
    advance_parser.add_argument("--team", required=True)
    advance_parser.add_argument("--agent", required=True)
    advance_parser.add_argument("--json", action="store_true")

    summary_parser = subparsers.add_parser("summary", help="Export the full debate summary.")
    summary_parser.add_argument("--team", required=True)
    summary_parser.add_argument("--json", action="store_true")
    summary_parser.add_argument("--include-bodies", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.command == "init":
        config = DebateConfig(
            team_name=args.team,
            workflow=args.workflow,
            goal=args.goal,
            evidence_policy=args.evidence_policy,
            proposer_count=args.proposer_count,
            max_review_rounds=args.max_review_rounds,
            max_rebuttal_rounds=args.max_rebuttal_rounds,
            max_epochs=args.max_epochs,
        )
        path = init_debate_state(config, reset=args.reset)
        print(path)
        return 0

    with connect(args.team) as conn:
        ensure_schema(conn)

        if args.command == "status":
            payload = summary_payload(conn)
            if args.agent:
                payload["agent_view"] = next_action_payload(conn, agent=args.agent)
            emit(payload, args.json)
            return 0

        if args.command == "next-action":
            payload = next_action_payload(conn, agent=args.agent)
            emit(payload, args.json)
            return 0

        if args.command == "submit-proposal":
            payload = submit_proposal(conn, agent=args.agent, file_path=Path(args.file))
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0

        if args.command == "submit-review":
            payload = submit_review(
                conn,
                agent=args.agent,
                target=args.target,
                blocking=args.blocking == "yes",
                file_path=Path(args.file),
                synthetic=args.synthetic,
                submitted_by=args.submitted_by,
                synthetic_reason=args.reason,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0

        if args.command == "submit-rebuttal":
            payload = submit_rebuttal(
                conn,
                agent=args.agent,
                file_path=Path(args.file),
                concede=args.concede,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0

        if args.command == "advance":
            payload = advance_state(conn, agent=args.agent)
            emit(payload, args.json)
            return 0

        if args.command == "summary":
            payload = summary_payload(conn, include_bodies=args.include_bodies)
            emit(payload, args.json)
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
