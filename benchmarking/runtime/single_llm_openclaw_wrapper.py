#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarking.core.convergence import (
    ConvergencePolicy,
    extract_latest_complete_answer_from_transcript_for_eval,
    is_complete_answer_for_eval,
    is_complete_rescue_answer,
    summarize_transcript_convergence,
)
from benchmarking.core.result_contract import contract_to_payload, parse_agent_stdout
from benchmarking.runtime.openclaw_env import build_openclaw_subprocess_env
from benchmarking.runtime.session_isolation import (
    SessionIsolationError,
    inspect_postflight_session,
    merge_preflight_postflight_audit,
    reset_agent_main_session_if_stale,
)

OPENCLAW_STREAM_READ_ERROR_TEXT = "stream_read_error"
OPENCLAW_AGENT_NO_RESPONSE_FRAGMENT = "Agent couldn't generate a response"
OPENCLAW_TIMEOUT_SENTINELS = (
    "Request timed out before a response was generated",
    "The model did not produce a response before the LLM idle timeout",
    "LLM request timed out.",
    "Request timed out.",
)
TIME_REMINDER_MIN_TIMEOUT_SECONDS = 600
TIME_REMINDER_ELAPSED_FRACTION = 2 / 3
TIME_REMINDER_POLL_SECONDS = 1.0
TIME_REMINDER_PROMPT = """TIME REMINDER:
Less than one third of the answer budget remains. Please quickly organize the reasoning chain already available in this session.
Converge on a complete final answer in the required format.
Do not start new tool chains or skill exploration unless one short decisive check is clearly necessary."""
def _answer_schema_from_args(args: argparse.Namespace) -> dict[str, Any]:
    raw = str(getattr(args, "answer_schema_json", "") or "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def build_finalization_rescue_prompt(eval_kind: str = "", answer_schema: dict[str, Any] | None = None) -> str:
    kind = str(eval_kind or "").strip()
    common = [
        "The previous turn did not organize a final answer that satisfies the benchmark output requirements.",
        "Do not call tools or inspect files.",
        "Use only the reasoning chain, calculations, tool verification results, and evidence already present in this session.",
        "Before answering, check consistency across the prior reasoning and then follow only the output requirements for this task type.",
    ]
    if kind == "frontierscience_research":
        specific = [
            "This is a FrontierScience research-track chemistry task scored against itemized reasoning criteria.",
            "Provide a complete, structured, multi-part research synthesis covering every requested condition, calculation, mechanism, protocol consequence, and conclusion.",
            "Keep rubric-relevant derivations, assumptions, units, evidence, and justifications visible before the final research section.",
            "Do not add the short-answer `FINAL ANSWER:` marker used by non-research tasks.",
            "End with exactly this Markdown heading and section:",
            "## FINAL RESEARCH ANSWER",
            "<rubric-complete final synthesis>",
        ]
    elif kind == "superchem_multiple_choice_rpf":
        specific = [
            "This is a chemistry multiple-choice question.",
            "Provide concise visible option checks that distinguish the candidates.",
            "End with exactly one line formatted as: FINAL ANSWER: <option letters>.",
            "Use only uppercase option letters; separate multiple correct letters with `|`.",
        ]
    elif kind == "chembench_open_ended":
        specific = [
            "Provide the necessary formulae, substitutions, units, rounding, or exact string evidence for the answer.",
            "End with exactly one line formatted as: FINAL ANSWER: <answer>.",
        ]
    elif kind == "frontierscience_olympiad":
        specific = [
            "End with exactly one line formatted as: FINAL ANSWER: <answer>.",
            "The final line should contain only the requested value, expression, formula, structure name, or entity, including required units or rounding.",
            "Do not provide multiple answer attempts.",
        ]
    elif kind == "hle":
        specific = [
            "Use the official HLE response format exactly:",
            "Explanation: <your visible derivation and checks>",
            "Answer: <your chosen answer>",
            "Confidence: <your confidence score between 0% and 100%>",
            "Do not add `FINAL ANSWER:` to HLE responses.",
        ]
    elif kind == "verifier_grounded":
        specific = [
            "Provide a complete verifier-grounded answer based on the existing session reasoning.",
            "End with the exact final answer format requested in the original question.",
        ]
    else:
        specific = [
            "Provide a complete answer based on the existing session reasoning.",
            "If a final answer line is needed, use: FINAL ANSWER: <answer>.",
        ]
    return "\n".join(common + specific)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-LLM OpenClaw turns with benchmark session isolation.")
    parser.add_argument("--agent", required=True, help="OpenClaw agent id.")
    parser.add_argument("--config-file", required=True, help="OpenClaw config path for this benchmark run.")
    parser.add_argument("--session-id", required=True, help="Run-scoped OpenClaw session id.")
    parser.add_argument("--message", required=True, help="Prompt to send to OpenClaw.")
    parser.add_argument("--thinking", help="Forward OpenClaw thinking override.")
    parser.add_argument("--timeout", type=int, help="Forward OpenClaw timeout override in seconds.")
    parser.add_argument("--eval-kind", default="", help="Benchmark eval kind for rescue-only answer recovery.")
    parser.add_argument("--answer-schema-json", default="", help="Optional benchmark answer schema JSON for schema-aware recovery.")
    parser.add_argument("--json", action="store_true", help="Forward OpenClaw JSON output and attach isolation audit.")
    return parser.parse_args()


def merge_isolation_audit(payload: Any, audit: dict[str, Any]) -> Any:
    if not isinstance(payload, dict):
        return payload
    target = payload.get("result") if isinstance(payload.get("result"), dict) else payload
    if isinstance(target, dict):
        meta = target.setdefault("meta", {})
        if isinstance(meta, dict):
            existing = meta.get("session_isolation")
            merged = dict(existing) if isinstance(existing, dict) else {}
            merged.update(audit)
            meta["session_isolation"] = merged
    return payload


def transcript_path_from_audit(audit: dict[str, Any]) -> Path | None:
    raw = str(audit.get("postflight_entry_session_file") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_file() else None


def _target_result_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    result = payload.get("result")
    if isinstance(result, dict):
        return result
    return payload


def _is_timeout_like_payload(
    target: dict[str, Any],
    *,
    eval_kind: str = "",
    answer_schema: dict[str, Any] | None = None,
) -> bool:
    payload_texts = _payload_texts(target)
    for text in payload_texts:
        if any(needle in text for needle in OPENCLAW_TIMEOUT_SENTINELS):
            return True
    if any(is_complete_answer_for_eval(text, eval_kind=eval_kind, answer_schema=answer_schema) for text in payload_texts):
        return False
    meta = target.get("meta") if isinstance(target.get("meta"), dict) else {}
    return bool(meta.get("aborted") is True or str(meta.get("livenessState") or "") == "blocked")


def _has_timeout_sentinel_payload(target: dict[str, Any]) -> bool:
    return any(any(needle in text for needle in OPENCLAW_TIMEOUT_SENTINELS) for text in _payload_texts(target))


def _payload_texts(target: dict[str, Any]) -> list[str]:
    payloads = target.get("payloads")
    texts: list[str] = []
    if isinstance(payloads, list):
        for item in payloads:
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                if text:
                    texts.append(text)
    return texts


def _has_error_payload_marker(target: dict[str, Any]) -> bool:
    payloads = target.get("payloads")
    if not isinstance(payloads, list):
        return False
    return any(isinstance(item, dict) and item.get("isError") is True for item in payloads)


def _extract_error_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("message", "error", "code", "type", "reason"):
            text = str(value.get(key) or "").strip()
            if text:
                return text
    text = str(value or "").strip()
    return text


def _replay_invalid_diagnostics(target: dict[str, Any]) -> dict[str, Any]:
    meta = target.get("meta") if isinstance(target.get("meta"), dict) else {}
    if meta.get("replayInvalid") is not True:
        return {}
    completion = meta.get("completion") if isinstance(meta.get("completion"), dict) else {}
    convergence = meta.get("convergence") if isinstance(meta.get("convergence"), dict) else {}
    diagnostic_candidates = [
        meta.get("replayInvalidReason"),
        meta.get("replayError"),
        meta.get("error"),
        meta.get("message"),
        convergence.get("latest_prompt_error"),
        convergence.get("finalization_rescue_error"),
    ]
    diagnostic_text = ""
    for candidate in diagnostic_candidates:
        diagnostic_text = _extract_error_text(candidate)
        if diagnostic_text:
            break
    return {
        "reason": "replay_invalid",
        "diagnostic_text": diagnostic_text,
        "stopReason": meta.get("stopReason"),
        "finishReason": completion.get("finishReason") or completion.get("stopReason"),
        "livenessState": meta.get("livenessState"),
        "payload_is_error": _has_error_payload_marker(target),
        "latest_prompt_error": convergence.get("latest_prompt_error"),
        "latest_prompt_error_is_timeout": convergence.get("latest_prompt_error_is_timeout"),
    }


def _classify_agent_error_payload(
    target: dict[str, Any],
    *,
    eval_kind: str = "",
    answer_schema: dict[str, Any] | None = None,
) -> str:
    payload_texts = _payload_texts(target)
    if _has_timeout_sentinel_payload(target):
        return ""
    meta = target.get("meta") if isinstance(target.get("meta"), dict) else {}
    completion = meta.get("completion") if isinstance(meta.get("completion"), dict) else {}
    stop_reason = str(meta.get("stopReason") or "").strip().lower()
    finish_reason = str(completion.get("finishReason") or completion.get("stopReason") or "").strip().lower()
    liveness_state = str(meta.get("livenessState") or "").strip()
    has_complete_answer = any(
        is_complete_answer_for_eval(text, eval_kind=eval_kind, answer_schema=answer_schema)
        for text in payload_texts
    )

    if (
        any(text == OPENCLAW_STREAM_READ_ERROR_TEXT for text in payload_texts)
        or stop_reason == "error"
        or finish_reason == "error"
    ):
        return "agent_stream_read_error"
    if (
        any(OPENCLAW_AGENT_NO_RESPONSE_FRAGMENT in text for text in payload_texts)
        or (meta.get("replayInvalid") is True and not has_complete_answer)
        or (liveness_state in {"abandoned", "blocked"} and (_has_error_payload_marker(target) or not has_complete_answer))
    ):
        return "agent_response_unavailable"
    return ""


def _session_isolation_ok(audit: dict[str, Any]) -> bool:
    return audit.get("session_isolation_ok") is True


def _merge_convergence(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    meta = target.setdefault("meta", {})
    if not isinstance(meta, dict):
        return {}
    existing = meta.get("convergence")
    convergence = dict(existing) if isinstance(existing, dict) else {}
    convergence.update(updates)
    meta["convergence"] = convergence
    return convergence


def _rescue_output_text(payload: Any) -> str:
    target = _target_result_payload(payload)
    if target is None:
        return ""
    return "\n\n".join(_payload_texts(target)).strip()


def _time_reminder_enabled(args: argparse.Namespace) -> bool:
    timeout = int(getattr(args, "timeout", 0) or 0)
    return timeout > TIME_REMINDER_MIN_TIMEOUT_SECONDS


def _time_reminder_threshold_seconds(timeout_seconds: int) -> int:
    if timeout_seconds <= 0:
        return TIME_REMINDER_MIN_TIMEOUT_SECONDS
    return max(1, math.ceil(timeout_seconds * TIME_REMINDER_ELAPSED_FRACTION))


def _base_time_reminder_meta(args: argparse.Namespace) -> dict[str, Any]:
    timeout_seconds = int(getattr(args, "timeout", 0) or 0)
    return {
        "enabled": _time_reminder_enabled(args),
        "threshold_seconds": _time_reminder_threshold_seconds(timeout_seconds),
        "due_before_primary_return": False,
        "primary_elapsed_seconds": 0.0,
        "applied": False,
        "skipped_reason": "disabled" if not _time_reminder_enabled(args) else "threshold_not_reached",
        "remaining_seconds_at_primary_return": float(max(0, int(getattr(args, "timeout", 0) or 0))),
    }


def _time_reminder_meta_from_result(result: subprocess.CompletedProcess[str], args: argparse.Namespace) -> dict[str, Any]:
    raw = getattr(result, "time_reminder_meta", None)
    if isinstance(raw, dict):
        meta = _base_time_reminder_meta(args)
        meta.update(raw)
        return meta
    return _base_time_reminder_meta(args)


def _has_complete_answer_in_payload(
    payload: Any,
    *,
    eval_kind: str = "",
    answer_schema: dict[str, Any] | None = None,
) -> bool:
    target = _target_result_payload(payload)
    if target is None:
        return False
    return any(is_complete_answer_for_eval(text, eval_kind=eval_kind, answer_schema=answer_schema) for text in _payload_texts(target))


def _has_complete_answer_in_result(
    result: subprocess.CompletedProcess[str],
    audit: dict[str, Any],
    *,
    eval_kind: str = "",
    answer_schema: dict[str, Any] | None = None,
) -> bool:
    output = (result.stdout or "").strip() or (result.stderr or "").strip()
    if output:
        try:
            if _has_complete_answer_in_payload(parse_openclaw_json_output(output), eval_kind=eval_kind, answer_schema=answer_schema):
                return True
        except Exception:
            pass
    transcript_path = transcript_path_from_audit(audit)
    return bool(
        transcript_path is not None
        and extract_latest_complete_answer_from_transcript_for_eval(
            transcript_path,
            eval_kind=eval_kind,
            answer_schema=answer_schema,
        )
    )


def _parse_remaining_seconds(value: Any) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0


def _try_finalization_rescue(
    target: dict[str, Any],
    *,
    args: argparse.Namespace,
    env: dict[str, str],
    answer_schema: dict[str, Any] | None = None,
) -> bool:
    try:
        result = run_openclaw(
            args,
            env=env,
            message_override=build_finalization_rescue_prompt(
                str(getattr(args, "eval_kind", "") or ""),
                answer_schema=answer_schema,
            ),
        )
    except Exception as exc:
        _merge_convergence(
            target,
            {
                "finalization_rescue_attempted": True,
                "finalization_rescue_succeeded": False,
                "finalization_rescue_error": str(exc)[:1000],
            },
        )
        return False

    if result.returncode != 0:
        _merge_convergence(
            target,
            {
                "finalization_rescue_attempted": True,
                "finalization_rescue_succeeded": False,
                "finalization_rescue_returncode": result.returncode,
                "finalization_rescue_stderr_excerpt": str(result.stderr or "")[:1000],
            },
        )
        return False

    rescue_payload = parse_openclaw_json_output((result.stdout or "").strip() or (result.stderr or "").strip())
    rescue_text = _rescue_output_text(rescue_payload)
    if not is_complete_rescue_answer(
        rescue_text,
        eval_kind=str(getattr(args, "eval_kind", "") or ""),
        answer_schema=answer_schema,
    ):
        _merge_convergence(
            target,
            {
                "finalization_rescue_attempted": True,
                "finalization_rescue_succeeded": False,
                "finalization_rescue_payload_excerpt": rescue_text[:1000],
            },
        )
        return False

    target["payloads"] = [{"text": rescue_text}]
    _merge_convergence(
        target,
        {
            "finalization_rescue_attempted": True,
            "finalization_rescue_succeeded": True,
            "recovery_source": "single-llm-finalization-rescue",
        },
    )
    return True


def merge_convergence_metadata(
    payload: Any,
    *,
    args: argparse.Namespace,
    audit: dict[str, Any],
    env: dict[str, str] | None = None,
    time_reminder_meta: dict[str, Any] | None = None,
    answer_schema: dict[str, Any] | None = None,
) -> Any:
    target = _target_result_payload(payload)
    if target is None:
        return payload
    policy = ConvergencePolicy(
        timeout_seconds=int(getattr(args, "timeout", 0) or 0),
    )
    convergence_meta: dict[str, Any] = {
        "policy": policy.to_meta(),
        "transcript_answer_recovered": False,
        "agent_error_payload_detected": False,
        "agent_error_kind": "",
        "finalization_rescue_attempted": False,
        "finalization_rescue_succeeded": False,
        "time_reminder": dict(time_reminder_meta or _base_time_reminder_meta(args)),
    }
    transcript_path = transcript_path_from_audit(audit)
    if transcript_path is not None:
        convergence_meta.update(summarize_transcript_convergence(transcript_path))
    eval_kind = str(getattr(args, "eval_kind", "") or "")
    agent_error_kind = _classify_agent_error_payload(target, eval_kind=eval_kind, answer_schema=answer_schema)
    if agent_error_kind:
        convergence_meta["agent_error_payload_detected"] = True
        convergence_meta["agent_error_kind"] = agent_error_kind
    replay_diagnostics = _replay_invalid_diagnostics(target)
    if replay_diagnostics:
        convergence_meta["replay_invalid_diagnostics"] = replay_diagnostics

    meta = target.setdefault("meta", {})
    if isinstance(meta, dict):
        existing = meta.get("convergence")
        merged = dict(existing) if isinstance(existing, dict) else {}
        merged.update(convergence_meta)
        meta["convergence"] = merged

    error_like = bool(agent_error_kind)
    timeout_like = _is_timeout_like_payload(target, eval_kind=eval_kind, answer_schema=answer_schema)
    if transcript_path is not None and (timeout_like or error_like):
        recovered = extract_latest_complete_answer_from_transcript_for_eval(
            transcript_path,
            eval_kind=eval_kind,
            answer_schema=answer_schema,
        )
        if recovered:
            target["payloads"] = [{"text": recovered}]
            meta = target.setdefault("meta", {})
            if isinstance(meta, dict):
                convergence = meta.setdefault("convergence", {})
                if isinstance(convergence, dict):
                    convergence["transcript_answer_recovered"] = True
                    convergence["recovery_source"] = "single-llm-session-transcript"
            return payload
    if (
        error_like
        and env is not None
        and transcript_path is not None
        and _session_isolation_ok(audit)
        and not any(
            is_complete_answer_for_eval(text, eval_kind=eval_kind, answer_schema=answer_schema)
            for text in _payload_texts(target)
        )
    ):
        _try_finalization_rescue(target, args=args, env=env, answer_schema=answer_schema)
    return payload


def parse_openclaw_json_output(output: str) -> Any:
    return contract_to_payload(parse_agent_stdout(output))


def resolve_openclaw_executable() -> str:
    executable = shutil.which("openclaw")
    if executable:
        return executable
    raise SessionIsolationError("Missing openclaw executable in PATH.")


def _build_openclaw_command(
    args: argparse.Namespace,
    *,
    message_override: str | None = None,
    timeout_override: int | None = None,
) -> list[str]:
    timeout = args.timeout if timeout_override is None else timeout_override
    command = [
        resolve_openclaw_executable(),
        "agent",
        "--local",
        "--agent",
        args.agent,
        "--session-id",
        args.session_id,
        "--message",
        args.message if message_override is None else message_override,
    ]
    if args.thinking:
        command.extend(["--thinking", args.thinking])
    if timeout is not None:
        command.extend(["--timeout", str(max(1, int(timeout)))])
    if args.json:
        command.append("--json")
    return command


def _benchmark_workspace_cwd(env: dict[str, str]) -> str | None:
    workspace = str(env.get("BENCHMARK_WORKSPACE_DIR") or "").strip()
    if not workspace:
        return None
    resolved = Path(workspace).expanduser().resolve()
    if not resolved.is_dir():
        raise SessionIsolationError(f"Benchmark workspace is unavailable: {resolved}")
    return str(resolved)


def _run_openclaw_with_time_reminder_tracking(
    command: list[str],
    *,
    args: argparse.Namespace,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    timeout_seconds = int(getattr(args, "timeout", 0) or 0)
    threshold_seconds = _time_reminder_threshold_seconds(timeout_seconds)
    start = time.monotonic()
    reminder_due = False
    proc = subprocess.Popen(
        command,
        env=env,
        cwd=_benchmark_workspace_cwd(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    while True:
        try:
            stdout, stderr = proc.communicate(timeout=TIME_REMINDER_POLL_SECONDS)
            break
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            if not reminder_due and elapsed >= threshold_seconds:
                reminder_due = True
    elapsed = time.monotonic() - start
    reminder_due = reminder_due or elapsed >= threshold_seconds
    remaining = max(0.0, float(timeout_seconds) - elapsed) if timeout_seconds > 0 else 0.0
    result = subprocess.CompletedProcess(command, proc.returncode, stdout=stdout, stderr=stderr)
    result.time_reminder_meta = {
        "enabled": True,
        "threshold_seconds": threshold_seconds,
        "due_before_primary_return": reminder_due,
        "primary_elapsed_seconds": elapsed,
        "applied": False,
        "skipped_reason": "" if reminder_due else "threshold_not_reached",
        "remaining_seconds_at_primary_return": remaining,
    }
    return result


def run_openclaw(
    args: argparse.Namespace,
    *,
    env: dict[str, str],
    message_override: str | None = None,
    timeout_override: int | None = None,
) -> subprocess.CompletedProcess[str]:
    command = _build_openclaw_command(args, message_override=message_override, timeout_override=timeout_override)
    if message_override is None and timeout_override is None and _time_reminder_enabled(args):
        return _run_openclaw_with_time_reminder_tracking(command, args=args, env=env)
    return subprocess.run(
        command,
        env=env,
        cwd=_benchmark_workspace_cwd(env),
        capture_output=True,
        text=True,
        check=False,
    )


def _maybe_run_time_reminder(
    primary_result: subprocess.CompletedProcess[str],
    *,
    args: argparse.Namespace,
    env: dict[str, str],
    audit: dict[str, Any],
    answer_schema: dict[str, Any] | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    reminder_meta = _time_reminder_meta_from_result(primary_result, args)
    if not reminder_meta.get("enabled"):
        reminder_meta["skipped_reason"] = "disabled"
        return primary_result, reminder_meta
    if not reminder_meta.get("due_before_primary_return"):
        reminder_meta["skipped_reason"] = "threshold_not_reached"
        return primary_result, reminder_meta
    if _has_complete_answer_in_result(
        primary_result,
        audit,
        eval_kind=str(getattr(args, "eval_kind", "") or ""),
        answer_schema=answer_schema,
    ):
        reminder_meta["skipped_reason"] = "complete_answer_available"
        return primary_result, reminder_meta
    remaining_seconds = _parse_remaining_seconds(reminder_meta.get("remaining_seconds_at_primary_return"))
    if remaining_seconds <= 0:
        reminder_meta["skipped_reason"] = "no_remaining_time"
        return primary_result, reminder_meta

    reminder_result = run_openclaw(
        args,
        env=env,
        message_override=TIME_REMINDER_PROMPT,
        timeout_override=remaining_seconds,
    )
    reminder_meta["applied"] = True
    reminder_meta["skipped_reason"] = ""
    if reminder_result.returncode != 0:
        reminder_meta["reminder_returncode"] = reminder_result.returncode
        reminder_meta["reminder_stderr_excerpt"] = str(reminder_result.stderr or "")[:1000]
        return primary_result, reminder_meta
    return reminder_result, reminder_meta


def main() -> int:
    args = parse_args()
    answer_schema = _answer_schema_from_args(args)
    config_path = Path(args.config_file).expanduser().resolve()
    env = build_openclaw_subprocess_env(base_env=os.environ.copy(), config_path=config_path)
    try:
        preflight_audit = reset_agent_main_session_if_stale(args.agent, args.session_id, config_path=config_path)
        result = run_openclaw(args, env=env)
        if result.returncode != 0:
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            return result.returncode
        primary_postflight_audit = inspect_postflight_session(args.agent, args.session_id, config_path=config_path)
        primary_audit = merge_preflight_postflight_audit(preflight_audit, primary_postflight_audit)
        result, time_reminder_meta = _maybe_run_time_reminder(
            result,
            args=args,
            env=env,
            audit=primary_audit,
            answer_schema=answer_schema,
        )
        postflight_audit = inspect_postflight_session(args.agent, args.session_id, config_path=config_path)
        audit = merge_preflight_postflight_audit(preflight_audit, postflight_audit)
        if args.json:
            output = result.stdout.strip() or result.stderr.strip()
            payload = parse_openclaw_json_output(output)
            payload = merge_convergence_metadata(
                payload,
                args=args,
                audit=audit,
                env=env,
                time_reminder_meta=time_reminder_meta,
                answer_schema=answer_schema,
            )
            payload = merge_isolation_audit(payload, audit)
            print(json.dumps(payload, ensure_ascii=False))
        else:
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
        return 0
    except Exception as exc:
        if args.json:
            payload = {
                "result": {
                    "payloads": [],
                    "meta": {
                        "session_isolation": {
                            "requested_session_id": args.session_id,
                            "agent_id": args.agent,
                            "session_store_path": "",
                            "preflight_removed_stale_main_entry": False,
                            "preflight_previous_session_id": "",
                            "postflight_entry_session_id": "",
                            "postflight_entry_session_file": "",
                            "session_isolation_ok": False,
                            "error": str(exc),
                        }
                    },
                }
            }
            print(json.dumps(payload, ensure_ascii=False))
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
