from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _opsin_common import (
    OPSIN_BASE_URL,
    compact_text,
    exit_code_for_status,
    finalize_payload,
    init_payload,
    invalid_request_payload,
    load_request,
    parse_args,
    run_opsin_lookup,
)


def _summary_provider_health(items: list[dict[str, Any]]) -> dict[str, Any]:
    source_trace = [entry for item in items for entry in item.get("source_trace", []) if entry.get("provider") == "opsin"]
    last = source_trace[-1] if source_trace else {}
    timeout = any(bool(entry.get("timeout")) for entry in source_trace)
    provider_failure_count = sum(1 for item in items if item.get("primary_result", {}).get("result_kind") == "provider_failure")
    no_result_count = sum(1 for item in items if item.get("primary_result", {}).get("result_kind") == "no_result")
    success_count = sum(1 for item in items if item.get("status") == "success")
    status = "healthy"
    if provider_failure_count:
        status = "degraded"
    elif no_result_count and success_count:
        status = "partial"
    elif no_result_count:
        status = "healthy"

    return {
        "provider_url": OPSIN_BASE_URL,
        "calls": len(source_trace),
        "http_status": last.get("http_status"),
        "elapsed_ms": last.get("elapsed_ms"),
        "timeout": timeout,
        "parse_status": "mixed" if len({compact_text(entry.get('parse_status')) for entry in source_trace if entry.get('parse_status')}) > 1 else compact_text(last.get("parse_status")) or None,
        "status": status,
        "success_count": success_count,
        "no_result_count": no_result_count,
        "provider_failure_count": provider_failure_count,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv, "Resolve multiple systematic chemical names with OPSIN")
    try:
        request = load_request(args.request_json)
    except Exception as exc:
        payload = invalid_request_payload({}, str(exc))
        finalize_payload(payload, output_dir=args.output_dir, emit_json=args.json)
        return 1

    names = request.get("names")
    if not isinstance(names, list) or not names:
        payload = invalid_request_payload(request, "request.names must be a non-empty list")
        finalize_payload(payload, output_dir=args.output_dir, emit_json=args.json)
        return 1

    payload = init_payload(request)
    items: list[dict[str, Any]] = []
    success_count = 0
    failure_count = 0
    no_result_count = 0
    provider_failure_count = 0

    for raw_name in names:
        name = compact_text(raw_name)
        item_request = dict(request)
        item_request["name"] = name
        item_payload = run_opsin_lookup(item_request, requests_get=requests.get)
        items.append(item_payload)
        payload["candidates"].append(item_payload)
        payload["diagnostics"].extend(item_payload.get("diagnostics", []))
        payload["warnings"].extend(item_payload.get("warnings", []))
        payload["errors"].extend(
            [{**error, "input_name": name} for error in item_payload.get("errors", [])]
        )
        payload["tool_trace"].extend(item_payload.get("tool_trace", []))
        payload["source_trace"].extend(item_payload.get("source_trace", []))
        if item_payload.get("status") == "success":
            success_count += 1
        else:
            failure_count += 1
            result_kind = item_payload.get("primary_result", {}).get("result_kind")
            if result_kind == "no_result":
                no_result_count += 1
            elif result_kind == "provider_failure":
                provider_failure_count += 1

    if success_count == len(items):
        payload["status"] = "success"
    elif success_count > 0:
        payload["status"] = "partial"
    else:
        payload["status"] = "error"

    payload["primary_result"] = {
        "result_kind": "batch_lookup",
        "success_count": success_count,
        "failure_count": failure_count,
        "no_result_count": no_result_count,
        "provider_failure_count": provider_failure_count,
    }
    payload["provider_health"] = {"opsin": _summary_provider_health(items)}

    finalize_payload(payload, output_dir=args.output_dir, emit_json=args.json)
    return exit_code_for_status(payload["status"])


if __name__ == "__main__":
    raise SystemExit(main())
