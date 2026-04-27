from __future__ import annotations

import sys
from pathlib import Path

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _opsin_common import exit_code_for_status, finalize_payload, load_request, parse_args, run_opsin_lookup


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv, "Resolve one systematic chemical name with OPSIN")
    try:
        request = load_request(args.request_json)
    except Exception as exc:
        request = {}
        payload = {
            "status": "error",
            "request": request,
            "primary_result": {
                "result_kind": "invalid_request",
                "validation_status": "invalid",
                "validation_errors": [str(exc)],
            },
            "candidates": [],
            "diagnostics": [],
            "warnings": [],
            "errors": [{"code": "invalid_request_json", "message": str(exc)}],
            "tool_trace": [],
            "source_trace": [],
            "provider_health": {},
        }
        finalize_payload(payload, output_dir=args.output_dir, emit_json=args.json)
        return 1

    payload = run_opsin_lookup(request, requests_get=requests.get)
    finalize_payload(payload, output_dir=args.output_dir, emit_json=args.json)
    return exit_code_for_status(payload["status"])


if __name__ == "__main__":
    raise SystemExit(main())
