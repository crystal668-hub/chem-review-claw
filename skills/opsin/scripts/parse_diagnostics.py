from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _opsin_common import (
    compact_text,
    exit_code_for_status,
    finalize_payload,
    init_payload,
    invalid_request_payload,
    load_request,
    normalize_diagnostic,
    parse_args,
)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv, "Normalize OPSIN diagnostics")
    try:
        request = load_request(args.request_json)
    except Exception as exc:
        payload = invalid_request_payload({}, str(exc))
        finalize_payload(payload, output_dir=args.output_dir, emit_json=args.json)
        return 1

    raw_diagnostics = request.get("diagnostics")
    if not isinstance(raw_diagnostics, list) or not raw_diagnostics:
        payload = invalid_request_payload(request, "request.diagnostics must be a non-empty list")
        finalize_payload(payload, output_dir=args.output_dir, emit_json=args.json)
        return 1

    payload = init_payload(request)
    normalized: list[dict] = []
    for item in raw_diagnostics:
        if not isinstance(item, dict):
            payload["warnings"].append({"message": "skipped non-object diagnostic item"})
            continue
        normalized.append(
            normalize_diagnostic(
                input_name=compact_text(item.get("input_name")),
                provider_status=compact_text(item.get("provider_status")),
                provider_message=item.get("provider_message"),
            )
        )

    payload["status"] = "success"
    payload["diagnostics"] = normalized
    payload["primary_result"] = {
        "result_kind": "diagnostic_summary",
        "normalized_count": len(normalized),
        "categories": [item["category"] for item in normalized],
    }
    payload["tool_trace"].append(
        {
            "tool": "parse_diagnostics",
            "input_count": len(raw_diagnostics),
            "normalized_count": len(normalized),
        }
    )
    finalize_payload(payload, output_dir=args.output_dir, emit_json=args.json)
    return exit_code_for_status(payload["status"])


if __name__ == "__main__":
    raise SystemExit(main())
