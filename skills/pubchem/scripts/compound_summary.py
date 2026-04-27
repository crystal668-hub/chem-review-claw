from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from cid_to_properties import run as run_cid_to_properties
from name_to_cid import run as run_name_to_cid
from synonyms import run as run_synonyms
from pubchem_common import finalize_success, handle_exception, load_request, parse_cli_args


RESULT_FILE = "compound_summary_result.json"


def run(request: dict[str, Any], *, output_dir: str | Path, requester=None, persist: bool = True) -> dict[str, Any]:
    name_lookup = run_name_to_cid(request, output_dir=output_dir, requester=requester, persist=False)
    if not name_lookup["primary_result"]:
        return finalize_success(name_lookup, output_dir=output_dir, filename=RESULT_FILE) if persist else name_lookup

    cid = name_lookup["primary_result"]["cid"]
    properties = run_cid_to_properties(
        {"cids": [cid], "timeout_seconds": request.get("timeout_seconds", 8.0), "retry_attempts": request.get("retry_attempts", 1)},
        output_dir=output_dir,
        requester=requester,
        persist=False,
    )
    synonyms = run_synonyms(
        {"cid": cid, "max_synonyms": request.get("synonym_limit", 10), "timeout_seconds": request.get("timeout_seconds", 8.0), "retry_attempts": request.get("retry_attempts", 1)},
        output_dir=output_dir,
        requester=requester,
        persist=False,
    )
    result = {
        "status": "success",
        "request": request,
        "primary_result": dict(properties.get("primary_result") or {}),
        "candidates": properties.get("candidates") or [],
        "diagnostics": name_lookup.get("diagnostics", []) + properties.get("diagnostics", []) + synonyms.get("diagnostics", []),
        "warnings": name_lookup.get("warnings", []) + properties.get("warnings", []) + synonyms.get("warnings", []),
        "errors": name_lookup.get("errors", []) + properties.get("errors", []) + synonyms.get("errors", []),
        "tool_trace": [],
        "source_trace": name_lookup.get("source_trace", []) + properties.get("source_trace", []) + synonyms.get("source_trace", []),
        "provider_health": synonyms.get("provider_health") or properties.get("provider_health") or name_lookup.get("provider_health"),
    }
    result["primary_result"]["synonyms"] = (synonyms.get("primary_result") or {}).get("synonyms", [])
    result["primary_result"]["cid"] = cid
    if any(status == "error" for status in [name_lookup["status"], properties["status"], synonyms["status"]]):
        result["status"] = "partial" if result["primary_result"] else "error"
    elif any(status == "partial" for status in [name_lookup["status"], properties["status"], synonyms["status"]]):
        result["status"] = "partial"
    return finalize_success(result, output_dir=output_dir, filename=RESULT_FILE) if persist else result


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_cli_args(argv)
    request = load_request(args.request_json)
    try:
        result = run(request, output_dir=args.output_dir, requester=None)
    except Exception as exc:
        handle_exception(request, output_dir=args.output_dir, filename=RESULT_FILE, exc=exc, emit_json=args.json)
        return 0
    if args.json:
        import json

        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
