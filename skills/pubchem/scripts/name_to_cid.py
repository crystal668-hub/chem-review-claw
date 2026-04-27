from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pubchem_common import (
    PubChemClient,
    PubChemHttpError,
    apply_http_error,
    base_result,
    finalize_success,
    handle_exception,
    is_pubchem_error,
    load_request,
    merge_health,
    parse_cli_args,
    require_fields,
    status_from_parts,
)


RESULT_FILE = "name_to_cid_result.json"


def run(request: dict[str, Any], *, output_dir: str | Path, requester=None, persist: bool = True) -> dict[str, Any]:
    result = base_result(request)
    require_fields(request, ["query"])
    client = PubChemClient(
        timeout_seconds=request.get("timeout_seconds", 8.0),
        retry_attempts=request.get("retry_attempts", 1),
        requester=requester,
    )
    query = str(request.get("query"))
    max_candidates = max(1, int(request.get("max_candidates", 5)))

    try:
        payload, trace = client.request_json(f"compound/name/{query}/cids")
    except Exception as exc:
        if not is_pubchem_error(exc):
            raise
        merge_health(result, client)
        apply_http_error(result, exc)
        result["source_trace"].extend(exc.trace or [])
        return finalize_success(result, output_dir=output_dir, filename=RESULT_FILE) if persist else result

    result["source_trace"].extend(trace)
    cid_values = list(((payload.get("IdentifierList") or {}).get("CID")) or [])[:max_candidates]
    candidates = [{"cid": int(cid), "query": query} for cid in cid_values]
    result["candidates"] = candidates
    if candidates:
        result["primary_result"] = candidates[0]
    else:
        result["errors"].append({"message": f"No PubChem CID candidates found for query={query}"})
    if len(candidates) > 1:
        result["warnings"].append({"message": "PubChem returned multiple CID candidates", "candidate_count": len(candidates)})

    merge_health(result, client)
    result["status"] = status_from_parts(usable=bool(candidates), partial=len(candidates) > 1)
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
        print(json_text(result))
    return 0


def json_text(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
