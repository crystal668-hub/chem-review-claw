from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from cid_to_properties import run as run_cid_to_properties
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


RESULT_FILE = "similarity_search_result.json"


def run(request: dict[str, Any], *, output_dir: str | Path, requester=None, persist: bool = True) -> dict[str, Any]:
    result = base_result(request)
    require_fields(request, ["query_smiles"])
    client = PubChemClient(
        timeout_seconds=request.get("timeout_seconds", 8.0),
        retry_attempts=request.get("retry_attempts", 1),
        requester=requester,
    )
    query_smiles = str(request.get("query_smiles"))
    threshold = int(request.get("threshold", 95))
    max_records = max(1, int(request.get("max_records", 10)))

    try:
        payload, trace = client.request_json(
            f"compound/fastsimilarity_2d/smiles/{query_smiles}/cids",
            params={"Threshold": threshold, "MaxRecords": max_records},
        )
    except Exception as exc:
        if not is_pubchem_error(exc):
            raise
        merge_health(result, client)
        apply_http_error(result, exc)
        result["source_trace"].extend(exc.trace or [])
        return finalize_success(result, output_dir=output_dir, filename=RESULT_FILE) if persist else result

    result["source_trace"].extend(trace)
    cid_values = list(((payload.get("IdentifierList") or {}).get("CID")) or [])[:max_records]
    result["candidates"] = [{"cid": int(cid)} for cid in cid_values]
    if cid_values:
        result["primary_result"] = result["candidates"][0]
    else:
        result["errors"].append({"message": f"No PubChem similarity candidates found for query_smiles={query_smiles}"})

    partial = False
    if cid_values:
        properties_payload = run_cid_to_properties(
            {"cids": cid_values, "timeout_seconds": request.get("timeout_seconds", 8.0), "retry_attempts": request.get("retry_attempts", 1)},
            output_dir=output_dir,
            requester=requester,
            persist=False,
        )
        if properties_payload["status"] == "error":
            partial = True
            result["warnings"].append({"message": "Similarity CIDs were resolved but property enrichment failed"})
            result["errors"].extend(properties_payload["errors"])
            result["source_trace"].extend(properties_payload["source_trace"])
            result["provider_health"] = properties_payload["provider_health"]
        else:
            partial = properties_payload["status"] == "partial"
            result["candidates"] = properties_payload["candidates"]
            result["primary_result"] = properties_payload["primary_result"]
            result["warnings"].extend(properties_payload["warnings"])
            result["source_trace"].extend(properties_payload["source_trace"])
            result["provider_health"] = properties_payload["provider_health"]
    else:
        merge_health(result, client)

    result["status"] = status_from_parts(usable=bool(cid_values), partial=partial)
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
