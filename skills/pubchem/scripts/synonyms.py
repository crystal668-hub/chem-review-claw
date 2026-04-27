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
    normalize_cids,
    parse_cli_args,
    require_fields,
    status_from_parts,
)


RESULT_FILE = "synonyms_result.json"


def run(request: dict[str, Any], *, output_dir: str | Path, requester=None, persist: bool = True) -> dict[str, Any]:
    result = base_result(request)
    cid_value = request.get("cid")
    if cid_value in (None, "", []):
        require_fields(request, ["query"])
        from name_to_cid import run as run_name_lookup

        lookup = run_name_lookup(request, output_dir=output_dir, requester=requester, persist=False)
        if not lookup["primary_result"]:
            return finalize_success(lookup, output_dir=output_dir, filename=RESULT_FILE) if persist else lookup
        cid_value = lookup["primary_result"]["cid"]
        result["diagnostics"].append({"message": "Resolved CID from query before synonym lookup", "cid": cid_value})
    client = PubChemClient(
        timeout_seconds=request.get("timeout_seconds", 8.0),
        retry_attempts=request.get("retry_attempts", 1),
        requester=requester,
    )
    cid = normalize_cids([cid_value])[0]
    max_synonyms = max(1, int(request.get("max_synonyms", 20)))

    try:
        payload, trace = client.request_json(f"compound/cid/{cid}/synonyms")
    except Exception as exc:
        if not is_pubchem_error(exc):
            raise
        merge_health(result, client)
        apply_http_error(result, exc)
        result["source_trace"].extend(exc.trace or [])
        return finalize_success(result, output_dir=output_dir, filename=RESULT_FILE) if persist else result

    result["source_trace"].extend(trace)
    info_rows = ((payload.get("InformationList") or {}).get("Information")) or []
    synonyms: list[str] = []
    if info_rows:
        synonyms = list((info_rows[0].get("Synonym")) or [])[:max_synonyms]
    result["primary_result"] = {"cid": cid, "synonyms": synonyms}
    result["candidates"] = [{"value": value} for value in synonyms]
    if not synonyms:
        result["errors"].append({"message": f"PubChem returned no synonyms for cid={cid}"})

    merge_health(result, client)
    result["status"] = status_from_parts(usable=bool(synonyms), partial=False)
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
