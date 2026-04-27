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
    normalize_properties,
    parse_cli_args,
    require_fields,
    status_from_parts,
)


RESULT_FILE = "cid_to_properties_result.json"


def _property_key_map() -> dict[str, str]:
    return {
        "CID": "cid",
        "MolecularFormula": "molecular_formula",
        "MolecularWeight": "molecular_weight",
        "CanonicalSMILES": "canonical_smiles",
        "IsomericSMILES": "isomeric_smiles",
        "InChI": "inchi",
        "InChIKey": "inchikey",
        "Charge": "charge",
    }


def run(request: dict[str, Any], *, output_dir: str | Path, requester=None, persist: bool = True) -> dict[str, Any]:
    result = base_result(request)
    require_fields(request, ["cids"])
    client = PubChemClient(
        timeout_seconds=request.get("timeout_seconds", 8.0),
        retry_attempts=request.get("retry_attempts", 1),
        requester=requester,
    )
    cids = normalize_cids(request.get("cids"))
    properties = normalize_properties(request.get("properties"))
    property_path = ",".join(properties)
    cid_path = ",".join(str(cid) for cid in cids)

    try:
        payload, trace = client.request_json(f"compound/cid/{cid_path}/property/{property_path}")
    except Exception as exc:
        if not is_pubchem_error(exc):
            raise
        result["source_trace"].extend(exc.trace or [])
        merge_health(result, client)
        apply_http_error(result, exc)
        return finalize_success(result, output_dir=output_dir, filename=RESULT_FILE) if persist else result

    result["source_trace"].extend(trace)
    table = ((payload.get("PropertyTable") or {}).get("Properties")) or []
    parsed_candidates: list[dict[str, Any]] = []
    property_key_map = _property_key_map()
    partial = False
    for row in table:
        candidate = {target: row.get(source) for source, target in property_key_map.items() if row.get(source) is not None}
        missing = [source for source in properties if row.get(source) is None]
        if missing:
            partial = True
            candidate["missing_properties"] = missing
            result["warnings"].append({"cid": row.get("CID"), "missing_properties": missing})
        parsed_candidates.append(candidate)

    if parsed_candidates:
        result["primary_result"] = parsed_candidates[0]
        result["candidates"] = parsed_candidates
    else:
        result["errors"].append({"message": "PubChem returned no property rows"})

    merge_health(result, client)
    result["status"] = status_from_parts(usable=bool(parsed_candidates), partial=partial)
    return finalize_success(result, output_dir=output_dir, filename=RESULT_FILE) if persist else result


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_cli_args(argv)
    request = load_request(args.request_json)
    try:
        run(request, output_dir=args.output_dir, requester=None)
    except Exception as exc:
        handle_exception(request, output_dir=args.output_dir, filename=RESULT_FILE, exc=exc, emit_json=args.json)
        return 0
    if args.json:
        print((Path(args.output_dir) / RESULT_FILE).read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
