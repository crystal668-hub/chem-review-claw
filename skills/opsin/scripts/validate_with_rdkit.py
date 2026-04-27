from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _opsin_common import exit_code_for_status, finalize_payload, init_payload, invalid_request_payload, load_request, parse_args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv, "Validate OPSIN structures with RDKit")
    try:
        request = load_request(args.request_json)
    except Exception as exc:
        payload = invalid_request_payload({}, str(exc))
        finalize_payload(payload, output_dir=args.output_dir, emit_json=args.json)
        return 1

    payload = init_payload(request)
    try:
        from rdkit import Chem
    except Exception as exc:
        payload["status"] = "error"
        payload["primary_result"] = {
            "result_kind": "dependency_missing",
            "validation_status": "invalid",
            "validation_errors": [str(exc)],
        }
        payload["errors"] = [{"code": "rdkit_unavailable", "message": str(exc), "provider": "rdkit"}]
        finalize_payload(payload, output_dir=args.output_dir, emit_json=args.json)
        return 1

    opsin_result = request.get("opsin_result")
    if not isinstance(opsin_result, dict):
        payload = invalid_request_payload(request, "request.opsin_result must be an object")
        finalize_payload(payload, output_dir=args.output_dir, emit_json=args.json)
        return 1

    input_name = str(request.get("name") or opsin_result.get("input_name") or "")
    smiles = str(opsin_result.get("smiles") or "").strip()
    stdinchi = str(opsin_result.get("stdinchi") or "").strip()

    mol = None
    used_input_format = None
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        used_input_format = "smiles"
    if mol is None and stdinchi and hasattr(Chem, "MolFromInchi"):
        mol = Chem.MolFromInchi(stdinchi)
        used_input_format = "stdinchi"

    if mol is None:
        payload["status"] = "error"
        payload["primary_result"] = {
            "input_name": input_name,
            "result_kind": "invalid_structure",
            "validation_status": "invalid",
            "validation_errors": ["RDKit could not parse the provided structure"],
        }
        payload["errors"] = [
            {
                "code": "rdkit_parse_failed",
                "message": "RDKit could not parse the provided structure",
                "provider": "rdkit",
            }
        ]
        payload["provider_health"] = {
            "rdkit": {
                "status": "available",
                "parse_status": "invalid",
                "timeout": False,
            }
        }
        finalize_payload(payload, output_dir=args.output_dir, emit_json=args.json)
        return 1

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    payload["status"] = "success"
    payload["primary_result"] = {
        "input_name": input_name,
        "result_kind": "validated_structure",
        "validation_status": "valid",
        "validation_errors": [],
        "canonical_smiles": canonical_smiles,
        "used_input_format": used_input_format,
    }
    payload["tool_trace"] = [{"tool": "rdkit", "action": "validate_structure"}]
    payload["source_trace"] = [
        {
            "provider": "rdkit",
            "used_input_format": used_input_format,
            "parse_status": "valid",
        }
    ]
    payload["provider_health"] = {
        "rdkit": {
            "status": "available",
            "parse_status": "valid",
            "timeout": False,
        }
    }
    finalize_payload(payload, output_dir=args.output_dir, emit_json=args.json)
    return exit_code_for_status(payload["status"])


if __name__ == "__main__":
    raise SystemExit(main())
