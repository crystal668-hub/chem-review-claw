from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable


RESULT_FILE_NAME = "result.json"


class SkillError(Exception):
    def __init__(self, code: str, message: str, *, primary_result: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.primary_result = primary_result or {}


class RequestError(SkillError):
    pass


class DependencyError(SkillError):
    pass


class ProcessingError(SkillError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def empty_payload(request: Any) -> dict[str, Any]:
    return {
        "status": "error",
        "request": request,
        "primary_result": {},
        "candidates": [],
        "diagnostics": [],
        "warnings": [],
        "errors": [],
        "tool_trace": [],
        "source_trace": [],
        "provider_health": {},
    }


def json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def append_error(payload: dict[str, Any], code: str, message: str) -> None:
    payload.setdefault("errors", []).append({"code": code, "message": message})


def append_warning(payload: dict[str, Any], code: str, message: str) -> None:
    payload.setdefault("warnings", []).append({"code": code, "message": message})


def append_diagnostic(payload: dict[str, Any], code: str, message: str) -> None:
    payload.setdefault("diagnostics", []).append({"code": code, "message": message})


def load_request_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RequestError("missing_request_json", f"Request JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RequestError("invalid_request_json", f"Request JSON is not valid JSON: {exc}") from exc


def write_payload(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / RESULT_FILE_NAME
    result_path.write_text(json_dumps(payload), encoding="utf-8")
    return result_path


def ensure_mapping_request(request: Any) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise RequestError("invalid_request_type", "Top-level request JSON must be an object.")
    return request


def safe_import_rdkit() -> dict[str, Any]:
    try:
        from rdkit import Chem, DataStructs, rdBase
        from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdChemReactions, rdFingerprintGenerator, rdMolDescriptors
    except ImportError as exc:
        raise DependencyError("rdkit_missing", "RDKit is not installed in the current Python environment.") from exc

    return {
        "Chem": Chem,
        "AllChem": AllChem,
        "Crippen": Crippen,
        "DataStructs": DataStructs,
        "Descriptors": Descriptors,
        "Lipinski": Lipinski,
        "rdBase": rdBase,
        "rdChemReactions": rdChemReactions,
        "rdFingerprintGenerator": rdFingerprintGenerator,
        "rdMolDescriptors": rdMolDescriptors,
        "version": str(getattr(rdBase, "rdkitVersion", "")),
    }


def provider_health_available(version: str) -> dict[str, Any]:
    return {
        "rdkit": {
            "available": True,
            "status": "available",
            "version": version,
        }
    }


def provider_health_missing(message: str) -> dict[str, Any]:
    return {
        "rdkit": {
            "available": False,
            "status": "missing_dependency",
            "message": message,
        }
    }


def get_required_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RequestError("missing_field", f"Request field `{key}` is required and must be a non-empty string.")
    return value.strip()


def load_molecule(rdkit_ctx: dict[str, Any], spec: Any, *, strip_atom_maps: bool = False) -> tuple[Any, dict[str, Any]]:
    if not isinstance(spec, dict):
        raise RequestError("invalid_molecule", "Molecule entry must be an object with `format` and `value`.")
    fmt = str(spec.get("format") or "smiles").strip().lower()
    value = str(spec.get("value") or "").strip()
    if not value:
        raise RequestError("missing_molecule_value", "Molecule `value` is required.")

    Chem = rdkit_ctx["Chem"]

    if fmt == "smiles":
        mol = Chem.MolFromSmiles(value, sanitize=False)
    elif fmt == "inchi":
        loader = getattr(Chem, "MolFromInchi", None)
        if loader is None:
            raise ProcessingError(
                "inchi_not_supported",
                "This RDKit build does not expose InChI parsing support.",
                primary_result={"format": fmt, "input_value": value, "valid": False},
            )
        try:
            mol = loader(value, sanitize=False, removeHs=False)
        except TypeError:
            mol = loader(value)
    else:
        raise RequestError("unsupported_format", f"Unsupported molecule format: {fmt}")

    if mol is None:
        raise ProcessingError(
            "molecule_parse_error",
            f"RDKit could not parse the provided {fmt} value.",
            primary_result={"format": fmt, "input_value": value, "valid": False},
        )

    if strip_atom_maps:
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum():
                atom.SetAtomMapNum(0)

    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        raise ProcessingError(
            "molecule_sanitize_error",
            f"RDKit sanitization failed: {exc}",
            primary_result={"format": fmt, "input_value": value, "valid": False},
        ) from exc

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    isomeric_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    metadata = {
        "format": fmt,
        "input_value": value,
        "canonical_smiles": canonical_smiles,
        "isomeric_smiles": isomeric_smiles,
        "valid": True,
    }
    return mol, metadata


def load_molecule_list(rdkit_ctx: dict[str, Any], specs: Any) -> list[tuple[Any, dict[str, Any]]]:
    if not isinstance(specs, list) or not specs:
        raise RequestError("invalid_molecule_list", "Request field `molecules` must be a non-empty list.")
    return [load_molecule(rdkit_ctx, spec) for spec in specs]


def finalize_payload(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    payload = dict(base)
    for key in ["status", "request", "primary_result", "candidates", "diagnostics", "warnings", "errors", "tool_trace", "source_trace", "provider_health"]:
        if key in update:
            payload[key] = update[key]
    return payload


def run_named_capability(capability_name: str, handler: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]) -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    request_path = Path(args.request_json).expanduser().resolve()

    request: Any = {}
    payload = empty_payload(request)
    start = time.time()
    try:
        request = load_request_json(request_path)
        payload = empty_payload(request)
        payload["tool_trace"].append({"step": "load_request_json", "status": "success", "path": str(request_path)})
        request_mapping = ensure_mapping_request(request)
        rdkit_ctx = safe_import_rdkit()
        payload["provider_health"] = provider_health_available(rdkit_ctx["version"])
        payload["tool_trace"].append({"step": "import_rdkit", "status": "success", "version": rdkit_ctx["version"]})
        update = handler(request_mapping, rdkit_ctx)
        payload = finalize_payload(payload, update)
    except SkillError as exc:
        payload = empty_payload(request)
        if isinstance(exc, DependencyError):
            payload["provider_health"] = provider_health_missing(exc.message)
        append_error(payload, exc.code, exc.message)
        payload["primary_result"] = dict(exc.primary_result)
        payload["status"] = "error"
    except Exception as exc:  # pragma: no cover - defensive contract guard
        payload = empty_payload(request)
        append_error(payload, "unexpected_error", str(exc))
        payload["status"] = "error"

    elapsed_ms = int((time.time() - start) * 1000)
    payload.setdefault("tool_trace", []).append(
        {"step": capability_name, "status": payload.get("status", "error"), "elapsed_ms": elapsed_ms}
    )
    result_path = write_payload(output_dir, payload)
    payload.setdefault("tool_trace", []).append({"step": "write_result", "status": "success", "path": str(result_path)})
    write_payload(output_dir, payload)
    if args.json:
        print(json_dumps(payload))
    return 0
