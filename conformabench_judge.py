from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


class ConformaBenchJudgeError(RuntimeError):
    pass


class ConformaBenchDependencyError(ConformaBenchJudgeError):
    pass


def ensure_rdkit_available() -> None:
    _import_rdkit()


def resolve_hidden_judge_spec_path(record_source_file: str | Path, hidden_judge_spec_ref: str) -> Path:
    if not hidden_judge_spec_ref:
        raise ConformaBenchJudgeError("ConformaBench record is missing `hidden_judge_spec_ref`.")
    source_path = Path(record_source_file).expanduser().resolve()
    return source_path.parent.parent / "items" / hidden_judge_spec_ref / "hidden_judge_spec.yaml"


def load_hidden_judge_spec(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConformaBenchJudgeError(f"Hidden judge spec not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConformaBenchJudgeError(f"Hidden judge spec must be a mapping: {path}")
    return payload


def evaluate_submission(*, final_answer_smiles: str, hidden_spec: dict[str, Any]) -> dict[str, Any]:
    Chem, AllChem, Lipinski = _import_rdkit()
    normalization = _mapping(hidden_spec.get("normalization"))
    protocol = _mapping(hidden_spec.get("rdkit_protocol"))
    acceptance = _predicate_list(_mapping(hidden_spec.get("acceptance_predicates")).get("all_of"))
    rejections = _predicate_list(_mapping(hidden_spec.get("rejection_predicates")).get("any_of"))

    details: dict[str, Any] = {
        "answer_parse": {
            "raw_final_answer": final_answer_smiles,
            "valid_final_answer_line": bool(final_answer_smiles and final_answer_smiles.strip()),
        },
        "normalization": {
            "strip_atom_mapping": bool(normalization.get("strip_atom_mapping")),
            "largest_fragment_only": bool(normalization.get("largest_fragment_only")),
            "reject_disconnected_graphs": bool(normalization.get("reject_disconnected_graphs")),
            "reject_invalid_valence": bool(normalization.get("reject_invalid_valence")),
            "allowed_formal_charge": list(normalization.get("allowed_formal_charge") or []),
        },
        "canonical_smiles": "",
        "topology_predicates": [],
        "rejection_predicates": [],
        "force_field": {
            "preferred_order": list(_mapping(protocol.get("force_field_policy")).get("preferred_order") or []),
            "per_seed": [],
        },
        "seed_runs": [],
        "stability": {
            "require_same_pass_result_across_all_seeds": bool(
                _mapping(_mapping(protocol.get("robustness"))).get("require_same_pass_result_across_all_seeds", False)
            ),
            "seed_passes": [],
            "stable": True,
            "unstable_fail": False,
        },
    }

    if not final_answer_smiles or not final_answer_smiles.strip():
        details["answer_parse"]["error"] = "Missing final answer SMILES."
        details["passed"] = False
        return details

    raw_mol = Chem.MolFromSmiles(final_answer_smiles, sanitize=False)
    if raw_mol is None:
        details["answer_parse"]["error"] = "SMILES could not be parsed by RDKit."
        details["passed"] = False
        return details

    if normalization.get("strip_atom_mapping"):
        for atom in raw_mol.GetAtoms():
            if atom.GetAtomMapNum():
                atom.SetAtomMapNum(0)

    sanitize_requested = bool(protocol.get("sanitize", True))
    sanitize_error = None
    if sanitize_requested:
        try:
            Chem.SanitizeMol(raw_mol)
        except Exception as exc:  # pragma: no cover - depends on RDKit runtime specifics
            sanitize_error = str(exc)
            if normalization.get("reject_invalid_valence", True):
                details["answer_parse"]["error"] = f"RDKit sanitization failed: {sanitize_error}"
                details["passed"] = False
                return details

    fragments = list(Chem.GetMolFrags(raw_mol, asMols=True, sanitizeFrags=False))
    details["normalization"]["fragment_count"] = len(fragments)
    if normalization.get("reject_disconnected_graphs") and len(fragments) > 1:
        details["answer_parse"]["error"] = "Disconnected graphs are not allowed."
        details["passed"] = False
        return details
    if normalization.get("largest_fragment_only") and len(fragments) > 1:
        raw_mol = max(fragments, key=lambda mol: (mol.GetNumHeavyAtoms(), mol.GetNumAtoms()))
        if sanitize_requested:
            Chem.SanitizeMol(raw_mol)

    canonical_smiles = Chem.MolToSmiles(raw_mol, canonical=True)
    details["canonical_smiles"] = canonical_smiles
    details["answer_parse"]["canonical_smiles"] = canonical_smiles
    details["answer_parse"]["sanitize_error"] = sanitize_error

    topology_results: list[dict[str, Any]] = []
    geometry_predicates: list[dict[str, Any]] = []
    for predicate in acceptance:
        kind = str(predicate.get("kind") or "").strip()
        if kind == "geometry_lowest_conformer":
            geometry_predicates.append(predicate)
            continue
        topology_results.append(
            _evaluate_predicate(predicate=predicate, raw_mol=raw_mol, canonical_smiles=canonical_smiles, Lipinski=Lipinski)
        )
    details["topology_predicates"] = topology_results
    if any(not item["passed"] for item in topology_results):
        details["rejection_predicates"] = [
            {"predicate_id": item["predicate_id"], "kind": item["kind"], "matched": not item["passed"], "source": "acceptance"}
            for item in topology_results
            if not item["passed"]
        ]
        details["passed"] = False
        return details

    rejection_results = [
        _evaluate_predicate(predicate=predicate, raw_mol=raw_mol, canonical_smiles=canonical_smiles, Lipinski=Lipinski)
        for predicate in rejections
        if str(predicate.get("kind") or "").strip() != "no_supported_force_field"
    ]
    details["rejection_predicates"] = [
        {
            "predicate_id": item["predicate_id"],
            "kind": item["kind"],
            "matched": item["passed"],
            "source": "rejection",
        }
        for item in rejection_results
    ]
    if any(item["passed"] for item in rejection_results):
        details["passed"] = False
        return details

    protocol_embedding = _mapping(protocol.get("embedding"))
    protocol_optimization = _mapping(protocol.get("optimization"))
    protocol_force_field = _mapping(protocol.get("force_field_policy"))
    robustness = _mapping(protocol.get("robustness"))
    seeds = [int(seed) for seed in list(protocol_embedding.get("random_seeds") or [0])]
    num_confs = int(protocol_embedding.get("num_confs") or 1)
    max_iters = int(protocol_optimization.get("max_iters") or 200)
    target_observables = {
        str(observable.get("observable_id") or ""): observable
        for observable in list(hidden_spec.get("target_observables") or [])
        if isinstance(observable, dict) and str(observable.get("observable_id") or "").strip()
    }

    h_mol_template = Chem.AddHs(raw_mol)
    seed_passes: list[bool] = []
    for seed in seeds:
        seed_result = {
            "seed": seed,
            "status": "pending",
            "force_field": None,
            "embedding": {
                "method": str(protocol_embedding.get("method") or "ETKDGv3"),
                "num_confs_requested": num_confs,
            },
            "lowest_energy_kcal_mol": None,
            "lowest_energy_conformer_id": None,
            "observables": {},
            "geometry_predicates": [],
            "rejection_predicates": [],
            "passed": False,
        }
        working_mol = Chem.Mol(h_mol_template)
        working_mol.RemoveAllConformers()
        conf_ids = _embed_conformers(
            working_mol,
            AllChem=AllChem,
            method=str(protocol_embedding.get("method") or "ETKDGv3"),
            num_confs=num_confs,
            random_seed=seed,
            prune_rms_thresh=float(protocol_embedding.get("prune_rms_thresh") or 0.0),
            use_random_coords=bool(protocol_embedding.get("use_random_coords", False)),
        )
        if not conf_ids:
            seed_result["status"] = "embed_failed"
            details["seed_runs"].append(seed_result)
            seed_passes.append(False)
            continue

        force_field_name = _select_force_field_name(working_mol, AllChem=AllChem, policy=protocol_force_field)
        seed_result["force_field"] = force_field_name
        details["force_field"]["per_seed"].append({"seed": seed, "force_field": force_field_name})
        if not force_field_name:
            no_ff_matched = any(str(predicate.get("kind") or "").strip() == "no_supported_force_field" for predicate in rejections)
            seed_result["status"] = "no_supported_force_field"
            seed_result["rejection_predicates"].append(
                {"predicate_id": "implicit_no_supported_force_field", "kind": "no_supported_force_field", "matched": True}
            )
            details["seed_runs"].append(seed_result)
            seed_passes.append(False)
            if no_ff_matched or protocol_force_field.get("fail_if_no_supported_force_field", True):
                continue
            continue

        energies: list[tuple[int, float]] = []
        for conf_id in conf_ids:
            force_field = _build_force_field(working_mol, AllChem=AllChem, force_field_name=force_field_name, conf_id=int(conf_id))
            if force_field is None:
                continue
            force_field.Minimize(maxIts=max_iters)
            energies.append((int(conf_id), float(force_field.CalcEnergy())))
        if not energies:
            seed_result["status"] = "optimization_failed"
            details["seed_runs"].append(seed_result)
            seed_passes.append(False)
            continue

        lowest_conf_id, lowest_energy = min(energies, key=lambda item: item[1])
        seed_result["status"] = "ok"
        seed_result["lowest_energy_conformer_id"] = lowest_conf_id
        seed_result["lowest_energy_kcal_mol"] = lowest_energy

        observable_values = {
            observable_id: _compute_observable(
                working_mol,
                conf_id=lowest_conf_id,
                observable=observable,
                Chem=Chem,
            )
            for observable_id, observable in target_observables.items()
        }
        seed_result["observables"] = observable_values

        geometry_results = [
            _evaluate_geometry_predicate(predicate, observable_values=observable_values)
            for predicate in geometry_predicates
        ]
        seed_result["geometry_predicates"] = geometry_results
        seed_result["passed"] = all(item["passed"] for item in geometry_results)
        details["seed_runs"].append(seed_result)
        seed_passes.append(bool(seed_result["passed"]))

    details["stability"]["seed_passes"] = seed_passes
    require_all = bool(robustness.get("require_same_pass_result_across_all_seeds", False))
    if require_all:
        details["stability"]["stable"] = len(set(seed_passes)) <= 1
        details["stability"]["unstable_fail"] = bool(seed_passes) and not details["stability"]["stable"]
        details["passed"] = bool(seed_passes) and all(seed_passes)
    else:
        details["passed"] = bool(seed_passes) and bool(seed_passes[0])
    return details


def _import_rdkit() -> tuple[Any, Any, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Lipinski
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via benchmark wrapper tests
        raise ConformaBenchDependencyError(
            "ConformaBench judge requires the optional `rdkit` dependency. "
            "Install RDKit in the benchmark runtime environment before running this eval kind."
        ) from exc
    return Chem, AllChem, Lipinski


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _predicate_list(value: Any) -> list[dict[str, Any]]:
    return [item for item in list(value or []) if isinstance(item, dict)]


def _evaluate_predicate(*, raw_mol: Any, canonical_smiles: str, predicate: dict[str, Any], Lipinski: Any) -> dict[str, Any]:
    kind = str(predicate.get("kind") or "").strip()
    predicate_id = str(predicate.get("predicate_id") or kind or "predicate")
    actual: Any = None
    passed = False
    if kind == "molecule_valid":
        actual = True
        passed = True
    elif kind == "single_fragment":
        actual = len(_get_fragment_atom_tuples(raw_mol))
        passed = actual == 1
    elif kind == "formal_charge_equals":
        actual = sum(int(atom.GetFormalCharge()) for atom in raw_mol.GetAtoms())
        passed = actual == int(predicate.get("value") or 0)
    elif kind == "max_heavy_atoms":
        actual = int(raw_mol.GetNumHeavyAtoms())
        passed = actual <= int(predicate.get("value") or 0)
    elif kind == "ring_count_equals":
        actual = int(raw_mol.GetRingInfo().NumRings())
        passed = actual == int(predicate.get("value") or 0)
    elif kind == "aromatic_ring_count_equals":
        actual = _count_aromatic_rings(raw_mol)
        passed = actual == int(predicate.get("value") or 0)
    elif kind == "aromatic_atom_count_equals":
        actual = sum(1 for atom in raw_mol.GetAtoms() if atom.GetIsAromatic())
        passed = actual == int(predicate.get("value") or 0)
    elif kind == "element_count_equals":
        symbol = str(predicate.get("element") or "").strip()
        actual = sum(1 for atom in raw_mol.GetAtoms() if atom.GetSymbol() == symbol)
        passed = actual == int(predicate.get("value") or 0)
    elif kind == "element_subset":
        allowed = {str(item).strip() for item in list(predicate.get("allowed") or []) if str(item).strip()}
        actual = sorted({atom.GetSymbol() for atom in raw_mol.GetAtoms()})
        passed = all(symbol in allowed for symbol in actual)
    elif kind == "hbd_count_equals":
        actual = int(Lipinski.NumHDonors(raw_mol))
        passed = actual == int(predicate.get("value") or 0)
    elif kind == "hba_count_equals":
        actual = int(Lipinski.NumHAcceptors(raw_mol))
        passed = actual == int(predicate.get("value") or 0)
    elif kind == "smarts_match_count_equals":
        matches = _count_smarts_matches(raw_mol, str(predicate.get("smarts") or ""))
        actual = matches
        passed = matches == int(predicate.get("value") or 0)
    elif kind == "smarts_match_count_at_least":
        matches = _count_smarts_matches(raw_mol, str(predicate.get("smarts") or ""))
        actual = matches
        passed = matches >= int(predicate.get("value") or 0)
    elif kind == "canonical_smiles_in":
        values = {str(item).strip() for item in list(predicate.get("values") or []) if str(item).strip()}
        actual = canonical_smiles
        passed = canonical_smiles in values
    elif kind == "no_supported_force_field":
        actual = False
        passed = False
    else:
        raise ConformaBenchJudgeError(f"Unsupported ConformaBench predicate kind: {kind}")
    return {
        "predicate_id": predicate_id,
        "kind": kind,
        "passed": bool(passed),
        "actual": actual,
        "expected": predicate.get("value"),
    }


def _evaluate_geometry_predicate(predicate: dict[str, Any], *, observable_values: dict[str, float | None]) -> dict[str, Any]:
    observable_id = str(predicate.get("observable") or "").strip()
    actual = observable_values.get(observable_id)
    operator = str(predicate.get("operator") or "").strip()
    expected = predicate.get("value")
    passed = False
    if actual is not None:
        if operator == "<=":
            passed = actual <= float(expected)
        elif operator == ">=":
            passed = actual >= float(expected)
        elif operator == "in_range":
            lower, upper = list(expected or [None, None])
            passed = float(lower) <= actual <= float(upper)
        else:
            raise ConformaBenchJudgeError(f"Unsupported geometry operator: {operator}")
    return {
        "predicate_id": str(predicate.get("predicate_id") or "geometry"),
        "kind": str(predicate.get("kind") or "geometry_lowest_conformer"),
        "observable": observable_id,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
        "operator": operator,
    }


def _embed_conformers(
    mol: Any,
    *,
    AllChem: Any,
    method: str,
    num_confs: int,
    random_seed: int,
    prune_rms_thresh: float,
    use_random_coords: bool,
) -> list[int]:
    if method != "ETKDGv3":
        raise ConformaBenchJudgeError(f"Unsupported embedding method: {method}")
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    params.pruneRmsThresh = float(prune_rms_thresh)
    params.useRandomCoords = bool(use_random_coords)
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=int(num_confs), params=params)
    return [int(conf_id) for conf_id in conf_ids]


def _select_force_field_name(mol: Any, *, AllChem: Any, policy: dict[str, Any]) -> str | None:
    preferred_order = [str(item).strip() for item in list(policy.get("preferred_order") or []) if str(item).strip()]
    require_coverage = bool(policy.get("require_parameter_coverage", True))
    for name in preferred_order:
        if name in {"MMFF94", "MMFF94s"}:
            has_params = bool(AllChem.MMFFHasAllMoleculeParams(mol))
            if require_coverage and not has_params:
                continue
            if has_params:
                return name
        elif name == "UFF":
            has_params = bool(AllChem.UFFHasAllMoleculeParams(mol))
            if require_coverage and not has_params:
                continue
            if has_params:
                return name
        else:
            raise ConformaBenchJudgeError(f"Unsupported force field policy entry: {name}")
    return None


def _build_force_field(mol: Any, *, AllChem: Any, force_field_name: str, conf_id: int) -> Any | None:
    if force_field_name in {"MMFF94", "MMFF94s"}:
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=force_field_name)
        if props is None:
            return None
        return AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
    if force_field_name == "UFF":
        return AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
    raise ConformaBenchJudgeError(f"Unsupported force field: {force_field_name}")


def _compute_observable(mol: Any, *, conf_id: int, observable: dict[str, Any], Chem: Any) -> float | None:
    kind = str(observable.get("kind") or "").strip()
    if kind == "max_heavy_atom_distance":
        return _compute_max_heavy_atom_distance(mol, conf_id=conf_id)
    if kind == "distance_from_smarts_match":
        return _compute_smarts_measurement(mol, Chem=Chem, conf_id=conf_id, observable=observable, arity=2)
    if kind == "angle_from_smarts_match":
        return _compute_smarts_measurement(mol, Chem=Chem, conf_id=conf_id, observable=observable, arity=3)
    if kind == "dihedral_from_smarts_match":
        return _compute_smarts_measurement(mol, Chem=Chem, conf_id=conf_id, observable=observable, arity=4)
    if kind == "distance_between_queries":
        return _compute_distance_between_queries(mol, Chem=Chem, conf_id=conf_id, observable=observable)
    raise ConformaBenchJudgeError(f"Unsupported ConformaBench observable kind: {kind}")


def _compute_max_heavy_atom_distance(mol: Any, *, conf_id: int) -> float | None:
    conf = mol.GetConformer(conf_id)
    heavy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if len(heavy_indices) < 2:
        return 0.0
    max_distance = 0.0
    for i, first_idx in enumerate(heavy_indices):
        first = conf.GetAtomPosition(first_idx)
        for second_idx in heavy_indices[i + 1 :]:
            second = conf.GetAtomPosition(second_idx)
            distance = first.Distance(second)
            if distance > max_distance:
                max_distance = distance
    return float(max_distance)


def _compute_smarts_measurement(mol: Any, *, Chem: Any, conf_id: int, observable: dict[str, Any], arity: int) -> float | None:
    smarts = str(observable.get("smarts") or "").strip()
    atom_map_numbers = [int(item) for item in list(observable.get("atom_map_numbers") or [])]
    if len(atom_map_numbers) != arity:
        raise ConformaBenchJudgeError(
            f"Observable `{observable.get('observable_id')}` requires {arity} atom map numbers, got {atom_map_numbers}."
        )
    matches = _get_mapped_matches(mol, Chem=Chem, smarts=smarts, atom_map_numbers=atom_map_numbers)
    values = []
    for atom_indices in matches:
        if arity == 2:
            values.append(_distance(mol, conf_id=conf_id, atom_indices=atom_indices))
        elif arity == 3:
            values.append(_angle(mol, conf_id=conf_id, atom_indices=atom_indices))
        elif arity == 4:
            wrap_mode = str(observable.get("wrap_mode") or "fold_to_180").strip()
            values.append(_dihedral(mol, conf_id=conf_id, atom_indices=atom_indices, wrap_mode=wrap_mode))
    return _aggregate_values(values, aggregation=str(observable.get("aggregation") or "min").strip())


def _compute_distance_between_queries(mol: Any, *, Chem: Any, conf_id: int, observable: dict[str, Any]) -> float | None:
    query_a = _mapping(observable.get("query_a"))
    query_b = _mapping(observable.get("query_b"))
    atoms_a = _resolve_query_atom_indices(mol, Chem=Chem, query=query_a)
    atoms_b = _resolve_query_atom_indices(mol, Chem=Chem, query=query_b)
    require_distinct = bool(observable.get("require_distinct_atoms", True))
    values = []
    for atom_a in atoms_a:
        for atom_b in atoms_b:
            if require_distinct and atom_a == atom_b:
                continue
            values.append(_distance(mol, conf_id=conf_id, atom_indices=(atom_a, atom_b)))
    return _aggregate_values(values, aggregation=str(observable.get("aggregation") or "min").strip())


def _resolve_query_atom_indices(mol: Any, *, Chem: Any, query: dict[str, Any]) -> list[int]:
    smarts = str(query.get("smarts") or "").strip()
    atom_map_number = int(query.get("atom_map_number") or 1)
    matches = _get_mapped_matches(mol, Chem=Chem, smarts=smarts, atom_map_numbers=[atom_map_number])
    return [match[0] for match in matches]


def _get_mapped_matches(mol: Any, *, Chem: Any, smarts: str, atom_map_numbers: list[int]) -> list[tuple[int, ...]]:
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        raise ConformaBenchJudgeError(f"Invalid SMARTS pattern in ConformaBench spec: {smarts}")
    map_positions: dict[int, int] = {}
    for atom in pattern.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num:
            map_positions[map_num] = atom.GetIdx()
    if any(map_num not in map_positions for map_num in atom_map_numbers):
        raise ConformaBenchJudgeError(
            f"SMARTS pattern `{smarts}` does not expose all requested atom maps: {atom_map_numbers}"
        )
    ordered_positions = [map_positions[map_num] for map_num in atom_map_numbers]
    matches = mol.GetSubstructMatches(pattern, uniquify=True)
    return [tuple(int(match[position]) for position in ordered_positions) for match in matches]


def _aggregate_values(values: list[float], *, aggregation: str) -> float | None:
    if not values:
        return None
    if aggregation == "first":
        return float(values[0])
    if aggregation == "min":
        return float(min(values))
    if aggregation == "max":
        return float(max(values))
    raise ConformaBenchJudgeError(f"Unsupported observable aggregation mode: {aggregation}")


def _distance(mol: Any, *, conf_id: int, atom_indices: tuple[int, int]) -> float:
    conf = mol.GetConformer(conf_id)
    first = conf.GetAtomPosition(int(atom_indices[0]))
    second = conf.GetAtomPosition(int(atom_indices[1]))
    return float(first.Distance(second))


def _angle(mol: Any, *, conf_id: int, atom_indices: tuple[int, int, int]) -> float:
    conf = mol.GetConformer(conf_id)
    a = conf.GetAtomPosition(int(atom_indices[0]))
    b = conf.GetAtomPosition(int(atom_indices[1]))
    c = conf.GetAtomPosition(int(atom_indices[2]))
    return float(_angle_between_points(a, b, c))


def _dihedral(mol: Any, *, conf_id: int, atom_indices: tuple[int, int, int, int], wrap_mode: str) -> float:
    from rdkit.Chem import rdMolTransforms

    value = float(rdMolTransforms.GetDihedralDeg(mol.GetConformer(conf_id), *[int(item) for item in atom_indices]))
    absolute = abs(value)
    folded = absolute if absolute <= 180.0 else absolute % 180.0
    if wrap_mode == "raw":
        return value
    if wrap_mode == "fold_to_180":
        return float(min(folded, abs(360.0 - folded)))
    if wrap_mode == "nearest_coplanar_deviation":
        folded_180 = min(folded, abs(360.0 - folded))
        return float(min(folded_180, abs(180.0 - folded_180)))
    raise ConformaBenchJudgeError(f"Unsupported dihedral wrap mode: {wrap_mode}")


def _angle_between_points(a: Any, b: Any, c: Any) -> float:
    ba = (a.x - b.x, a.y - b.y, a.z - b.z)
    bc = (c.x - b.x, c.y - b.y, c.z - b.z)
    norm_ba = math.sqrt(sum(component * component for component in ba))
    norm_bc = math.sqrt(sum(component * component for component in bc))
    if norm_ba <= 1e-12 or norm_bc <= 1e-12:
        return 0.0
    cosine = sum(lhs * rhs for lhs, rhs in zip(ba, bc)) / (norm_ba * norm_bc)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _count_aromatic_rings(mol: Any) -> int:
    ring_info = mol.GetRingInfo()
    count = 0
    for atom_indices in ring_info.AtomRings():
        if atom_indices and all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in atom_indices):
            count += 1
    return count


def _count_smarts_matches(mol: Any, smarts: str) -> int:
    from rdkit import Chem

    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        raise ConformaBenchJudgeError(f"Invalid SMARTS pattern in predicate: {smarts}")
    return len(mol.GetSubstructMatches(pattern, uniquify=True))


def _get_fragment_atom_tuples(mol: Any) -> tuple[tuple[int, ...], ...]:
    from rdkit import Chem

    return tuple(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False))
