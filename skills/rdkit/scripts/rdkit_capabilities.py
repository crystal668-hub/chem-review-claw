from __future__ import annotations

from typing import Any

from rdkit_skill_common import ProcessingError, RequestError, append_warning, get_required_string, load_molecule, load_molecule_list


def canonicalize(request: dict[str, Any], rdkit_ctx: dict[str, Any]) -> dict[str, Any]:
    Chem = rdkit_ctx["Chem"]
    strip_atom_maps = bool(request.get("strip_atom_maps") or request.get("strip_atom_mapping"))
    mol, metadata = load_molecule(rdkit_ctx, request.get("molecule"), strip_atom_maps=strip_atom_maps)
    primary_result = {
        **metadata,
        "atom_count": int(mol.GetNumAtoms()),
        "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
        "formal_charge": int(Chem.GetFormalCharge(mol)),
        "strip_atom_maps_applied": strip_atom_maps,
    }
    return {
        "status": "success",
        "primary_result": primary_result,
        "source_trace": [{"provider": "rdkit", "operation": "canonicalize"}],
    }


def descriptors(request: dict[str, Any], rdkit_ctx: dict[str, Any]) -> dict[str, Any]:
    Chem = rdkit_ctx["Chem"]
    Crippen = rdkit_ctx["Crippen"]
    Descriptors = rdkit_ctx["Descriptors"]
    Lipinski = rdkit_ctx["Lipinski"]
    rdMolDescriptors = rdkit_ctx["rdMolDescriptors"]

    mol, metadata = load_molecule(rdkit_ctx, request.get("molecule"))
    primary_result = {
        **metadata,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": round(float(Descriptors.MolWt(mol)), 6),
        "exact_mass": round(float(rdMolDescriptors.CalcExactMolWt(mol)), 6),
        "formal_charge": int(Chem.GetFormalCharge(mol)),
        "atom_count": int(mol.GetNumAtoms()),
        "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
        "h_bond_donor_count": int(Lipinski.NumHDonors(mol)),
        "h_bond_acceptor_count": int(Lipinski.NumHAcceptors(mol)),
        "rotatable_bond_count": int(Lipinski.NumRotatableBonds(mol)),
        "tpsa": round(float(rdMolDescriptors.CalcTPSA(mol)), 6),
        "logp": round(float(Crippen.MolLogP(mol)), 6),
        "ring_count": int(len(mol.GetRingInfo().AtomRings())),
    }
    return {
        "status": "success",
        "primary_result": primary_result,
        "source_trace": [{"provider": "rdkit", "operation": "descriptors"}],
    }


def _functional_group_patterns(rdkit_ctx: dict[str, Any]) -> list[dict[str, Any]]:
    Chem = rdkit_ctx["Chem"]
    entries = [
        ("alcohol", "[OX2H][CX4;!$(C[O,N,S]=O)]"),
        ("amine", "[NX3;!$(NC=O);!$([N+](=O)[O-])]"),
        ("carbonyl", "[CX3]=[OX1]"),
        ("carboxylic_acid", "[CX3](=O)[OX2H1]"),
        ("ester", "[CX3](=O)[OX2H0][#6]"),
        ("amide", "[CX3](=O)[NX3]"),
        ("alkene", "[CX3]=[CX3]"),
        ("alkyne", "[CX2]#[CX2,CH1,CH0]"),
        ("aryl_halide", "[c][F,Cl,Br,I]"),
        ("nitrile", "[CX2]#N"),
        ("nitro", "[$([NX3](=O)=O),$([N+](=O)[O-])]"),
        ("thiol", "[SX2H]"),
    ]
    patterns: list[dict[str, Any]] = []
    for name, smarts in entries:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            raise ProcessingError("invalid_internal_smarts", f"Internal SMARTS pattern failed to parse for `{name}`.")
        patterns.append({"name": name, "smarts": smarts, "pattern": pattern})
    return patterns


def functional_groups(request: dict[str, Any], rdkit_ctx: dict[str, Any]) -> dict[str, Any]:
    mol, metadata = load_molecule(rdkit_ctx, request.get("molecule"))
    candidates: list[dict[str, Any]] = []
    matched_groups: list[str] = []
    for entry in _functional_group_patterns(rdkit_ctx):
        matches = mol.GetSubstructMatches(entry["pattern"])
        matched = bool(matches)
        if matched:
            matched_groups.append(entry["name"])
        candidates.append(
            {
                "name": entry["name"],
                "smarts": entry["smarts"],
                "matched": matched,
                "match_count": len(matches),
                "matches": [list(match) for match in matches],
            }
        )
    return {
        "status": "success",
        "primary_result": {
            **metadata,
            "matched_groups": matched_groups,
            "matched_group_count": len(matched_groups),
        },
        "candidates": candidates,
        "source_trace": [{"provider": "rdkit", "operation": "functional_groups"}],
    }


def _builtin_substructure_query(name: str, rdkit_ctx: dict[str, Any]) -> tuple[Any, str]:
    Chem = rdkit_ctx["Chem"]
    library = {
        "benzene": "c1ccccc1",
        "hydroxyl": "[OX2H]",
        "carbonyl": "[CX3]=[OX1]",
        "amine": "[NX3;!$(NC=O)]",
    }
    smarts = library.get(name)
    if smarts is None:
        raise RequestError("unknown_query_name", f"Unknown built-in substructure query: {name}")
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        raise ProcessingError("invalid_internal_smarts", f"Built-in SMARTS failed to parse for `{name}`.")
    return query, smarts


def substructure(request: dict[str, Any], rdkit_ctx: dict[str, Any]) -> dict[str, Any]:
    Chem = rdkit_ctx["Chem"]
    query_spec = request.get("query")
    if not isinstance(query_spec, dict):
        raise RequestError("missing_query", "Request field `query` must be an object.")
    query_smarts = str(query_spec.get("smarts") or "").strip()
    if query_smarts:
        query = Chem.MolFromSmarts(query_smarts)
        if query is None:
            raise ProcessingError("query_parse_error", "The provided SMARTS query could not be parsed.")
    else:
        query_name = get_required_string(query_spec, "name")
        query, query_smarts = _builtin_substructure_query(query_name, rdkit_ctx)

    raw_molecules = request.get("molecules")
    if raw_molecules is None:
        raw_molecules = [request.get("molecule")]
    molecules = load_molecule_list(rdkit_ctx, raw_molecules)
    match_rows: list[dict[str, Any]] = []
    for index, (mol, metadata) in enumerate(molecules):
        matches = mol.GetSubstructMatches(query)
        match_rows.append(
            {
                "index": index,
                "input_value": metadata["input_value"],
                "canonical_smiles": metadata["canonical_smiles"],
                "matched": bool(matches),
                "match_count": len(matches),
                "matches": [list(match) for match in matches],
            }
        )
    return {
        "status": "success",
        "primary_result": {
            "query_smarts": query_smarts,
            "matched_count": sum(1 for row in match_rows if row["matched"]),
            "matches": match_rows,
        },
        "candidates": match_rows,
        "source_trace": [{"provider": "rdkit", "operation": "substructure"}],
    }


def rings_aromaticity(request: dict[str, Any], rdkit_ctx: dict[str, Any]) -> dict[str, Any]:
    mol, metadata = load_molecule(rdkit_ctx, request.get("molecule"))
    atom_rings = [tuple(ring) for ring in mol.GetRingInfo().AtomRings()]
    aromatic_atom_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIsAromatic()]
    aromatic_rings = [ring for ring in atom_rings if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)]
    heteroaromatic_rings = [
        ring for ring in aromatic_rings if any(mol.GetAtomWithIdx(idx).GetAtomicNum() not in {1, 6} for idx in ring)
    ]
    fused_ring_pair_count = 0
    for left_index, left_ring in enumerate(atom_rings):
        left_atoms = set(left_ring)
        for right_ring in atom_rings[left_index + 1 :]:
            if len(left_atoms.intersection(right_ring)) >= 2:
                fused_ring_pair_count += 1

    primary_result = {
        **metadata,
        "ring_count": len(atom_rings),
        "ring_sizes": sorted(len(ring) for ring in atom_rings),
        "aromatic_ring_count": len(aromatic_rings),
        "aromatic_atom_count": len(aromatic_atom_indices),
        "aromatic_atom_indices": aromatic_atom_indices,
        "heteroaromatic_ring_count": len(heteroaromatic_rings),
        "fused_ring_pair_count": fused_ring_pair_count,
    }
    return {
        "status": "success",
        "primary_result": primary_result,
        "source_trace": [{"provider": "rdkit", "operation": "rings_aromaticity"}],
    }


def stereochemistry(request: dict[str, Any], rdkit_ctx: dict[str, Any]) -> dict[str, Any]:
    Chem = rdkit_ctx["Chem"]
    mol, metadata = load_molecule(rdkit_ctx, request.get("molecule"))
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, includeCIP=True)
    stereo_entries = []
    specified_chiral_center_count = 0
    for atom_index, label in chiral_centers:
        is_specified = label != "?"
        specified_chiral_center_count += int(is_specified)
        stereo_entries.append({"atom_index": int(atom_index), "label": label, "specified": is_specified})

    double_bond_entries = []
    potential = Chem.FindPotentialStereo(mol)
    for entry in potential:
        if str(entry.type) != "Bond_Double":
            continue
        specified = str(entry.specified) == "Specified"
        bond = mol.GetBondWithIdx(entry.centeredOn)
        double_bond_entries.append(
            {
                "bond_index": int(entry.centeredOn),
                "begin_atom_index": int(bond.GetBeginAtomIdx()),
                "end_atom_index": int(bond.GetEndAtomIdx()),
                "specified": specified,
                "stereo": str(bond.GetStereo()),
            }
        )

    primary_result = {
        **metadata,
        "chiral_center_count": len(stereo_entries),
        "specified_chiral_center_count": specified_chiral_center_count,
        "unspecified_chiral_center_count": len(stereo_entries) - specified_chiral_center_count,
        "double_bond_stereo_count": len(double_bond_entries),
        "specified_double_bond_stereo_count": sum(1 for item in double_bond_entries if item["specified"]),
        "unspecified_double_bond_stereo_count": sum(1 for item in double_bond_entries if not item["specified"]),
        "chiral_centers": stereo_entries,
        "double_bond_stereochemistry": double_bond_entries,
    }
    return {
        "status": "success",
        "primary_result": primary_result,
        "source_trace": [{"provider": "rdkit", "operation": "stereochemistry"}],
    }


def similarity(request: dict[str, Any], rdkit_ctx: dict[str, Any]) -> dict[str, Any]:
    DataStructs = rdkit_ctx["DataStructs"]
    rdFingerprintGenerator = rdkit_ctx["rdFingerprintGenerator"]

    query_mol, query_metadata = load_molecule(rdkit_ctx, request.get("query"))
    raw_candidates = request.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise RequestError("invalid_candidates", "Request field `candidates` must be a non-empty list.")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    query_fp = generator.GetFingerprint(query_mol)

    ranked_candidates = []
    for index, spec in enumerate(raw_candidates):
        mol, metadata = load_molecule(rdkit_ctx, spec)
        candidate_id = str(spec.get("id") or f"candidate_{index}")
        similarity_score = float(DataStructs.TanimotoSimilarity(query_fp, generator.GetFingerprint(mol)))
        ranked_candidates.append(
            {
                "id": candidate_id,
                "input_value": metadata["input_value"],
                "canonical_smiles": metadata["canonical_smiles"],
                "isomeric_smiles": metadata["isomeric_smiles"],
                "similarity": round(similarity_score, 12),
                "_index": index,
            }
        )

    ranked_candidates.sort(key=lambda item: (-item["similarity"], item["canonical_smiles"], item["id"], item["_index"]))
    for rank, item in enumerate(ranked_candidates, start=1):
        item["rank"] = rank
        item.pop("_index", None)

    primary_result = {
        "query": query_metadata,
        "candidate_count": len(ranked_candidates),
        "fingerprint": {"type": "morgan", "radius": 2, "fp_size": 2048},
        "top_candidate": ranked_candidates[0] if ranked_candidates else None,
    }
    return {
        "status": "success",
        "primary_result": primary_result,
        "candidates": ranked_candidates,
        "source_trace": [{"provider": "rdkit", "operation": "similarity"}],
    }


def reaction_smarts(request: dict[str, Any], rdkit_ctx: dict[str, Any]) -> dict[str, Any]:
    Chem = rdkit_ctx["Chem"]
    rdChemReactions = rdkit_ctx["rdChemReactions"]

    reaction_smarts_text = get_required_string(request, "reaction_smarts")
    reaction = rdChemReactions.ReactionFromSmarts(reaction_smarts_text)
    if reaction is None:
        raise ProcessingError("reaction_parse_error", "The provided reaction SMARTS could not be parsed.")

    reactant_specs = request.get("reactants")
    if not isinstance(reactant_specs, list) or not reactant_specs:
        raise RequestError("invalid_reactants", "Request field `reactants` must be a non-empty list.")
    reactants = [load_molecule(rdkit_ctx, spec)[0] for spec in reactant_specs]

    raw_product_sets = reaction.RunReactants(tuple(reactants))
    if not raw_product_sets:
        raise ProcessingError(
            "reaction_no_match",
            "The supplied reactants did not produce any product candidates for the reaction SMARTS.",
            primary_result={"reaction_smarts": reaction_smarts_text, "product_sets": []},
        )

    seen: set[tuple[str, ...]] = set()
    product_sets = []
    for products in raw_product_sets:
        normalized_products = []
        smiles_key = []
        for product in products:
            try:
                Chem.SanitizeMol(product)
            except Exception as exc:
                raise ProcessingError("product_sanitize_error", f"Reaction product sanitization failed: {exc}") from exc
            canonical_smiles = Chem.MolToSmiles(product, canonical=True)
            isomeric_smiles = Chem.MolToSmiles(product, canonical=True, isomericSmiles=True)
            smiles_key.append(canonical_smiles)
            normalized_products.append(
                {
                    "canonical_smiles": canonical_smiles,
                    "isomeric_smiles": isomeric_smiles,
                }
            )
        key = tuple(smiles_key)
        if key in seen:
            continue
        seen.add(key)
        product_sets.append({"products": normalized_products})

    product_sets.sort(key=lambda item: [product["canonical_smiles"] for product in item["products"]])
    return {
        "status": "success",
        "primary_result": {
            "reaction_smarts": reaction_smarts_text,
            "reactant_count": len(reactants),
            "product_sets": product_sets,
        },
        "candidates": product_sets,
        "source_trace": [{"provider": "rdkit", "operation": "reaction_smarts"}],
    }


def conformer_embed(request: dict[str, Any], rdkit_ctx: dict[str, Any]) -> dict[str, Any]:
    Chem = rdkit_ctx["Chem"]
    AllChem = rdkit_ctx["AllChem"]

    num_conformers = int(request.get("num_conformers", 1))
    if num_conformers <= 0:
        raise RequestError(
            "invalid_num_conformers",
            "Request field `num_conformers` must be a positive integer.",
            primary_result={"embedded_conformer_count": 0},
        )
    random_seed = int(request.get("random_seed", 20260427))

    mol, metadata = load_molecule(rdkit_ctx, request.get("molecule"))
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed

    if num_conformers == 1:
        conf_id = AllChem.EmbedMolecule(mol, params)
        conformer_ids = [] if conf_id < 0 else [int(conf_id)]
    else:
        conformer_ids = [int(conf_id) for conf_id in AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)]

    if not conformer_ids:
        raise ProcessingError(
            "embed_failed",
            "RDKit could not embed the requested conformers.",
            primary_result={"embedded_conformer_count": 0},
        )

    force_field_name = ""
    energies: list[float] = []
    if AllChem.MMFFHasAllMoleculeParams(mol):
        force_field_name = "MMFF94"
        mmff_properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
        optimize_results = AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant="MMFF94")
        for conf_id in conformer_ids:
            force_field = AllChem.MMFFGetMoleculeForceField(mol, mmff_properties, confId=conf_id)
            energies.append(round(float(force_field.CalcEnergy()), 6))
    elif AllChem.UFFHasAllMoleculeParams(mol):
        force_field_name = "UFF"
        optimize_results = AllChem.UFFOptimizeMoleculeConfs(mol)
        for conf_id in conformer_ids:
            force_field = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            energies.append(round(float(force_field.CalcEnergy()), 6))
    else:
        optimize_results = [(1, 0.0) for _ in conformer_ids]

    status = "success" if force_field_name else "partial"
    warnings = []
    if not force_field_name:
        warning_payload = {"code": "force_field_unavailable", "message": "No RDKit force field was available for optimization."}
        warnings.append(warning_payload)

    primary_result = {
        **metadata,
        "embedded_conformer_count": len(conformer_ids),
        "optimized_conformer_count": len(conformer_ids),
        "force_field": force_field_name,
        "random_seed": random_seed,
        "conformer_ids": conformer_ids,
        "optimization_status_codes": [int(item[0]) for item in optimize_results],
        "energies_kcal_mol": energies,
        "lowest_energy_kcal_mol": min(energies) if energies else None,
    }
    return {
        "status": status,
        "primary_result": primary_result,
        "warnings": warnings,
        "source_trace": [{"provider": "rdkit", "operation": "conformer_embed"}],
    }


def nmr_symmetry_heuristics(request: dict[str, Any], rdkit_ctx: dict[str, Any]) -> dict[str, Any]:
    Chem = rdkit_ctx["Chem"]
    mol, metadata = load_molecule(rdkit_ctx, request.get("molecule"))
    mol_h = Chem.AddHs(mol)

    carbon_ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    proton_ranks = list(Chem.CanonicalRankAtoms(mol_h, breakTies=False))
    carbon_classes: dict[int, list[int]] = {}
    proton_classes: dict[int, list[int]] = {}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        carbon_classes.setdefault(int(carbon_ranks[atom.GetIdx()]), []).append(int(atom.GetIdx()))
    for atom in mol_h.GetAtoms():
        if atom.GetSymbol() != "H":
            continue
        proton_classes.setdefault(int(proton_ranks[atom.GetIdx()]), []).append(int(atom.GetIdx()))

    warnings: list[dict[str, Any]] = []
    placeholder_payload = {"status": "partial"}
    append_warning(placeholder_payload, "heuristic_only", "Graph symmetry heuristics do not replace expert NMR interpretation.")
    append_warning(
        placeholder_payload,
        "missing_effects",
        "Conformation, tautomerism, solvent effects, and accidental equivalence are not modeled by this heuristic.",
    )
    warnings.extend(placeholder_payload["warnings"])

    primary_result = {
        **metadata,
        "proton_equivalence_class_count": len(proton_classes),
        "carbon_equivalence_class_count": len(carbon_classes),
        "proton_equivalence_classes": sorted(
            ({"rank": rank, "atom_indices": atom_indices} for rank, atom_indices in proton_classes.items()),
            key=lambda item: (len(item["atom_indices"]), item["atom_indices"]),
        ),
        "carbon_equivalence_classes": sorted(
            ({"rank": rank, "atom_indices": atom_indices} for rank, atom_indices in carbon_classes.items()),
            key=lambda item: (len(item["atom_indices"]), item["atom_indices"]),
        ),
    }
    return {
        "status": "partial",
        "primary_result": primary_result,
        "warnings": warnings,
        "source_trace": [{"provider": "rdkit", "operation": "nmr_symmetry_heuristics"}],
    }
