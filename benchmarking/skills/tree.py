from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = ROOT / "skills" / "chemistry-routing-matrix.json"


SKILL_TREE: tuple[dict[str, Any], ...] = (
    {
        "id": "chemist-sop",
        "label": "Chemist benchmark SOP",
        "families": (
            {
                "id": "chemistry-reasoning-sop",
                "label": "Chemist-style reasoning, verification, and answer tracing",
                "skills": ("act-like-a-chemist",),
            },
        ),
    },
    {
        "id": "calculation-math",
        "label": "Calculation and formula math",
        "families": (
            {
                "id": "deterministic-chemistry-calculation",
                "label": "Deterministic chemistry calculation",
                "skills": ("chem-calculator",),
            },
        ),
    },
    {
        "id": "molecular-structure-identity",
        "label": "Molecular structure, identity, and cheminformatics",
        "families": (
            {
                "id": "structure-toolkit",
                "label": "Structure parsing, descriptors, fingerprints, and scaffolds",
                "skills": ("rdkit", "datamol", "molfeat"),
            },
            {
                "id": "name-resolution",
                "label": "Chemical name and identifier resolution",
                "skills": ("opsin", "pubchem"),
            },
            {
                "id": "compound-profile",
                "label": "Compound public records and profiles",
                "skills": ("pubchem-database", "chemistry-query"),
            },
            {
                "id": "medchem-filters",
                "label": "Medicinal chemistry filters and drug-likeness checks",
                "skills": ("medchem",),
            },
        ),
    },
    {
        "id": "literature-evidence",
        "label": "Literature evidence and paper processing",
        "families": (
            {
                "id": "paper-pipeline",
                "label": "Paper retrieval, access, and parsing",
                "skills": ("paper-retrieval", "paper-access", "paper-parse"),
            },
            {
                "id": "literature-databases",
                "label": "Bibliographic databases",
                "skills": ("pubmed-database", "openalex-database"),
            },
            {
                "id": "review-synthesis",
                "label": "Literature review and synthesis",
                "skills": ("literature-review", "synthesize-literature"),
            },
        ),
    },
    {
        "id": "bioactivity-safety-discovery",
        "label": "Bioactivity, safety, and discovery databases",
        "families": (
            {
                "id": "bioactivity-databases",
                "label": "Bioactivity, screening, and purchasable compounds",
                "skills": ("chembl-database", "zinc-database"),
            },
            {
                "id": "tooluniverse-discovery",
                "label": "ToolUniverse chemical safety, retrieval, and small-molecule discovery",
                "skills": (
                    "tooluniverse-chemical-safety",
                    "tooluniverse-small-molecule-discovery",
                    "tooluniverse-chemical-compound-retrieval",
                ),
            },
            {
                "id": "biological-databases",
                "label": "Protein, pathway, and structure databases",
                "skills": ("pdb-database", "alphafold-database", "reactome-database"),
            },
        ),
    },
    {
        "id": "materials-crystals",
        "label": "Materials, crystals, and solid-state data",
        "families": (
            {
                "id": "crystal-structure",
                "label": "Crystal structure parsing, analysis, and editing",
                "skills": ("pymatgen", "ase", "cod"),
            },
            {
                "id": "materials-databases",
                "label": "Materials databases and property lookup",
                "skills": ("materials-project", "oqmd", "jarvis"),
            },
            {
                "id": "specialist-materials",
                "label": "Specialist materials workflows",
                "skills": ("doped-perovskite-structure-analysis",),
            },
        ),
    },
    {
        "id": "simulation-forcefield-md",
        "label": "Simulation, force fields, and molecular dynamics",
        "families": (
            {
                "id": "md-analysis",
                "label": "Molecular dynamics setup and trajectory analysis",
                "skills": ("molecular-dynamics", "openmm"),
            },
            {
                "id": "forcefield-parameterization",
                "label": "Force-field assignment and parameterization",
                "skills": ("open-forcefield-toolkit", "atb"),
            },
        ),
    },
    {
        "id": "quantum-hpc",
        "label": "Quantum chemistry, HPC engines, and parsed outputs",
        "families": (
            {
                "id": "quantum-inputs",
                "label": "Quantum chemistry input preparation and engine guidance",
                "skills": (
                    "q-chem",
                    "hpc-orca",
                    "hpc-gaussian",
                    "hpc-pyscf",
                    "hpc-xtb",
                    "hpc-vasp",
                    "hpc-nwchem",
                    "hpc-cp2k",
                    "hpc-quantum-espresso",
                ),
            },
            {
                "id": "local-xtb-cli",
                "label": "Local xTB CLI calculations and property extraction",
                "skills": ("xtb-cli",),
            },
            {
                "id": "quantum-output-analysis",
                "label": "Quantum output parsing and benchmark references",
                "skills": ("cclib", "qc-output-analysis", "cccbdb", "molssi-qca"),
            },
        ),
    },
    {
        "id": "spectra-formats-visualization",
        "label": "Spectra, chemical formats, and visualization",
        "families": (
            {
                "id": "spectra-analysis",
                "label": "Spectral formats and spectral interpretation",
                "skills": ("spectral-analysis", "jcamp-dx"),
            },
            {
                "id": "chemical-file-formats",
                "label": "Chemical and crystallographic file formats",
                "skills": ("cif", "cml", "blue-obelisk"),
            },
            {
                "id": "structure-visualization",
                "label": "Crystal and structure visualization",
                "skills": ("crystal-viewer", "xtal2png"),
            },
        ),
    },
    {
        "id": "ml-generative-modeling",
        "label": "ML potentials, generative modeling, and property prediction",
        "families": (
            {
                "id": "ml-potentials",
                "label": "Atomistic ML potentials",
                "skills": ("mace", "chgnet", "mattersim", "schnet", "nequip", "orb", "reann", "torchmd-net"),
            },
            {
                "id": "generative-materials",
                "label": "Generative crystal and materials models",
                "skills": ("mattergen", "diffcsp", "crystalflow"),
            },
            {
                "id": "molecular-ml",
                "label": "Molecular ML models",
                "skills": ("chemprop",),
            },
            {
                "id": "materials-ml",
                "label": "Materials ML datasets and models",
                "skills": ("matformer", "matminer", "matbench", "modnet", "crabnet", "xenonpy"),
            },
        ),
    },
    {
        "id": "workflow-automation",
        "label": "Workflow automation and interoperable infrastructure",
        "families": (
            {
                "id": "materials-api-interoperability",
                "label": "Materials API interoperability",
                "skills": ("optimade", "optimade-python-tools"),
            },
            {
                "id": "workflow-engines",
                "label": "Workflow engines and job orchestration",
                "skills": ("aiida", "atomate", "fireworks", "custodian", "quacc", "pyiron", "qmflows", "qmforge"),
            },
        ),
    },
)


@lru_cache(maxsize=1)
def load_chemistry_skill_inventory() -> dict[str, Any]:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def benchmark_skill_allowlist() -> tuple[str, ...]:
    return tuple(
        str(entry["skill"])
        for entry in load_chemistry_skill_inventory().get("skills", [])
        if entry.get("single_agent_exposure") is True
    )


def load_skill_tree() -> tuple[dict[str, Any], ...]:
    return SKILL_TREE


def lookup_skill_family(family_id: str) -> dict[str, Any]:
    normalized = str(family_id or "").strip().lower()
    for domain in SKILL_TREE:
        for family in domain["families"]:
            if str(family["id"]).lower() == normalized:
                return family
    raise KeyError(f"unknown skill family: {family_id}")


def render_top_level_skill_tree(available_skills: set[str] | None = None) -> str:
    inventory_by_skill = {
        str(entry["skill"]): entry
        for entry in load_chemistry_skill_inventory().get("skills", [])
        if entry.get("single_agent_exposure") is True
    }
    lines = [
        "Chemistry skill catalog:",
        "The catalog describes available capabilities; whether and how to use a skill is your choice.",
    ]
    if available_skills is None:
        lines.append("All single-agent chemistry skills are listed below.")
    else:
        lines.append("Only health-checked skills available in this run are listed below.")
    for domain in SKILL_TREE:
        rendered_families: list[tuple[dict[str, Any], list[str]]] = []
        for family in domain["families"]:
            family_skills = [
                str(skill)
                for skill in family["skills"]
                if available_skills is None or str(skill) in available_skills
            ]
            if not family_skills:
                continue
            rendered_families.append((family, family_skills))
        if not rendered_families:
            continue
        lines.append(f"- Domain `{domain['id']}`: {domain['label']}")
        for family, family_skills in rendered_families:
            lines.append(f"  - Family `{family['id']}`: {family['label']}")
            for skill in family_skills:
                summary = str(inventory_by_skill.get(skill, {}).get("route_summary") or "").strip()
                lines.append(f"    - `{skill}`: {summary}")
    return "\n".join(lines)
