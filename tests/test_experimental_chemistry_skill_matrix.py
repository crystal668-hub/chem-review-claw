from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = ROOT / "skills"
RUNTIME_OR_ORCHESTRATION_SKILLS = {"benchmark-cleanroom", "debateclaw-v1", "chemqa-review"}
ALLOWED_CAPABILITY_DOMAINS = {
    "protocol",
    "numeric_calculation",
    "molecular_structure",
    "literature",
    "materials_database",
    "spectra",
    "protein",
    "md",
    "hpc",
    "ml",
    "drug_safety",
    "workflow_infrastructure",
}
ALLOWED_PROVIDER_ROLES = {"sop", "primary", "specialized", "inventory_only", "excluded_runtime"}


EXPECTED_EXPERIMENTAL_SKILLS = {
    "pymatgen",
    "ase",
    "cclib",
    "datamol",
    "molfeat",
    "chembl-database",
    "zinc-database",
    "materials-project",
    "cod",
    "oqmd",
    "jarvis",
    "cccbdb",
    "molssi-qca",
    "molecular-dynamics",
    "openmm",
    "open-forcefield-toolkit",
    "tooluniverse-chemical-safety",
    "tooluniverse-small-molecule-discovery",
    "tooluniverse-chemical-compound-retrieval",
    "hpc-orca",
    "hpc-pyscf",
    "hpc-xtb",
    "xtb-cli",
    "hpc-vasp",
    "hpc-gaussian",
    "q-chem",
    "hpc-nwchem",
    "hpc-cp2k",
    "hpc-quantum-espresso",
    "qc-output-analysis",
    "spectral-analysis",
    "cif",
    "jcamp-dx",
    "cml",
    "crystal-viewer",
    "xtal2png",
    "doped-perovskite-structure-analysis",
    "mace",
    "chgnet",
    "mattersim",
    "mattergen",
    "diffcsp",
    "crystalflow",
    "chemprop",
    "schnet",
    "nequip",
    "matformer",
    "orb",
    "reann",
    "torchmd-net",
    "chemistry-query",
    "pubchem-database",
    "medchem",
    "atb",
    "pdb-database",
    "alphafold-database",
    "reactome-database",
    "pubmed-database",
    "openalex-database",
    "paper-retrieval",
    "paper-access",
    "paper-parse",
    "literature-review",
    "synthesize-literature",
    "matminer",
    "matbench",
    "modnet",
    "crabnet",
    "xenonpy",
    "optimade",
    "optimade-python-tools",
    "aiida",
    "atomate",
    "fireworks",
    "custodian",
    "quacc",
    "pyiron",
    "qmflows",
    "qmforge",
    "blue-obelisk",
}


def test_experimental_matrix_covers_selected_mid_plus_skills() -> None:
    from benchmarking.skills.tree import (
        benchmark_skill_allowlist,
        load_chemistry_skill_inventory,
    )

    inventory = load_chemistry_skill_inventory()
    skill_names = set(benchmark_skill_allowlist())

    assert len(inventory["skills"]) == 85
    assert len(skill_names) == len(inventory["skills"])
    assert "act-like-a-chemist" in skill_names
    assert skill_names >= EXPECTED_EXPERIMENTAL_SKILLS
    assert {"rdkit", "opsin", "pubchem", "chem-calculator"} <= skill_names
    assert inventory["mode"] == "experimental_mid_plus"


def test_experimental_matrix_entries_define_provider_inventory_contract() -> None:
    from benchmarking.skills.tree import load_chemistry_skill_inventory

    inventory = load_chemistry_skill_inventory()

    for entry in inventory["skills"]:
        assert isinstance(entry.get("capability_domain"), str), entry["skill"]
        assert entry["capability_domain"] in ALLOWED_CAPABILITY_DOMAINS, entry["skill"]
        assert isinstance(entry.get("provider_role"), str), entry["skill"]
        assert entry["provider_role"] in ALLOWED_PROVIDER_ROLES, entry["skill"]
        assert isinstance(entry.get("single_agent_exposure"), bool), entry["skill"]

    by_skill = {entry["skill"]: entry for entry in inventory["skills"]}
    assert by_skill["act-like-a-chemist"]["capability_domain"] == "protocol"
    assert by_skill["act-like-a-chemist"]["provider_role"] == "sop"
    assert by_skill["act-like-a-chemist"]["single_agent_exposure"] is True
    assert "Optional chemistry reasoning workflow" in by_skill["act-like-a-chemist"]["route_summary"]
    assert "mandatory verification triggers" not in by_skill["act-like-a-chemist"]["route_summary"].lower()

    expected_primary = {
        "chem-calculator": "numeric_calculation",
        "rdkit": "molecular_structure",
        "opsin": "molecular_structure",
        "pubchem": "molecular_structure",
        "pymatgen": "materials_database",
        "chembl-database": "drug_safety",
        "medchem": "drug_safety",
    }
    for skill, domain in expected_primary.items():
        assert by_skill[skill]["capability_domain"] == domain
        assert by_skill[skill]["provider_role"] == "primary"
        assert by_skill[skill]["single_agent_exposure"] is True

    expected_specialized = {
        "materials-project": "materials_database",
        "cclib": "numeric_calculation",
        "qc-output-analysis": "numeric_calculation",
        "xtb-cli": "numeric_calculation",
        "matminer": "ml",
        "tooluniverse-chemical-safety": "drug_safety",
    }
    for skill, domain in expected_specialized.items():
        assert by_skill[skill]["capability_domain"] == domain
        assert by_skill[skill]["provider_role"] == "specialized"
        assert by_skill[skill]["single_agent_exposure"] is True

    assert not (RUNTIME_OR_ORCHESTRATION_SKILLS & set(by_skill))


def test_act_like_a_chemist_skill_bundle_is_installed() -> None:
    skill_root = SKILLS_ROOT / "act-like-a-chemist"

    assert (skill_root / "SKILL.md").is_file()


def test_selected_experimental_skills_are_installed_as_skill_bundles() -> None:
    for skill in EXPECTED_EXPERIMENTAL_SKILLS:
        skill_root = SKILLS_ROOT / skill
        assert (skill_root / "SKILL.md").is_file(), skill


def test_core_new_skill_wrappers_are_installed() -> None:
    expected_wrappers = {
        "cclib": "scripts/parse_output.py",
        "chembl-database": "scripts/bioactivity_query.py",
        "pymatgen": "scripts/structure_summary.py",
        "molecular-dynamics": "scripts/trajectory_summary.py",
        "open-forcefield-toolkit": "scripts/parameterize_molecule.py",
        "xtb-cli": "scripts/xtb_runner.py",
    }

    for skill, wrapper in expected_wrappers.items():
        assert (SKILLS_ROOT / skill / wrapper).is_file(), f"{skill}/{wrapper}"


def test_top_level_skill_tree_is_grouped_not_full_skill_docs() -> None:
    from benchmarking.skills.tree import render_top_level_skill_tree

    tree = render_top_level_skill_tree()

    for domain_or_family in (
        "chemist-sop",
        "chemistry-reasoning-sop",
        "calculation-math",
        "molecular-structure-identity",
        "literature-evidence",
        "paper-pipeline",
        "materials-crystals",
        "quantum-hpc",
        "workflow-automation",
    ):
        assert domain_or_family in tree

    assert "Quick Start Guide" not in tree
    assert "fact ledger" not in tree
    assert "Organic mechanism SOP" not in tree
    assert "Core Workflow: OpenMM Simulation" not in tree
    assert "Installation and Setup" not in tree
    assert "first matching primary route" not in tree
    assert "Read `act-like-a-chemist` first" not in tree
    assert "whether and how to use a skill is your choice" in tree
    assert "`rdkit`" in tree
    assert "`paper-retrieval`" in tree
    assert len(tree.splitlines()) < 150


def test_single_agent_skills_on_prompt_exposes_neutral_skill_tree() -> None:
    from benchmarking.core.datasets import BenchmarkRecord
    from benchmarking.workflow.prompts import build_single_llm_prompt

    record = BenchmarkRecord(
        record_id="route-cif",
        dataset="hle",
        source_file="/tmp/hle.jsonl",
        eval_kind="hle",
        prompt="What coordination polyhedra does this CIF crystal structure contain?",
        reference_answer="Al, Re2Al13",
    )

    prompt = build_single_llm_prompt(record, websearch_enabled=False)

    assert "Chemistry skill catalog:" in prompt
    assert "act-like-a-chemist" in prompt
    assert "Atomic Coverage Checklist" not in prompt
    assert "materials-crystals" in prompt
    assert "paper-pipeline" in prompt
    assert "Read `act-like-a-chemist` first" not in prompt
    assert "Organic mechanism SOP" not in prompt
    assert "Experimental chemistry skill routing rules" not in prompt
    assert "--workspace-root /Users/xutao/.openclaw/workspace" not in prompt
    assert "--execution-cwd" not in prompt
    assert "tool name must be exactly `exec`" not in prompt
    assert 'exec {"command":' not in prompt
    assert "run_skill.py" not in prompt
    assert "`python3` tool call" not in prompt
    assert "TOOLS.md" not in prompt
    assert "python3 <skill-root>" not in prompt
    assert "benchmark-solving-protocol" not in prompt


def test_act_like_a_chemist_defines_coverage_checklist_contract() -> None:
    text = (SKILLS_ROOT / "act-like-a-chemist" / "SKILL.md").read_text(encoding="utf-8")

    assert "## Atomic Coverage Checklist" in text
    assert "## Benchmark Coverage Checklist" not in text
    assert "Atomic Coverage Checklist" in text
    assert "Standard Answering Flow" in text
    assert "If provider skills would help, use `contract/skill-triggers.md` as a capability reference" in text
    assert "all atoms are `done` or `blocked`" in text
    assert "`todo`" in text
    assert "`done`" in text
    assert "`blocked`" in text
    assert "known givens" in text
    assert "scoped evidence" in text
    assert "Numeric, Formula, Or Table Tasks" in text
    assert "Multiple-Choice Tasks" in text
    assert "Research Or Open-Ended Tasks" in text
    assert "HLE Tasks" in text
    assert "Do not use `python`, `python3`, `pip`" not in text
    assert "usage error" not in text
    assert "`done` only after its derivation or evidence is complete" in text
    assert "supports`, `partially supports`, `contradicts`, or `only verifies an intermediate step`" in text
    assert "a useful tool result does not close neighboring atoms" in text
    assert "## Candidate / Hypothesis Verification" in text
    assert "Do not verify only the first candidate that gives a usable tool result" in text
    assert "solve for the unknown directly" in text
    assert "compare residuals for nearby or chemically plausible competitors" in text
    assert "A database hit, formula match, approximate numeric match, valid structure, or retrieved source" in text
    assert "It is not sufficient final-answer evidence" in text
    assert "Do not stop only because one tool call returned a useful or promising intermediate result" in text
    assert "done`: a gap already satisfied" not in text
    assert "Mark an item `done` only when prompt evidence, derivation, source evidence, or tool output actually supports it" not in text


def test_single_agent_skills_off_prompt_does_not_expose_chemist_sop() -> None:
    from benchmarking.core.datasets import BenchmarkRecord
    from benchmarking.workflow.prompts import build_single_llm_prompt

    record = BenchmarkRecord(
        record_id="skills-off",
        dataset="chembench",
        source_file="/tmp/chembench.jsonl",
        eval_kind="chembench_open_ended",
        prompt="Calculate the pH.",
        reference_answer="7",
    )

    prompt = build_single_llm_prompt(record, websearch_enabled=False, skills_enabled=False)

    assert "Do not use OpenClaw skills" not in prompt
    assert "act-like-a-chemist" not in prompt
    assert "Chemistry skill catalog:" not in prompt
    assert "tool name must be exactly `exec`" not in prompt
    assert "`python3` tool call" not in prompt
    assert "Organic mechanism SOP" not in prompt


def test_experimental_skill_dependencies_are_optional_and_scoped() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional = pyproject["project"]["optional-dependencies"]

    assert {
        "chem-materials",
        "chem-quantum-parse",
        "chem-bioactivity",
        "chem-md",
        "chem-cheminformatics-ml",
        "chem-materials-ml",
        "chem-workflows",
        "chem-experimental",
    } <= set(optional)

    expected_by_extra = {
        "chem-materials": {"pymatgen==2026.5.4", "mp-api==0.46.1", "ase==3.28.0"},
        "chem-quantum-parse": {"cclib==1.8.1"},
        "chem-bioactivity": {"chembl_webresource_client==0.10.9", "pubchempy==1.0.5"},
        "chem-md": {"openmm==8.5.1", "MDAnalysis==2.10.0"},
        "chem-cheminformatics-ml": {"datamol==0.12.5", "molfeat==0.11.0"},
        "chem-materials-ml": {"matminer==0.10.1", "jarvis-tools==2026.4.2"},
        "chem-workflows": {"custodian==2025.12.14", "fireworks==2.1.3", "quacc==1.2.6"},
    }

    for extra, dependencies in expected_by_extra.items():
        assert dependencies <= set(optional[extra])

    full = set(optional["full"])
    experimental = set(optional["chem-experimental"])
    assert "chemqa[chem-experimental]" not in full
    assert {
        "chemqa[chem-materials]",
        "chemqa[chem-quantum-parse]",
        "chemqa[chem-bioactivity]",
        "chemqa[chem-md]",
        "chemqa[chem-cheminformatics-ml]",
        "chemqa[chem-materials-ml]",
        "chemqa[chem-workflows]",
    } <= experimental

    all_optional_items = {dependency for dependencies in optional.values() for dependency in dependencies}
    assert not any("openff" in dependency.lower() for dependency in all_optional_items)
