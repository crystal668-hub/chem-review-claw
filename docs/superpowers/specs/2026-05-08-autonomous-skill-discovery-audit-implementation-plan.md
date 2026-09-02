# Autonomous Skill Discovery and Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace deterministic record-level skill routing with full benchmark skill availability, a lightweight Hierarchical Skill Tree, model-driven skill use, and post-run audit metrics.

**Architecture:** Keep `single_llm_skills_on` configured with the full benchmark skill allowlist for every record. Replace the compact router and the earlier flat `benchmarking/skill_inventory.py` design with `benchmarking/skill_tree.py`, which renders a three-layer capability tree: Domain -> Skill Family -> Concrete Skill. The model sees only compact domain/family guidance in the prompt, can call any loaded skill directly when useful, and post-run audit reports actual tool execution separately from answer quality.

**Tech Stack:** Python 3.12, pytest/unittest, OpenClaw config JSON, existing benchmark modules under `benchmarking/`, local skill bundles under `skills/`, ChemQA prompt contracts under `skills/chemqa-review/prompts/`.

---

## Current Branch And Old Branch Decision

New implementation branch:

- Branch: `skill-autonomous-discovery-audit`
- Worktree: `/Users/xutao/.config/superpowers/worktrees/workspace/skill-autonomous-discovery-audit`
- Base: local `master` at `d02027d`
- Baseline check already run: `uv run pytest tests/test_benchmark_prompts.py tests/test_benchmark_config_runtime.py tests/test_experimental_chemistry_skill_matrix.py -q`
- Baseline result: `19 passed`

Old router repair branch:

- Branch: `skill-injection-routing-repair`
- Worktree: `/Users/xutao/.config/superpowers/worktrees/workspace/skill-injection-routing-repair`
- Status: clean
- Merge status: not merged into `master`
- Unique commits: 9 commits from `f5b3151` through `caa3e60`
- Decision: the branch can be deleted after this plan is committed if the team accepts discarding the record-scoped router implementation. It should not be merged because its core architecture conflicts with the approved full-availability design.

Deletion commands when approved:

```bash
git -C /Users/xutao/.openclaw/workspace tag archive/skill-injection-routing-repair caa3e60
git -C /Users/xutao/.openclaw/workspace worktree remove /Users/xutao/.config/superpowers/worktrees/workspace/skill-injection-routing-repair
git -C /Users/xutao/.openclaw/workspace branch -D skill-injection-routing-repair
```

The archive tag preserves a named recovery point before deleting the unmerged branch ref.

## Design Direction

The approved design is:

- Full benchmark skill allowlist is always available to `single_llm_skills_on`.
- There is no record-scoped `selected_skills`, no `SkillPlan`, and no preexecution controller that chooses tools before the model answers.
- There is no deterministic keyword router and no prompt language saying a triggered route can be satisfied by a skipped trace.
- The model gets a compact capability map, not 84 full skill documents.
- The capability map is hierarchical so the model first orients to a domain, then a family, then calls a concrete skill only when useful.
- Post-run audit records whether tool calls actually happened, whether the answer text declared a skip, and whether the run had no tool calls.

The historical JSON file `skills/chemistry-routing-matrix.json` remains as the initial source of 84 skill names and tier metadata. Runtime code must treat it as inventory data only. New code must not use its historical `primary_triggers` as routing rules.

## Hierarchical Skill Tree

Layer 1 is the broad capability domain. Layer 2 is the skill family the model can use to narrow attention. Layer 3 is the concrete skill bundle that OpenClaw can call.

Initial tree:

```python
SKILL_TREE = (
    {
        "id": "calculation-math",
        "label": "Calculation and formula math",
        "families": (
            {"id": "deterministic-chemistry-calculation", "label": "Deterministic chemistry calculation", "skills": ("chem-calculator",)},
        ),
    },
    {
        "id": "molecular-structure-identity",
        "label": "Molecular structure, identity, and cheminformatics",
        "families": (
            {"id": "structure-toolkit", "label": "Structure parsing, descriptors, fingerprints, and scaffolds", "skills": ("rdkit", "datamol", "molfeat")},
            {"id": "name-resolution", "label": "Chemical name and identifier resolution", "skills": ("opsin", "pubchem")},
            {"id": "compound-profile", "label": "Compound public records and profiles", "skills": ("pubchem-database", "chemistry-query")},
            {"id": "medchem-filters", "label": "Medicinal chemistry filters and drug-likeness checks", "skills": ("medchem",)},
        ),
    },
    {
        "id": "literature-evidence",
        "label": "Literature evidence and paper processing",
        "families": (
            {"id": "paper-pipeline", "label": "Paper retrieval, access, and parsing", "skills": ("paper-retrieval", "paper-access", "paper-parse")},
            {"id": "literature-databases", "label": "Bibliographic databases", "skills": ("pubmed-database", "openalex-database")},
            {"id": "review-synthesis", "label": "Literature review and synthesis", "skills": ("literature-review", "synthesize-literature")},
        ),
    },
    {
        "id": "bioactivity-safety-discovery",
        "label": "Bioactivity, safety, and discovery databases",
        "families": (
            {"id": "bioactivity-databases", "label": "Bioactivity, screening, and purchasable compounds", "skills": ("chembl-database", "zinc-database")},
            {"id": "tooluniverse-discovery", "label": "ToolUniverse chemical safety, retrieval, and small-molecule discovery", "skills": ("tooluniverse-chemical-safety", "tooluniverse-small-molecule-discovery", "tooluniverse-chemical-compound-retrieval")},
            {"id": "biological-databases", "label": "Protein, pathway, and structure databases", "skills": ("pdb-database", "alphafold-database", "reactome-database")},
        ),
    },
    {
        "id": "materials-crystals",
        "label": "Materials, crystals, and solid-state data",
        "families": (
            {"id": "crystal-structure", "label": "Crystal structure parsing, analysis, and editing", "skills": ("pymatgen", "ase", "cod")},
            {"id": "materials-databases", "label": "Materials databases and property lookup", "skills": ("materials-project", "oqmd", "jarvis")},
            {"id": "specialist-materials", "label": "Specialist materials workflows", "skills": ("doped-perovskite-structure-analysis",)},
        ),
    },
    {
        "id": "simulation-forcefield-md",
        "label": "Simulation, force fields, and molecular dynamics",
        "families": (
            {"id": "md-analysis", "label": "Molecular dynamics setup and trajectory analysis", "skills": ("molecular-dynamics", "openmm")},
            {"id": "forcefield-parameterization", "label": "Force-field assignment and parameterization", "skills": ("open-forcefield-toolkit", "atb")},
        ),
    },
    {
        "id": "quantum-hpc",
        "label": "Quantum chemistry, HPC engines, and parsed outputs",
        "families": (
            {"id": "quantum-inputs", "label": "Quantum chemistry input preparation and engine guidance", "skills": ("q-chem", "hpc-orca", "hpc-gaussian", "hpc-pyscf", "hpc-xtb", "hpc-vasp", "hpc-nwchem", "hpc-cp2k", "hpc-quantum-espresso")},
            {"id": "quantum-output-analysis", "label": "Quantum output parsing and benchmark references", "skills": ("cclib", "qc-output-analysis", "cccbdb", "molssi-qca")},
        ),
    },
    {
        "id": "spectra-formats-visualization",
        "label": "Spectra, chemical formats, and visualization",
        "families": (
            {"id": "spectra-analysis", "label": "Spectral formats and spectral interpretation", "skills": ("spectral-analysis", "jcamp-dx")},
            {"id": "chemical-file-formats", "label": "Chemical and crystallographic file formats", "skills": ("cif", "cml", "blue-obelisk")},
            {"id": "structure-visualization", "label": "Crystal and structure visualization", "skills": ("crystal-viewer", "xtal2png")},
        ),
    },
    {
        "id": "ml-generative-modeling",
        "label": "ML potentials, generative modeling, and property prediction",
        "families": (
            {"id": "ml-potentials", "label": "Atomistic ML potentials", "skills": ("mace", "chgnet", "mattersim", "schnet", "nequip", "orb", "reann", "torchmd-net")},
            {"id": "generative-materials", "label": "Generative crystal and materials models", "skills": ("mattergen", "diffcsp", "crystalflow")},
            {"id": "molecular-ml", "label": "Molecular ML models", "skills": ("chemprop",)},
            {"id": "materials-ml", "label": "Materials ML datasets and models", "skills": ("matformer", "matminer", "matbench", "modnet", "crabnet", "xenonpy")},
        ),
    },
    {
        "id": "workflow-automation",
        "label": "Workflow automation and interoperable infrastructure",
        "families": (
            {"id": "materials-api-interoperability", "label": "Materials API interoperability", "skills": ("optimade", "optimade-python-tools")},
            {"id": "workflow-engines", "label": "Workflow engines and job orchestration", "skills": ("aiida", "atomate", "fireworks", "custodian", "quacc", "pyiron", "qmflows", "qmforge")},
        ),
    },
)
```

This first version intentionally assigns each of the 84 benchmark allowlist skills to at least one family. The tree is a discovery aid, not a gate: family membership must never restrict runtime skill availability.

## File Structure

Create:

- `benchmarking/skill_tree.py`
  - Owns loading the historical chemistry skill inventory, deriving the full benchmark skill allowlist, defining the three-layer skill tree, rendering compact top-level discovery guidance, and looking up a family when the model or future tooling needs details.
- `benchmarking/skill_audit.py`
  - Owns conservative post-run tool-use audit extraction from runner metadata and final answer text.
- `tests/test_benchmark_skill_tree.py`
  - Tests allowlist coverage, paper pipeline availability, tree coverage, compact rendering, and lookup behavior.
- `tests/test_benchmark_skill_audit.py`
  - Tests audit classification for actual tool calls, model-declared skipped traces, and no-tool-call runs.

Modify:

- `benchmarking/prompts.py`
  - Replace `render_compact_skill_routing_table()` with `render_top_level_skill_tree()`.
  - Remove route-selection and skipped-route wording.
- `benchmarking/config_renderer.py`
  - Preserve full runner skill allowlists for skills-on groups; add regression tests that no per-record selected subset exists.
- `benchmarking/runners/single_llm.py`
  - Add `skill_use_audit` metadata after OpenClaw returns a result.
- `benchmarking/reporting.py`
  - Add aggregate counts for actual skill tool use and skipped-trace declarations.
- `benchmark_test.py`
  - Import inventory and allowlist helpers from `benchmarking.skill_tree`.
- `tests/test_benchmark_prompts.py`
  - Update expectations from routing table to hierarchical skill tree guidance.
- `tests/test_benchmark_config_runtime.py`
  - Assert full paper pipeline and provider skills remain available in rendered skills-on configs.
- `tests/test_benchmark_test.py`
  - Update loader import expectations, direct runner construction, and reporting summary expectations.
- `tests/test_experimental_chemistry_skill_matrix.py`
  - Replace route-scenario tests with tree coverage and discovery-rendering tests.
- `skills/chemqa-review/scripts/provider_trace_policy.py`
  - Remove deterministic route-trigger requirements and skipped-as-valid acceptance.
  - Keep provider trace validation as an audit/enforcement mechanism for traces the model actually submits, not as a keyword classifier that decides a skill was required.
- `skills/chemqa-review/prompts/contracts/proposer-main.md`
  - Remove `not applicable` / `intentionally skipped` as a normal compliance path.
- `skills/chemqa-review/prompts/contracts/reviewer-evidence-trace.md`
  - Treat missing tool evidence as a review finding when a candidate claims tool-backed evidence without artifacts; do not accept skipped trace as equivalent.
- `skills/chemqa-review/prompts/contracts/reviewer-counterevidence.md`
  - Same trace-policy cleanup for counterevidence reviews.
- `skills/chemqa-review/prompts/contracts/reviewer-reasoning-consistency.md`
  - Same trace-policy cleanup for reasoning reviews.
- `skills/chemqa-review/prompts/modules/context/required-skills.md`
  - Replace routing-policy wording with full-availability hierarchical discovery wording.
- `GLOBAL_DEV_SPEC.md`
  - Update architecture and feature matrix after implementation.

Delete or retire:

- `benchmarking/chemistry_routing.py`
  - Delete after all imports move to `benchmarking.skill_tree`.
  - Do not create `benchmarking/skill_inventory.py`; the previous flat inventory design is replaced by `skill_tree.py`.

## Task 1: Introduce Hierarchical Skill Tree

**Files:**

- Create: `benchmarking/skill_tree.py`
- Modify: `benchmark_test.py`
- Create: `tests/test_benchmark_skill_tree.py`

- [ ] **Step 1: Write failing skill tree tests**

Create `tests/test_benchmark_skill_tree.py`:

```python
from __future__ import annotations

from benchmarking.skill_tree import (
    benchmark_skill_allowlist,
    load_chemistry_skill_inventory,
    load_skill_tree,
    lookup_skill_family,
    render_top_level_skill_tree,
)


def _all_tree_skills() -> list[str]:
    skills: list[str] = []
    for domain in load_skill_tree():
        for family in domain["families"]:
            skills.extend(str(skill) for skill in family["skills"])
    return skills


def test_benchmark_skill_allowlist_includes_all_matrix_skills_and_paper_pipeline() -> None:
    inventory = load_chemistry_skill_inventory()
    allowlist = benchmark_skill_allowlist()

    assert len(allowlist) == 84
    assert len(allowlist) == len(set(allowlist))
    assert allowlist == tuple(str(entry["skill"]) for entry in inventory["skills"])
    assert {"paper-retrieval", "paper-access", "paper-parse"} <= set(allowlist)
    assert {"chem-calculator", "rdkit", "opsin", "pubchem"} <= set(allowlist)


def test_skill_tree_covers_every_allowlisted_skill() -> None:
    allowlist = set(benchmark_skill_allowlist())
    tree_skills = _all_tree_skills()

    assert allowlist <= set(tree_skills)
    assert not (set(tree_skills) - allowlist)


def test_skill_tree_has_three_layers_and_paper_pipeline_family() -> None:
    tree = load_skill_tree()

    assert any(domain["id"] == "literature-evidence" for domain in tree)
    family = lookup_skill_family("paper-pipeline")
    assert family["id"] == "paper-pipeline"
    assert family["skills"] == ("paper-retrieval", "paper-access", "paper-parse")


def test_top_level_skill_tree_is_compact_and_not_a_router() -> None:
    rendered = render_top_level_skill_tree()

    assert "Skill capability tree" in rendered
    assert "First choose a capability domain" in rendered
    assert "paper-pipeline" in rendered
    assert "literature-evidence" in rendered
    assert "Experimental chemistry skill routing rules" not in rendered
    assert "first matching primary route" not in rendered
    assert "selected skill route" not in rendered.lower()
    assert "SKILL TRACE: skipped" not in rendered
    assert "If you skip" not in rendered
    assert "`chem-calculator`" not in rendered
    assert "`rdkit`" not in rendered
    assert "`paper-retrieval`" not in rendered
    assert len(rendered.splitlines()) < 60
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/test_benchmark_skill_tree.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarking.skill_tree'`.

- [ ] **Step 3: Implement `benchmarking/skill_tree.py`**

Create `benchmarking/skill_tree.py`:

```python
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "skills" / "chemistry-routing-matrix.json"


SKILL_TREE: tuple[dict[str, Any], ...] = (
    {
        "id": "calculation-math",
        "label": "Calculation and formula math",
        "families": (
            {"id": "deterministic-chemistry-calculation", "label": "Deterministic chemistry calculation", "skills": ("chem-calculator",)},
        ),
    },
    {
        "id": "molecular-structure-identity",
        "label": "Molecular structure, identity, and cheminformatics",
        "families": (
            {"id": "structure-toolkit", "label": "Structure parsing, descriptors, fingerprints, and scaffolds", "skills": ("rdkit", "datamol", "molfeat")},
            {"id": "name-resolution", "label": "Chemical name and identifier resolution", "skills": ("opsin", "pubchem")},
            {"id": "compound-profile", "label": "Compound public records and profiles", "skills": ("pubchem-database", "chemistry-query")},
            {"id": "medchem-filters", "label": "Medicinal chemistry filters and drug-likeness checks", "skills": ("medchem",)},
        ),
    },
    {
        "id": "literature-evidence",
        "label": "Literature evidence and paper processing",
        "families": (
            {"id": "paper-pipeline", "label": "Paper retrieval, access, and parsing", "skills": ("paper-retrieval", "paper-access", "paper-parse")},
            {"id": "literature-databases", "label": "Bibliographic databases", "skills": ("pubmed-database", "openalex-database")},
            {"id": "review-synthesis", "label": "Literature review and synthesis", "skills": ("literature-review", "synthesize-literature")},
        ),
    },
    {
        "id": "bioactivity-safety-discovery",
        "label": "Bioactivity, safety, and discovery databases",
        "families": (
            {"id": "bioactivity-databases", "label": "Bioactivity, screening, and purchasable compounds", "skills": ("chembl-database", "zinc-database")},
            {"id": "tooluniverse-discovery", "label": "ToolUniverse chemical safety, retrieval, and small-molecule discovery", "skills": ("tooluniverse-chemical-safety", "tooluniverse-small-molecule-discovery", "tooluniverse-chemical-compound-retrieval")},
            {"id": "biological-databases", "label": "Protein, pathway, and structure databases", "skills": ("pdb-database", "alphafold-database", "reactome-database")},
        ),
    },
    {
        "id": "materials-crystals",
        "label": "Materials, crystals, and solid-state data",
        "families": (
            {"id": "crystal-structure", "label": "Crystal structure parsing, analysis, and editing", "skills": ("pymatgen", "ase", "cod")},
            {"id": "materials-databases", "label": "Materials databases and property lookup", "skills": ("materials-project", "oqmd", "jarvis")},
            {"id": "specialist-materials", "label": "Specialist materials workflows", "skills": ("doped-perovskite-structure-analysis",)},
        ),
    },
    {
        "id": "simulation-forcefield-md",
        "label": "Simulation, force fields, and molecular dynamics",
        "families": (
            {"id": "md-analysis", "label": "Molecular dynamics setup and trajectory analysis", "skills": ("molecular-dynamics", "openmm")},
            {"id": "forcefield-parameterization", "label": "Force-field assignment and parameterization", "skills": ("open-forcefield-toolkit", "atb")},
        ),
    },
    {
        "id": "quantum-hpc",
        "label": "Quantum chemistry, HPC engines, and parsed outputs",
        "families": (
            {"id": "quantum-inputs", "label": "Quantum chemistry input preparation and engine guidance", "skills": ("q-chem", "hpc-orca", "hpc-gaussian", "hpc-pyscf", "hpc-xtb", "hpc-vasp", "hpc-nwchem", "hpc-cp2k", "hpc-quantum-espresso")},
            {"id": "quantum-output-analysis", "label": "Quantum output parsing and benchmark references", "skills": ("cclib", "qc-output-analysis", "cccbdb", "molssi-qca")},
        ),
    },
    {
        "id": "spectra-formats-visualization",
        "label": "Spectra, chemical formats, and visualization",
        "families": (
            {"id": "spectra-analysis", "label": "Spectral formats and spectral interpretation", "skills": ("spectral-analysis", "jcamp-dx")},
            {"id": "chemical-file-formats", "label": "Chemical and crystallographic file formats", "skills": ("cif", "cml", "blue-obelisk")},
            {"id": "structure-visualization", "label": "Crystal and structure visualization", "skills": ("crystal-viewer", "xtal2png")},
        ),
    },
    {
        "id": "ml-generative-modeling",
        "label": "ML potentials, generative modeling, and property prediction",
        "families": (
            {"id": "ml-potentials", "label": "Atomistic ML potentials", "skills": ("mace", "chgnet", "mattersim", "schnet", "nequip", "orb", "reann", "torchmd-net")},
            {"id": "generative-materials", "label": "Generative crystal and materials models", "skills": ("mattergen", "diffcsp", "crystalflow")},
            {"id": "molecular-ml", "label": "Molecular ML models", "skills": ("chemprop",)},
            {"id": "materials-ml", "label": "Materials ML datasets and models", "skills": ("matformer", "matminer", "matbench", "modnet", "crabnet", "xenonpy")},
        ),
    },
    {
        "id": "workflow-automation",
        "label": "Workflow automation and interoperable infrastructure",
        "families": (
            {"id": "materials-api-interoperability", "label": "Materials API interoperability", "skills": ("optimade", "optimade-python-tools")},
            {"id": "workflow-engines", "label": "Workflow engines and job orchestration", "skills": ("aiida", "atomate", "fireworks", "custodian", "quacc", "pyiron", "qmflows", "qmforge")},
        ),
    },
)


@lru_cache(maxsize=1)
def load_chemistry_skill_inventory() -> dict[str, Any]:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def benchmark_skill_allowlist() -> tuple[str, ...]:
    return tuple(str(entry["skill"]) for entry in load_chemistry_skill_inventory().get("skills", []))


def load_skill_tree() -> tuple[dict[str, Any], ...]:
    return SKILL_TREE


def lookup_skill_family(family_id: str) -> dict[str, Any]:
    normalized = str(family_id or "").strip().lower()
    for domain in SKILL_TREE:
        for family in domain["families"]:
            if str(family["id"]).lower() == normalized:
                return family
    raise KeyError(f"unknown skill family: {family_id}")


def render_top_level_skill_tree() -> str:
    lines = [
        "Skill capability tree:",
        "First choose a capability domain, then a skill family, then call a concrete skill only when it helps answer the record.",
        "All benchmark skills remain available; this tree is a discovery aid, not a router or allowlist filter.",
    ]
    for domain in SKILL_TREE:
        families = ", ".join(f"`{family['id']}`" for family in domain["families"])
        lines.append(f"- `{domain['id']}`: {domain['label']} Families: {families}.")
    lines.append("When a family is relevant, inspect or call the concrete skill from the loaded OpenClaw skill list.")
    lines.append("Use tool outputs, artifact paths, or cited retrieved evidence in the answer when a skill contributes.")
    return "\n".join(lines)
```

- [ ] **Step 4: Replace benchmark allowlist imports**

In `benchmark_test.py`, replace both import blocks:

```python
from benchmarking.skill_tree import benchmark_skill_allowlist, load_chemistry_skill_inventory
```

and:

```python
from workspace.benchmarking.skill_tree import benchmark_skill_allowlist, load_chemistry_skill_inventory
```

Replace `BENCHMARK_SKILLS_ALLOWLIST` with:

```python
BENCHMARK_SKILLS_ALLOWLIST = list(benchmark_skill_allowlist())
```

If tests still call `benchmark_test.load_chemistry_routing_matrix()`, replace those expectations with `load_chemistry_skill_inventory()`.

- [ ] **Step 5: Update benchmark module test name and assertions**

In `tests/test_benchmark_test.py`, rename `test_benchmark_skills_allowlist_comes_from_routing_matrix` to `test_benchmark_skills_allowlist_comes_from_skill_tree` and replace the body with:

```python
def test_benchmark_skills_allowlist_comes_from_skill_tree(self) -> None:
    inventory_skills = [
        str(entry["skill"])
        for entry in benchmark_test.load_chemistry_skill_inventory().get("skills", [])
    ]

    self.assertEqual(inventory_skills, benchmark_test.BENCHMARK_SKILLS_ALLOWLIST)
    self.assertEqual(84, len(benchmark_test.BENCHMARK_SKILLS_ALLOWLIST))
    self.assertIn("chem-calculator", benchmark_test.BENCHMARK_SKILLS_ALLOWLIST)
    self.assertIn("pymatgen", benchmark_test.BENCHMARK_SKILLS_ALLOWLIST)
    self.assertIn("paper-retrieval", benchmark_test.BENCHMARK_SKILLS_ALLOWLIST)
    self.assertIn("paper-access", benchmark_test.BENCHMARK_SKILLS_ALLOWLIST)
    self.assertIn("paper-parse", benchmark_test.BENCHMARK_SKILLS_ALLOWLIST)
    self.assertNotIn("benchmark-cleanroom", benchmark_test.BENCHMARK_SKILLS_ALLOWLIST)
    self.assertNotIn("chemqa-review", benchmark_test.BENCHMARK_SKILLS_ALLOWLIST)
    self.assertNotIn("debateclaw-v1", benchmark_test.BENCHMARK_SKILLS_ALLOWLIST)
```

- [ ] **Step 6: Run skill tree tests**

Run:

```bash
uv run pytest tests/test_benchmark_skill_tree.py tests/test_benchmark_test.py::BenchmarkTestModuleTests::test_benchmark_skills_allowlist_comes_from_skill_tree -q
```

Expected: PASS.

- [ ] **Step 7: Commit skill tree changes**

Run:

```bash
git add benchmark_test.py benchmarking/skill_tree.py tests/test_benchmark_skill_tree.py tests/test_benchmark_test.py
git commit -m "feat: add hierarchical benchmark skill tree"
```

## Task 2: Replace Routing Prompt With Hierarchical Discovery Prompt

**Files:**

- Modify: `benchmarking/prompts.py`
- Modify: `tests/test_benchmark_prompts.py`
- Modify: `tests/test_experimental_chemistry_skill_matrix.py`

- [ ] **Step 1: Write failing prompt tests**

In `tests/test_benchmark_prompts.py`, update `test_single_llm_prompt_respects_skills_enabled_flag`:

```python
def test_single_llm_prompt_respects_skills_enabled_flag(self) -> None:
    record = BenchmarkRecord(
        record_id="fs-1",
        dataset="frontierscience",
        source_file="/tmp/frontierscience.jsonl",
        eval_kind="frontierscience_olympiad",
        prompt="Calculate the pH.",
        reference_answer="4.7",
        payload={"track": "olympiad"},
    )

    skills_on = build_single_llm_prompt(record, websearch_enabled=True, skills_enabled=True)
    skills_off = build_single_llm_prompt(record, websearch_enabled=True, skills_enabled=False)

    self.assertIn("Skill capability tree", skills_on)
    self.assertIn("First choose a capability domain", skills_on)
    self.assertIn("paper-pipeline", skills_on)
    self.assertIn("calculation-math", skills_on)
    self.assertNotIn("Experimental chemistry skill routing rules", skills_on)
    self.assertNotIn("first matching primary route", skills_on)
    self.assertNotIn("SKILL TRACE: skipped", skills_on)
    self.assertNotIn("Do not use OpenClaw skills", skills_on)
    self.assertNotIn("Skill capability tree", skills_off)
    self.assertIn("Do not use OpenClaw skills", skills_off)
```

- [ ] **Step 2: Run prompt tests and verify failure**

Run:

```bash
uv run pytest tests/test_benchmark_prompts.py::BenchmarkPromptsTests::test_single_llm_prompt_respects_skills_enabled_flag -q
```

Expected: FAIL because `build_single_llm_prompt()` still imports and injects the compact routing table.

- [ ] **Step 3: Update `benchmarking/prompts.py`**

Replace:

```python
from .chemistry_routing import render_compact_skill_routing_table
```

with:

```python
from .skill_tree import render_top_level_skill_tree
```

Replace:

```python
instructions.append(render_compact_skill_routing_table())
```

with:

```python
instructions.append(render_top_level_skill_tree())
```

Do not add no-route, selected-route, preexecution, or skipped-route wording.

- [ ] **Step 4: Replace routing matrix tests with tree tests**

In `tests/test_experimental_chemistry_skill_matrix.py`, remove `test_route_scenarios_use_unique_primary_skills` and `test_single_agent_prompt_injects_same_compact_matrix`.

Update `test_experimental_matrix_covers_selected_mid_plus_skills` to import from `benchmarking.skill_tree`:

```python
from benchmarking.skill_tree import benchmark_skill_allowlist, load_chemistry_skill_inventory

inventory = load_chemistry_skill_inventory()
skill_names = set(benchmark_skill_allowlist())

assert len(inventory["skills"]) == 84
assert len(skill_names) == len(inventory["skills"])
assert EXPECTED_EXPERIMENTAL_SKILLS <= skill_names
assert {"rdkit", "opsin", "pubchem", "chem-calculator"} <= skill_names
assert inventory["mode"] == "experimental_mid_plus"
```

Replace `test_compact_prompt_routing_is_grouped_not_full_skill_docs` with:

```python
def test_top_level_skill_tree_is_grouped_not_full_skill_docs() -> None:
    from benchmarking.skill_tree import render_top_level_skill_tree

    tree = render_top_level_skill_tree()

    for domain_or_family in (
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
    assert "Core Workflow: OpenMM Simulation" not in tree
    assert "Installation and Setup" not in tree
    assert "first matching primary route" not in tree
    assert len(tree.splitlines()) < 60
```

Add:

```python
def test_single_agent_prompt_injects_skill_tree() -> None:
    from benchmarking.datasets import BenchmarkRecord
    from benchmarking.prompts import build_single_llm_prompt

    record = BenchmarkRecord(
        record_id="route-cif",
        dataset="hle",
        source_file="/tmp/hle.jsonl",
        eval_kind="hle",
        prompt="What coordination polyhedra does this CIF crystal structure contain?",
        reference_answer="Al, Re2Al13",
    )

    prompt = build_single_llm_prompt(record, websearch_enabled=False)

    assert "Skill capability tree:" in prompt
    assert "materials-crystals" in prompt
    assert "paper-pipeline" in prompt
    assert "Experimental chemistry skill routing rules" not in prompt
```

- [ ] **Step 5: Run prompt and matrix tests**

Run:

```bash
uv run pytest tests/test_benchmark_prompts.py tests/test_experimental_chemistry_skill_matrix.py tests/test_benchmark_skill_tree.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit prompt changes**

Run:

```bash
git add benchmarking/prompts.py tests/test_benchmark_prompts.py tests/test_experimental_chemistry_skill_matrix.py
git commit -m "feat: use hierarchical skill discovery prompt"
```

## Task 3: Delete Deterministic Router Interfaces

**Files:**

- Delete: `benchmarking/chemistry_routing.py`
- Modify: `skills/chemqa-review/scripts/provider_trace_policy.py`
- Modify: `skills/chemqa-review/tests/test_chemqa_review_runtime.py`

- [ ] **Step 1: Verify remaining imports**

Run:

```bash
rg -n "chemistry_routing|route_skill_for_text|requirements_for_text|render_compact_skill_routing_table" .
```

Expected before this task: matches remain in `provider_trace_policy.py`, tests, and docs.

- [ ] **Step 2: Update provider trace tests**

In `skills/chemqa-review/tests/test_chemqa_review_runtime.py`, replace `test_artifact_outcome_surfaces_provider_trace_audit_warnings` with:

```python
def test_artifact_outcome_accepts_no_provider_trace_when_model_makes_no_tool_claim(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        driver = driver_module.ChemQAReviewDriver.__new__(driver_module.ChemQAReviewDriver)
        proposal_path = Path(tmpdir) / transport.proposal_filename()
        proposal_path.write_text(
            "\n".join(
                [
                    "artifact_kind: candidate_submission",
                    "artifact_contract_version: react-reviewed-v2",
                    "phase: propose",
                    "owner: proposer-1",
                    "direct_answer: '42'",
                    "summary: numeric answer without provider trace.",
                    "submission_trace:",
                    "- step: reasoning",
                    "  status: success",
                    "  detail: mental arithmetic only.",
                ]
            ),
            encoding="utf-8",
        )

        outcome = driver_module.ChemQAReviewDriver._observe_artifact_outcome(
            driver,
            file_path=proposal_path,
            filename=transport.proposal_filename(),
            checker=lambda text: transport.check_candidate_submission(
                text,
                owner="proposer-1",
                answer_kind="numeric_short_answer",
                provider_trace_mode="audit",
            ),
            pre_call_snapshot=None,
            pre_call_normalized=None,
            require_file_change=False,
        )

        self.assertEqual("present_valid", outcome.state)
        self.assertFalse(any("chem-calculator" in warning for warning in outcome.validation_warnings))
```

Replace `test_candidate_submission_enforces_provider_trace_mode` with:

```python
def test_candidate_submission_rejects_skipped_provider_trace_even_in_enforce_mode(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        driver = driver_module.ChemQAReviewDriver.__new__(driver_module.ChemQAReviewDriver)
        driver.args = argparse.Namespace(role="proposer-1", provider_trace_mode="enforce")
        driver.answer_kind = lambda: "numeric_short_answer"
        proposal_path = Path(tmpdir) / transport.proposal_filename()
        proposal_path.write_text(
            "\n".join(
                [
                    "artifact_kind: candidate_submission",
                    "artifact_contract_version: react-reviewed-v2",
                    "phase: propose",
                    "owner: proposer-1",
                    "direct_answer: '42'",
                    "summary: claims skipped provider trace.",
                    "submission_trace:",
                    "- skill: chem-calculator",
                    "  status: skipped",
                    "  trigger: numeric_or_formula_math",
                    "  reason: model chose ordinary reasoning",
                    "  risk: calculation may be wrong",
                ]
            ),
            encoding="utf-8",
        )

        outcome = driver_module.ChemQAReviewDriver._observe_artifact_outcome(
            driver,
            file_path=proposal_path,
            filename=transport.proposal_filename(),
            checker=driver_module.ChemQAReviewDriver.candidate_submission_checker(driver),
            pre_call_snapshot=None,
            pre_call_normalized=None,
            require_file_change=False,
        )

        self.assertEqual("present_invalid", outcome.state)
        self.assertTrue(any("skipped" in error.lower() for error in outcome.validation_errors))
```

- [ ] **Step 3: Remove router dependency from `provider_trace_policy.py`**

Delete:

```python
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from benchmarking.chemistry_routing import requirements_for_text  # noqa: E402
```

Remove the now-unused `sys` import.

Replace `requirements_for_candidate()` with:

```python
def requirements_for_candidate(
    *,
    answer_kind: str,
    prompt: str = "",
    eval_kind: str = "",
    dataset: str = "",
) -> list[ProviderTraceRequirement]:
    return []
```

Add this helper:

```python
def _submitted_trace_requirements(trace_entries: list[dict[str, Any]]) -> list[ProviderTraceRequirement]:
    requirements: list[ProviderTraceRequirement] = []
    for entry in trace_entries:
        skill = str(entry.get("skill") or entry.get("tool") or entry.get("provider") or "").strip()
        if not skill:
            continue
        trigger = str(entry.get("trigger") or entry.get("step") or "submitted_trace").strip()
        requirements.append(
            ProviderTraceRequirement(
                skill=skill,
                trigger=trigger,
                reason="Model submitted a provider trace for this skill.",
            )
        )
    return _dedupe_requirements(requirements)
```

In `validate_provider_traces()`, after `trace_entries = _provider_trace_entries(payload)`, extend requirements from submitted traces:

```python
requirements = _dedupe_requirements(requirements + _submitted_trace_requirements(trace_entries))
```

Replace the skipped handling in `_entry_is_acceptable()`:

```python
if status == "skipped":
    return False
```

Update the incomplete-trace error text to:

```python
"provide status success/partial with a provider result JSON artifact path or structured tool_trace conclusion."
```

This preserves auditing for model-submitted provider traces without creating a hidden question-class router.

- [ ] **Step 4: Delete `benchmarking/chemistry_routing.py`**

Run:

```bash
git rm benchmarking/chemistry_routing.py
```

- [ ] **Step 5: Verify no router imports remain**

Run:

```bash
rg -n "chemistry_routing|route_skill_for_text|requirements_for_text|render_compact_skill_routing_table" benchmarking tests skills/chemqa-review/scripts skills/chemqa-review/tests
```

Expected: no matches.

- [ ] **Step 6: Run relevant tests**

Run:

```bash
uv run pytest tests/test_experimental_chemistry_skill_matrix.py tests/test_benchmark_prompts.py skills/chemqa-review/tests/test_chemqa_review_runtime.py::RunStatusShapeTest -q
```

Expected: PASS.

- [ ] **Step 7: Commit router deletion**

Run:

```bash
git add skills/chemqa-review/scripts/provider_trace_policy.py skills/chemqa-review/tests/test_chemqa_review_runtime.py
git commit -m "refactor: remove deterministic skill router"
```

## Task 4: Guard Full Skill Availability In Runtime Config

**Files:**

- Modify: `tests/test_benchmark_config_runtime.py`
- Modify: `tests/test_benchmark_test.py`

- [ ] **Step 1: Add config regression test for full skills-on availability**

In `tests/test_benchmark_config_runtime.py`, add or update a test that renders the `single_llm_skills_on` config and asserts full availability:

```python
def test_single_llm_skills_on_config_keeps_full_benchmark_skill_allowlist(self) -> None:
    payload = render_run_config(
        base_payload=self.base_payload,
        spec=self.experiment_specs["single_llm_skills_on"],
        provisioned=self.provisioned_single,
        judge_model="judge-model",
        runner_model="runner-model",
    )

    agents = payload["agents"]["list"]
    runner = next(agent for agent in agents if agent["id"] == "benchmark-single-skills-on")
    skills = runner["skills"]

    self.assertIn("chem-calculator", skills)
    self.assertIn("rdkit", skills)
    self.assertIn("paper-retrieval", skills)
    self.assertIn("paper-access", skills)
    self.assertIn("paper-parse", skills)
    self.assertGreaterEqual(len(skills), 80)
```

Adjust fixture names to match the existing test class setup exactly.

- [ ] **Step 2: Run config regression test**

Run:

```bash
uv run pytest tests/test_benchmark_config_runtime.py -q
```

Expected: PASS on current `master` behavior, because full group allowlist is already used.

- [ ] **Step 3: Add negative grep guard for old per-record override**

Add a test in `tests/test_benchmark_test.py`:

```python
def test_single_llm_runner_does_not_use_record_scoped_skill_config(self) -> None:
    source = Path("benchmarking/runners/single_llm.py").read_text(encoding="utf-8")

    self.assertNotIn("selected_skills", source)
    self.assertNotIn("config_for_record", source)
    self.assertNotIn("SkillPlan", source)
```

`Path` is already imported in this test file.

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/test_benchmark_config_runtime.py tests/test_benchmark_test.py::BenchmarkTestModuleTests::test_single_llm_runner_does_not_use_record_scoped_skill_config -q
```

Expected: PASS.

- [ ] **Step 5: Commit config guards**

Run:

```bash
git add tests/test_benchmark_config_runtime.py tests/test_benchmark_test.py
git commit -m "test: guard full skill availability"
```

## Task 5: Add Post-Run Skill Use Audit

**Files:**

- Create: `benchmarking/skill_audit.py`
- Modify: `benchmarking/runners/single_llm.py`
- Modify: `benchmark_test.py`
- Create: `tests/test_benchmark_skill_audit.py`
- Modify: `tests/test_benchmark_test.py`

- [ ] **Step 1: Write failing audit tests**

Create `tests/test_benchmark_skill_audit.py`:

```python
from __future__ import annotations

from benchmarking.skill_audit import build_skill_use_audit


def test_skill_use_audit_detects_tool_calls() -> None:
    audit = build_skill_use_audit(
        skills_enabled=True,
        configured_skills=("chem-calculator", "rdkit"),
        runner_meta={"toolSummary": {"calls": 2, "tools": ["write", "exec"], "failures": 0}},
        final_response_text="FINAL ANSWER: 7.59",
    )

    assert audit["skills_enabled"] is True
    assert audit["available_skill_count"] == 2
    assert audit["tool_call_count"] == 2
    assert audit["skill_tool_executed"] is True
    assert audit["model_declared_skip"] is False
    assert audit["no_tool_call"] is False


def test_skill_use_audit_detects_declared_skip_without_counting_execution() -> None:
    audit = build_skill_use_audit(
        skills_enabled=True,
        configured_skills=("rdkit",),
        runner_meta={"toolSummary": {"calls": 0, "tools": [], "failures": 0}},
        final_response_text="SKILL TRACE: skipped rdkit because this is qualitative.",
    )

    assert audit["skill_tool_executed"] is False
    assert audit["model_declared_skip"] is True
    assert audit["no_tool_call"] is True


def test_skill_use_audit_handles_missing_tool_summary() -> None:
    audit = build_skill_use_audit(
        skills_enabled=True,
        configured_skills=("paper-retrieval",),
        runner_meta={},
        final_response_text="FINAL ANSWER: A",
    )

    assert audit["tool_call_count"] == 0
    assert audit["tool_names"] == []
    assert audit["no_tool_call"] is True
```

- [ ] **Step 2: Run audit tests and verify failure**

Run:

```bash
uv run pytest tests/test_benchmark_skill_audit.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarking.skill_audit'`.

- [ ] **Step 3: Implement `benchmarking/skill_audit.py`**

Create `benchmarking/skill_audit.py`:

```python
from __future__ import annotations

import re
from typing import Any


SKIP_TRACE_RE = re.compile(r"\bskill\s+trace\s*:\s*skipped\b", re.IGNORECASE)


def build_skill_use_audit(
    *,
    skills_enabled: bool,
    configured_skills: tuple[str, ...] | list[str],
    runner_meta: dict[str, Any],
    final_response_text: str,
) -> dict[str, Any]:
    tool_summary = runner_meta.get("toolSummary") or {}
    calls = int(tool_summary.get("calls") or 0) if isinstance(tool_summary, dict) else 0
    raw_tools = tool_summary.get("tools") if isinstance(tool_summary, dict) else []
    tool_names = [str(item) for item in raw_tools] if isinstance(raw_tools, list) else []
    configured = [str(skill) for skill in configured_skills]
    declared_skip = bool(SKIP_TRACE_RE.search(str(final_response_text or "")))
    return {
        "skills_enabled": bool(skills_enabled),
        "available_skill_count": len(configured),
        "available_skills": configured,
        "tool_call_count": calls,
        "tool_names": tool_names,
        "tool_failure_count": int(tool_summary.get("failures") or 0) if isinstance(tool_summary, dict) else 0,
        "skill_tool_executed": bool(calls > 0),
        "model_declared_skip": declared_skip,
        "no_tool_call": bool(calls == 0),
    }
```

- [ ] **Step 4: Wire audit into `SingleLLMRunner`**

In `benchmarking/runners/single_llm.py`, import:

```python
from ..skill_audit import build_skill_use_audit
```

Add a `configured_skills` constructor parameter:

```python
configured_skills: tuple[str, ...] | list[str] = (),
```

Store it:

```python
self.configured_skills = tuple(str(skill) for skill in configured_skills)
```

After `runner_meta = dict(result_payload.get("meta") or {})`, add:

```python
runner_meta["skill_use_audit"] = build_skill_use_audit(
    skills_enabled=bool(getattr(group, "skills_enabled", True)),
    configured_skills=self.configured_skills,
    runner_meta=runner_meta,
    final_response_text=full_response_text,
)
```

In the wrapper class `benchmark_test.SingleLLMRunner.__init__`, accept the same optional `configured_skills` parameter and pass it to `super().__init__`.

In `run_group()`, pass the group allowlist when constructing the single runner:

```python
configured_skills=tuple(EXPERIMENT_SPECS[group.id].skill_allowlist or ()),
```

- [ ] **Step 5: Update runner test**

In `tests/test_benchmark_test.py::BenchmarkRunnerTests::test_single_llm_runner_invokes_openclaw_with_high_thinking`, change the fake stdout to include a tool summary:

```python
stdout=json.dumps(
    {
        "result": {
            "payloads": [{"text": "Reasoning\nFINAL ANSWER: 5"}],
            "meta": {"toolSummary": {"calls": 1, "tools": ["exec"], "failures": 0}},
        }
    }
),
```

Pass configured skills:

```python
runner = benchmark_test.SingleLLMRunner(
    agent_id="benchmark-single-skills-on",
    timeout_seconds=30,
    config_path=Path("/tmp/single.json"),
    runtime_bundle_root=Path("/tmp"),
    configured_skills=("chem-calculator", "paper-retrieval"),
)
```

After `out = runner.run(...)`, add:

```python
audit = out.runner_meta["skill_use_audit"]
self.assertEqual(2, audit["available_skill_count"])
self.assertTrue(audit["skill_tool_executed"])
self.assertEqual(1, audit["tool_call_count"])
```

- [ ] **Step 6: Run audit and runner tests**

Run:

```bash
uv run pytest tests/test_benchmark_skill_audit.py tests/test_benchmark_test.py::BenchmarkRunnerTests::test_single_llm_runner_invokes_openclaw_with_high_thinking -q
```

Expected: PASS.

- [ ] **Step 7: Commit audit changes**

Run:

```bash
git add benchmark_test.py benchmarking/runners/single_llm.py benchmarking/skill_audit.py tests/test_benchmark_skill_audit.py tests/test_benchmark_test.py
git commit -m "feat: audit actual skill tool use"
```

## Task 6: Add Aggregate Skill-Use Metrics

**Files:**

- Modify: `benchmarking/reporting.py`
- Modify: `tests/test_benchmark_test.py`

- [ ] **Step 1: Write failing reporting test**

In `tests/test_benchmark_test.py`, update the existing aggregate reporting test that builds `GroupRecordResult` items. Add `runner_meta` examples:

```python
runner_meta={
    "skill_use_audit": {
        "skills_enabled": True,
        "tool_call_count": 2,
        "skill_tool_executed": True,
        "model_declared_skip": False,
        "no_tool_call": False,
    }
}
```

Add assertions on the group bucket:

```python
self.assertEqual(1, bucket["skill_tool_executed_count"])
self.assertEqual(1, bucket["skill_model_declared_skip_count"])
self.assertEqual(1, bucket["skill_no_tool_call_count"])
self.assertEqual(2, bucket["skill_tool_call_total"])
```

- [ ] **Step 2: Run reporting test and verify failure**

Run the exact aggregate test:

```bash
uv run pytest tests/test_benchmark_test.py::BenchmarkReportingTests -q
```

Expected: FAIL because the new aggregate keys do not exist.

- [ ] **Step 3: Add aggregate fields**

In `benchmarking/reporting.py`, add helper functions:

```python
def skill_audit(item: GroupRecordResult) -> dict[str, Any]:
    audit = (item.runner_meta or {}).get("skill_use_audit") or {}
    return audit if isinstance(audit, dict) else {}


def skill_tool_call_count(item: GroupRecordResult) -> int:
    value = skill_audit(item).get("tool_call_count")
    return int(value) if isinstance(value, (int, float)) else 0
```

Add to `aggregate_bucket()`:

```python
"skill_tool_executed_count": sum(1 for item in items if skill_audit(item).get("skill_tool_executed")),
"skill_model_declared_skip_count": sum(1 for item in items if skill_audit(item).get("model_declared_skip")),
"skill_no_tool_call_count": sum(1 for item in items if skill_audit(item).get("no_tool_call")),
"skill_tool_call_total": sum(skill_tool_call_count(item) for item in items),
```

- [ ] **Step 4: Run reporting tests**

Run:

```bash
uv run pytest tests/test_benchmark_test.py::BenchmarkReportingTests tests/test_benchmark_skill_audit.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit reporting metrics**

Run:

```bash
git add benchmarking/reporting.py tests/test_benchmark_test.py
git commit -m "feat: report skill tool-use metrics"
```

## Task 7: Clean ChemQA Prompt Policy

**Files:**

- Modify: `skills/chemqa-review/prompts/contracts/proposer-main.md`
- Modify: `skills/chemqa-review/prompts/contracts/reviewer-evidence-trace.md`
- Modify: `skills/chemqa-review/prompts/contracts/reviewer-counterevidence.md`
- Modify: `skills/chemqa-review/prompts/contracts/reviewer-reasoning-consistency.md`
- Modify: `skills/chemqa-review/prompts/modules/context/required-skills.md`
- Modify: `skills/chemqa-review/tests/test_chemqa_review_runtime.py`

- [ ] **Step 1: Add text regression test**

In `skills/chemqa-review/tests/test_chemqa_review_runtime.py`, add:

```python
def test_chemqa_prompts_do_not_accept_skipped_skill_trace_as_compliance(self):
    prompt_root = SKILL_ROOT / "prompts"
    checked = [
        prompt_root / "contracts" / "proposer-main.md",
        prompt_root / "contracts" / "reviewer-evidence-trace.md",
        prompt_root / "contracts" / "reviewer-counterevidence.md",
        prompt_root / "contracts" / "reviewer-reasoning-consistency.md",
        prompt_root / "modules" / "context" / "required-skills.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in checked)

    self.assertNotIn("intentionally skipped", combined)
    self.assertNotIn("not applicable after inspecting the prompt", combined)
    self.assertNotIn("status: skipped", combined)
    self.assertNotIn("valid `submission_trace` entry with `status: skipped`", combined)
    self.assertNotIn("triggered route is skipped", combined)
```

- [ ] **Step 2: Run prompt-policy test and verify failure**

Run:

```bash
uv run pytest skills/chemqa-review/tests/test_chemqa_review_runtime.py::ChemQAReviewRuntimeTests::test_chemqa_prompts_do_not_accept_skipped_skill_trace_as_compliance -q
```

Expected: FAIL because current prompts contain skipped-trace language.

- [ ] **Step 3: Update proposer prompt**

In `skills/chemqa-review/prompts/contracts/proposer-main.md`, replace skipped-trace compliance language with:

```markdown
- If you use a provider skill and the call fails because the tool is unavailable or returns an error, record `status: error`, the skill name, request summary, error text, and the risk to the answer. Do not treat an unexecuted skill as equivalent to a provider result.
```

Remove any wording that says a route is triggered by prompt class or that a skipped trace is valid compliance.

- [ ] **Step 4: Update reviewer prompts**

In each reviewer prompt, replace skipped-trace acceptance clauses with:

```markdown
- Treat a missing provider artifact or structured `tool_trace` as a finding when the candidate explicitly relies on tool-backed calculation, molecular structure, compound identity, literature evidence, database lookup, spectra, materials, simulation, or workflow evidence.
```

Apply this to:

- `skills/chemqa-review/prompts/contracts/reviewer-evidence-trace.md`
- `skills/chemqa-review/prompts/contracts/reviewer-counterevidence.md`
- `skills/chemqa-review/prompts/contracts/reviewer-reasoning-consistency.md`

- [ ] **Step 5: Update required-skills context**

In `skills/chemqa-review/prompts/modules/context/required-skills.md`, replace routing policy lines with:

```markdown
Skill discovery policy:
- Benchmark chemistry skills are available from the sibling `skills/` directory.
- Use provider skills directly when they help answer a calculation, structure, identity, database, literature, spectra, materials, simulation, ML, or workflow subproblem.
- First orient by capability domain and family; read full `SKILL.md` files only for skills you are about to use.
- Do not treat an unexecuted skill as a valid provider trace.
- Literature or external-fact claims that require paper evidence can use the `paper-pipeline` family: `paper-retrieval` -> `paper-access` -> `paper-parse`.
```

- [ ] **Step 6: Run ChemQA prompt-policy tests**

Run:

```bash
uv run pytest skills/chemqa-review/tests/test_chemqa_review_runtime.py::ChemQAReviewRuntimeTests::test_chemqa_prompts_do_not_accept_skipped_skill_trace_as_compliance -q
```

Expected: PASS.

- [ ] **Step 7: Commit ChemQA policy cleanup**

Run:

```bash
git add skills/chemqa-review/prompts/contracts/proposer-main.md skills/chemqa-review/prompts/contracts/reviewer-evidence-trace.md skills/chemqa-review/prompts/contracts/reviewer-counterevidence.md skills/chemqa-review/prompts/contracts/reviewer-reasoning-consistency.md skills/chemqa-review/prompts/modules/context/required-skills.md skills/chemqa-review/tests/test_chemqa_review_runtime.py
git commit -m "fix: remove skipped skill compliance path"
```

## Task 8: Update Global Dev Spec

**Files:**

- Modify: `GLOBAL_DEV_SPEC.md`

- [ ] **Step 1: Update architecture text**

In `GLOBAL_DEV_SPEC.md`, replace the existing routing-matrix architecture bullet with:

```markdown
- `chemistry-routing-matrix.json`
  - Historical experimental chemistry skill inventory for medium-or-higher-value chemistry capabilities. Despite the historical filename, runtime benchmark prompts treat this as inventory data, not as a deterministic router.
  - `workspace/benchmarking/skill_tree.py` defines the benchmark skill allowlist and a three-layer discovery tree: Domain -> Skill Family -> Concrete Skill.
  - Single-agent skills-on runs expose the full benchmark skill allowlist to the model. Prompts include a lightweight hierarchical skill tree and rely on the model to choose and call relevant skills when they help answer the record.
  - Post-run reporting records actual tool-use audit metadata such as tool-call counts, model-declared skipped traces, and no-tool-call outcomes. Skipped traces are diagnostic only and do not count as executed skill use.
```

- [ ] **Step 2: Update feature matrix**

Add or update a feature entry:

```markdown
- Name: Autonomous benchmark skill discovery and audit
  - Description: Skills-on benchmark runs keep the full benchmark skill allowlist available for each single-LLM record, use a compact Hierarchical Skill Tree instead of deterministic selected-skill routing, and report post-run skill-use audit counters from actual tool execution metadata.
  - Input / Output:
    - Input: benchmark record prompt plus full configured benchmark skill allowlist.
    - Output: normal benchmark answer plus `runner_meta.skill_use_audit` and aggregate skill tool-use counters.
  - Implementation location: `workspace/benchmarking/skill_tree.py`, `workspace/benchmarking/skill_audit.py`, `workspace/benchmarking/prompts.py`, `workspace/benchmarking/reporting.py`
  - Status: `DONE`
```

- [ ] **Step 3: Remove stale router references**

Run:

```bash
rg -n "chemistry_routing|selected skill route|first matching primary route|skipped triggered route|record-scoped route|SkillPlan" GLOBAL_DEV_SPEC.md
```

Expected after edits: no matches.

- [ ] **Step 4: Commit spec update**

Run:

```bash
git add GLOBAL_DEV_SPEC.md
git commit -m "docs: document hierarchical skill discovery"
```

## Task 9: Final Verification And Benchmark Smoke

**Files:**

- No source edits unless a verification failure reveals a task-specific bug.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
uv run pytest tests/test_benchmark_prompts.py tests/test_benchmark_config_runtime.py tests/test_benchmark_skill_tree.py tests/test_benchmark_skill_audit.py tests/test_experimental_chemistry_skill_matrix.py tests/test_benchmark_test.py -q
```

Expected: PASS.

- [ ] **Step 2: Run ChemQA-focused tests touched by prompt/policy changes**

Run:

```bash
uv run pytest skills/chemqa-review/tests/test_chemqa_review_runtime.py tests/test_chemqa_artifact_flow.py -q
```

Expected: PASS.

- [ ] **Step 3: Generate one prompt preview**

Run:

```bash
uv run python - <<'PY'
from benchmarking.datasets import BenchmarkRecord
from benchmarking.prompts import build_single_llm_prompt

record = BenchmarkRecord(
    record_id="demo",
    dataset="frontierscience",
    source_file="/tmp/demo.jsonl",
    eval_kind="frontierscience_olympiad",
    prompt="Calculate the mass of dissolved Sr2+ from Ksp and concentrations.",
    reference_answer="7.59",
    payload={"track": "olympiad"},
)
prompt = build_single_llm_prompt(record, websearch_enabled=True, skills_enabled=True)
print(prompt.split("QUESTION:", 1)[0])
PY
```

Expected output contains `Skill capability tree`, `First choose a capability domain`, and `paper-pipeline`. It must not contain `Experimental chemistry skill routing rules`, `first matching primary route`, or `SKILL TRACE: skipped`.

- [ ] **Step 4: Render one runtime config preview**

Run:

```bash
uv run python - <<'PY'
import benchmark_test

skills = benchmark_test.BENCHMARK_SKILLS_ALLOWLIST
print(len(skills))
print("paper-retrieval" in skills, "paper-access" in skills, "paper-parse" in skills)
PY
```

Expected output has a count of `84`, followed by `True True True True`.

- [ ] **Step 5: Inspect final diff for router residue**

Run:

```bash
rg -n "selected_skills|config_for_record|SkillPlan|preexecute_skill_plan|first matching primary route|Use ordinary reasoning and avoid loading unrelated skills|render_compact_skill_routing_table|route_skill_for_text|requirements_for_text|SKILL TRACE: skipped|status: skipped|intentionally skipped|not applicable after inspecting" benchmark_test.py benchmarking skills/chemqa-review/prompts skills/chemqa-review/scripts GLOBAL_DEV_SPEC.md
```

Expected: no matches in source, prompts, or `GLOBAL_DEV_SPEC.md`. `SKILL TRACE: skipped` may remain only inside `tests/test_benchmark_skill_audit.py` as a diagnostic audit fixture, not as compliance or routing prompt language.

- [ ] **Step 6: Commit any verification fixes**

If Step 5 required edits, commit them:

```bash
git add .
git commit -m "fix: remove routing residue"
```

If no edits were required, do not create an empty commit.

## Task 10: Optional Old Branch Deletion

**Files:**

- No source files.

- [ ] **Step 1: Confirm current branch is not the old branch**

Run:

```bash
git -C /Users/xutao/.config/superpowers/worktrees/workspace/skill-autonomous-discovery-audit branch --show-current
```

Expected: `skill-autonomous-discovery-audit`.

- [ ] **Step 2: Confirm old branch has no local modifications**

Run:

```bash
git -C /Users/xutao/.config/superpowers/worktrees/workspace/skill-injection-routing-repair status --short
```

Expected: no output.

- [ ] **Step 3: Archive old branch head**

Run:

```bash
git -C /Users/xutao/.openclaw/workspace tag archive/skill-injection-routing-repair caa3e60
```

Expected: command succeeds. If the tag already exists at `caa3e60`, keep it and continue.

- [ ] **Step 4: Remove old worktree**

Run:

```bash
git -C /Users/xutao/.openclaw/workspace worktree remove /Users/xutao/.config/superpowers/worktrees/workspace/skill-injection-routing-repair
```

Expected: worktree removed.

- [ ] **Step 5: Delete old branch ref**

Run:

```bash
git -C /Users/xutao/.openclaw/workspace branch -D skill-injection-routing-repair
```

Expected: branch deleted. The archive tag remains as the recovery point.
