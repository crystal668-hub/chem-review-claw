from __future__ import annotations

from benchmarking.skills.tree import (
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


RUNTIME_OR_ORCHESTRATION_SKILLS = {"benchmark-cleanroom", "debateclaw-v1", "chemqa-review"}


def test_benchmark_skill_allowlist_includes_all_matrix_skills_and_paper_pipeline() -> None:
    inventory = load_chemistry_skill_inventory()
    allowlist = benchmark_skill_allowlist()

    assert len(allowlist) == 85
    assert len(allowlist) == len(set(allowlist))
    assert allowlist == tuple(
        str(entry["skill"]) for entry in inventory["skills"] if entry["single_agent_exposure"] is True
    )
    assert "act-like-a-chemist" in allowlist
    assert "xtb-cli" in allowlist
    assert {"paper-retrieval", "paper-access", "paper-parse"} <= set(allowlist)
    assert {"chem-calculator", "rdkit", "opsin", "pubchem"} <= set(allowlist)
    assert not (RUNTIME_OR_ORCHESTRATION_SKILLS & set(allowlist))


def test_skill_tree_covers_every_allowlisted_skill() -> None:
    allowlist = set(benchmark_skill_allowlist())
    tree_skills = _all_tree_skills()

    assert allowlist <= set(tree_skills)
    assert not (set(tree_skills) - allowlist)
    assert not (RUNTIME_OR_ORCHESTRATION_SKILLS & set(tree_skills))


def test_skill_tree_has_three_layers_and_paper_pipeline_family() -> None:
    tree = load_skill_tree()

    assert tree[0]["id"] == "chemist-sop"
    sop_family = lookup_skill_family("chemistry-reasoning-sop")
    assert sop_family["id"] == "chemistry-reasoning-sop"
    assert sop_family["skills"] == ("act-like-a-chemist",)
    assert any(domain["id"] == "literature-evidence" for domain in tree)
    family = lookup_skill_family("paper-pipeline")
    assert family["id"] == "paper-pipeline"
    assert family["skills"] == ("paper-retrieval", "paper-access", "paper-parse")
    family = lookup_skill_family("local-xtb-cli")
    assert family["id"] == "local-xtb-cli"
    assert "xtb-cli" in family["skills"]


def test_top_level_skill_tree_is_a_neutral_full_catalog() -> None:
    rendered = render_top_level_skill_tree()
    inventory = load_chemistry_skill_inventory()

    assert "Chemistry skill catalog" in rendered
    assert "whether and how to use a skill is your choice" in rendered
    assert "Read `act-like-a-chemist` first" not in rendered
    assert "Atomic Coverage Checklist" not in rendered
    assert "Benchmark Coverage Checklist" not in rendered
    assert "benchmark-solving-protocol" not in rendered
    assert "skills/benchmark-solving-protocol" not in rendered
    assert "chemist-sop" in rendered
    assert "chemistry-reasoning-sop" in rendered
    assert "paper-pipeline" in rendered
    assert "literature-evidence" in rendered
    assert "--workspace-root /Users/xutao/.openclaw/workspace" not in rendered
    assert "--execution-cwd" not in rendered
    assert "--script skills/<skill>/scripts/<script>.py --" not in rendered
    assert "tool name must be exactly `exec`" not in rendered
    assert 'exec {"command":' not in rendered
    assert "run_skill.py" not in rendered
    assert "`python3` tool call" not in rendered
    assert "`script`, `cmd`, or `command` tool call" not in rendered
    assert "`exec` with `{}`" not in rendered
    assert "direct `python skills/" not in rendered
    assert "`system-event-scheduler`" not in rendered
    assert "TOOLS.md" not in rendered
    assert "fact ledger" not in rendered
    assert "Organic mechanism SOP" not in rendered
    assert "Experimental chemistry skill routing rules" not in rendered
    assert "first matching primary route" not in rendered
    assert "Provider Skill Trigger Rules" not in rendered
    assert "Capability Routing Matrix" not in rendered
    assert "single-agent-exposed provider inventory" not in rendered
    assert "selected skill route" not in rendered.lower()
    assert "SKILL TRACE: skipped" not in rendered
    assert "If you skip" not in rendered
    assert "find run_skill" not in rendered.lower()
    assert "python3 <skill-root>" not in rendered
    for entry in inventory["skills"]:
        if entry["single_agent_exposure"] is True:
            assert f"`{entry['skill']}`" in rendered
            assert entry["route_summary"] in rendered
    for runtime_skill in RUNTIME_OR_ORCHESTRATION_SKILLS:
        assert runtime_skill not in rendered
    assert len(rendered.splitlines()) < 150


def test_top_level_skill_tree_reflects_health_filtered_availability() -> None:
    rendered = render_top_level_skill_tree(available_skills={"act-like-a-chemist", "rdkit", "paper-access"})

    assert "Only health-checked skills available in this run are listed below" in rendered
    assert "benchmark-solving-protocol" not in rendered
    assert "chemist-sop" in rendered
    assert "molecular-structure-identity" in rendered
    assert "literature-evidence" in rendered
    assert "`act-like-a-chemist`" in rendered
    assert "`rdkit`" in rendered
    assert "`paper-access`" in rendered
    assert "`chem-calculator`" not in rendered
    assert "`paper-retrieval`" not in rendered
