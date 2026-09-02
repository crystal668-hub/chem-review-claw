import json
import tempfile
import unittest
from pathlib import Path

from benchmarking.core.experiments import ExperimentSpec
from benchmarking.runtime.agent_workspace import (
    AttemptWorkspaceManager,
    WorkspaceTemplate,
)
from benchmarking.runtime.config import render_run_config
from benchmarking.runtime.config_pool import (
    BENCHMARK_WORKDIR_GUARD_PLUGIN_ID,
    BENCHMARK_WORKDIR_GUARD_PLUGIN_ROOT,
    ConfigPool,
    RuntimeConfigContext,
    RuntimeConfigError,
    actual_slot_ids,
    build_run_scoped_config_payload,
)
from benchmarking.runtime.provisioning import (
    ProvisionedAgent,
    ProvisionedExperiment,
    provision_slot_workspace,
)
from benchmarking.runtime.workspace_policy import ProtectedRoot
from benchmarking.skills.tree import benchmark_skill_allowlist


class RuntimeConfigGroup:
    def __init__(self, *, id: str, label: str, runner: str, websearch: bool, skills_enabled: bool = True) -> None:
        self.id = id
        self.label = label
        self.runner = runner
        self.websearch = websearch
        self.skills_enabled = skills_enabled


class BenchmarkConfigRuntimeTests(unittest.TestCase):
    @staticmethod
    def _workspace_manager(root: Path) -> AttemptWorkspaceManager:
        template = root / "template"
        template.mkdir(parents=True, exist_ok=True)
        (template / "AGENTS.md").write_text("# test\n", encoding="utf-8")
        return AttemptWorkspaceManager(
            runtime_root=root / "managed-workspaces" / "runs",
            output_root=root / "runs",
            run_id="run-1",
            invocation_id="invocation-1",
            templates={"test-v1": WorkspaceTemplate(template_id="test-v1", source_dir=template)},
            protected_roots=(
                ProtectedRoot("benchmark_runtime_root", root / "managed-workspaces" / "runs", "test.runtime_root"),
                ProtectedRoot("current_output_root", root / "runs", "test.output_root"),
            ),
        )

    def test_render_run_config_is_pure_and_does_not_mutate_base_payload(self) -> None:
        base = {
            "agents": {"list": []},
            "tools": {"web": {"search": {"enabled": False}, "fetch": {"enabled": False}}},
            "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
        }
        spec = ExperimentSpec(
            id="single_llm_skills_on",
            label="Single LLM with skills",
            runner_kind="single_llm",
            websearch_enabled=True,
            skills_enabled=True,
            single_agent_id="benchmark-single-skills-on",
            skill_allowlist=("chem-calculator", "rdkit"),
        )
        provisioned = ProvisionedExperiment(
            judge=ProvisionedAgent("benchmark-judge", Path("/tmp/judge"), Path("/tmp/agents/judge")),
            runner_agents=(
                ProvisionedAgent(
                    "benchmark-single-skills-on",
                    Path("/tmp/single"),
                    Path("/tmp/agents/single"),
                ),
            ),
        )

        rendered = render_run_config(
            base_payload=base,
            spec=spec,
            provisioned=provisioned,
            judge_model="su8/gpt-5.4",
            runner_model="qwen3.5-plus",
        )

        self.assertEqual([], base["agents"]["list"])
        self.assertTrue(rendered["tools"]["web"]["search"]["enabled"])
        self.assertTrue(rendered["tools"]["web"]["fetch"]["enabled"])
        self.assertTrue(rendered["plugins"]["entries"]["duckduckgo"]["enabled"])
        self.assertEqual(["chem-calculator", "rdkit"], rendered["agents"]["list"][1]["skills"])

    def test_render_run_config_forces_single_llm_web_search_and_fetch_off(self) -> None:
        for group_id, skills_enabled, skill_allowlist in (
            ("single_llm_skills_on", True, ("chem-calculator", "rdkit")),
            ("single_llm_skills_off", False, ()),
        ):
            with self.subTest(group_id=group_id):
                base = {
                    "agents": {"list": []},
                    "tools": {"web": {"search": {"enabled": True}, "fetch": {"enabled": True}}},
                    "plugins": {"entries": {"duckduckgo": {"enabled": True, "config": {}}}},
                }
                spec = ExperimentSpec(
                    id=group_id,
                    label=group_id,
                    runner_kind="single_llm",
                    websearch_enabled=False,
                    skills_enabled=skills_enabled,
                    single_agent_id=f"benchmark-{group_id}",
                    skill_allowlist=skill_allowlist,
                )
                provisioned = ProvisionedExperiment(
                    judge=ProvisionedAgent("benchmark-judge", Path("/tmp/judge"), Path("/tmp/agents/judge")),
                    runner_agents=(
                        ProvisionedAgent(
                            f"benchmark-{group_id}",
                            Path("/tmp/single"),
                            Path("/tmp/agents/single"),
                        ),
                    ),
                )

                rendered = render_run_config(
                    base_payload=base,
                    spec=spec,
                    provisioned=provisioned,
                    judge_model="judge-model",
                    runner_model="runner-model",
                )

                self.assertIs(False, rendered["tools"]["web"]["search"]["enabled"])
                self.assertIs(False, rendered["tools"]["web"]["fetch"]["enabled"])
                self.assertIs(False, rendered["plugins"]["entries"]["duckduckgo"]["enabled"])

    def test_render_run_config_disables_runner_skills_with_empty_allowlist(self) -> None:
        base = {
            "agents": {"list": []},
            "tools": {"web": {"search": {"enabled": False}}},
            "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
        }
        spec = ExperimentSpec(
            id="single_llm_skills_off",
            label="Single LLM without skills",
            runner_kind="single_llm",
            websearch_enabled=True,
            skills_enabled=False,
            single_agent_id="benchmark-single-skills-off",
        )
        provisioned = ProvisionedExperiment(
            judge=ProvisionedAgent("benchmark-judge", Path("/tmp/judge"), Path("/tmp/agents/judge")),
            runner_agents=(
                ProvisionedAgent(
                    "benchmark-single-skills-off",
                    Path("/tmp/single"),
                    Path("/tmp/agents/single"),
                ),
            ),
        )

        rendered = render_run_config(
            base_payload=base,
            spec=spec,
            provisioned=provisioned,
            judge_model="su8/gpt-5.4",
            runner_model="qwen3.5-plus",
        )

        agents = {entry["id"]: entry for entry in rendered["agents"]["list"]}
        self.assertNotIn("skills", agents["benchmark-judge"])
        self.assertEqual([], agents["benchmark-single-skills-off"]["skills"])

    def test_single_llm_skills_on_config_keeps_full_benchmark_skill_allowlist(self) -> None:
        base = {
            "agents": {"list": []},
            "tools": {"web": {"search": {"enabled": False}}},
            "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
        }
        spec = ExperimentSpec(
            id="single_llm_skills_on",
            label="Single LLM with skills",
            runner_kind="single_llm",
            websearch_enabled=True,
            skills_enabled=True,
            single_agent_id="benchmark-single-skills-on",
            skill_allowlist=benchmark_skill_allowlist(),
        )
        provisioned = ProvisionedExperiment(
            judge=ProvisionedAgent("benchmark-judge", Path("/tmp/judge"), Path("/tmp/agents/judge")),
            runner_agents=(
                ProvisionedAgent(
                    "benchmark-single-skills-on",
                    Path("/tmp/single"),
                    Path("/tmp/agents/single"),
                ),
            ),
        )

        payload = render_run_config(
            base_payload=base,
            spec=spec,
            provisioned=provisioned,
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

    def test_render_run_config_replaces_managed_agent_and_strips_thinking(self) -> None:
        base = {
            "agents": {
                "list": [
                    {
                        "id": "benchmark-judge",
                        "name": "old judge",
                        "workspace": "/tmp/old-judge",
                        "agentDir": "/tmp/old-agent-dir",
                        "model": "old-model",
                        "thinking": "high",
                    },
                    {
                        "id": "benchmark-single-skills-on",
                        "name": "old single",
                        "workspace": "/tmp/old-single",
                        "agentDir": "/tmp/old-single-agent-dir",
                        "model": "old-runner",
                        "thinking": "high",
                    },
                ]
            },
            "tools": {"web": {"search": {"enabled": False}}},
            "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
        }
        spec = ExperimentSpec(
            id="single_llm_skills_on",
            label="Single LLM with skills",
            runner_kind="single_llm",
            websearch_enabled=True,
            skills_enabled=True,
            single_agent_id="benchmark-single-skills-on",
            skill_allowlist=("chem-calculator", "rdkit"),
        )
        provisioned = ProvisionedExperiment(
            judge=ProvisionedAgent("benchmark-judge", Path("/tmp/judge"), Path("/tmp/agents/judge")),
            runner_agents=(
                ProvisionedAgent(
                    "benchmark-single-skills-on",
                    Path("/tmp/single"),
                    Path("/tmp/agents/single"),
                ),
            ),
        )

        rendered = render_run_config(
            base_payload=base,
            spec=spec,
            provisioned=provisioned,
            judge_model="su8/gpt-5.4",
            runner_model="qwen3.5-plus",
        )

        agents = {entry["id"]: entry for entry in rendered["agents"]["list"]}
        self.assertEqual("su8/gpt-5.4", agents["benchmark-judge"]["model"])
        self.assertEqual("qwen3.5-plus", agents["benchmark-single-skills-on"]["model"])
        self.assertNotIn("thinking", agents["benchmark-judge"])
        self.assertNotIn("thinking", agents["benchmark-single-skills-on"])
        self.assertEqual(["chem-calculator", "rdkit"], agents["benchmark-single-skills-on"]["skills"])
        self.assertEqual("old-model", base["agents"]["list"][0]["model"])
        self.assertEqual("high", base["agents"]["list"][0]["thinking"])

    def test_provision_slot_workspace_creates_agents_md_and_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "debateA-1"
            workspace_root = workspace.parent

            provision_slot_workspace(
                workspace=workspace,
                workspace_root=workspace_root,
                slot_id="debateA-1",
                agents_template_text="# demo\n",
                last_session_id="session-123",
            )

            self.assertEqual("# demo\n", (workspace / "AGENTS.md").read_text(encoding="utf-8"))
            sentinel = json.loads((workspace / ".debateclaw-slot.json").read_text(encoding="utf-8"))
            self.assertEqual("debateclaw-slot-workspace", sentinel["kind"])
            self.assertEqual(1, sentinel["version"])
            self.assertEqual("debateA-1", sentinel["slot"])
            self.assertEqual(str(workspace.resolve()), sentinel["workspace"])
            self.assertEqual(str(workspace_root.resolve()), sentinel["workspace_root"])
            self.assertEqual("session-123", sentinel["last_session_id"])
            self.assertEqual("debateclaw", sentinel["managed_by"])

    def test_build_run_scoped_config_payload_provisions_single_agent_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = RuntimeConfigContext(
                agents_root=root / "agents",
                judge_agent_id="benchmark-judge",
                chemqa_slot_sets={"chemqa_skills_on": "A"},
                experiment_specs={
                    "single_llm_skills_on": ExperimentSpec(
                        id="single_llm_skills_on",
                        label="Single LLM with skills",
                        runner_kind="single_llm",
                        websearch_enabled=True,
                        skills_enabled=True,
                        single_agent_id="benchmark-single-skills-on",
                        skill_allowlist=("chem-calculator", "rdkit"),
                    )
                },
                benchmark_skills_root=root / "workspace" / "skills",
            )
            base = {
                "agents": {"list": []},
                "tools": {"web": {"search": {"enabled": False}}},
                "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
            }
            group = RuntimeConfigGroup(
                id="single_llm_skills_on",
                label="Single LLM with skills",
                runner="single_llm",
                websearch=True,
                skills_enabled=True,
            )

            payload = build_run_scoped_config_payload(
                base,
                context=context,
                group=group,
                single_agent_model="qwen3.5-plus",
                judge_model="su8/gpt-5.4",
                workspace_manager=self._workspace_manager(root),
            )

            agents = {entry["id"]: entry for entry in payload["agents"]["list"]}
            self.assertEqual("qwen3.5-plus", agents["benchmark-single-skills-on"]["model"])
            self.assertEqual("su8/gpt-5.4", agents["benchmark-judge"]["model"])
            self.assertEqual(["chem-calculator", "rdkit"], agents["benchmark-single-skills-on"]["skills"])
            self.assertIn(str((root / "workspace" / "skills").resolve()), payload["skills"]["load"]["extraDirs"])
            self.assertIn(
                str(BENCHMARK_WORKDIR_GUARD_PLUGIN_ROOT.resolve()),
                payload["plugins"]["load"]["paths"],
            )
            guard = payload["plugins"]["entries"][BENCHMARK_WORKDIR_GUARD_PLUGIN_ID]
            self.assertIs(guard["enabled"], True)
            policy = guard["config"]["agentPolicies"]["benchmark-single-skills-on"]
            self.assertEqual("single_llm", policy["role"])
            self.assertTrue(policy["skills_enabled"])
            self.assertEqual(agents["benchmark-single-skills-on"]["workspace"], policy["read_scopes"][0]["path"])
            self.assertRegex(policy["policy_digest"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                str(
                    self._workspace_manager(root).active_workspace_path(
                        group_id="single_llm_skills_on",
                        agent_id="benchmark-single-skills-on",
                    )
                ),
                agents["benchmark-single-skills-on"]["workspace"],
            )
            self.assertFalse(Path(agents["benchmark-single-skills-on"]["workspace"]).exists())

    def test_skills_off_guard_policy_excludes_skills_and_wrapper_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skills_root = root / "workspace" / "skills"
            context = RuntimeConfigContext(
                agents_root=root / "agents",
                judge_agent_id="benchmark-judge",
                chemqa_slot_sets={},
                experiment_specs={
                    "single_llm_skills_off": ExperimentSpec(
                        id="single_llm_skills_off",
                        label="skills off",
                        runner_kind="single_llm",
                        websearch_enabled=False,
                        skills_enabled=False,
                        single_agent_id="benchmark-single-skills-off",
                    )
                },
                benchmark_skills_root=skills_root,
            )
            payload = build_run_scoped_config_payload(
                {"agents": {"list": []}, "plugins": {"entries": {}}},
                context=context,
                group=RuntimeConfigGroup(
                    id="single_llm_skills_off",
                    label="skills off",
                    runner="single_llm",
                    websearch=False,
                    skills_enabled=False,
                ),
                single_agent_model="runner",
                judge_model="judge",
                workspace_manager=self._workspace_manager(root),
            )

            guard = payload["plugins"]["entries"][BENCHMARK_WORKDIR_GUARD_PLUGIN_ID]
            policy = guard["config"]["agentPolicies"]["benchmark-single-skills-off"]
            serialized = json.dumps(policy)
            self.assertFalse(policy["skills_enabled"])
            self.assertNotIn(str(skills_root.resolve()), serialized)
            self.assertNotIn("run_skill.py", serialized)
            self.assertNotIn(str(skills_root.resolve()), payload.get("skills", {}).get("load", {}).get("extraDirs", []))
    def test_build_run_scoped_config_payload_does_not_write_skill_exec_tools_md_for_skills_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = RuntimeConfigContext(
                agents_root=root / "agents",
                judge_agent_id="benchmark-judge",
                chemqa_slot_sets={},
                experiment_specs={
                    "single_llm_skills_off": ExperimentSpec(
                        id="single_llm_skills_off",
                        label="Single LLM without skills",
                        runner_kind="single_llm",
                        websearch_enabled=True,
                        skills_enabled=False,
                        single_agent_id="benchmark-single-skills-off",
                    )
                },
                benchmark_skills_root=root / "workspace" / "skills",
            )
            base = {
                "agents": {"list": []},
                "tools": {"web": {"search": {"enabled": False}}},
                "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
            }
            group = RuntimeConfigGroup(
                id="single_llm_skills_off",
                label="Single LLM without skills",
                runner="single_llm",
                websearch=True,
                skills_enabled=False,
            )

            build_run_scoped_config_payload(
                base,
                context=context,
                group=group,
                single_agent_model="qwen3.5-plus",
                judge_model="su8/gpt-5.4",
                workspace_manager=self._workspace_manager(root),
            )

            tools_md = root / "benchmark-runtime" / "benchmark-single-skills-off" / "TOOLS.md"
            self.assertFalse(tools_md.exists())

    def test_build_run_scoped_config_payload_provisions_chemqa_slots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = RuntimeConfigContext(
                agents_root=root / "agents",
                judge_agent_id="benchmark-judge",
                chemqa_slot_sets={"chemqa_skills_on": "A"},
                experiment_specs={
                    "chemqa_skills_on": ExperimentSpec(
                        id="chemqa_skills_on",
                        label="ChemQA with skills",
                        runner_kind="chemqa",
                        websearch_enabled=True,
                        skills_enabled=True,
                        slot_set="A",
                        skill_allowlist=("chem-calculator", "rdkit"),
                    )
                },
                benchmark_skills_root=root / "workspace" / "skills",
            )
            base = {
                "agents": {"list": []},
                "tools": {"web": {"search": {"enabled": False}}},
                "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
            }
            group = RuntimeConfigGroup(
                id="chemqa_skills_on",
                label="ChemQA with skills",
                runner="chemqa",
                websearch=True,
                skills_enabled=True,
            )

            payload = build_run_scoped_config_payload(
                base,
                context=context,
                group=group,
                single_agent_model="qwen3.5-plus",
                judge_model="su8/gpt-5.4",
                workspace_manager=self._workspace_manager(root),
            )

            agents = {entry["id"]: entry for entry in payload["agents"]["list"]}
            self.assertEqual("qwen3.5-plus", agents["debateA-coordinator"]["model"])
            self.assertEqual("qwen3.5-plus", agents["debateA-5"]["model"])
            self.assertEqual(["chem-calculator", "rdkit"], agents["debateA-coordinator"]["skills"])
            self.assertEqual(["chem-calculator", "rdkit"], agents["debateA-5"]["skills"])
            self.assertFalse(Path(agents["debateA-1"]["workspace"]).exists())
            self.assertNotEqual(agents["debateA-1"]["workspace"], agents["debateA-2"]["workspace"])
            guard = payload["plugins"]["entries"][BENCHMARK_WORKDIR_GUARD_PLUGIN_ID]
            policies = guard["config"]["agentPolicies"]
            self.assertEqual(set(actual_slot_ids("A").values()), set(policies))
            self.assertTrue(all(policy["role"] == "chemqa" for policy in policies.values()))

    def test_build_run_scoped_config_payload_wraps_renderer_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = RuntimeConfigContext(
                agents_root=root / "agents",
                judge_agent_id="benchmark-judge",
                chemqa_slot_sets={},
                experiment_specs={
                    "single_llm_skills_on": ExperimentSpec(
                        id="single_llm_skills_on",
                        label="Single LLM with skills",
                        runner_kind="single_llm",
                        websearch_enabled=True,
                        skills_enabled=True,
                        single_agent_id="benchmark-single-skills-on",
                    )
                },
                benchmark_skills_root=root / "workspace" / "skills",
            )
            group = RuntimeConfigGroup(
                id="single_llm_skills_on",
                label="Single LLM with skills",
                runner="single_llm",
                websearch=True,
                skills_enabled=True,
            )

            with self.assertRaises(RuntimeConfigError):
                build_run_scoped_config_payload(
                    {"agents": {"list": {}}},
                    context=context,
                    group=group,
                    single_agent_model="qwen3.5-plus",
                    judge_model="su8/gpt-5.4",
                    workspace_manager=self._workspace_manager(root),
                )

    def test_config_pool_falls_back_to_openai_gpt_55(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_config = root / "openclaw.json"
            base_config.write_text(
                json.dumps(
                    {
                        "agents": {"list": []},
                        "tools": {"web": {"search": {"enabled": False}}},
                        "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
                    }
                ),
                encoding="utf-8",
            )
            context = RuntimeConfigContext(
                agents_root=root / "agents",
                judge_agent_id="benchmark-judge",
                chemqa_slot_sets={},
                experiment_specs={
                    "single_llm_skills_on": ExperimentSpec(
                        id="single_llm_skills_on",
                        label="Single LLM with skills",
                        runner_kind="single_llm",
                        websearch_enabled=True,
                        skills_enabled=True,
                        single_agent_id="benchmark-single-skills-on",
                    )
                },
                benchmark_skills_root=root / "workspace" / "skills",
            )
            group = RuntimeConfigGroup(
                id="single_llm_skills_on",
                label="Single LLM with skills",
                runner="single_llm",
                websearch=True,
                skills_enabled=True,
            )

            manager = self._workspace_manager(root)
            pool = ConfigPool(
                base_config_path=base_config,
                output_root=root / "runs",
                context=context,
                run_id="run-1",
                invocation_id="invocation-1",
                workspace_manager=manager,
            )
            config_path = pool.config_for_group(group)
            payload = json.loads(config_path.read_text(encoding="utf-8"))

            agents = {entry["id"]: entry for entry in payload["agents"]["list"]}
            self.assertEqual("openai/gpt-5.5", agents["benchmark-single-skills-on"]["model"])
            self.assertEqual("openai/gpt-5.5", agents["benchmark-judge"]["model"])

    def test_same_agent_override_remains_isolated_by_group_and_never_uses_legacy_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = self._workspace_manager(root)
            specs = {
                group_id: ExperimentSpec(
                    id=group_id,
                    label=group_id,
                    runner_kind="single_llm",
                    websearch_enabled=False,
                    skills_enabled=group_id.endswith("on"),
                    single_agent_id=f"default-{group_id}",
                )
                for group_id in ("single_llm_skills_on", "single_llm_skills_off")
            }
            context = RuntimeConfigContext(
                agents_root=root / "agents",
                judge_agent_id="benchmark-judge",
                chemqa_slot_sets={},
                experiment_specs=specs,
                benchmark_skills_root=root / "workspace" / "skills",
            )
            base = {
                "agents": {"list": []},
                "tools": {"web": {"search": {"enabled": False}}},
                "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
            }
            paths = []
            for group_id in specs:
                group = RuntimeConfigGroup(
                    id=group_id,
                    label=group_id,
                    runner="single_llm",
                    websearch=False,
                    skills_enabled=group_id.endswith("on"),
                )
                payload = build_run_scoped_config_payload(
                    base,
                    context=context,
                    group=group,
                    single_agent_model="runner",
                    judge_model="judge",
                    workspace_manager=manager,
                    single_agent_id_override="shared-agent",
                )
                agent = next(entry for entry in payload["agents"]["list"] if entry["id"] == "shared-agent")
                paths.append(Path(agent["workspace"]))

            self.assertNotEqual(paths[0], paths[1])
            for path in paths:
                self.assertTrue(str(path).startswith(str(manager.invocation_runtime_root)))
                self.assertNotIn(str(root / "legacy"), str(path))


if __name__ == "__main__":
    unittest.main()
