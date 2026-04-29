import json
import tempfile
import unittest
from pathlib import Path

from benchmarking.config_renderer import render_run_config
from benchmarking.experiments import ExperimentSpec
from benchmarking.provisioning import (
    ProvisionedAgent,
    ProvisionedExperiment,
    provision_slot_workspace,
)
from benchmarking.runtime_config import (
    RuntimeConfigContext,
    RuntimeConfigError,
    build_run_scoped_config_payload,
)


class RuntimeConfigGroup:
    def __init__(self, *, id: str, label: str, runner: str, websearch: bool) -> None:
        self.id = id
        self.label = label
        self.runner = runner
        self.websearch = websearch


class BenchmarkConfigRuntimeTests(unittest.TestCase):
    def test_render_run_config_is_pure_and_does_not_mutate_base_payload(self) -> None:
        base = {
            "agents": {"list": []},
            "tools": {"web": {"search": {"enabled": False}}},
            "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
        }
        spec = ExperimentSpec(
            id="single_llm_web_on",
            label="Single LLM with web",
            runner_kind="single_llm",
            websearch_enabled=True,
            single_agent_id="benchmark-single-web-on",
        )
        provisioned = ProvisionedExperiment(
            judge=ProvisionedAgent("benchmark-judge", Path("/tmp/judge"), Path("/tmp/agents/judge")),
            runner_agents=(
                ProvisionedAgent(
                    "benchmark-single-web-on",
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
        self.assertTrue(rendered["plugins"]["entries"]["duckduckgo"]["enabled"])

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
                        "id": "benchmark-single-web-on",
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
            id="single_llm_web_on",
            label="Single LLM with web",
            runner_kind="single_llm",
            websearch_enabled=False,
            single_agent_id="benchmark-single-web-on",
        )
        provisioned = ProvisionedExperiment(
            judge=ProvisionedAgent("benchmark-judge", Path("/tmp/judge"), Path("/tmp/agents/judge")),
            runner_agents=(
                ProvisionedAgent(
                    "benchmark-single-web-on",
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
        self.assertEqual("qwen3.5-plus", agents["benchmark-single-web-on"]["model"])
        self.assertNotIn("thinking", agents["benchmark-judge"])
        self.assertNotIn("thinking", agents["benchmark-single-web-on"])
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
                baseline_workspace_root=root / "benchmark-runtime",
                chemqa_workspace_roots={
                    "A": root / "benchmark-runtime" / "chemqa_web_on",
                    "B": root / "benchmark-runtime" / "chemqa_web_off",
                },
                agents_root=root / "agents",
                judge_agent_id="benchmark-judge",
                chemqa_slot_sets={"chemqa_web_on": "A", "chemqa_web_off": "B"},
                experiment_specs={
                    "single_llm_web_on": ExperimentSpec(
                        id="single_llm_web_on",
                        label="Single LLM with web",
                        runner_kind="single_llm",
                        websearch_enabled=True,
                        single_agent_id="benchmark-single-web-on",
                    )
                },
                load_slot_agents_template=lambda: "# slot template\n",
            )
            base = {
                "agents": {"list": []},
                "tools": {"web": {"search": {"enabled": False}}},
                "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
            }
            group = RuntimeConfigGroup(
                id="single_llm_web_on",
                label="Single LLM with web",
                runner="single_llm",
                websearch=True,
            )

            payload = build_run_scoped_config_payload(
                base,
                context=context,
                group=group,
                single_agent_model="qwen3.5-plus",
                judge_model="su8/gpt-5.4",
            )

            agents = {entry["id"]: entry for entry in payload["agents"]["list"]}
            self.assertEqual("qwen3.5-plus", agents["benchmark-single-web-on"]["model"])
            self.assertEqual("su8/gpt-5.4", agents["benchmark-judge"]["model"])
            self.assertEqual(
                str((root / "benchmark-runtime" / "benchmark-single-web-on").resolve()),
                agents["benchmark-single-web-on"]["workspace"],
            )

    def test_build_run_scoped_config_payload_provisions_chemqa_slots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = RuntimeConfigContext(
                baseline_workspace_root=root / "benchmark-runtime",
                chemqa_workspace_roots={
                    "A": root / "benchmark-runtime" / "chemqa_web_on",
                    "B": root / "benchmark-runtime" / "chemqa_web_off",
                },
                agents_root=root / "agents",
                judge_agent_id="benchmark-judge",
                chemqa_slot_sets={"chemqa_web_on": "A", "chemqa_web_off": "B"},
                experiment_specs={},
                load_slot_agents_template=lambda: "# slot template\n",
            )
            base = {
                "agents": {"list": []},
                "tools": {"web": {"search": {"enabled": False}}},
                "plugins": {"entries": {"duckduckgo": {"enabled": False, "config": {}}}},
            }
            group = RuntimeConfigGroup(
                id="chemqa_web_off",
                label="ChemQA without web",
                runner="chemqa",
                websearch=False,
            )

            payload = build_run_scoped_config_payload(
                base,
                context=context,
                group=group,
                single_agent_model="qwen3.5-plus",
                judge_model="su8/gpt-5.4",
            )

            agents = {entry["id"]: entry for entry in payload["agents"]["list"]}
            self.assertEqual("qwen3.5-plus", agents["debateB-coordinator"]["model"])
            self.assertEqual("qwen3.5-plus", agents["debateB-5"]["model"])
            self.assertTrue((root / "benchmark-runtime" / "chemqa_web_off" / "debateB-1" / "AGENTS.md").is_file())
            self.assertTrue((root / "benchmark-runtime" / "chemqa_web_off" / "debateB-1" / ".debateclaw-slot.json").is_file())

    def test_build_run_scoped_config_payload_wraps_renderer_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            context = RuntimeConfigContext(
                baseline_workspace_root=root / "benchmark-runtime",
                chemqa_workspace_roots={},
                agents_root=root / "agents",
                judge_agent_id="benchmark-judge",
                chemqa_slot_sets={},
                experiment_specs={
                    "single_llm_web_on": ExperimentSpec(
                        id="single_llm_web_on",
                        label="Single LLM with web",
                        runner_kind="single_llm",
                        websearch_enabled=True,
                        single_agent_id="benchmark-single-web-on",
                    )
                },
                load_slot_agents_template=lambda: "# slot template\n",
            )
            group = RuntimeConfigGroup(
                id="single_llm_web_on",
                label="Single LLM with web",
                runner="single_llm",
                websearch=True,
            )

            with self.assertRaises(RuntimeConfigError):
                build_run_scoped_config_payload(
                    {"agents": {"list": {}}},
                    context=context,
                    group=group,
                    single_agent_model="qwen3.5-plus",
                    judge_model="su8/gpt-5.4",
                )


if __name__ == "__main__":
    unittest.main()
