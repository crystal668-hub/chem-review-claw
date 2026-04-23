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


if __name__ == "__main__":
    unittest.main()
