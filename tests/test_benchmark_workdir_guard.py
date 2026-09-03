from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

PLUGIN_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarking"
    / "runtime"
    / "openclaw_plugins"
    / "benchmark-workdir-guard"
    / "index.js"
)


@unittest.skipUnless(shutil.which("node"), "Node.js is required for the OpenClaw plugin test")
class BenchmarkWorkdirGuardTests(unittest.TestCase):
    def _run_hook(
        self,
        *,
        workspace: Path,
        workdir: str | None = None,
        tool_name: str = "exec",
        params: dict[str, object] | None = None,
        protected_roots: list[dict[str, str]] | None = None,
        attempt_python: bool = False,
    ) -> dict[str, object] | None:
        policy = {
            "policy_digest": "test-policy",
            "read_scopes": [
                {"scope_id": "active_workspace", "path": str(workspace), "kind": "directory"}
            ],
            "write_scopes": [
                {"scope_id": "attempt_scratch", "path": str(workspace / "scratch"), "kind": "directory"}
            ],
            "exec_workdir_scopes": [
                {"scope_id": "active_workspace", "path": str(workspace), "kind": "directory"}
            ],
            "protected_roots": protected_roots or [],
        }
        event_params = params if params is not None else ({} if workdir is None else {"workdir": workdir})
        script = """
import plugin from %s;
let handler;
plugin.register({
  pluginConfig: { agentPolicies: { "benchmark-agent": %s } },
  on(name, candidate) {
    if (name === "before_tool_call") handler = candidate;
  },
});
const result = handler(
  { toolName: %s, params: %s },
  { agentId: "benchmark-agent" },
);
process.stdout.write(JSON.stringify(result ?? null));
""" % (
            json.dumps(PLUGIN_PATH.as_uri()),
            json.dumps(policy),
            json.dumps(tool_name),
            json.dumps(event_params),
        )
        environment = None
        if attempt_python:
            environment = {**os.environ, "BENCHMARK_ATTEMPT_PYTHON": "/attempt/venv/bin/python"}
        completed = subprocess.run(
            ["node", "--input-type=module", "-e", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        return json.loads(completed.stdout)

    def test_missing_explicit_workdir_is_blocked_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()

            result = self._run_hook(workspace=workspace, workdir=str(workspace / "missing"))

            self.assertIs(result["block"], True)
            self.assertIn("does not exist", str(result["blockReason"]))
            self.assertIn("The operation was not executed", str(result["blockReason"]))

    def test_existing_workdir_inside_attempt_workspace_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            scratch = workspace / ".benchmark-scratch" / "record" / "session"
            scratch.mkdir(parents=True)

            self.assertIsNone(self._run_hook(workspace=workspace, workdir=str(scratch)))

    def test_workdir_outside_attempt_workspace_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace = root / "workspace"
            outside = root / "outside"
            workspace.mkdir()
            outside.mkdir()

            result = self._run_hook(workspace=workspace, workdir=str(outside))

            self.assertIs(result["block"], True)
            self.assertIn("outside the policy scope", str(result["blockReason"]))

    def test_file_workdir_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()
            file_path = workspace / "not-a-directory"
            file_path.write_text("data", encoding="utf-8")

            result = self._run_hook(workspace=workspace, workdir=str(file_path))

            self.assertIs(result["block"], True)
            self.assertIn("not a directory", str(result["blockReason"]))

    def test_symlink_escape_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace = root / "workspace"
            outside = root / "outside"
            workspace.mkdir()
            outside.mkdir()
            symlink = workspace / "outside-link"
            symlink.symlink_to(outside, target_is_directory=True)

            result = self._run_hook(workspace=workspace, workdir=str(symlink))

            self.assertIs(result["block"], True)
            self.assertIn("outside the policy scope", str(result["blockReason"]))

    def test_omitted_workdir_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()

            self.assertIsNone(self._run_hook(workspace=workspace, workdir=None))

    def test_exec_command_referencing_protected_root_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace = root / "workspace"
            protected = root / "runtime" / "runs"
            (workspace / "scratch").mkdir(parents=True)
            protected.mkdir(parents=True)
            policy_path = protected / "other-attempt" / "secret.txt"

            result = self._run_hook(
                workspace=workspace,
                params={"command": f"ls -la {policy_path}"},
                protected_roots=[
                    {"policy_id": "benchmark_runtime_root", "path": str(root / "runtime"), "source": "test"}
                ],
            )

            self.assertIs(result["block"], True)
            self.assertIn("exec command references a protected path", str(result["blockReason"]))

    def test_exec_command_outside_workspace_is_blocked_and_system_binary_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace = root / "workspace"
            outside = root / "outside"
            (workspace / "scratch").mkdir(parents=True)
            outside.mkdir()

            blocked = self._run_hook(
                workspace=workspace,
                params={"command": f"cat {outside / 'secret.txt'}"},
            )
            self.assertIs(blocked["block"], True)

            self.assertIsNone(
                self._run_hook(
                    workspace=workspace,
                    params={"command": "/usr/bin/printf ok"},
                )
            )

            for command in ("cat ../outside/secret.txt", "cat ~/secret.txt"):
                with self.subTest(command=command):
                    result = self._run_hook(workspace=workspace, params={"command": command})
                    self.assertIs(result["block"], True)

    def test_exec_command_inside_workspace_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            (workspace / "scratch").mkdir(parents=True)

            self.assertIsNone(
                self._run_hook(
                    workspace=workspace,
                    params={"command": "cat scratch/notes.txt"},
                )
            )

    def test_attempt_dependency_policy_blocks_pip_mutation_and_non_registry_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            (workspace / "scratch").mkdir(parents=True)
            for command in (
                "pip install rdkit",
                "python -m pip uninstall rdkit",
                "uv pip install git+https://example.com/repo.git",
                "uv pip install --extra-index-url https://example.com/simple rdkit",
                "uv pip install verifier-grounded-benchmark",
                "uv pip install --python /tmp/other-python rdkit",
                "uv sync",
                "uv run --with rdkit python scratch/tmp/calc.py",
            ):
                with self.subTest(command=command):
                    result = self._run_hook(
                        workspace=workspace,
                        params={"command": command},
                        attempt_python=True,
                    )
                    self.assertIs(result["block"], True)
                    self.assertIn("access=dependency", str(result["blockReason"]))

    def test_attempt_dependency_policy_allows_uv_registry_install(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            (workspace / "scratch").mkdir(parents=True)
            self.assertIsNone(
                self._run_hook(
                    workspace=workspace,
                    params={"command": "uv pip install rdkit"},
                    attempt_python=True,
                )
            )

    def test_structured_write_inside_scratch_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            (workspace / "scratch").mkdir(parents=True)

            self.assertIsNone(
                self._run_hook(
                    workspace=workspace,
                    tool_name="write",
                    params={"path": "scratch/notes.txt", "content": "ok"},
                )
            )

    def test_structured_write_outside_scratch_is_blocked_with_stable_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace = root / "workspace"
            outside = root / "outside"
            (workspace / "scratch").mkdir(parents=True)
            outside.mkdir()

            result = self._run_hook(
                workspace=workspace,
                tool_name="write",
                params={"path": str(outside / "answer.txt"), "content": "no"},
            )

            self.assertIs(result["block"], True)
            self.assertIn("benchmark_workspace_guard_blocked", str(result["blockReason"]))
            self.assertIn("access=write", str(result["blockReason"]))

    def test_structured_read_symlink_escape_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workspace = root / "workspace"
            outside = root / "outside"
            (workspace / "scratch").mkdir(parents=True)
            outside.mkdir()
            (outside / "secret.txt").write_text("secret", encoding="utf-8")
            (workspace / "scratch" / "link").symlink_to(outside, target_is_directory=True)

            result = self._run_hook(
                workspace=workspace,
                tool_name="read",
                params={"path": "scratch/link/secret.txt"},
            )

            self.assertIs(result["block"], True)
            self.assertIn("access=read", str(result["blockReason"]))
