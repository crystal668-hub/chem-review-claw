from __future__ import annotations

import io
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_LEASE_PATH = ROOT / "skills" / "benchmark-cleanroom" / "scripts" / "runtime_lease.py"
CLEANUP_PATH = ROOT / "skills" / "benchmark-cleanroom" / "scripts" / "cleanup_benchmark_run.py"


def load_module(name: str, path: Path):
    script_dir = str(path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runtime_lease = load_module("benchmark_cleanroom_runtime_lease_test", RUNTIME_LEASE_PATH)
cleanup_benchmark_run = load_module("benchmark_cleanroom_cleanup_test", CLEANUP_PATH)


class BenchmarkCleanroomTests(unittest.TestCase):
    def test_parse_process_snapshot_handles_macos_style_ps_output(self) -> None:
        payload = "\n".join(
            [
                "  101   100 /usr/bin/python3 script.py --run-id demo-run",
                "  202   200 /opt/homebrew/bin/openclaw --session demo-session",
                "not-a-valid-line",
                "",
            ]
        )
        parsed = cleanup_benchmark_run.parse_process_snapshot(payload)
        self.assertEqual(
            [
                cleanup_benchmark_run.ProcessInfo(pid=101, pgid=100, command="/usr/bin/python3 script.py --run-id demo-run"),
                cleanup_benchmark_run.ProcessInfo(pid=202, pgid=200, command="/opt/homebrew/bin/openclaw --session demo-session"),
            ],
            parsed,
        )

    def test_process_targets_scans_process_snapshot_for_run_and_session_matches(self) -> None:
        manifest = {
            "run_id": "demo-run",
            "session_assignments": {"debate-1": "demo-session"},
        }
        snapshot = [
            cleanup_benchmark_run.ProcessInfo(pid=12345, pgid=12340, command="python worker.py --run-id demo-run"),
            cleanup_benchmark_run.ProcessInfo(pid=22345, pgid=22340, command="openclaw resume demo-session"),
            cleanup_benchmark_run.ProcessInfo(pid=os.getpid(), pgid=0, command="current-process"),
            cleanup_benchmark_run.ProcessInfo(pid=32345, pgid=32340, command="unrelated command"),
        ]
        with mock.patch.object(cleanup_benchmark_run, "process_snapshot", return_value=(snapshot, [])):
            groups, targets, warnings = cleanup_benchmark_run.process_targets(manifest, [])
        self.assertEqual([], warnings)
        self.assertEqual({12340, 22340}, {item["pgid"] for item in groups})
        self.assertEqual({12345, 22345}, {item["pid"] for item in targets})
        self.assertEqual({"proc-scan"}, {item["source"] for item in targets})

    def test_postcheck_remaining_processes_handles_missing_cmdline(self) -> None:
        manifest = {"run_id": "demo-run", "session_assignments": {}}
        context = cleanup_benchmark_run.CleanupContext(manifest=manifest, manifest_path=None)
        lease_payloads = [{"pid": 43210, "pgid": 43200, "role": "driver", "slot": "debate-1", "session_id": "demo-session"}]
        with mock.patch.object(cleanup_benchmark_run, "iter_lease_payloads", return_value=lease_payloads):
            with mock.patch.object(cleanup_benchmark_run, "process_targets", return_value=([], [{"pid": 43210, "pgid": 43200, "source": "lease", "cmdline": ""}], [])):
                with mock.patch.object(cleanup_benchmark_run, "candidate_session_stores", return_value=[]):
                    with mock.patch.object(cleanup_benchmark_run, "session_paths_from_manifest", return_value=[]):
                        with mock.patch.object(cleanup_benchmark_run, "terminate_process_groups", return_value=[]):
                            with mock.patch.object(cleanup_benchmark_run, "terminate_pids", return_value=[]):
                                with mock.patch.object(cleanup_benchmark_run, "wait_for_exit", return_value=[43210]):
                                    with mock.patch.object(cleanup_benchmark_run, "pid_exists", side_effect=lambda pid: pid == 43210):
                                        with mock.patch.object(cleanup_benchmark_run, "process_snapshot", return_value=([], ["ps unavailable for postcheck"])):
                                            report = cleanup_benchmark_run.cleanup(
                                                context,
                                                grace_seconds=0.0,
                                                kill_after_seconds=0.0,
                                                dry_run=False,
                                            )
        self.assertFalse(report["success"])
        self.assertEqual([{"pid": 43210, "cmdline": ""}], report["postcheck"]["remaining_processes"])
        self.assertIn("ps unavailable for postcheck", report["warnings"])

    def test_main_json_mode_returns_structured_error_payload(self) -> None:
        with mock.patch.object(cleanup_benchmark_run, "load_context", side_effect=RuntimeError("boom")):
            with mock.patch.object(
                sys,
                "argv",
                [
                    "cleanup_benchmark_run.py",
                    "--run-id",
                    "demo-run",
                    "--kind",
                    "chemqa",
                    "--output-root",
                    "/tmp/demo-cleanroom",
                    "--json",
                ],
            ):
                with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    exit_code = cleanup_benchmark_run.main()
        payload = json.loads(stdout.getvalue())
        self.assertEqual(1, exit_code)
        self.assertFalse(payload["success"])
        self.assertIn("Unhandled cleanup failure: boom", payload["errors"])

    def test_manual_dry_run_json_does_not_require_procfs(self) -> None:
        script = str(CLEANUP_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    script,
                    "--run-id",
                    "demo-run",
                    "--kind",
                    "chemqa",
                    "--output-root",
                    tmpdir,
                    "--dry-run",
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual("", result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["success"])
        self.assertEqual("demo-run", payload["run_id"])

    def test_manifest_payload_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "out"
            payload = runtime_lease.build_manifest_payload(
                run_id="demo-run",
                benchmark_kind="chemqa",
                group_id="chemqa_web_off",
                output_root=output_root,
                launch_home=output_root / "launch-home",
                clawteam_data_dir=output_root / "clawteam-data",
                session_assignments={"debateB-1": "demo-session"},
                control_roots=[str(output_root / "control" / "runplans" / "demo-run.json")],
                generated_roots=[str(output_root / "generated" / "runtime-context" / "demo-run-context.json")],
                artifact_roots=[str(output_root / "artifacts" / "demo-run")],
                lease_dir=output_root / "cleanroom" / "leases",
            )
            path = runtime_lease.manifest_path(output_root, "demo-run")
            runtime_lease.write_manifest(path, payload)
            loaded = runtime_lease.read_json(path)
            self.assertEqual(runtime_lease.MANIFEST_KIND, loaded["kind"])
            self.assertEqual("demo-run", loaded["run_id"])
            self.assertEqual("demo-session", loaded["session_assignments"]["debateB-1"])

    def test_scrub_session_store_removes_only_matching_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "sessions.json"
            payload = {
                "agent:debateB-1:main": {
                    "sessionId": "demo-session",
                    "sessionFile": "/tmp/demo-session.jsonl",
                },
                "agent:debateB-2:main": {
                    "sessionId": "other-session",
                    "sessionFile": "/tmp/other-session.jsonl",
                },
            }
            store_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            result = cleanup_benchmark_run.scrub_session_store(
                store_path,
                run_id="demo-run",
                session_ids={"demo-session"},
                dry_run=False,
            )
            self.assertTrue(result["changed"])
            updated = json.loads(store_path.read_text(encoding="utf-8"))
            self.assertNotIn("agent:debateB-1:main", updated)
            self.assertIn("agent:debateB-2:main", updated)

    def test_session_paths_from_manifest_collects_jsonl_checkpoint_and_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            launch_home = Path(tmpdir) / "launch-home"
            session_dir = launch_home / ".openclaw" / "agents" / "debate-1" / "sessions"
            session_dir.mkdir(parents=True, exist_ok=True)
            (session_dir / "demo-session.jsonl").write_text("{}", encoding="utf-8")
            (session_dir / "demo-session.jsonl.lock").write_text("", encoding="utf-8")
            (session_dir / "demo-session.checkpoint.001.jsonl").write_text("{}", encoding="utf-8")
            manifest = {
                "run_id": "demo-run",
                "launch_home": str(launch_home),
                "session_assignments": {"debate-1": "demo-session"},
            }
            paths = cleanup_benchmark_run.session_paths_from_manifest(manifest)
            names = sorted(path.name for path in paths)
            self.assertEqual(
                ["demo-session.checkpoint.001.jsonl", "demo-session.jsonl", "demo-session.jsonl.lock"],
                names,
            )


if __name__ == "__main__":
    unittest.main()
