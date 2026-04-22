from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
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
