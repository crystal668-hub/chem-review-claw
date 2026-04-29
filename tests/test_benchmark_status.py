from __future__ import annotations

import unittest

from benchmarking.status import is_chemqa_terminal_status, normalize_chemqa_run_status


class BenchmarkStatusTests(unittest.TestCase):
    def test_normalize_chemqa_run_status_maps_completed_with_artifact_errors(self) -> None:
        payload = normalize_chemqa_run_status({"status": "completed_with_artifact_errors"})

        self.assertEqual("done", payload["status"])
        self.assertEqual("completed", payload["terminal_state"])
        self.assertEqual("artifact_collection_error", payload["terminal_reason_code"])
        self.assertEqual("error", payload["artifact_collection"]["status"])
        self.assertEqual("completed_with_artifact_errors", payload["legacy_status"])

    def test_normalize_chemqa_run_status_keeps_artifact_finalizing_non_terminal(self) -> None:
        payload = normalize_chemqa_run_status(
            {
                "status": "done",
                "protocol_terminal_state": "completed",
                "artifact_flow_state": "finalizing",
                "benchmark_terminal_state": "running",
                "terminal_state": "running",
            }
        )

        self.assertEqual("running", payload["status"])
        self.assertEqual("completed", payload["protocol_terminal_state"])
        self.assertEqual("finalizing", payload["artifact_flow_state"])
        self.assertEqual("running", payload["benchmark_terminal_state"])
        self.assertFalse(is_chemqa_terminal_status(payload))
