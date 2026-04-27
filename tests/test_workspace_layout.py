from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class WorkspaceLayoutTests(unittest.TestCase):
    def test_review_loop_benchmark_entrypoint_is_removed(self) -> None:
        self.assertFalse((ROOT / "benchmark_rl.py").exists())


if __name__ == "__main__":
    unittest.main()
