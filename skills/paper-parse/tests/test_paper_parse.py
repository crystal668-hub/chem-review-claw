from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "paper_parse.py"
MODULE_NAME = "paper_parse_under_test"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert SPEC and SPEC.loader
paper_parse = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = paper_parse
SPEC.loader.exec_module(paper_parse)


class PaperParseMineruAPITests(unittest.TestCase):
    def tearDown(self) -> None:
        paper_parse._read_dotenv.cache_clear()

    def test_parser_config_reads_mineru_api_url(self) -> None:
        config = paper_parse.ParserConfig.from_dict(
            {
                "mineru_api_url": "  http://127.0.0.1:8000  ",
                "primary_backend": "MinerU",
                "secondary_backend": "fitz",
            }
        )

        self.assertEqual(config.mineru_api_url, "http://127.0.0.1:8000")
        self.assertEqual(config.primary_backend, "mineru")
        self.assertEqual(config.secondary_backend, "pymupdf")

    def test_parser_config_reads_mineru_api_url_from_env(self) -> None:
        with mock.patch.dict(paper_parse.os.environ, {"MINERU_API_URL": "http://127.0.0.1:8000"}, clear=False):
            config = paper_parse.ParserConfig.from_dict({})

        self.assertEqual(config.mineru_api_url, "http://127.0.0.1:8000")

    def test_parser_config_reads_mineru_api_url_from_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            env_path = Path(temp_dir_name) / ".env"
            env_path.write_text("MINERU_API_URL=http://127.0.0.1:8000\n", encoding="utf-8")
            with mock.patch.object(paper_parse, "DEFAULT_ENV_FILE", env_path):
                with mock.patch.dict(paper_parse.os.environ, {}, clear=True):
                    paper_parse._read_dotenv.cache_clear()
                    config = paper_parse.ParserConfig.from_dict({})

        self.assertEqual(config.mineru_api_url, "http://127.0.0.1:8000")

    def test_extract_with_mineru_passes_api_url_to_cli(self) -> None:
        engine = paper_parse.PaperParseEngine(
            config=paper_parse.ParserConfig(
                mineru_api_url="http://127.0.0.1:8000",
                min_total_chars=1,
                min_chars_per_text_page=1,
                min_text_page_ratio=0.0,
                min_printable_ratio=0.0,
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir_name:
            pdf_path = Path(temp_dir_name) / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")

            def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
                self.assertIn("--api-url", command)
                self.assertEqual(command[command.index("--api-url") + 1], "http://127.0.0.1:8000")
                self.assertIn("env", kwargs)
                self.assertIn("127.0.0.1", kwargs["env"]["NO_PROXY"])
                self.assertIn("127.0.0.1", kwargs["env"]["no_proxy"])
                output_dir = Path(command[command.index("-o") + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "sample.md").write_text("# Abstract\nhello world", encoding="utf-8")
                (output_dir / "sample_content_list.json").write_text(
                    json.dumps([{"page_idx": 0, "type": "text", "text": "Abstract"}]),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with mock.patch.object(paper_parse.shutil, "which", return_value="/usr/bin/mineru"):
                with mock.patch.object(paper_parse.subprocess, "run", side_effect=fake_run):
                    attempt = engine._extract_with_mineru(pdf_path)

        self.assertTrue(attempt.succeeded)
        self.assertTrue(attempt.usable)

    def test_extract_with_mineru_omits_api_url_when_unset(self) -> None:
        engine = paper_parse.PaperParseEngine(
            config=paper_parse.ParserConfig(
                min_total_chars=1,
                min_chars_per_text_page=1,
                min_text_page_ratio=0.0,
                min_printable_ratio=0.0,
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir_name:
            pdf_path = Path(temp_dir_name) / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")

            def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
                self.assertNotIn("--api-url", command)
                self.assertNotIn("env", kwargs)
                output_dir = Path(command[command.index("-o") + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "sample.md").write_text("# Abstract\nhello world", encoding="utf-8")
                (output_dir / "sample_content_list.json").write_text(
                    json.dumps([{"page_idx": 0, "type": "text", "text": "Abstract"}]),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with mock.patch.object(paper_parse.shutil, "which", return_value="/usr/bin/mineru"):
                with mock.patch.object(paper_parse.subprocess, "run", side_effect=fake_run):
                    attempt = engine._extract_with_mineru(pdf_path)

        self.assertTrue(attempt.succeeded)
        self.assertTrue(attempt.usable)


if __name__ == "__main__":
    unittest.main()
