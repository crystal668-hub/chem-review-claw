from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "paper_parse.py"
MODULE_NAME = "paper_parse_under_test"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert SPEC and SPEC.loader
paper_parse = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = paper_parse
SPEC.loader.exec_module(paper_parse)


class FakeResponse:
    def __init__(self, *, status_code: int = 200, payload: dict | None = None, text: str = "", content: bytes | None = None, headers: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.content = content if content is not None else text.encode("utf-8")
        self.headers = headers or {}

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


class PaperParseBackendTests(unittest.TestCase):
    def tearDown(self) -> None:
        paper_parse._read_dotenv.cache_clear()

    def test_parser_config_defaults_to_cloud_auto_backend(self) -> None:
        config = paper_parse.ParserConfig.from_dict({})

        self.assertEqual(config.backend, "auto")
        self.assertEqual(config.agent_api_url, "https://mineru.net/api/v1/agent")
        self.assertEqual(config.precision_api_url, "https://mineru.net/api/v4")
        self.assertEqual(config.precision_token_env, "MINERU_API_TOKEN")

    def test_parser_config_reads_agent_url_from_env(self) -> None:
        with mock.patch.dict(paper_parse.os.environ, {"MINERU_AGENT_API_URL": " https://example.test/api/v1/agent "}, clear=False):
            config = paper_parse.ParserConfig.from_dict({})

        self.assertEqual(config.agent_api_url, "https://example.test/api/v1/agent")

    def test_auto_routes_small_documents_to_agent_then_precision_then_pymupdf(self) -> None:
        engine = paper_parse.PaperParseEngine(config=paper_parse.ParserConfig(poll_interval_seconds=0.1))
        with mock.patch.object(engine, "_precision_token", return_value="token"):
            self.assertEqual(
                engine._backend_order(pdf_size=1024, page_count=2),
                ["mineru-agent-api", "mineru-precision-api", "pymupdf"],
            )

    def test_auto_routes_large_documents_to_precision_when_token_exists(self) -> None:
        engine = paper_parse.PaperParseEngine(config=paper_parse.ParserConfig())
        with mock.patch.object(engine, "_precision_token", return_value="token"):
            self.assertEqual(engine._backend_order(pdf_size=paper_parse.LIGHTWEIGHT_MAX_BYTES + 1, page_count=2), ["mineru-precision-api", "pymupdf"])
            self.assertEqual(engine._backend_order(pdf_size=1024, page_count=21), ["mineru-precision-api", "pymupdf"])

    def test_auto_routes_large_documents_to_pymupdf_without_token(self) -> None:
        engine = paper_parse.PaperParseEngine(config=paper_parse.ParserConfig())
        with mock.patch.object(engine, "_precision_token", return_value=None):
            self.assertEqual(engine._backend_order(pdf_size=paper_parse.LIGHTWEIGHT_MAX_BYTES + 1, page_count=2), ["pymupdf"])

    def test_agent_file_upload_poll_and_markdown_download(self) -> None:
        responses = [
            FakeResponse(payload={"code": 0, "trace_id": "trace-1", "data": {"task_id": "task-1", "file_url": "https://upload.test/file"}}),
            FakeResponse(status_code=200),
            FakeResponse(payload={"code": 0, "data": {"task_id": "task-1", "state": "running"}}),
            FakeResponse(payload={"code": 0, "data": {"task_id": "task-1", "state": "done", "markdown_url": "https://cdn.test/full.md"}}),
            FakeResponse(text="# Abstract\n" + "word " * 300),
        ]
        with tempfile.TemporaryDirectory() as temp_dir_name:
            pdf_path = Path(temp_dir_name) / "paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            with mock.patch.object(paper_parse.requests, "request", side_effect=responses) as request:
                markdown, metadata = paper_parse.MineruAgentClient(
                    timeout_seconds=2,
                    poll_interval_seconds=0.1,
                    max_retries=0,
                ).extract(pdf_path=pdf_path, config=paper_parse.ParserConfig(agent_timeout_seconds=2, poll_interval_seconds=0.1))

        self.assertTrue(markdown.startswith("# Abstract"))
        self.assertEqual(metadata["task_id"], "task-1")
        self.assertEqual(metadata["trace_id"], "trace-1")
        self.assertEqual(request.call_args_list[0].args[:2], ("POST", "https://mineru.net/api/v1/agent/parse/file"))
        self.assertEqual(request.call_args_list[1].args[:2], ("PUT", "https://upload.test/file"))
        self.assertNotIn("Authorization", request.call_args_list[0].kwargs.get("headers", {}))

    def test_precision_batch_upload_and_zip_adaptation(self) -> None:
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as archive:
            archive.writestr("full.md", "# Introduction\n" + "word " * 300)
            archive.writestr("content_list.json", json.dumps([{"page_idx": 0, "type": "text", "text": "Introduction"}]))
        responses = [
            FakeResponse(payload={"code": 0, "trace_id": "trace-2", "data": {"batch_id": "batch-1", "file_urls": ["https://upload.test/file"]}}),
            FakeResponse(status_code=200),
            FakeResponse(payload={"code": 0, "data": {"batch_id": "batch-1", "extract_result": [{"file_name": "paper.pdf", "state": "done", "full_zip_url": "https://cdn.test/result.zip"}]}}),
            FakeResponse(content=archive_buffer.getvalue()),
        ]
        with tempfile.TemporaryDirectory() as temp_dir_name:
            pdf_path = Path(temp_dir_name) / "paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            config = paper_parse.ParserConfig(precision_timeout_seconds=2, poll_interval_seconds=0.1)
            with mock.patch.object(paper_parse.requests, "request", side_effect=responses):
                markdown, content_list, metadata = paper_parse.MineruPrecisionClient(
                    timeout_seconds=2,
                    poll_interval_seconds=0.1,
                    max_retries=0,
                ).extract(pdf_path=pdf_path, config=config, token="secret")

        self.assertIn("Introduction", markdown)
        self.assertEqual(content_list[0]["page_idx"], 0)
        self.assertEqual(metadata["batch_id"], "batch-1")

    def test_cloud_attempt_metadata_is_written_to_parse_result(self) -> None:
        engine = paper_parse.PaperParseEngine(
            config=paper_parse.ParserConfig(
                backend="mineru-agent-api",
                min_total_chars=1,
                min_chars_per_text_page=1,
                min_text_page_ratio=0.0,
                min_printable_ratio=0.0,
            )
        )
        fake_attempt = paper_parse.ExtractionAttempt(
            extractor="mineru-agent-api",
            succeeded=True,
            fulltext="# Abstract\nhello world",
            sections=[paper_parse.ExtractedSection("Abstract", "abstract", "hello world", 1, 1)],
            metrics={"reasons": []},
            usable=True,
            metadata={"task_id": "task-1", "trace_id": "trace-1"},
        )
        with tempfile.TemporaryDirectory() as temp_dir_name:
            with mock.patch.object(engine, "_extract_with_agent_api", return_value=fake_attempt):
                result = engine.process_pdf_bytes(document_id="paper", pdf_bytes=b"%PDF-1.4\n", output_dir=temp_dir_name)
            report = json.loads(Path(temp_dir_name, "paper.extraction_report.json").read_text(encoding="utf-8"))

        self.assertEqual(result["extractor"], "mineru-agent-api")
        self.assertEqual(report["attempts"][0]["metadata"]["task_id"], "task-1")

    def test_pymupdf_fallback_does_not_invoke_mineru_cli(self) -> None:
        engine = paper_parse.PaperParseEngine(config=paper_parse.ParserConfig(backend="pymupdf"))
        with mock.patch.object(paper_parse.requests, "request") as request, tempfile.TemporaryDirectory() as temp_dir_name:
            result = engine.process_pdf_bytes(document_id="invalid", pdf_bytes=b"not-a-pdf", output_dir=temp_dir_name)

        self.assertEqual(result["fulltext_status"], "fulltext_unusable")
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
