from __future__ import annotations

import argparse
import html
import ipaddress
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit


SECTION_HEADING_BODY = (
    r"(?:#{1,6}\s*)?"
    r"(?:\d+(?:\.\d+)*[.)]?\s+)?"
    r"(abstract|introduction|background|materials?\s+and\s+methods|methods?|experimental|results?|discussion|conclusions?|limitations?)"
    r"\s*:?"
)
SECTION_HEADING_PATTERN = re.compile(rf"(?mi)^{SECTION_HEADING_BODY}\s*$")
SECTION_HEADING_LINE_PATTERN = re.compile(rf"(?i)^{SECTION_HEADING_BODY}\s*$")
SECTION_TYPE_MAP = {
    "abstract": "abstract",
    "introduction": "introduction",
    "background": "introduction",
    "materials and methods": "methods",
    "material and methods": "methods",
    "methods": "methods",
    "method": "methods",
    "experimental": "methods",
    "result": "results",
    "results": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "limitation": "limitations",
    "limitations": "limitations",
}
UNKNOWN_SECTION_TITLE = "Body"
UNKNOWN_SECTION_TYPE = "unknown"
SUPPORTED_PDF_BACKENDS = {"mineru", "pymupdf"}
PDF_BACKEND_DISPLAY_NAMES = {
    "mineru": "MinerU",
    "pymupdf": "PyMuPDF",
}
PDF_BACKEND_ALIASES = {
    "fitz": "pymupdf",
    "magic-pdf": "mineru",
    "magic_pdf": "mineru",
}
MINERU_PAGE_AUXILIARY_TYPES = {"header", "footer", "page_number", "aside_text", "page_footnote", "seal"}
MINERU_TIMEOUT_SECONDS = 600
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


def _compact_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()


@lru_cache(maxsize=4)
def _read_dotenv(path_str: str) -> dict[str, str]:
    path = Path(path_str)
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = _compact_text(key)
        if not normalized_key:
            continue
        normalized_value = str(value).strip()
        if len(normalized_value) >= 2 and normalized_value[0] == normalized_value[-1] and normalized_value[0] in {'"', "'"}:
            normalized_value = normalized_value[1:-1]
        values[normalized_key] = normalized_value
    return values


def _default_env_value(name: str) -> Optional[str]:
    runtime_value = _compact_text(os.environ.get(name))
    if runtime_value:
        return runtime_value
    dotenv_value = _compact_text(_read_dotenv(str(DEFAULT_ENV_FILE)).get(name))
    return dotenv_value or None


def _normalize_text(value: Any) -> str:
    text = str(value or "").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_html(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return _normalize_text(text)


def _normalize_backend_name(value: Any) -> str:
    cleaned = _compact_text(value).lower()
    return PDF_BACKEND_ALIASES.get(cleaned, cleaned)


def _text_chunks(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        chunks: list[str] = []
        for item in value:
            chunks.extend(_text_chunks(item))
        return chunks
    normalized = _normalize_text(value)
    return [normalized] if normalized else []


def _extract_section_heading(value: Any) -> Optional[dict[str, str]]:
    candidate = _compact_text(value)
    if not candidate:
        return None
    match = SECTION_HEADING_LINE_PATTERN.match(candidate)
    if not match:
        return None
    key = _compact_text(match.group(1)).lower()
    return {
        "key": key,
        "heading": _compact_text(match.group(1)).title(),
        "section_type": SECTION_TYPE_MAP.get(key, UNKNOWN_SECTION_TYPE),
    }


def _printable_text_ratio(text: str) -> float:
    cleaned = str(text or "")
    if not cleaned:
        return 0.0
    printable = sum(1 for char in cleaned if char.isprintable() or char in "\n\t")
    return printable / max(1, len(cleaned))


def _repeated_line_ratio(text: str) -> float:
    lines = [_compact_text(line) for line in str(text or "").splitlines() if _compact_text(line)]
    if not lines:
        return 0.0
    counts = Counter(lines)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / max(1, len(lines))


@dataclass
class ParserConfig:
    enabled: bool = True
    primary_backend: str = "mineru"
    secondary_backend: str = "pymupdf"
    mineru_backend: str = "pipeline"
    mineru_method: str = "auto"
    mineru_api_url: Optional[str] = None
    min_total_chars: int = 800
    min_chars_per_text_page: int = 80
    min_text_page_ratio: float = 0.5
    min_printable_ratio: float = 0.95
    snippet_target_chars: int = 1000
    snippet_overlap_chars: int = 120
    preserve_page_blocks: bool = True

    @classmethod
    def from_dict(cls, payload: Optional[dict[str, Any]]) -> "ParserConfig":
        raw = dict(payload or {})
        config = cls(
            enabled=bool(raw.get("enabled", True)),
            primary_backend=_normalize_backend_name(raw.get("primary_backend") or "mineru"),
            secondary_backend=_normalize_backend_name(raw.get("secondary_backend") or "pymupdf"),
            mineru_backend=_compact_text(raw.get("mineru_backend") or "pipeline").lower() or "pipeline",
            mineru_method=_compact_text(raw.get("mineru_method") or "auto").lower() or "auto",
            mineru_api_url=_compact_text(raw.get("mineru_api_url") or _default_env_value("MINERU_API_URL")) or None,
            min_total_chars=max(0, int(raw.get("min_total_chars", 800) or 800)),
            min_chars_per_text_page=max(1, int(raw.get("min_chars_per_text_page", 80) or 80)),
            min_text_page_ratio=min(1.0, max(0.0, float(raw.get("min_text_page_ratio", 0.5) or 0.5))),
            min_printable_ratio=min(1.0, max(0.0, float(raw.get("min_printable_ratio", 0.95) or 0.95))),
            snippet_target_chars=max(200, int(raw.get("snippet_target_chars", 1000) or 1000)),
            snippet_overlap_chars=max(0, int(raw.get("snippet_overlap_chars", 120) or 120)),
            preserve_page_blocks=bool(raw.get("preserve_page_blocks", True)),
        )
        if config.snippet_overlap_chars >= config.snippet_target_chars:
            config.snippet_overlap_chars = max(0, config.snippet_target_chars // 4)
        return config


@dataclass
class ExtractedBlock:
    page_no: int
    text: str


@dataclass
class ExtractedSection:
    heading: str
    section_type: str
    text: str
    page_start: int
    page_end: int
    fulltext_char_start: int = 0
    fulltext_char_end: int = 0


@dataclass
class ExtractionAttempt:
    extractor: str
    succeeded: bool
    fulltext: str = ""
    page_texts: list[str] = field(default_factory=list)
    blocks: list[ExtractedBlock] = field(default_factory=list)
    sections: list[ExtractedSection] = field(default_factory=list)
    page_count: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    usable: bool = False
    failure_reason: Optional[str] = None
    ocr_applied: bool = False


class PaperParseEngine:
    def __init__(self, *, config: Optional[ParserConfig | dict[str, Any]] = None) -> None:
        if isinstance(config, ParserConfig):
            self.config = config
        else:
            self.config = ParserConfig.from_dict(config)

    def process_document(self, *, input_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        path = Path(input_path)
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        document_id = path.stem or "document"
        if path.suffix.lower() == ".pdf":
            result = self.process_pdf_bytes(document_id=document_id, pdf_bytes=path.read_bytes(), output_dir=destination)
        else:
            text = path.read_text(encoding="utf-8")
            result = self.process_text(document_id=document_id, text=text, output_dir=destination, source_extension=path.suffix.lower())
        (destination / "parse_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def process_text(
        self,
        *,
        document_id: str,
        text: str,
        output_dir: str | Path,
        source_extension: str = ".txt",
    ) -> dict[str, Any]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        source_path = destination / f"{document_id}{source_extension or '.txt'}"
        source_path.write_text(text, encoding="utf-8")
        normalized = _normalize_text(text)
        sections = self._apply_section_offsets(self._sections_from_fulltext(fulltext=normalized, page_spans=[]))
        snippets = self._build_snippets(normalized)
        fulltext_path = destination / f"{document_id}.fulltext.txt"
        sections_path = destination / f"{document_id}.sections.json"
        snippets_path = destination / f"{document_id}.snippets.json"
        report_path = destination / f"{document_id}.extraction_report.json"
        fulltext_path.write_text(normalized, encoding="utf-8")
        sections_payload = [self._section_payload(section) for section in sections]
        sections_path.write_text(json.dumps(sections_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        snippets_path.write_text(json.dumps(snippets, ensure_ascii=False, indent=2), encoding="utf-8")
        report = {
            "document_id": document_id,
            "status": "fulltext_indexed",
            "selected_extractor": "text",
            "attempts": [],
            "warnings": [],
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "document_id": document_id,
            "fulltext_status": "fulltext_indexed",
            "source_artifact_path": str(source_path),
            "fulltext_artifact_path": str(fulltext_path),
            "sections_artifact_path": str(sections_path),
            "snippets_artifact_path": str(snippets_path),
            "extraction_report_path": str(report_path),
            "sections": sections_payload,
            "warnings": [],
            "extractor": "text",
            "ocr_applied": False,
            "report": report,
        }

    def process_pdf_bytes(
        self,
        *,
        document_id: str,
        pdf_bytes: bytes,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        source_path = destination / f"{document_id}.pdf"
        source_path.write_bytes(pdf_bytes)
        warnings: list[str] = []
        attempts: list[ExtractionAttempt] = []

        if not self.config.enabled:
            report = {
                "document_id": document_id,
                "status": "binary_only",
                "selected_extractor": None,
                "attempts": [],
                "warnings": [],
            }
            report_path = destination / f"{document_id}.extraction_report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return {
                "document_id": document_id,
                "fulltext_status": "binary_only",
                "source_artifact_path": str(source_path),
                "fulltext_artifact_path": str(source_path),
                "sections_artifact_path": None,
                "snippets_artifact_path": None,
                "extraction_report_path": str(report_path),
                "sections": [],
                "warnings": [],
                "extractor": None,
                "ocr_applied": False,
                "report": report,
            }

        if not self._is_true_pdf(pdf_bytes):
            warning = f"{document_id}: invalid PDF header; parsing skipped."
            warnings.append(warning)
            report = {
                "document_id": document_id,
                "status": "fulltext_unusable",
                "selected_extractor": None,
                "attempts": [],
                "warnings": warnings,
                "failure_reason": "invalid_pdf_header",
            }
            report_path = destination / f"{document_id}.extraction_report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return {
                "document_id": document_id,
                "fulltext_status": "fulltext_unusable",
                "source_artifact_path": str(source_path),
                "fulltext_artifact_path": str(source_path),
                "sections_artifact_path": None,
                "snippets_artifact_path": None,
                "extraction_report_path": str(report_path),
                "sections": [],
                "warnings": warnings,
                "extractor": None,
                "ocr_applied": False,
                "report": report,
            }

        warnings.extend(self._pdf_backend_config_warnings(document_id))
        configured_backends = self._configured_pdf_backends()
        if not configured_backends:
            report = {
                "document_id": document_id,
                "status": "fulltext_unusable",
                "selected_extractor": None,
                "attempts": [],
                "warnings": warnings,
                "failure_reason": "no_supported_pdf_backend_configured",
            }
            report_path = destination / f"{document_id}.extraction_report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return {
                "document_id": document_id,
                "fulltext_status": "fulltext_unusable",
                "source_artifact_path": str(source_path),
                "fulltext_artifact_path": str(source_path),
                "sections_artifact_path": None,
                "snippets_artifact_path": None,
                "extraction_report_path": str(report_path),
                "sections": [],
                "warnings": warnings,
                "extractor": None,
                "ocr_applied": False,
                "report": report,
            }

        for backend in configured_backends:
            attempt = self._extract_with_backend(backend=backend, source_path=source_path, pdf_bytes=pdf_bytes)
            attempts.append(attempt)
            if attempt.usable:
                return self._finalize_success(
                    document_id=document_id,
                    source_path=source_path,
                    destination=destination,
                    attempt=attempt,
                    attempts=attempts,
                    warnings=warnings,
                )

        for attempt in attempts:
            attempt_warning = self._attempt_warning(document_id=document_id, attempt=attempt)
            if attempt_warning:
                warnings.append(attempt_warning)
        report = {
            "document_id": document_id,
            "status": "fulltext_unusable",
            "selected_extractor": None,
            "attempts": [self._attempt_payload(item) for item in attempts],
            "warnings": warnings,
        }
        report_path = destination / f"{document_id}.extraction_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "document_id": document_id,
            "fulltext_status": "fulltext_unusable",
            "source_artifact_path": str(source_path),
            "fulltext_artifact_path": str(source_path),
            "sections_artifact_path": None,
            "snippets_artifact_path": None,
            "extraction_report_path": str(report_path),
            "sections": [],
            "warnings": warnings,
            "extractor": None,
            "ocr_applied": False,
            "report": report,
        }

    def _extract_with_backend(self, *, backend: str, source_path: Path, pdf_bytes: bytes) -> ExtractionAttempt:
        if backend == "mineru":
            return self._extract_with_mineru(source_path)
        if backend == "pymupdf":
            return self._extract_with_pymupdf(pdf_bytes)
        return ExtractionAttempt(extractor=backend, succeeded=False, failure_reason=f"unsupported backend: {backend}")

    def _finalize_success(
        self,
        *,
        document_id: str,
        source_path: Path,
        destination: Path,
        attempt: ExtractionAttempt,
        attempts: list[ExtractionAttempt],
        warnings: list[str],
    ) -> dict[str, Any]:
        sections = list(attempt.sections or [])
        if not sections:
            sections = self._apply_section_offsets(self._sections_from_fulltext(fulltext=attempt.fulltext, page_spans=[]))
        fulltext = attempt.fulltext or self._fulltext_from_sections(sections)
        fulltext_path = destination / f"{document_id}.fulltext.txt"
        sections_path = destination / f"{document_id}.sections.json"
        snippets_path = destination / f"{document_id}.snippets.json"
        report_path = destination / f"{document_id}.extraction_report.json"
        fulltext_path.write_text(fulltext, encoding="utf-8")
        sections_payload = [self._section_payload(section) for section in sections]
        snippets_payload = self._build_snippets(fulltext)
        sections_path.write_text(json.dumps(sections_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        snippets_path.write_text(json.dumps(snippets_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        report = {
            "document_id": document_id,
            "status": "fulltext_indexed",
            "selected_extractor": attempt.extractor,
            "attempts": [self._attempt_payload(item) for item in attempts],
            "warnings": warnings,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "document_id": document_id,
            "fulltext_status": "fulltext_indexed",
            "source_artifact_path": str(source_path),
            "fulltext_artifact_path": str(fulltext_path),
            "sections_artifact_path": str(sections_path),
            "snippets_artifact_path": str(snippets_path),
            "extraction_report_path": str(report_path),
            "sections": sections_payload,
            "warnings": warnings,
            "extractor": attempt.extractor,
            "ocr_applied": bool(attempt.ocr_applied),
            "report": report,
        }

    def _extract_with_mineru(self, pdf_path: Path) -> ExtractionAttempt:
        executable = shutil.which("mineru")
        if not executable:
            return ExtractionAttempt(extractor="mineru", succeeded=False, failure_reason="mineru CLI unavailable on PATH")
        with tempfile.TemporaryDirectory(prefix="paper-parse-mineru-") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            command = [
                executable,
                "-p",
                str(pdf_path),
                "-o",
                str(temp_dir),
                "-b",
                self.config.mineru_backend,
                "-m",
                self.config.mineru_method,
            ]
            if self.config.mineru_api_url:
                command.extend(["--api-url", self.config.mineru_api_url])
            run_kwargs: dict[str, Any] = {
                "capture_output": True,
                "text": True,
                "timeout": MINERU_TIMEOUT_SECONDS,
                "check": False,
            }
            subprocess_env = self._mineru_subprocess_env()
            if subprocess_env is not None:
                run_kwargs["env"] = subprocess_env
            try:
                completed = subprocess.run(command, **run_kwargs)
            except subprocess.TimeoutExpired:
                return ExtractionAttempt(
                    extractor="mineru",
                    succeeded=False,
                    failure_reason=f"mineru timed out after {MINERU_TIMEOUT_SECONDS}s",
                )
            except Exception as exc:
                return ExtractionAttempt(extractor="mineru", succeeded=False, failure_reason=str(exc))
            if completed.returncode != 0:
                detail = _compact_text(completed.stderr or completed.stdout) or f"exit code {completed.returncode}"
                return ExtractionAttempt(extractor="mineru", succeeded=False, failure_reason=detail)

            markdown_path = self._find_mineru_markdown(temp_dir=temp_dir, document_stem=pdf_path.stem)
            if markdown_path is None:
                return ExtractionAttempt(extractor="mineru", succeeded=False, failure_reason="markdown output not found")
            content_list_path = self._find_mineru_content_list(temp_dir=temp_dir, document_stem=pdf_path.stem)
            if content_list_path is None:
                return ExtractionAttempt(extractor="mineru", succeeded=False, failure_reason="content_list.json output not found")

            try:
                markdown_text = _normalize_text(markdown_path.read_text(encoding="utf-8"))
            except Exception as exc:
                return ExtractionAttempt(extractor="mineru", succeeded=False, failure_reason=f"failed to read markdown output: {exc}")
            if not markdown_text:
                return ExtractionAttempt(extractor="mineru", succeeded=False, failure_reason="markdown output was empty")

            try:
                page_texts, blocks, heading_page_hints, page_count = self._load_mineru_content_list(content_list_path)
            except Exception as exc:
                return ExtractionAttempt(extractor="mineru", succeeded=False, failure_reason=f"failed to parse content_list.json: {exc}")

        sections = self._apply_section_offsets(
            self._sections_from_fulltext(
                fulltext=markdown_text,
                page_spans=[],
                heading_page_hints=heading_page_hints,
                fallback_page_count=max(1, page_count),
            )
        )
        metrics = self._evaluate_quality(fulltext=markdown_text, page_texts=page_texts)
        return ExtractionAttempt(
            extractor="mineru",
            succeeded=True,
            fulltext=markdown_text,
            page_texts=page_texts,
            blocks=blocks,
            sections=sections,
            page_count=max(page_count, len(page_texts)),
            metrics=metrics,
            usable=not metrics["reasons"],
        )

    def _extract_with_pymupdf(self, pdf_bytes: bytes) -> ExtractionAttempt:
        try:
            import pymupdf as fitz
        except Exception:
            try:
                import fitz  # type: ignore[no-redef]
            except Exception as exc:
                return ExtractionAttempt(extractor="pymupdf", succeeded=False, failure_reason=f"pymupdf unavailable: {exc}")
        try:
            document = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            return ExtractionAttempt(extractor="pymupdf", succeeded=False, failure_reason=str(exc))
        page_texts: list[str] = []
        blocks: list[ExtractedBlock] = []
        try:
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                page_blocks: list[str] = []
                for block in page.get_text("blocks", sort=True):
                    if len(block) < 5:
                        continue
                    text = _normalize_text(block[4])
                    if not text:
                        continue
                    page_blocks.append(text)
                    if self.config.preserve_page_blocks:
                        blocks.append(ExtractedBlock(page_no=page_index + 1, text=text))
                page_text = "\n\n".join(page_blocks)
                if not page_text:
                    page_text = _normalize_text(page.get_text("text", sort=True))
                page_texts.append(page_text)
        finally:
            document.close()
        fulltext, page_spans = self._join_page_texts(page_texts)
        sections = self._apply_section_offsets(self._sections_from_fulltext(fulltext=fulltext, page_spans=page_spans))
        metrics = self._evaluate_quality(fulltext=fulltext, page_texts=page_texts)
        return ExtractionAttempt(
            extractor="pymupdf",
            succeeded=True,
            fulltext=fulltext,
            page_texts=page_texts,
            blocks=blocks,
            sections=sections,
            page_count=len(page_texts),
            metrics=metrics,
            usable=not metrics["reasons"],
        )

    def _join_page_texts(self, page_texts: list[str]) -> tuple[str, list[dict[str, int]]]:
        fulltext_parts: list[str] = []
        page_spans: list[dict[str, int]] = []
        cursor = 0
        for index, page_text in enumerate(page_texts):
            normalized = _normalize_text(page_text)
            if fulltext_parts:
                fulltext_parts.append("\n\n")
                cursor += 2
            start = cursor
            fulltext_parts.append(normalized)
            cursor += len(normalized)
            page_spans.append({"page_no": index + 1, "start": start, "end": cursor})
        return "".join(fulltext_parts), page_spans

    def _sections_from_fulltext(
        self,
        *,
        fulltext: str,
        page_spans: list[dict[str, int]],
        heading_page_hints: Optional[list[dict[str, int]]] = None,
        fallback_page_count: int = 1,
    ) -> list[ExtractedSection]:
        normalized = _normalize_text(fulltext)
        if not normalized:
            return []
        matches = list(SECTION_HEADING_PATTERN.finditer(normalized))
        if not matches:
            sections = [
                ExtractedSection(
                    heading=UNKNOWN_SECTION_TITLE,
                    section_type=UNKNOWN_SECTION_TYPE,
                    text=normalized,
                    page_start=self._page_for_offset(0, page_spans),
                    page_end=self._page_for_offset(len(normalized), page_spans) if page_spans else max(1, fallback_page_count),
                )
            ]
            return self._apply_heading_page_hints(sections, heading_page_hints or [], fallback_page_count)
        sections: list[ExtractedSection] = []
        if matches[0].start() > 0:
            prefix = normalized[: matches[0].start()].strip()
            if prefix:
                sections.append(
                    ExtractedSection(
                        heading=UNKNOWN_SECTION_TITLE,
                        section_type=UNKNOWN_SECTION_TYPE,
                        text=prefix,
                        page_start=self._page_for_offset(0, page_spans),
                        page_end=self._page_for_offset(matches[0].start(), page_spans) if page_spans else 1,
                    )
                )
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
            heading = _compact_text(match.group(1)).title()
            text = normalized[start:end].strip()
            if not text:
                continue
            key = _compact_text(match.group(1)).lower()
            sections.append(
                ExtractedSection(
                    heading=heading,
                    section_type=SECTION_TYPE_MAP.get(key, UNKNOWN_SECTION_TYPE),
                    text=text,
                    page_start=self._page_for_offset(match.start(), page_spans),
                    page_end=self._page_for_offset(end, page_spans) if page_spans else max(1, fallback_page_count),
                )
            )
        return self._apply_heading_page_hints(sections, heading_page_hints or [], fallback_page_count)

    def _page_for_offset(self, offset: int, page_spans: list[dict[str, int]]) -> int:
        if not page_spans:
            return 1
        for page_span in page_spans:
            if int(page_span["start"]) <= offset <= int(page_span["end"]):
                return int(page_span["page_no"])
        return int(page_spans[-1]["page_no"])

    def _apply_section_offsets(self, sections: list[ExtractedSection]) -> list[ExtractedSection]:
        cursor = 0
        applied: list[ExtractedSection] = []
        for section in sections:
            text = _normalize_text(section.text)
            start = cursor
            end = start + len(text)
            applied.append(
                ExtractedSection(
                    heading=section.heading,
                    section_type=section.section_type,
                    text=text,
                    page_start=max(1, int(section.page_start or 1)),
                    page_end=max(int(section.page_start or 1), int(section.page_end or section.page_start or 1)),
                    fulltext_char_start=start,
                    fulltext_char_end=end,
                )
            )
            cursor = end + 2
        return applied

    def _apply_heading_page_hints(
        self,
        sections: list[ExtractedSection],
        heading_page_hints: list[dict[str, int]],
        page_count: int,
    ) -> list[ExtractedSection]:
        if not sections:
            return []
        if not heading_page_hints:
            return sections
        remaining_hints = list(heading_page_hints)
        hinted_sections: list[ExtractedSection] = []
        for section in sections:
            page_start = max(1, int(section.page_start or 1))
            heading_key = _compact_text(section.heading).lower()
            if heading_key and heading_key != UNKNOWN_SECTION_TITLE.lower():
                for index, hint in enumerate(remaining_hints):
                    if hint["heading_key"] == heading_key:
                        page_start = max(1, int(hint["page_no"]))
                        remaining_hints = remaining_hints[index + 1 :]
                        break
            hinted_sections.append(
                ExtractedSection(
                    heading=section.heading,
                    section_type=section.section_type,
                    text=section.text,
                    page_start=page_start,
                    page_end=page_start,
                )
            )
        resolved_page_count = max(page_count, max((section.page_start for section in hinted_sections), default=1))
        resolved_sections: list[ExtractedSection] = []
        for index, section in enumerate(hinted_sections):
            next_start = hinted_sections[index + 1].page_start if index + 1 < len(hinted_sections) else resolved_page_count
            resolved_sections.append(
                ExtractedSection(
                    heading=section.heading,
                    section_type=section.section_type,
                    text=section.text,
                    page_start=section.page_start,
                    page_end=max(section.page_start, next_start),
                )
            )
        return resolved_sections

    def _fulltext_from_sections(self, sections: list[ExtractedSection]) -> str:
        return "\n\n".join(section.text for section in sections if _normalize_text(section.text))

    def _build_snippets(self, fulltext: str) -> list[dict[str, Any]]:
        text = _normalize_text(fulltext)
        if not text:
            return []
        snippets: list[dict[str, Any]] = []
        step = max(1, self.config.snippet_target_chars - self.config.snippet_overlap_chars)
        start = 0
        snippet_index = 1
        while start < len(text):
            end = min(len(text), start + self.config.snippet_target_chars)
            snippet_text = text[start:end].strip()
            if snippet_text:
                snippets.append(
                    {
                        "snippet_id": f"snippet-{snippet_index}",
                        "char_start": start,
                        "char_end": end,
                        "text": snippet_text,
                    }
                )
                snippet_index += 1
            if end >= len(text):
                break
            start += step
        return snippets

    def _section_payload(self, section: ExtractedSection) -> dict[str, Any]:
        return {
            "section_id": f"section-{section.fulltext_char_start}-{section.fulltext_char_end}",
            "section_type": section.section_type,
            "heading": section.heading,
            "page_start": section.page_start,
            "page_end": max(section.page_start, section.page_end),
            "fulltext_char_start": section.fulltext_char_start,
            "fulltext_char_end": section.fulltext_char_end,
            "text": section.text,
        }

    def _evaluate_quality(self, *, fulltext: str, page_texts: list[str]) -> dict[str, Any]:
        total_chars = len(_compact_text(fulltext))
        page_count = len(page_texts)
        text_pages = [page for page in page_texts if len(_compact_text(page)) >= self.config.min_chars_per_text_page]
        text_page_ratio = len(text_pages) / max(1, page_count)
        printable_ratio = _printable_text_ratio(fulltext)
        repeated_ratio = _repeated_line_ratio(fulltext)
        reasons: list[str] = []
        if total_chars < self.config.min_total_chars:
            reasons.append("total_chars_below_threshold")
        if text_page_ratio < self.config.min_text_page_ratio:
            reasons.append("text_page_ratio_below_threshold")
        if printable_ratio < self.config.min_printable_ratio:
            reasons.append("printable_ratio_below_threshold")
        if repeated_ratio > 0.35:
            reasons.append("repeated_line_ratio_above_threshold")
        return {
            "total_chars": total_chars,
            "page_count": page_count,
            "text_page_ratio": round(text_page_ratio, 4),
            "printable_ratio": round(printable_ratio, 4),
            "repeated_line_ratio": round(repeated_ratio, 4),
            "reasons": reasons,
        }

    def _attempt_payload(self, attempt: ExtractionAttempt) -> dict[str, Any]:
        return {
            "extractor": attempt.extractor,
            "succeeded": attempt.succeeded,
            "usable": attempt.usable,
            "failure_reason": attempt.failure_reason,
            "metrics": dict(attempt.metrics or {}),
            "page_count": attempt.page_count,
        }

    def _is_true_pdf(self, content: bytes) -> bool:
        return bytes(content[:5]) == b"%PDF-"

    def _configured_pdf_backends(self) -> list[str]:
        ordered = [self.config.primary_backend, self.config.secondary_backend]
        backends: list[str] = []
        for backend in ordered:
            normalized = _normalize_backend_name(backend)
            if not normalized or normalized in backends:
                continue
            if normalized in SUPPORTED_PDF_BACKENDS:
                backends.append(normalized)
        return backends

    def _pdf_backend_config_warnings(self, document_id: str) -> list[str]:
        warnings: list[str] = []
        seen: set[str] = set()
        for backend in [self.config.primary_backend, self.config.secondary_backend]:
            normalized = _normalize_backend_name(backend)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            if normalized not in SUPPORTED_PDF_BACKENDS:
                warnings.append(
                    f"{document_id}: unsupported PDF backend configured: {normalized}; supported backends are mineru, pymupdf."
                )
        return warnings

    def _attempt_warning(self, *, document_id: str, attempt: ExtractionAttempt) -> Optional[str]:
        backend_label = PDF_BACKEND_DISPLAY_NAMES.get(attempt.extractor, attempt.extractor)
        if attempt.failure_reason:
            return f"{document_id}: {backend_label} extraction failed: {attempt.failure_reason}"
        if attempt.metrics.get("reasons"):
            return (
                f"{document_id}: {backend_label} extraction was rejected by quality gates: "
                + ", ".join(attempt.metrics.get("reasons", []))
                + "."
            )
        return None

    def _mineru_subprocess_env(self) -> Optional[dict[str, str]]:
        if not self.config.mineru_api_url:
            return None
        hostname = _compact_text(urlsplit(self.config.mineru_api_url).hostname).lower()
        if not hostname:
            return None
        if hostname != "localhost":
            try:
                if not ipaddress.ip_address(hostname).is_loopback:
                    return None
            except ValueError:
                return None
        env = dict(os.environ)
        no_proxy_hosts = [hostname, "localhost", "127.0.0.1", "::1"]

        def _merged_no_proxy(value: Any) -> str:
            entries = [item.strip() for item in str(value or "").split(",") if item.strip()]
            for host in no_proxy_hosts:
                if host not in entries:
                    entries.append(host)
            return ",".join(entries)

        env["NO_PROXY"] = _merged_no_proxy(env.get("NO_PROXY"))
        env["no_proxy"] = _merged_no_proxy(env.get("no_proxy"))
        return env

    def _find_mineru_markdown(self, *, temp_dir: Path, document_stem: str) -> Optional[Path]:
        candidates = [path for path in temp_dir.rglob("*.md") if path.is_file()]
        if not candidates:
            return None
        exact_name = [path for path in candidates if path.stem == document_stem]
        selected = exact_name or candidates
        return max(selected, key=lambda path: (path.stat().st_size, -len(path.parts), str(path)))

    def _find_mineru_content_list(self, *, temp_dir: Path, document_stem: str) -> Optional[Path]:
        candidates = [
            path
            for path in temp_dir.rglob("*.json")
            if path.is_file() and path.name in {"content_list.json", f"{document_stem}_content_list.json"}
        ]
        if not candidates:
            return None
        exact_name = [path for path in candidates if path.name == f"{document_stem}_content_list.json"]
        selected = exact_name or candidates
        return max(selected, key=lambda path: (path.stat().st_size, -len(path.parts), str(path)))

    def _load_mineru_content_list(
        self,
        content_list_path: Path,
    ) -> tuple[list[str], list[ExtractedBlock], list[dict[str, int]], int]:
        try:
            payload = json.loads(content_list_path.read_text(encoding="utf-8"))
        except Exception:
            payload = json.loads(content_list_path.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(payload, list):
            raise ValueError("content_list.json must be a JSON array")
        page_buckets: dict[int, list[str]] = {}
        blocks: list[ExtractedBlock] = []
        heading_page_hints: list[dict[str, int]] = []
        max_page_index = -1
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                page_index = max(0, int(item.get("page_idx", 0) or 0))
            except Exception:
                page_index = 0
            max_page_index = max(max_page_index, page_index)
            text = self._mineru_item_text(item)
            if text:
                page_buckets.setdefault(page_index, []).append(text)
                if self.config.preserve_page_blocks:
                    blocks.append(ExtractedBlock(page_no=page_index + 1, text=text))
            heading_info = _extract_section_heading(item.get("text"))
            if heading_info:
                heading_page_hints.append({"heading_key": heading_info["key"], "page_no": page_index + 1})
        page_count = max_page_index + 1 if max_page_index >= 0 else 0
        page_texts = ["\n\n".join(page_buckets.get(index, [])) for index in range(page_count)]
        return page_texts, blocks, heading_page_hints, page_count

    def _mineru_item_text(self, item: dict[str, Any]) -> str:
        item_type = _compact_text(item.get("type")).lower()
        if item_type in MINERU_PAGE_AUXILIARY_TYPES:
            return ""
        text_parts: list[str] = []
        if item_type == "text":
            text_parts.extend(_text_chunks(item.get("text")))
        elif item_type == "equation":
            text_parts.extend(_text_chunks(item.get("text")))
        elif item_type in {"image", "chart"}:
            text_parts.extend(_text_chunks(item.get("image_caption")))
            text_parts.extend(_text_chunks(item.get("image_footnote")))
        elif item_type == "table":
            text_parts.extend(_text_chunks(item.get("table_caption")))
            text_parts.extend(_text_chunks(_strip_html(item.get("table_body"))))
            text_parts.extend(_text_chunks(item.get("table_footnote")))
        elif item_type == "code":
            text_parts.extend(_text_chunks(item.get("code_caption")))
            text_parts.extend(_text_chunks(item.get("code_body")))
            text_parts.extend(_text_chunks(item.get("code_footnote")))
        elif item_type == "list":
            text_parts.extend(_text_chunks(item.get("list_items")))
        else:
            text_parts.extend(_text_chunks(item.get("text")))
            text_parts.extend(_text_chunks(item.get("content")))
        return "\n\n".join(chunk for chunk in text_parts if chunk)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portable paper parsing skill")
    parser.add_argument("--input", required=True, help="Local PDF/text path to parse")
    parser.add_argument("--output-dir", required=True, help="Directory for emitted artifacts")
    parser.add_argument("--config-json", default=None, help="Optional JSON object overriding parser config")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    config = ParserConfig.from_dict(json.loads(args.config_json) if args.config_json else None)
    engine = PaperParseEngine(config=config)
    result = engine.process_document(input_path=args.input, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
