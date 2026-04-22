---
name: paper-parse
description: Use when an agent needs to parse a local paper PDF or text artifact into fulltext, sections, snippets, and an extraction report.
---

# Paper Parse

## Overview

Parse a local document into structured text artifacts. This skill is self-contained and assumes only a file path plus optional parser settings. The script always writes a canonical `parse_result.json` to the output directory.

The default PDF stack is:
- Primary parser: `MinerU`
- Fallback parser: `PyMuPDF`

Text-like inputs do not use the PDF backends. A local `.txt`, `.md`, or `.html` artifact can be parsed even when `mineru` and `pymupdf` are not installed.

## When to Use

Use this skill when:
- a paper has already been downloaded locally
- the next step needs clean fulltext or section boundaries
- an agent needs portable parsing behavior outside the current ChemQA runtime

Do not use this skill for:
- remote paper search
- OA resolution or HTTP downloading
- GROBID TEI/profile generation for reranking

## Execution

Run the parser script with a local input path and output directory:

```bash
python <skill-root>/scripts/paper_parse.py \
  --input /path/to/paper.pdf \
  --output-dir /tmp/paper-parse-out
```

The script writes JSON to stdout and stores artifacts in the output directory, including `parse_result.json`.

For PDF parsing, the preferred runtime is a locally installed `mineru` CLI on `PATH`. The parser invokes MinerU in local `pipeline` mode and falls back to `PyMuPDF` if MinerU is unavailable, fails, or is rejected by quality gates.

To reuse a long-lived MinerU service, pass `mineru_api_url` in `--config-json`. When set, `paper-parse` forwards `--api-url` to the `mineru` CLI instead of relying on a temporary local `mineru-api` process per run.

For a stable default, `paper-parse` also reads `MINERU_API_URL` from the process environment or the repo-root `.env` file.

## Inputs And Outputs

- Input: local `.pdf` path or a UTF-8 text artifact such as `.txt`, `.md`, or `.html`
- Output: normalized `fulltext`, `sections`, `snippets`, extraction report, warnings, and parser metadata

Read `references/contracts.md` for the JSON contract and failure semantics.

## Failure Modes

- Text artifacts parse without importing PDF-only modules
- Invalid PDF header returns structured `fulltext_unusable`
- MinerU extraction rejection or unavailability automatically triggers `PyMuPDF`
- If both PDF backends are unavailable or fail, the script returns structured `fulltext_unusable` with attempt metadata instead of crashing on import
