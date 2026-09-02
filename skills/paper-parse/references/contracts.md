# Paper Parse Contract

## Input

- `--input`: local `.pdf` path or a UTF-8 text artifact such as `.txt`, `.md`, or `.html`
- `--output-dir`: directory for generated artifacts
- optional config fields can be supplied via `--config-json`
- supported PDF backend config keys: `backend`, `agent_api_url`, `precision_api_url`, `precision_token_env`, `agent_timeout_seconds`, `precision_timeout_seconds`, `poll_interval_seconds`, `max_retries`, `language`, `enable_table`, `enable_formula`, `is_ocr`
- stable default env keys: `MINERU_AGENT_API_URL`, `MINERU_PRECISION_API_URL`, and optional `MINERU_API_TOKEN`

## Output JSON

- Canonical file: `parse_result.json`
- `document_id`
- `fulltext_status`
- `source_artifact_path`
- `fulltext_artifact_path`
- `sections_artifact_path`
- `snippets_artifact_path`
- `extraction_report_path`
- `sections`
- `warnings`
- `extractor`
- `ocr_applied`
- `report`

## Status Values

- `fulltext_indexed`
- `fulltext_unusable`
- `binary_only`

## Parser Policy

- Text inputs bypass PDF backends and can succeed without `mineru` or `pymupdf`
- For PDF inputs, `backend=auto` is the default
- Files at or below 10 MB and 20 pages try the MinerU Agent Lightweight API first
- Larger files try the MinerU Precision API only when its token is configured
- Both cloud paths use asynchronous submit/upload/poll/download flows
- If a cloud backend is unavailable, over limits, rate-limited, or rejected by quality gates, PyMuPDF is attempted
- A local MinerU CLI and local `mineru-api` process are not used
- Unsupported PDF backend names are ignored with a structured warning instead of crashing
- If all configured cloud backends and PyMuPDF are unavailable or fail, the script returns a structured `fulltext_unusable` result instead of failing at import time
- No repository-local imports or runtime state are required
