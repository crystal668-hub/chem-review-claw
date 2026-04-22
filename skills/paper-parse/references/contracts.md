# Paper Parse Contract

## Input

- `--input`: local `.pdf` path or a UTF-8 text artifact such as `.txt`, `.md`, or `.html`
- `--output-dir`: directory for generated artifacts
- optional config fields can be supplied via `--config-json`
- supported PDF backend config keys: `primary_backend`, `secondary_backend`, `mineru_backend`, `mineru_method`, `mineru_api_url`
- stable default env key: `MINERU_API_URL` from process env or repo-root `.env`

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
- For PDF inputs, the primary backend defaults to `mineru`
- The secondary PDF backend defaults to `pymupdf`
- MinerU is invoked via local CLI in `pipeline` mode by default with `mineru_method=auto`
- If `mineru_api_url` is set, `paper-parse` forwards it as `mineru --api-url ...` so repeated runs can reuse a long-lived `mineru-api`
- If `mineru_api_url` is omitted, `paper-parse` falls back to `MINERU_API_URL` from the runtime environment or repo-root `.env`
- If MinerU is unusable or rejected by quality gates, PyMuPDF is attempted when configured
- Unsupported PDF backend names are ignored with a structured warning instead of crashing
- If both PDF backends are unavailable or fail, the script returns a structured `fulltext_unusable` result instead of failing at import time
- No repository-local imports or runtime state are required
