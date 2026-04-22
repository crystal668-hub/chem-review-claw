# Paper Rerank Contract

## Request Shape

```json
{
  "question": "Which papers best support the HER claim?",
  "max_candidates": 3,
  "grobid": {
    "url": "http://localhost:8070"
  },
  "llm": {
    "base_url": "https://api.openai.com/v1",
    "api_key_env": "OPENAI_API_KEY",
    "model": "gpt-4.1-mini"
  },
  "candidates": [
    {
      "paper_id": "paper-1",
      "title": "Paper 1",
      "retrieval_score": 7.2,
      "pdf_path": "/tmp/paper-1.pdf"
    }
  ]
}
```

## Environment

- `GROBID_URL` is optional. If `grobid.url` is omitted from the request, the script uses `GROBID_URL`, then falls back to `http://localhost:8070`.
- `OPENAI_API_KEY` is optional only when `llm.api_key` is provided directly. Otherwise the script reads the variable named by `llm.api_key_env` and defaults that name to `OPENAI_API_KEY`.

## Output JSON

- Canonical file: `rerank_result.json`
- `locked_paper_ids`
- `dropped_paper_ids`
- `ranked_candidates`
- `paper_profiles`
- `screen_status`
- `failure_domain`

Per-candidate status fields:
- `ranked_candidates[].rerank_status` is `ready` or `skipped`
- `ranked_candidates[].profile_status` mirrors the profile stage result
- `paper_profiles[].profile_status` is one of `ready`, `missing_local_pdf`, `non_pdf_artifact`, or `profile_error`
- `screen_status` can be `ready`, `no_locks`, or `skipped`

## Failure Policy

- Missing readable PDF: structured candidate skip with `profile_status=missing_local_pdf`
- Non-PDF local artifact: structured candidate skip with `profile_status=non_pdf_artifact`
- GROBID/profile build failure: affected candidate becomes `profile_error`; if no candidates remain rerankable, the run returns `screen_status=skipped`
- LLM config missing or invalid: hard fail only when at least one candidate is rerankable
- Invalid or empty structured decisions: hard fail once LLM rerank is attempted
