# Paper Access Contract

## Request Shape

```json
{
  "documents": [
    {
      "paper_id": "doi-10-1000-example",
      "doi": "10.1000/example",
      "open_access_pdf_url": "https://example.org/download/example.pdf",
      "oa_url": "https://example.org/paper.pdf"
    }
  ],
  "prefer_unpaywall": true,
  "probe_pdf_urls": true,
  "unpaywall_email": "name@example.org"
}
```

## URL Resolution Order

- `documents[].open_access_pdf_url`
- `documents[].oa_url`
- Unpaywall `best_oa_location.url_for_pdf`
- Unpaywall landing-page and generic URL fallbacks

## Output JSON

- Canonical file: `access_result.json`
- Top-level fields: `documents`, `warnings`

Each document includes:
- `paper_id`
- `doi`
- `source_url`
- `source_url_kind`
- `final_url`
- `artifact_path`
- `artifact_kind`
- `is_readable_local_pdf`
- `content_type`
- `redirect_count`
- `fulltext_status`
- `pdf_probe`

Common downstream fields:
- `source_url_kind` is one of `open_access_pdf_url`, `oa_url`, `unpaywall_pdf`, `unpaywall_landing_page`, or `unpaywall_url`
- `artifact_kind` is one of `pdf`, `text`, or `binary`
- `is_readable_local_pdf` is the safe gate for passing the artifact to `paper-rerank`
- `fulltext_status` is `binary_only` for downloaded binary artifacts and `text_only` for fetched text artifacts
