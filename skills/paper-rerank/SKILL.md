---
name: paper-rerank
description: Use when an agent needs to build portable paper profiles from local PDFs and run listwise LLM reranking with explicit inputs, without relying on ChemQA workspace state or hidden carry-over data.
---

# Paper Rerank

## Overview

Build profile artifacts from local paper PDFs and ask an LLM to produce `lock` or `drop` rerank decisions. This skill is self-contained and requires explicit candidate inputs plus explicit external service configuration. Candidates without a readable local PDF are skipped structurally in `rerank_result.json` instead of crashing the whole batch.

For GROBID, the script uses this precedence order:
- `request_json.grobid.url`
- `GROBID_URL` from the process environment
- `http://localhost:8070`

## When to Use

Use this skill when:
- candidate papers are already downloaded locally and `paper-access` marked them as readable PDFs
- reranking should consider more than title or abstract metadata
- the caller can provide GROBID and LLM configuration explicitly

Do not use this skill for remote search, downloading, or fulltext parsing.

## Execution

```bash
python <skill-root>/scripts/paper_rerank.py \
  --request-json /path/to/request.json \
  --output-dir /tmp/paper-rerank-out
```

Read `references/contracts.md` for request fields and supported environment values.

The output directory always includes `rerank_result.json`. Mixed batches are allowed: ready PDFs are reranked, while skipped candidates remain in `ranked_candidates` with `rerank_status: skipped` and an explicit `profile_status`.
