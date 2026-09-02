## Verifier-grounded OpenClaw usage

The integration follows the public `verifier-grounded-benchmark` API. Dataset
provisioning reads `track.prompts()`, OpenClaw acts as the external model
caller, and isolated scoring calls `track.evaluate_one({task_id, response})`.
No VGB compatibility CLI or parameter-translation wrapper is added.

Use the canonical project benchmark CLI directly. Preview one RDKit task
without calling a model:

```bash
cd ~/.openclaw/workspace
uv run python -m benchmarking.workflow.cli \
  --groups single_llm_skills_on \
  --datasets verifier_grounded_rdkit \
  --limit 1 \
  --print-selected-records
```

Run the same selection and skip optional post-run analysis:

```bash
uv run python -m benchmarking.workflow.cli \
  --groups single_llm_skills_on \
  --datasets verifier_grounded_rdkit \
  --limit 1 \
  --no-analysis
```

Select an exact package task ID:

```bash
uv run python -m benchmarking.workflow.cli \
  --groups single_llm_skills_on \
  --datasets verifier_grounded_xtb_xyz \
  --record-ids xtb_gap_window_001 \
  --no-analysis
```

Without `--exact-output-dir`, runs are classified under
`state/benchmark-runs/<formal|temporary>/<benchmark>/<model>/<run-id>`.

Use `single_llm_skills_off` for the skills-disabled condition, or pass both
single-LLM group IDs to compare them. Omit `--limit` and `--record-ids` to run
the complete selected dataset. The three dataset names are
`verifier_grounded_rdkit` (11 tasks), `verifier_grounded_xtb_xyz` (18 tasks),
and `verifier_grounded_property_calculation` (2 tasks).

The complete integration contract is documented in
`docs/superpowers/specs/2026-07-15-verifier-grounded-openclaw-single-llm-integration-usage-spec.md`.

## Local paper-processing

The paper pipeline is `paper-retrieval` -> `paper-access` -> `paper-parse`.
`paper-parse` uses the official asynchronous MinerU Agent API for small PDFs,
the Precision API for larger PDFs when `MINERU_API_TOKEN` is configured, and
PyMuPDF as the final local fallback. It does not require a local MinerU CLI,
model cache, or `mineru-api` process.

Optional endpoint overrides are configured with `MINERU_AGENT_API_URL` and
`MINERU_PRECISION_API_URL`; the Precision API token is read from
`MINERU_API_TOKEN` by default. See the [MinerU API documentation](https://mineru.net/apiManage/docs)
for account, quota, and API details.
