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

## Local paper-processing service

The paper pipeline is `paper-retrieval` -> `paper-access` -> `paper-parse`.
`paper-parse` can use PyMuPDF locally and optionally a long-lived MinerU API
service at `http://127.0.0.1:8000`.

The service is referenced by the default environment variable in
`~/.openclaw/.env`:

- `MINERU_API_URL=http://127.0.0.1:8000`

### Native MinerU

On macOS, MinerU should run natively instead of through Docker. Install the CLI/runtime, pre-download models, then start the long-lived API:

```bash
cd ~/.openclaw/workspace
bash scripts/mineru_service.sh install
bash scripts/mineru_service.sh download-models
bash scripts/mineru_service.sh up
bash scripts/mineru_service.sh health
```

Common operations:

```bash
bash scripts/mineru_service.sh ps
bash scripts/mineru_service.sh logs
bash scripts/mineru_service.sh restart
bash scripts/mineru_service.sh down
```

Notes:

- The service binds to loopback only and is not exposed on the LAN.
- `paper-parse` reads `MINERU_API_URL` and passes it to the local `mineru` CLI.
- `mineru_service.sh up` defaults to `MINERU_MODEL_SOURCE=local`, so run `mineru_service.sh download-models` before the first service start.
- `mineru_service.sh download-models` defaults to `MINERU_DOWNLOAD_SOURCE=modelscope`; set `MINERU_DOWNLOAD_SOURCE=huggingface` if that source is preferred.
