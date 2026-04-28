# GLOBAL DEV SPEC

## 1. Project Overview
- Purpose
  - `.openclaw/` is a local OpenClaw runtime home that also contains a Python workspace for chemistry benchmark orchestration, DebateClaw debate workflows, ChemQA-style review workflows, paper retrieval/access/parse/rerank utilities, and benchmark cleanup tooling.
  - The main executable source code lives under `workspace/`.
  - The repo root also stores live runtime state for OpenClaw and ClawTeam: agent configs, generated workspaces, SQLite state, logs, device/auth files, and task/session registries.
- Current capabilities (ONLY what works)
  - `DONE`: Run benchmark batches across four experiment groups: `chemqa_web_on`, `chemqa_web_off`, `single_llm_web_on`, `single_llm_web_off` via `workspace/benchmark_test.py`.
  - `DONE`: Load benchmark JSONL datasets into a normalized `BenchmarkRecord` model via `workspace/benchmarking/datasets.py`.
  - `DONE`: Score outputs with registered evaluators for ChemBench, FrontierScience Olympiad/Research, ConformaBench, SuperChem, and generic semantic matching via `workspace/benchmark_test.py` and `workspace/benchmarking/evaluation.py`.
  - `DONE`: Provision run-scoped OpenClaw configs and DebateClaw/ChemQA slot workspaces via `workspace/benchmarking/config_renderer.py`, `workspace/benchmarking/provisioning.py`, and `workspace/benchmark_test.py`.
  - `DONE`: Run a single-agent OpenClaw baseline by shelling out to `openclaw agent` via `workspace/benchmarking/runners/single_llm.py`.
  - `DONE`: Run a ChemQA multi-agent workflow by compiling/materializing a ChemQA launch, monitoring benchmark-visible run-status, consuming canonical Artifact Flow outputs, archiving outputs, and cleaning runtime leftovers via `workspace/benchmarking/runners/chemqa.py`.
  - `DONE`: Manage DebateClaw V1 runtime, slot provisioning, prompt/materialization, and launch commands via `workspace/skills/debateclaw-v1/scripts/*.py`.
  - `DONE`: Maintain live debate protocol state in SQLite and expose CLI commands for init/status/next-action/submit/advance via `workspace/skills/debateclaw-v1/scripts/debate_state.py`.
  - `DONE`: Drive ChemQA reviewer/proposer/coordinator loops on top of DebateClaw state via `workspace/skills/chemqa-review/scripts/chemqa_review_openclaw_driver.py`.
  - `DONE`: Recover stalled ChemQA runs, respawn dead workers, and repair invalid protocol state via `workspace/skills/chemqa-review/scripts/recover_run.py`.
  - `DONE`: Collect ChemQA protocol outputs through Artifact Flow into canonical terminal artifacts, `artifact_manifest.json`, and legacy-compatible `qa_result.json` via `workspace/skills/chemqa-review/scripts/chemqa_artifact_flow.py` and `collect_artifacts.py`.
  - `DONE`: Provide deterministic chemistry provider skills for local structure reasoning, name resolution, public compound lookup, and numeric chemistry calculations via `workspace/skills/rdkit`, `workspace/skills/opsin`, `workspace/skills/pubchem`, and `workspace/skills/chem-calculator`.
  - `DONE`: Retrieve literature candidates from OpenAlex, Semantic Scholar, and Crossref via `workspace/skills/paper-retrieval/scripts/paper_retrieval.py`.
  - `DONE`: Resolve accessible paper artifacts using direct OA URLs and optional Unpaywall lookup via `workspace/skills/paper-access/scripts/paper_access.py`.
  - `DONE`: Parse local PDF/text documents with MinerU or PyMuPDF fallback via `workspace/skills/paper-parse/scripts/paper_parse.py`.
  - `DONE`: Rerank papers by building GROBID profiles and calling an OpenAI-compatible chat-completions endpoint via `workspace/skills/paper-rerank/scripts/paper_rerank.py`.
  - `DONE`: Clean up benchmark processes, session files, run-scoped artifacts, and leases via `workspace/skills/benchmark-cleanroom/scripts/cleanup_benchmark_run.py`.
  - `DONE`: Manage local Docker-backed GROBID and MinerU services via `workspace/scripts/docker_services.sh`.
  - `PARTIAL`: Native workflow-package support exists for `chemqa-review@1`, but the package implementation is skeletal and is not the primary runtime path.
  - `NOT_IMPLEMENTED`: No actual web UI/server is implemented in the repo despite `web-ui` optional dependencies in `workspace/pyproject.toml`.

## 2. System Architecture
- Top-level repo roles
  - `workspace/`
    - Main Python package and scripts.
    - Contains benchmark orchestration, skill bundles, dataset prep scripts, tests, docs, and Docker helpers.
  - `agents/`
    - OpenClaw agent runtime directories with `agent/models.json` and `sessions/sessions.json`.
    - Used as live runtime/config state, not source modules.
  - `benchmark/workspaces/`
    - Generated benchmark slot workspaces for chemqa/baseline/judge runs.
    - Used by benchmark scripts as runtime workspace roots.
  - `debateclaw/workspaces/`
    - Generated DebateClaw slot workspaces for live debate runs.
  - `flows/`, `tasks/`, `memory/`
    - SQLite runtime stores.
  - `logs/`, `devices/`, `identity/`, `qqbot/`
    - Operational state and logs; not code modules.
  - `openclaw.json`
    - Base OpenClaw config used and rewritten into run-scoped configs by benchmark launchers.

- Source modules
  - `workspace/benchmarking/`
    - `contracts.py`
      - Defines `RunStatus`, `AnswerPayload`, `FailureInfo`, `RecoveryInfo`, `RunnerResult`.
    - `datasets.py`
      - Normalizes benchmark records from JSONL.
    - `evaluation.py`
      - Registry/dispatch for evaluator functions.
    - `experiments.py`
      - Defines `ExperimentSpec`.
    - `config_renderer.py`
      - Produces run-scoped OpenClaw configs, toggles web search, injects agent entries.
    - `provisioning.py`
      - Creates slot workspaces and `.debateclaw-slot.json` sentinels.
    - `reporting.py`
      - Defines the per-record benchmark result schema and aggregates per-record results into summary buckets.
    - `runners/`
      - `single_llm.py`: baseline single-agent runner.
      - `chemqa.py`: ChemQA launch/monitor/archive/cleanup runner.
  - `workspace/benchmark_test.py`
    - Main four-group benchmark CLI.
    - Also contains benchmark-specific evaluators, runtime bundle helpers, config pools, cleanup registration, and runner wiring.
  - `workspace/conformabench_judge.py`
    - RDKit-based hidden judge for constructive molecular answers.
  - `workspace/runtime_paths.py`
    - Central path resolution for repo, skills, benchmarks, runtime roots, and config files.

- Skill bundles under `workspace/skills/`
  - `debateclaw-v1/`
    - Installable DebateClaw runtime bundle.
    - Owns preset compilation/materialization, slot provisioning, launch helpers, runtime checks, model profiles, and live debate state CLI.
  - `chemqa-review/`
    - Installable ChemQA review protocol bundle layered on top of DebateClaw V1.
    - Owns ChemQA launch pipeline, driver loop, artifact reconstruction, liveness/recovery tooling, a minimal native workflow package, and prompt/runtime dependency wiring for sibling chemistry provider skills.
  - `rdkit/`, `pubchem/`, `opsin/`, `chem-calculator/`
    - First-batch chemistry provider bundles used for deterministic structure, nomenclature, compound lookup, and numeric subproblems.
  - `benchmark-cleanroom/`
    - Run-scoped cleanup manifests and lease management plus cleanup executor.
  - `paper-retrieval/`, `paper-access/`, `paper-parse/`, `paper-rerank/`
    - Standalone paper-processing pipeline stages.

- Dataset prep modules
  - `workspace/benchmarks/chembench/extract_open_ended_reasoning_pool.py`
  - `workspace/benchmarks/frontierscience/extract_chemistry_pool.py`
  - `workspace/benchmarks/superchem/extract_superchem_pool.py`
  - `workspace/benchmarks/conformabench/`
    - Contains prepared pool/manifests/tests, but no extractor script in this repo.

- Module interactions
  - `benchmark_test.py` -> `benchmarking/*`
    - Uses dataset loading, config rendering, provisioning, runner construction, and reporting.
  - `benchmark_test.py` -> `skills/chemqa-review`
    - Launches ChemQA preset flow, passes resolved `answer_kind`, polls benchmark-visible run status, prefers canonical Artifact Flow paths, archives outputs.
  - `benchmark_test.py` -> `skills/benchmark-cleanroom`
    - Writes cleanup manifests and runs cleanup hooks on exit/failure.
  - `chemqa_review_openclaw_driver.py` -> `debate_state.py`
    - Subprocess-driven control loop; asks for next action, submits artifacts, advances state.
  - `collect_artifacts.py` -> protocol YAML/JSON emitted by coordinator
    - Converts protocol state through `chemqa_artifact_flow.py` into `final_answer_artifact.json` or `failure_artifact.json`, `artifact_manifest.json`, `candidate_view.json`, diagnostics, and legacy-compatible `qa_result.json`.
  - `paper-rerank.py` -> `paper-access`/`paper-parse` outputs
    - Expects local PDFs and calls GROBID + OpenAI-compatible LLM endpoint.

## 3. Feature Matrix
- Name: Four-group benchmark batch runner
  - Description: Runs ChemQA and single-agent baselines across websearch on/off groups, wave-batches groups, saves per-record and aggregate outputs.
  - Input / Output:
    - Input: benchmark root or dataset files, group list, timeouts, config path, model/profile overrides.
    - Output: `results.json`, `results.partial.json`, `runtime-manifest.json`, `runtime-config/*.json`, `per-record/*/*.json`, CSV summaries.
    - Per-record JSON entries are on schema version `2` and include explicit evaluability axes such as run lifecycle status, protocol completion/acceptance status, answer availability/reliability, evaluable/scored flags, recovery mode, degraded execution, and execution error kind.
    - Aggregate summaries in `results.json` and CSV exports retain legacy score fields and also expose operational counters such as completed vs failed runs, protocol completion, evaluable/scored counts, recovered-evaluable counts, and degraded execution counts.
  - Implementation location: `workspace/benchmark_test.py`, `workspace/benchmarking/*`
  - Status: `DONE`

### Benchmark Result Status Axes
- `results.json` now carries top-level `schema_version = 2` and a `status_axes_description` block that documents the evaluability axes used by per-record entries.
- `run_lifecycle_status` reports whether the benchmark run finished operationally, while `protocol_completion_status` reports whether the ChemQA protocol itself completed, failed, or is missing.
- `answer_availability` and `answer_reliability` distinguish native final answers from recovered candidate answers, preview-only fallbacks, and missing answers.
- `evaluable` means the system preserved a trustworthy answer that should count for benchmark scoring. `scored` means the evaluator actually ran. `passed` remains the task-quality outcome inside `evaluation`.
- `pass_count` remains a legacy score summary and should not be treated as an operational stability metric. Operational reporting should use the explicit status/evaluability counters instead.

- Name: Benchmark record normalization
  - Description: Loads JSONL records, validates prompt/answer presence, derives grading config and subset labels.
  - Input / Output:
    - Input: benchmark JSONL files.
    - Output: `BenchmarkRecord` objects with `GradingSpec`.
  - Implementation location: `workspace/benchmarking/datasets.py`
  - Status: `DONE`

- Name: Evaluator registry and dispatch
  - Description: Maps `eval_kind` to evaluator function with `generic_semantic` fallback.
  - Input / Output:
    - Input: `BenchmarkRecord`, short/full answer text, judge object.
    - Output: evaluator payload/dataclass.
  - Implementation location: `workspace/benchmarking/evaluation.py`
  - Status: `DONE`

- Name: ChemBench open-ended scoring
  - Description: Scores numeric or text answers for ChemBench open-ended tasks.
  - Input / Output:
    - Input: `BenchmarkRecord`, model answer text.
    - Output: `EvaluationResult`.
  - Implementation location: `workspace/benchmark_test.py`
  - Status: `DONE`

- Name: FrontierScience Olympiad scoring
  - Description: Evaluates olympiad-style short answers.
  - Input / Output:
    - Input: record + answer text.
    - Output: `EvaluationResult`.
  - Implementation location: `workspace/benchmark_test.py`
  - Status: `DONE`

- Name: FrontierScience Research scoring
  - Description: Uses rubric parsing plus judge support for research track outputs.
  - Input / Output:
    - Input: record + answer text.
    - Output: `EvaluationResult`.
  - Implementation location: `workspace/benchmark_test.py`
  - Status: `DONE`

- Name: SuperChem multimodal scoring
  - Description: Extracts option answer/checkpoints and computes score/RPF-style metrics.
  - Input / Output:
    - Input: record + answer text.
    - Output: `EvaluationResult`.
  - Implementation location: `workspace/benchmark_test.py`
  - Status: `DONE`

- Name: ConformaBench hidden judge
  - Description: Parses SMILES, applies RDKit normalization/topology predicates, embeds conformers, optimizes force fields, checks geometric predicates.
  - Input / Output:
    - Input: final answer SMILES + hidden judge spec.
    - Output: detailed pass/fail payload.
  - Implementation location: `workspace/conformabench_judge.py`
  - Status: `DONE`

- Name: Run-scoped OpenClaw config rendering
  - Description: Copies base OpenClaw config, toggles web search/plugin state, injects judge/runner agent entries, strips `thinking` from managed agents.
  - Input / Output:
    - Input: base config payload, experiment spec, provisioned agents.
    - Output: modified config payload written under run output.
  - Implementation location: `workspace/benchmarking/config_renderer.py`, `workspace/benchmark_test.py`
  - Status: `DONE`

- Name: Slot workspace provisioning
  - Description: Creates workspaces with `AGENTS.md` and `.debateclaw-slot.json`.
  - Input / Output:
    - Input: workspace path, slot id, template text.
    - Output: initialized runtime workspace.
  - Implementation location: `workspace/benchmarking/provisioning.py`
  - Status: `DONE`

- Name: Single-agent OpenClaw baseline runner
  - Description: Builds prompt, shells out to `openclaw agent --local`, unwraps JSON payload, normalizes answer tracks.
  - Input / Output:
    - Input: benchmark record, group config, runtime bundle root.
    - Output: `RunnerResult`.
  - Implementation location: `workspace/benchmarking/runners/single_llm.py`
  - Status: `DONE`

- Name: ChemQA benchmark runner
  - Description: Launches ChemQA preset flow, derives an immutable benchmark answer kind, waits for benchmark-visible terminal run-status, triggers bounded recovery when run-status stops changing, prefers canonical Artifact Flow paths, archives artifacts, keeps legacy reconstruction/fallback for compatibility, marks evaluable recovered candidate submissions as scoreable degraded executions, and writes cleanup manifest.
  - Input / Output:
    - Input: benchmark record, ChemQA skill root, config path, slot set, profile/round overrides.
    - Output: `RunnerResult` plus archived artifact tree including canonical final/failure artifacts when available.
  - Implementation location: `workspace/benchmarking/runners/chemqa.py`
  - Status: `DONE`

- Name: DebateClaw preset compile/materialize/launch
  - Description: Compiles run plan from preset, materializes prompt bundles/command map/template, optionally prints or runs `clawteam launch`.
  - Input / Output:
    - Input: preset, goal, optional run/model/round overrides.
    - Output: compiled run-plan JSON, materialized runtime files, optional launch command/result.
  - Implementation location: `workspace/skills/debateclaw-v1/scripts/compile_runplan.py`, `materialize_runplan.py`, `launch_from_preset.py`, `launch_from_config.py`
  - Status: `DONE`

- Name: DebateClaw fixed-slot provisioning
  - Description: Ensures OpenClaw debate slots exist, injects provider/model config, writes command-map payload.
  - Input / Output:
    - Input: provider families, proposer count, env/config paths.
    - Output: slot workspaces, updated OpenClaw config, command map.
  - Implementation location: `workspace/skills/debateclaw-v1/scripts/ensure_openclaw_debate.py`
  - Status: `DONE`

- Name: Debate state machine CLI
  - Description: Stores debate state in SQLite, handles proposal/review/rebuttal submission, computes next action, advances phases/epochs, renders summaries.
  - Input / Output:
    - Input: CLI subcommands plus team/agent/file arguments.
    - Output: JSON/text protocol state and stored artifacts under ClawTeam data dir.
  - Implementation location: `workspace/skills/debateclaw-v1/scripts/debate_state.py`
  - Status: `DONE`

- Name: ChemQA coordinator/worker driver
  - Description: Runs long-lived coordinator/worker loops for each role, queries debate state, updates task status, saves sessions, writes cleanroom leases, and separates DebateClaw protocol terminal state from ChemQA benchmark terminal state while Artifact Flow finalizes outputs.
  - Input / Output:
    - Input: team, role, slot, session id, prompt/config/runtime paths.
    - Output: live task/session side effects and protocol artifacts.
  - Implementation location: `workspace/skills/chemqa-review/scripts/chemqa_review_openclaw_driver.py`
  - Status: `DONE`

- Name: ChemQA Artifact Flow
  - Description: Validates typed candidate/review/rebuttal artifacts, resolves answer-kind-specific projections, applies answer-revision rebuttals to a current candidate view, tracks review item closure state, writes canonical final/failure artifacts, writes an artifact manifest with hashes, and projects legacy-compatible `qa_result.json`.
  - Input / Output:
    - Input: resolved `answer_kind`, protocol payloads, candidate/review/rebuttal artifacts, finalization metadata.
    - Output: `final_answer_artifact.json` or `failure_artifact.json`, `candidate_view.json`, `validation_summary.json`, `artifact_manifest.json`, `qa_result.json`, and run-status overlay fields.
  - Implementation location: `workspace/skills/chemqa-review/scripts/chemqa_artifact_flow.py`, `workspace/skills/chemqa-review/scripts/collect_artifacts.py`
  - Status: `DONE`

- Name: ChemQA artifact reconstruction
  - Description: Validates `chemqa_review_protocol`, preserves legacy artifact files, and delegates terminal answer/failure projection to Artifact Flow.
  - Input / Output:
    - Input: protocol file/source directory.
    - Output: normalized artifact directory with legacy files plus canonical terminal artifacts and manifest.
  - Implementation location: `workspace/skills/chemqa-review/scripts/collect_artifacts.py`
  - Status: `DONE`

- Name: ChemQA run recovery
  - Description: Repairs invalid review state, respawns missing workers and a dead coordinator control loop, replays placeholder transport reviews, advances stalled runs. Respawn budget tracking is stored in `spawn_registry.json` and is updated while iterating a snapshot of role entries so missing process recovery can initialize budget metadata without aborting the recovery pass.
  - Input / Output:
    - Input: team id, workspace/runtime roots, max steps/respawn budget.
    - Output: JSON recovery summary plus runtime mutations.
  - Implementation location: `workspace/skills/chemqa-review/scripts/recover_run.py`
  - Status: `DONE`

- Name: ChemQA liveness check
  - Description: Fetches compact state snapshot and ClawTeam task list, reports missing roles and recommendation.
  - Input / Output:
    - Input: skill root, team, agent.
    - Output: JSON health payload.
  - Implementation location: `workspace/skills/chemqa-review/scripts/check_run_liveness.py`
  - Status: `DONE`

- Name: RDKit provider skill
  - Description: Runs deterministic local cheminformatics helpers for canonicalization, descriptors, substructure, stereochemistry, similarity, reactions, conformers, and symmetry heuristics.
  - Input / Output:
    - Input: request JSON plus output directory.
    - Output: stable `result.json` payloads with structured diagnostics/tool traces.
  - Implementation location: `workspace/skills/rdkit/*`
  - Status: `DONE`

- Name: PubChem provider skill
  - Description: Resolves names, CIDs, formulas, properties, synonyms, and similarity lookups through PubChem PUG REST.
  - Input / Output:
    - Input: request JSON plus output directory.
    - Output: stable `result.json` payloads with provider health, source traces, and structured diagnostics.
  - Implementation location: `workspace/skills/pubchem/*`
  - Status: `DONE`

- Name: OPSIN provider skill
  - Description: Resolves systematic chemical names to structures through OPSIN with structured diagnostics and optional RDKit validation handoff.
  - Input / Output:
    - Input: request JSON plus output directory.
    - Output: stable `result.json` payloads with parse diagnostics and optional validation status.
  - Implementation location: `workspace/skills/opsin/*`
  - Status: `DONE`

- Name: Chemistry calculator skill
  - Description: Runs deterministic local chemistry calculations for stoichiometry, concentration, equilibria, acid/base, gas-law, electrochemistry, units, and answer checks.
  - Input / Output:
    - Input: request JSON plus output directory.
    - Output: stable `result.json` payloads with structured calculation traces and diagnostics.
  - Implementation location: `workspace/skills/chem-calculator/*`
  - Status: `DONE`

- Name: Native ChemQA workflow package
  - Description: Declares workflow package class with hooks for initialize/next-action/submit/advance/status/summary/finalize.
  - Input / Output:
    - Input: run config/state/role/payload.
    - Output: updated state or action/status payload.
  - Implementation location: `workspace/skills/chemqa-review/runtime/workflow.py`, `workspace/skills/chemqa-review/workflows/chemqa-review@1.json`
  - Status: `PARTIAL`

- Name: Workflow package loader
  - Description: Loads a workflow package from module/path and validates required attributes/methods.
  - Input / Output:
    - Input: workflow package spec payload.
    - Output: instantiated workflow object.
  - Implementation location: `workspace/skills/debateclaw-v1/scripts/workflow_loader.py`
  - Status: `PARTIAL`

- Name: Paper retrieval
  - Description: Queries OpenAlex, Semantic Scholar, and Crossref; deduplicates candidates and scores them heuristically.
  - Input / Output:
    - Input: query, must/exclude terms, year range, preferred sources, limit.
    - Output: `papers`, provider diagnostics, provider health.
  - Implementation location: `workspace/skills/paper-retrieval/scripts/paper_retrieval.py`
  - Status: `DONE`

- Name: Paper access
  - Description: Resolves open-access source URL, optionally probes for PDF, downloads PDF/text/binary artifact, writes `access_result.json`.
  - Input / Output:
    - Input: request JSON with document candidates and optional Unpaywall email.
    - Output: localized artifacts and access metadata.
  - Implementation location: `workspace/skills/paper-access/scripts/paper_access.py`
  - Status: `DONE`

- Name: Paper parsing
  - Description: Parses PDF/text documents, chooses MinerU/PyMuPDF backends, extracts sections/blocks/snippets, writes `parse_result.json`.
  - Input / Output:
    - Input: local file path, output dir, optional parser config JSON.
    - Output: normalized parsed document artifacts.
  - Implementation location: `workspace/skills/paper-parse/scripts/paper_parse.py`
  - Status: `DONE`

- Name: Paper reranking
  - Description: Builds GROBID XML profiles for local PDFs, then calls an OpenAI-compatible chat-completions API to lock/drop candidates.
  - Input / Output:
    - Input: request JSON with candidate list, local PDFs, GROBID config, LLM config.
    - Output: `rerank_result.json` with decisions and profile status.
  - Implementation location: `workspace/skills/paper-rerank/scripts/paper_rerank.py`
  - Status: `DONE`

- Name: Benchmark cleanroom manifest and lease tracking
  - Description: Tracks per-run processes/session assignments/artifact roots and writes/removes lease files.
  - Input / Output:
    - Input: run metadata, role/slot/session identifiers.
    - Output: manifest JSON and lease JSON files.
  - Implementation location: `workspace/skills/benchmark-cleanroom/scripts/runtime_lease.py`
  - Status: `DONE`

- Name: Benchmark cleanup executor
  - Description: Terminates related processes, scrubs session stores, removes run-scoped artifacts, verifies no leftovers remain.
  - Input / Output:
    - Input: cleanup manifest or explicit run parameters.
    - Output: cleanup report JSON.
  - Implementation location: `workspace/skills/benchmark-cleanroom/scripts/cleanup_benchmark_run.py`
  - Status: `DONE`

- Name: Docker service control
  - Description: Starts/stops/checks GROBID and MinerU API compose projects.
  - Input / Output:
    - Input: subcommand such as `up`, `down`, `health`, `logs`.
    - Output: compose actions and health checks.
  - Implementation location: `workspace/scripts/docker_services.sh`
  - Status: `DONE`

- Name: ChemBench dataset extraction
  - Description: Pulls ChemBench rows from Hugging Face datasets-server and extracts open-ended reasoning tasks into a pool.
  - Input / Output:
    - Input: dataset name and output paths.
    - Output: JSONL pool + manifest.
  - Implementation location: `workspace/benchmarks/chembench/extract_open_ended_reasoning_pool.py`
  - Status: `DONE`

- Name: FrontierScience dataset extraction
  - Description: Merges olympiad and research JSONL inputs into a chemistry-only pool.
  - Input / Output:
    - Input: olympiad/research JSONL files.
    - Output: JSONL pool + manifest.
  - Implementation location: `workspace/benchmarks/frontierscience/extract_chemistry_pool.py`
  - Status: `DONE`

- Name: SuperChem dataset extraction
  - Description: Reads SUPERChem rows from datasets-server or zip/parquet fallback, localizes assets, emits a multimodal pool.
  - Input / Output:
    - Input: dataset name, output JSONL/assets paths.
    - Output: JSONL pool + manifest + assets.
  - Implementation location: `workspace/benchmarks/superchem/extract_superchem_pool.py`
  - Status: `DONE`

- Name: ConformaBench pool generation
  - Description: Prepared data/manifests/tests exist in-repo.
  - Input / Output:
    - Input: N/A in current repo.
    - Output: `workspace/benchmarks/conformabench/data/*`.
  - Implementation location: `workspace/benchmarks/conformabench/data/*`, `workspace/benchmarks/conformabench/tests/*`
  - Status: `PARTIAL`

- Name: Web UI / API server
  - Description: Optional dependencies suggest planned FastAPI/Gradio/OpenAI-based UI surfaces.
  - Input / Output:
    - Input: UNKNOWN
    - Output: UNKNOWN
  - Implementation location: `workspace/pyproject.toml` only
  - Status: `NOT_IMPLEMENTED`

## 4. Actual Behavior
- Primary execution flow: four-group benchmark
  - `workspace/benchmark_test.py` parses CLI args and discovers benchmark JSONL files under `workspace/benchmarks/*/data/*.jsonl` unless explicit files/datasets are provided.
  - It normalizes records through `benchmarking.datasets.load_records`.
  - It builds per-group run-scoped OpenClaw configs in `output_root/runtime-config/`.
  - For `single_llm_*` groups:
    - The runner shells out directly to `openclaw agent --local ... --json`.
    - It does not use a native Python OpenClaw API.
  - For `chemqa_*` groups:
    - The runner shells out to ChemQA skill scripts to compile/materialize/launch the run.
    - It monitors run status via files under `chemqa-review/control/run-status/`.
    - If run-status remains unchanged across polling intervals, it invokes `chemqa-review/scripts/recover_run.py` with the run-scoped `CLAWTEAM_DATA_DIR`; repeated recovery attempts are rate-limited while the status signature remains unchanged.
    - It treats DebateClaw `phase=done/status=done` as protocol terminal only while Artifact Flow is still `finalizing`; benchmark-visible `status=done/terminal_state=completed|failed` is published only after canonical final/failure artifacts, manifest, and `qa_result.json` are readable.
    - It prefers canonical `qa_result_path`, `final_answer_artifact_path`, `failure_artifact_path`, and `artifact_manifest_path` from run status. If artifacts are missing, it tries to rebuild them from protocol files with `collect_artifacts.py`.
    - If the final `qa_result.json` is still missing or unusable, it can fall back to the latest archived `proposer-1` proposal or `final_answer_preview`.
  - All per-record outputs are persisted immediately under `per-record/<group>/<slug>.json`.
  - Cleanup manifests are registered and benchmark-cleanroom cleanup runs in `finally`/signal/atexit paths.

- Real ChemQA control path
  - The operational state machine is `workspace/skills/debateclaw-v1/scripts/debate_state.py`, not `workspace/skills/chemqa-review/runtime/workflow.py`.
  - `chemqa_review_openclaw_driver.py` loops by repeatedly calling `debate_state.py` subcommands in subprocesses.
  - The driver updates ClawTeam task state, saves sessions, opens/removes cleanup leases, and emits role-specific artifacts.
  - When DebateClaw reports protocol terminal conditions, the driver publishes `artifact_flow_state=finalizing` while keeping legacy `status=running`; after `collect_artifacts.py` / Artifact Flow writes terminal artifacts, run status carries `artifact_flow_state=finalized|finalization_failed`, `benchmark_terminal_state`, canonical paths, and legacy-compatible terminal fields.
  - Rebuttal artifacts now carry explicit `mode`: `response_only`, `answer_revision`, or `concession`. Only `answer_revision` updates the Artifact Flow current candidate view.
  - `chemqa-review/scripts/bundle_common.py` and the prompt pack now treat `rdkit`, `pubchem`, `opsin`, and `chem-calculator` as required sibling skills alongside DebateClaw and the paper pipeline.
  - Prompt routing now tells `proposer-1` to prefer `chem-calculator` for `FrontierScience` numeric questions before web search, and to extract available SMILES/name text before routing `SuperChem` structure questions to `rdkit`, `opsin`, and `pubchem`.
  - Reviewer prompt contracts now require numeric and structural challenges to cite script `result.json` artifacts or structured `tool_trace` entries instead of unsupported tool-use claims.
  - This integration phase does not add a dedicated image-reading or OCSR skill to ChemQA prompt routing.
  - Recovery is externalized:
    - `recover_run.py` inspects the same runtime files and database,
    - repairs invalid review phases,
    - respawns missing role processes from `spawn_registry.json`, including the coordinator when the protocol is not terminal and the coordinator action is `advance` or `wait`,
    - writes respawn stdout/stderr to per-role files under `spawn-logs/`,
    - may inject placeholder/transport artifacts to keep the run moving.

- Real DebateClaw control path
  - Debate runs are compiled from JSON presets and materialized into:
    - runplans,
    - prompt bundles,
    - command maps,
    - template files,
    - run-scoped OpenClaw configs.
  - `launch_from_preset.py` and `launch_from_config.py` are wrappers around compile/materialize/launch subprocesses.
  - Slot isolation is enforced by `.debateclaw-slot.json` sentinel files plus workspace resets when session id changes.

- Real paper-processing path
  - Retrieval -> access -> parse -> rerank are independent scripts, not a single orchestrated service.
  - `paper-rerank.py` requires already-downloaded local PDFs.
  - `paper-parse.py` can use a long-lived MinerU API URL from env/config or local backend fallback logic.
  - GROBID and MinerU are treated as required long-lived local HTTP services by the docs and Docker helper.

- Shortcuts, hacks, implicit logic
  - Benchmark scripts duplicate a large amount of logic that also exists in `workspace/benchmarking/*`; the package is not the sole orchestration layer.
  - `benchmark_test.py` contains direct JSON parsing, subprocess wrappers, config pools, and answer extraction helpers instead of delegating all logic to package modules.
  - Native workflow package support exists, but current live ChemQA execution bypasses it in favor of CLI/state-script orchestration.
  - Run-scoped OpenClaw configs are produced by mutating a copy of the user’s local `~/.openclaw/openclaw.json`.
  - Recovery and artifact collection rely on specific file naming conventions such as `proposer-1.md`, `chemqa_review_protocol.yaml`, `qa_result.json`.
  - Cleanup correctness depends on manifests being written before launch and on command/session naming matching run ids.

## 5. Gap Analysis
- Missing features
  - `NOT_IMPLEMENTED`: No actual FastAPI/Gradio/uvicorn application code despite optional `web-ui` dependencies in `workspace/pyproject.toml`.
  - `NOT_IMPLEMENTED`: No source-side ConformaBench pool extractor script in `workspace/benchmarks/conformabench/`; only prepared data/tests are present.
  - `NOT_IMPLEMENTED`: No active code path that uses `workflow_loader.py` to load `chemqa-review` native workflow packages.

- Incomplete implementations
  - `PARTIAL`: `workspace/skills/chemqa-review/runtime/workflow.py`
    - `advance()` returns the state unchanged.
    - `submit_artifact()` just appends generic artifacts.
    - No real review/rebuttal/acceptance logic.
  - `PARTIAL`: `workspace/skills/chemqa-review/runtime/state_models.py`
    - Provides only initial state defaults.
    - Does not implement transitions or validation.
  - `PARTIAL`: Workflow JSON under `workspace/skills/chemqa-review/workflows/chemqa-review@1.json`
    - Declares package loading and parameters, but the operational runtime still depends on `debate_state.py` and driver scripts.
  - `PARTIAL`: `workspace/skills/debateclaw-v1/scripts/workflow_loader.py`
    - Implemented loader/validator, but repository search shows no active caller.
  - `PARTIAL`: `workspace/benchmarks/conformabench/`
    - Data and tests exist, but generation pipeline is absent from this repo.

- Architectural inconsistencies
  - Intended architecture suggests package-based workflows and reusable modules.
  - Actual behavior is still script-heavy and subprocess-heavy:
    - `benchmark_test.py` is a monolithic entrypoint with embedded orchestration logic.
    - ChemQA runs are controlled through external state scripts instead of the native workflow package.
  - `workspace/benchmarking/` exists as a reusable layer, but benchmark entry scripts still duplicate significant behavior.
  - `workspace/pyproject.toml` advertises `web-ui` extras, but there is no corresponding app module.
  - Top-level repo contains a mix of source, runtime state, generated artifacts, logs, and secret-bearing config in one tree; module boundaries are not clean at the repository level.

## 6. Risks & Technical Debt
- Fragile logic
  - Artifact recovery depends on specific filenames and directory heuristics in `workspace/benchmarking/runners/chemqa.py`.
  - Cleanup depends on manifests, process command-line matching, and session store scrubbing heuristics in `workspace/skills/benchmark-cleanroom/scripts/cleanup_benchmark_run.py`.
  - ChemQA recovery depends on `spawn_registry.json`, `/proc`-style process inspection when available, and workspace naming conventions in `workspace/skills/chemqa-review/scripts/recover_run.py`.

- Hardcoded values
  - Default OpenClaw home/config roots are hardcoded in `workspace/runtime_paths.py`.
  - Default model ids, agent ids, workspace roots, slot sets, and timeouts are hardcoded in `workspace/benchmark_test.py`.
  - GROBID and MinerU default URLs are hardcoded in docs/scripts.
  - ChemQA role topology is fixed to one candidate owner plus four reviewer lanes in `workspace/skills/chemqa-review/runtime/state_models.py` and associated scripts.

- Missing abstractions
  - Benchmark CLI scripts combine CLI parsing, orchestration, evaluation, config generation, and fallback handling in single files.
  - Native workflow-package abstraction exists but is not the live control plane.
  - Paper tools are standalone scripts with no shared higher-level orchestrator.
  - OpenClaw/ClawTeam integration is done through subprocess calls everywhere; there is no local adapter interface.

- Operational risks
  - `openclaw.json` at repo root contains live gateway/auth/provider configuration and is reused as a mutable base for runtime configs.
  - Repo root stores live runtime state, backups, SQLite DBs, session logs, and generated artifacts beside source.
  - Optional dependencies listed in `pyproject.toml` may imply capabilities that do not actually exist in code.

## 7. Suggested Next Steps
- Replace or retire the skeletal native workflow package:
  - Either make `workspace/skills/chemqa-review/runtime/workflow.py` the real execution engine or explicitly treat it as deprecated scaffolding.
- Collapse duplicated benchmark orchestration logic:
  - Move more logic from `workspace/benchmark_test.py` into `workspace/benchmarking/`.
- Separate source from runtime state:
  - Move generated workspaces, logs, DBs, and mutable OpenClaw runtime state outside the analyzed source tree or document them as runtime-only roots.
- Remove or implement misleading declared surfaces:
  - Either add a real web UI/API module for the `web-ui` extras or drop those extras from the project metadata.
  - Either add a ConformaBench pool extraction script or document the dataset as imported/static.
- Harden artifact and cleanup flows:
  - Continue reducing filename/path guessing in legacy ChemQA artifact recovery paths now that canonical Artifact Flow paths exist.
  - Centralize run manifest/session/process metadata contracts used by runners, drivers, and cleanup.
- Add clearer ownership boundaries:
  - Separate DebateClaw engine logic, ChemQA protocol logic, benchmark orchestration, and paper pipeline into smaller modules with fewer embedded subprocess wrappers.
