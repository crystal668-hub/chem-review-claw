# GLOBAL DEV SPEC

This document describes only the current implemented system. Source code and
runtime contracts are the source of truth when they differ from this document.

Maintain this file by updating the relevant existing section in place. Do not
append release history, migration narratives, individual benchmark results,
planned work, or speculative features. Keep volatile details in code, generated
manifests, run artifacts, skill documentation, or the linked specifications and
runbooks.

## 1. Project and Repository Boundaries

### Canonical project root

- `/Users/xutao/.openclaw/workspace` is the Git repository and canonical source
  root.
- The project is a Python 3.12+ workspace managed by `uv`; project commands and
  tests run from this directory with `uv run ...` or its `.venv`.
- Persistent source changes belong under this root. The primary source surfaces
  are `benchmarking/`, `skills/`, `scripts/`, `docs/`, `tests/`,
  `pyproject.toml`, and `uv.lock`.
- All documentation and subdirectories under `docs/superpowers/` are tracked
  source content; other generated or local documentation remains subject to the
  repository ignore rules.

### OpenClaw runtime home

- `/Users/xutao/.openclaw` is the local OpenClaw runtime home, not the Git
  repository.
- `agents/`, `benchmark/`, `debateclaw/`, `flows/`, `tasks/`, `logs/`,
  `devices/`, `identity/`, and the runtime-home `memory/` contain live runtime
  state, generated workspaces, sessions, databases, or logs. They are not source
  modules.
- `openclaw.json` and `.env` are live local configuration inputs. Benchmark
  launchers read them to produce run-scoped configuration; they are not copied
  into this repository as canonical source configuration.

### Data and generated project state

- `benchmarking.runtime.paths` owns default path resolution.
  `OPENCLAW_PROJECT_ROOT`, `OPENCLAW_DATA_ROOT`, `OPENCLAW_SKILLS_ROOT`, and
  `OPENCLAW_BENCHMARKS_ROOT` provide supported overrides.
- Formal benchmark datasets default to
  `/Users/xutao/.openclaw/data/formal-benchmarks`; temporary datasets default to
  `/Users/xutao/.openclaw/data/temp-benchmarks`.
- Benchmark run records are generated under
  `workspace/state/benchmark-runs/<formal|temporary>/<benchmark>/<model>/<run-id>`.
  Formal and temporary inputs determine the top-level category; benchmark and
  single-LLM model slugs provide the next two levels. Verifier-grounded isolated
  runtimes and dashboard metadata also live under `workspace/state/`.
- Explicitly retained fixed-workspace evidence lives under
  `workspace/state/benchmark-runs/legacy-workspace-archives/<workspace>-<timestamp>`.
  These snapshots are maintenance artifacts, not classified benchmark runs or
  attempt workspace archives.
- Active attempt workspaces default to `.openclaw/benchmark/workspaces`; live
  DebateClaw workspaces default to `.openclaw/debateclaw/workspaces`.

## 2. Module Ownership

### Benchmark package

| Module | Ownership |
| --- | --- |
| `benchmarking/core/` | Dataset normalization, runner/result dataclasses, convergence and answer recovery, stateless answer/agent-response processing, result status axes, reporting, and stdout result validation. |
| `benchmarking/scoring/` | Evaluator registry plus per-track implementations and result/error contracts for ChemBench, FrontierScience, SuperChem, HLE, verifier-grounded tracks, and generic semantic fallback. |
| `benchmarking/runtime/` | Shared path resolution, run-scoped OpenClaw configuration, attempt workspace lifecycle, access policy and adjudication, transcript audit and typed recovery, structured execution-error capture, cancellation and owned process groups, session isolation, visual input bundles, subprocess execution utilities, judge execution, verifier-grounded isolation, cleanroom integration, web-search preflight, historical adjudication replay, and verified legacy-workspace evidence archival. |
| `benchmarking/skills/` | Benchmark skill inventory projection, health checks, fixed skill-script runtime, and post-run tool/skill diagnostics. |
| `benchmarking/workflow/` | CLI entrypoint and top-level scheduling, experiment definitions, dataset selection, persisted run state, prompts, wave/group orchestration, runner adapters, and ChemQA response reconstruction. |
| `benchmarking/analysis/` | Detached post-run evidence bundling and automated analysis reports. |
| `benchmarking/dashboard/` | Local FastAPI dashboard, progress reconciliation, immutable run inspection, asset containment, dashboard-only annotations, and synchronized dataset/subset facets across filters, run summaries, and record details. |

`benchmarking.runtime.paths` is the shared path authority used by the package
and scripts. The benchmark CLI is owned directly by `benchmarking.workflow.cli`;
there is no root-level compatibility facade.

Attempt workspace responsibilities are split by dependency direction:
`benchmarking.runtime.workspace_policy` owns immutable access policy and audit
adjudication, `benchmarking.runtime.workspace_audit` owns transcript and path
evidence parsing, and `benchmarking.runtime.agent_workspace` owns workspace
templates, leases, recovery, sealing, quarantine, and audit orchestration.

Benchmark workflow responsibilities follow the same ownership rule:
`benchmarking.workflow.experiments` owns group definitions and effective specs,
`benchmarking.workflow.dataset_selection` owns discovery, filtering, sampling,
and output-root classification, `benchmarking.workflow.run_state` owns persisted
results and run metadata, and `benchmarking.workflow.runner_adapters` binds the
generic runners to runtime bundles, cleanroom, sessions, and workspace policy.
`benchmarking.runtime.subprocess_utils` owns shared subprocess and stdout helpers,
`benchmarking.runtime.error_capture` owns execution-error evidence extraction,
provider/config error classification, and preservation of original upstream
status codes, error codes, messages, and matched log events,
`benchmarking.runtime.cancellation` owns run cancellation tokens, reasons, and
owned process-group termination,
`benchmarking.runtime.judge` owns judge execution and isolation, and
`benchmarking.runtime.vgb_bridge` owns the pinned verifier-grounded release,
isolated process bridge, and public package API calls. The scoring evaluator
only maps benchmark records and verifier results.
`benchmarking.runtime.cleanroom.CleanroomRuntime` is the cleanroom dependency
binding. `benchmarking.workflow.cli` does not re-export these component APIs.

Scoring responsibilities are split by dependency direction:
`benchmarking.core.answer_processing` owns answer-track normalization and pure
agent-response JSON extraction, `benchmarking.scoring.registry` owns evaluator
registration and dispatch, and `benchmarking.scoring.evaluators/` owns only
benchmark-specific scoring strategies. `benchmarking.scoring.results` owns the
stable `EvaluationResult` shape and execution-error construction;
`benchmarking.scoring.errors` owns scoring and registry exceptions.

### Skill bundles

- `skills/debateclaw-v1/` owns the DebateClaw state machine, preset/run-plan
  compilation, prompt and command materialization, slot provisioning, launch
  wrappers, and one-turn OpenClaw wrapper.
- `skills/chemqa-review/` owns the fixed-lane ChemQA protocol, role driver,
  shared spawn-registry policy, liveness and recovery tools, typed Artifact
  Flow, and terminal artifact reconstruction.
- `skills/benchmark-cleanroom/` owns cleanup manifests, runtime leases, and
  benchmark-owned process termination.
- Chemistry provider skills live as independent bundles under `skills/`.
  `skills/chemistry-routing-matrix.json` is the machine-readable capability and
  exposure inventory; it is not a deterministic router.
- The RDKit skill exposes neutral, explicit conformer force-field selection:
  its generic conformer entrypoint requires `MMFF` or `UFF`, and dedicated MMFF
  and UFF scripts implement each family without cross-family fallback. Every
  conformer request also requires explicit `num_conformers` and `random_seed`
  values; the skill defines no sampling defaults or preferred values.
- `skills/paper-retrieval/`, `paper-access/`, and `paper-parse/` are independent
  paper-processing stages.

ChemQA runtime checks expose a redacted process-environment report for the
MinerU API settings. The current process environment takes precedence over
the configured `OPENCLAW_ENV_FILE` (defaulting to the runtime home's `.env`)
when determining effective endpoint and token availability; values are never
printed in the report.

### Project scripts and resources

- `scripts/` is an importable project package containing the maintenance
  entrypoints below; its modules can be reused by tests without resolving to an
  unrelated installed package with the same name.
- `scripts/run_skill.py` is the fixed entrypoint for benchmark-agent execution
  of local skill scripts through the workspace `uv` environment.
- `scripts/sync_verifier_grounded_datasets.py` validates a pinned release and
  synchronizes public prompt datasets and isolated scoring runtime metadata.
- `scripts/replay_workspace_adjudication.py` replays stored transcript evidence
  without a model call, recovers archived final answers from per-record data,
  runner metadata, or the session transcript, reconstructs a missing
  `results.json` from per-record payloads, and can apply record-selective
  recovery only after writing a snapshot. Explicit manual adjudication requires
  selected record IDs and a reason, preserves the original audit and error, and
  cannot override confirmed contamination.
- `scripts/archive_legacy_benchmark_workspaces.py` copies complete fixed legacy
  workspaces into an independent evidence archive, records a path/metadata/SHA-256
  inventory, verifies every archive and unchanged source, and deletes sources
  only when all requested archives pass those checks.
- `scripts/sync_openclaw_qwen_provider.py` updates the live runtime-home Qwen
  provider configuration for `qwen3.6-plus`, `deepseek-v4-pro`,
  `qwen3.7-max`, `qwen3.7-plus`, and `qwen3.8-flash` using the
  `openai-responses` API, and removes applicable stale agent provider caches.
- `scripts/patch_openclaw_minimax_ui.py` applies the local OpenClaw 2026.6.9
  Control UI runtime patch: the model picker exposes only `off`/`adaptive` for
  MiniMax-M3, derives that picker from the current per-session model override
  (including the `minimax-m3` alias), selects `adaptive` when entering M3, and
  clears stale thinking overrides when leaving M3. It also versions the
  service-worker registration and main bundle URL so stale cache-first assets
  cannot keep serving the old picker. The installed bundle remains runtime
  state rather than canonical project source.
- The benchmark CLI and fixed-lane OpenClaw drivers accept the `adaptive`
  thinking level required by MiniMax-M3; the Benchmark Orchestrator validates
  the model-specific level before launching a run.
- VGB `single-LLM` attempts create a fresh `scratch/venv` from the bootstrap
  Python via `uv venv --seed --no-project`; the workspace `.venv` remains the
  bootstrap environment for the runner and non-VGB records.
- `benchmarking/resources/agent-workspace-templates/` contains the canonical
  benchmark workspace base contract and role overlays.
- `benchmarking/resources/verifier_grounded/` contains the pinned release
  identity and sanitized public dataset snapshots.

## 3. Core Execution Flows

### Benchmark CLI

The canonical entrypoint is:

```bash
uv run python -m benchmarking.workflow.cli
```

The implemented default experiment groups are:

- `single_llm_skills_on`: one OpenClaw agent with the health-filtered benchmark
  skill allowlist.
- `single_llm_skills_off`: one OpenClaw agent with an explicit empty skill list.
- `chemqa_skills_on`: the fixed-lane ChemQA workflow with the health-filtered
  benchmark skill allowlist.

All three current group definitions disable generic web search and web fetch.
For each invocation, the CLI:

1. Uses `benchmarking.workflow.dataset_selection` to discover or accept JSONL
   datasets, normalize them to `BenchmarkRecord`, apply record selection, and
   classify the run output root. Runner adapters materialize run-local visual
   bundles when required.
2. Runs skill health checks, filters skills-on allowlists, prepares a unique
   invocation identity, captures the verifier-grounded release identity for the
   lifetime of the invocation, recovers sentinel-proven stale active workspaces,
   and writes run-scoped OpenClaw configs.
3. Installs `SIGINT`/`SIGTERM` cancellation handlers, then dispatches groups in
   waves through `benchmarking.workflow.orchestration` and
   `benchmarking.workflow.runner_adapters`. Each record runs through either the
   single-LLM runner or the ChemQA runner, then through the registered evaluator
   when the runner result is scoreable.
4. Uses `benchmarking.workflow.run_state` to persist each record immediately,
   update run artifacts, aggregate only `scored=true` records, and support
   historical per-record resume data; the CLI writes the final results and
   runtime manifest.
5. Starts detached automated analysis unless `--no-analysis` is selected. A
   cancelled run never launches detached analysis.

Cancellation is run-scoped and cooperative. The first signal fixes the stable
cancellation reason; later signals shorten process termination grace. Scheduling
stops before another record, retry, wave, judge, or analysis launch. Registered
subprocesses start in owned process groups so termination covers descendants.
Active runners still audit and seal their attempt workspaces; cleanroom handles
manifest-owned ChemQA processes. Progress, waves, results, and the runtime
manifest finish as `cancelled` or `cancelled_with_errors`, and cancelled records
are non-evaluable, unscored, and use `execution_error_kind=cancelled`.

### Single-LLM runner

- Bounded single-LLM attempts default to 7200 seconds (2 hours). The runner
  forwards this budget to OpenClaw as `--timeout`; the wrapper subprocess guard
  adds the 90-second finalization safety window and 30-second process margin,
  for a default outer limit of 7320 seconds.
- The runner prepends the effective budget to the agent prompt as `Time budget:
  <seconds> seconds for the whole answer attempt.` For bounded positive
  budgets, the wrapper tracks the primary turn and, when it returns without a
  complete answer after roughly five sixths of the budget (6000 seconds at the
  default), sends a same-session reminder with the remaining time.
- Every primary or timeout-retry attempt receives a fresh sentinel-managed
  workspace and run-scoped session id.
- Records with `eval_kind=verifier_grounded` additionally receive a fresh
  attempt-local Python environment and uv cache. All attempts in an invocation
  share its run-start PyPI cutoff, while each retry starts from a new empty
  environment. The agent may install registry packages with `uv pip`; pip
  mutations, direct URLs, local/editable sources, alternate indexes, dependency
  target overrides, and the pinned verifier distribution are blocked for
  explicit commands under the cooperative-agent threat model.
- The runner materializes the role contract, attaches current scratch paths,
  invokes `benchmarking.runtime.single_llm_openclaw_wrapper`, validates OpenClaw
  JSON stdout, and enforces the eval-aware candidate-answer contract.
- The canonical workspace contract requires agent-created Python virtual
  environments under `scratch/` to use `python3 -m venv --copies venv`.
  Runner-created VGB environments use `uv venv --seed --no-project`, are
  inventoried after the agent returns, and are removed before archival.
- Nonzero OpenClaw subprocess results are classified from structured error
  evidence before diagnostic excerpts are truncated. Provider failures retain
  a terminal `primary_error` plus ordered `observed_errors`; internal error
  categories and retry policy do not replace the original upstream status code,
  error code, message, or matched log text.
- Timeout-family failures may create a fresh attempt. Transcript recovery and a
  same-session finalization repair can preserve a complete answer; incomplete or
  unreliable output remains non-scoreable.
- Canonical skill scripts continue through `scripts/run_skill.py`. Within a VGB
  attempt it executes them directly with `BENCHMARK_ATTEMPT_PYTHON`, without
  resolving the workspace project or implicitly installing project extras.
- After a VGB attempt returns, the runner records dependency commands from the
  transcript, the installed distribution inventory, RECORD hashes, a hashed
  replay requirements file, the run-start PyPI cutoff, credential names, and
  allowlisted native-tool fingerprints. It removes any detected exact-denylist
  distributions, then deletes the venv, uv cache, and native-tool wrappers
  before sealing the workspace; the manifest remains in archived scratch and
  runner metadata.
- The transcript is audited under the attempt access policy before the complete
  workspace is archived. A `non_evaluable` adjudication or archive failure
  rejects an otherwise complete answer; `scoreable_degraded` preserves it with
  degraded-execution metadata. If the runner already has a terminal execution
  failure and the audit is unavailable with indeterminate contamination, the
  audit remains attached as diagnostic workspace-isolation metadata and does
  not replace the original failure. Confirmed contamination and archive
  failures retain precedence. Exec auditing first builds a structured shell
  projection that tracks quotes, escapes, heredocs, command substitutions,
  nested substitutions, backticks, and arithmetic substitutions without
  executing transcript commands. Parser recovery is represented by stable
  recovery codes and versions; a successful `exec` result may be retained as a
  warning only when the original shell syntax is valid, the projection is
  complete, and the normal protected-path scan (including literal paths inside
  nested substitutions) finds no forbidden access. Unknown or incomplete shell
  constructs, missing results, syntax failures, unresolved recovery, and
  protected-root references remain non-evaluable or contaminated according to
  the normal audit rules.

### ChemQA runner

- Each attempt prepares one coordinator workspace and five role workspaces as an
  all-or-fail lease set.
- The runner compiles and materializes a `chemqa-review@1` launch, then the role
  drivers advance the DebateClaw SQLite state machine through candidate, review,
  rebuttal, and finalization phases.
- The fixed semantic topology is one candidate owner (`proposer-1`) and four
  reviewer lanes (`proposer-2` through `proposer-5`).
- Stalled status can invoke `recover_run.py`; a new recovery attempt uses a new
  workspace lease set.
- Artifact Flow validates typed protocol artifacts and publishes benchmark
  terminal status only after canonical terminal artifacts are readable.
- Default scoring consumes `final_answer_artifact.json`. Preview text and failure
  projections remain diagnostic. Cleanup, transcript audit, and archive complete
  before the final runner result is accepted.

### Evaluation, reporting, and review

- `benchmarking.scoring.registry` dispatches by `record.grading.kind` with
  `generic_semantic` fallback. LLM-judge calls use a fresh isolated judge
  session and attempt workspace; pure answer and agent-response parsing lives in
  `benchmarking.core.answer_processing`.
- Verifier-grounded tasks use `benchmarking.runtime.vgb_bridge` to call the
  pinned package through a hash-addressed, non-agent virtual environment and
  `python -I`; agent-visible datasets contain public prompts and answer schemas,
  not hidden verifier material. Final reporting references for every
  release-declared property-calculation track come from that pinned release's
  public `task(..., include_gold=True)` view; scoring-profile identifiers are
  removed before the references enter reporting artifacts.
- Completed aggregation writes run-local evidence and may launch
  `benchmarking.analysis.automated`. Analysis failure is diagnostic and does not
  change benchmark scoring or the CLI exit outcome.
- The dashboard recursively discovers classified run directories and stops
  scanning below each detected run. It skips the reserved
  `legacy-workspace-archives` maintenance tree rather than traversing retained
  workspace evidence. It writes its annotation SQLite database and may persist a
  `cancelled_with_errors` terminal projection when a progress owner PID proves
  that a `running` or `cancelling` run is stale. It does not rewrite record scores
  or launch benchmark processes. When an aggregate `results.json` is present,
  per-record outputs are merged into the dashboard view and take precedence for
  duplicate group/record keys so active or resumed runs expose results written
  after the last aggregate snapshot, while an unenriched verifier placeholder
  cannot replace an aggregate reporting reference. For active verifier-grounded
  property-calculation runs, the detail view derives the standard answer from
  the scored result's release-specific `properties.gold_answers` when the
  per-record reporting reference is still the public-data placeholder. Dataset
  facets use the canonical
  `source_file` dataset segment when it follows the standard
  `<dataset>/data/<file>.jsonl` layout, correcting inconsistent persisted result
  labels without rewriting run artifacts. Manual dashboard refreshes expose
  their pending state through the refresh control and restore the control after
  either success or failure. Favorited runs are pinned to the top of the run
  list; within favorited and non-favorited groups, discovery keeps the existing
  newest-first ordering. Record detail timing prefers the agent execution
  duration from `runner_meta.durationMs`, converted from milliseconds to
  seconds, and falls back to persisted `elapsed_seconds` for legacy results
  without that metadata.

### Paper pipeline

Paper processing is an explicit fixed sequence of independent scripts:

```text
retrieval -> access -> parse
```

Parsing uses the official MinerU Agent API for small documents, the optional
Precision API for larger documents, and PyMuPDF as the final local fallback.
The stages exchange explicit JSON artifacts and are not exposed as one
transactional orchestration service. No local MinerU CLI or `mineru-api`
service is required.

## 4. Stable Data and Isolation Contracts

### Runner and result contracts

- Runners return `RunnerResult` with `RunStatus`, `AnswerPayload`, `runner_meta`,
  raw provider data, and optional `FailureInfo` or `RecoveryInfo`.
- `RunnerResult.should_score()` is the gate into evaluator execution. Completed
  results score; recovered results score only when their recovery metadata marks
  them both evaluable and scoreable.
- Current per-record and top-level result writers use schema version `3`.
- Stable result axes are `run_lifecycle_status`,
  `protocol_completion_status`, `answer_availability`, `answer_reliability`,
  `evaluable`, `scored`, `recovery_mode`, `degraded_execution`, and
  `execution_error_kind`.
- Structured runner execution errors retain a stable internal `code`, `layer`,
  and `retryable` decision alongside original `primary_error` and
  `observed_errors` evidence. Primary provider evidence is selected by
  structured-field and parser specificity rather than terminal log position;
  punctuation-only diagnostic fragments are not error evidence. Provider
  transport failures such as `stream_read_error` are retryable. Retry attempt
  history retains the complete structured execution error for each failed
  attempt. Unsupported OpenClaw thinking levels are classified as explicit,
  non-retryable configuration failures and retain the original diagnostic.
- `benchmarking.runtime.subprocess_utils.summarize_payloads` excludes payloads
  marked `isError=true` and the OpenClaw fallback warning shape
  `⚠️ 🛠️ \`...\` failed` from formal answer text. Raw provider payloads,
  transcripts, and tool-failure audit counts remain unchanged.
- `passed` is an evaluator quality outcome, not a runtime-health field.
  Verifier-grounded continuous scores use `passed = null`.
- Aggregate score denominators contain only records with `scored=true`.
- Run, wave, and group lifecycle projections distinguish `cancelling`,
  `cancelled`, and `cancelled_with_errors`; records use `cancelled` without a
  fabricated evaluator score.

The final run artifact set includes:

- `results.json` and `runtime-manifest.json`;
- `per-record/<group>/<record>.json`;
- `progress/events.jsonl` and `progress/state.json`;
- `runtime-config/*.json`, `input-bundles/`, and archived attempt workspaces;
- `skill-health.json` and `web-search-preflight.json`;
- `analysis/` status, evidence, and reports when automated analysis is enabled.

Legacy fixed-workspace evidence uses a separate archive kind and schema. It
retains the complete source tree, including Git metadata, plus an inventory of
directories and regular files with modes, modification times, sizes, and SHA-256
digests. The maintenance command rejects symlinks and special files, refuses
overwrites, detects source mutation during copying, verifies the completed
archive independently, and rechecks every source before optional deletion. It
does not synthesize attempt identities or place legacy snapshots inside a run's
`agent-workspace-archives/` tree.

### Attempt workspace contract

- Attempt workspaces use scratch contract version `2` with stable
  `scratch/requests`, `scratch/outputs`, `scratch/notes`, and `scratch/tmp`.
- Workspace tree validation permits regular files named `.git` anywhere under
  `scratch/tmp/cache/uv/`, which `uv` may create in a scratch-local cache. It
  also permits relative symbolic links located under `scratch/` when their
  strict resolved targets remain under the same attempt scratch tree and are
  regular files or directories. A dangling relative link is permitted only when
  its lexically normalized target is strictly inside that same scratch tree and
  its existing target-parent components are real directories. Control-plane,
  absolute, escaping, chained-dangling, cyclic, special-file-targeting, and
  other `.git` paths remain forbidden.
- Attempt archives preserve validated scratch-relative symbolic links rather
  than dereferencing them, revalidate the relocated archive tree, and record a
  count plus deterministic link-manifest digest, with separate count and digest
  fields for dangling links. Cross-filesystem copies must match both regular-file
  statistics and the symbolic-link inventory.
- Structured file tools use workspace-relative `scratch/...` paths. Shell
  commands enter scratch through runner-provided environment variables.
- A canonical base `AGENTS.md` plus a minimal role overlay defines the same
  isolation behavior for single-LLM, judge, and ChemQA roles.
- The canonical base keeps native `exec` available for single-line commands and
  directs multiline scripts through a structured write to `scratch/tmp` before
  native execution; heredocs, here-strings, and inline multiline interpreter
  commands are discouraged in the workspace contract rather than record prompts.
- Immutable `WorkspaceAccessPolicy` objects define normalized read, write, and
  exec-workdir scopes, exact-file scopes, protected roots, and a deterministic
  digest. Skills-off and judge policies do not grant access to the skill source
  tree or `scripts/run_skill.py`.
- The `benchmark-workdir-guard` plugin preflights structured path arguments,
  explicit exec working directories, and absolute paths embedded in exec
  commands. Exec command paths inside the active workspace are allowed, known
  system executable/device paths are allowed, and other absolute paths or
  protected roots are blocked before execution. Transcript audit independently
  correlates tool calls and results and records access mode, outcome, resolved
  path, policy, and matched protected root.

Workspace audit has four independent axes:

- `audit_execution_status`: `complete` or `unavailable`;
- `boundary_status`: `clean`, `warning`, `violated`, or `unknown`;
- `contamination_status`: `clear`, `confirmed`, or `indeterminate`;
- `adjudication`: `scoreable`, `scoreable_degraded`, or `non_evaluable`.

Confirmed or indeterminate external information exposure is `non_evaluable`.
Guard-blocked operations are recorded as boundary violations with
`operation_outcome=blocked` and `information_exposure=none`; they do not by
themselves prove contamination and remain scoreable as degraded execution.
Write-only, other failed, or allowed-fallback boundary events do not by
themselves prove information contamination. A write-only boundary violation can
be `scoreable_degraded`; an allowed fallback is a warning and remains
`scoreable`. Audit evidence recovery is attempted before an unavailable audit is
finalized. Archive failure remains fail closed.

Known audit parser conditions use stable codes and an in-code recovery-handler
registry. `exec_unterminated_heredoc_eof` projects the complete EOF heredoc body
through recovery version 1, emits `transcript_audit_recovered`, and continues the
normal protected-path scan. Dynamic shell constructs use a quote-aware, nested
projection before `shlex` tokenization; literal paths discovered inside command
substitutions are audited with the same immutable policy. Recovery success is a
boundary warning; an unknown condition, incomplete projection, unresolved
dynamic construct, or recovery exception remains unavailable. Historical
dry-run replay can use a persisted per-record workspace policy when an
interrupted legacy run lacks final aggregate artifacts; apply mode still
requires `results.json` and `runtime-manifest.json`.

This lifecycle, guard, and transcript audit is not an operating-system security
boundary. Processes still run as the same local user.

### Session, skill, artifact, and cleanup contracts

- Single-LLM and judge calls clear only stale main-session pointers, use explicit
  run-scoped session ids, and verify the requested session and transcript after
  the turn. Historical transcripts remain available for audit.
- Skills-on exposure is the intersection of the inventory allowlist and startup
  health results. Skills-off runner configs contain `skills: []`. Skill choice is
  left to the model; tool and skill diagnostics do not change answer scores.
- Agent-invoked local skill scripts run through `scripts/run_skill.py`, which uses
  the canonical workspace for dependency resolution and the attempt scratch
  directory for relative artifacts.
- ChemQA terminal output is `final_answer_artifact.json` or
  `failure_artifact.json`, accompanied by `artifact_manifest.json`,
  `candidate_view.json`, validation diagnostics, and the compatibility projection
  `qa_result.json`.
- Benchmark cleanroom cleanup terminates benchmark-owned processes from manifests
  and leases. It intentionally retains session stores, transcripts, run artifacts,
  manifests, and archived workspaces for audit.

## 5. Current Risks and Non-goals

### Current risks

- Attempt isolation detects and adjudicates filesystem evidence but cannot prevent
  every same-user filesystem access performed inside arbitrary subprocesses.
- The benchmark CLI still owns argument parsing, wave scheduling, final
  aggregation, and runtime-manifest composition; changes to these concerns can
  therefore affect the whole benchmark entrypoint.
- OpenClaw and ClawTeam integration is subprocess- and file-contract-based;
  correctness depends on session identifiers, manifests, status files, and
  process metadata remaining consistent.
- ChemQA recovery and artifact reconstruction retain compatibility with specific
  protocol filenames and directory layouts.
- The live runtime-home `openclaw.json` is the mutable base for run-scoped configs
  and may contain provider and gateway configuration. It must be treated as local
  operational state.
- Many chemistry skills require optional Python packages, external executables,
  API credentials, network providers, or optional MinerU API access. Startup
  health filtering is the runtime authority for benchmark exposure.

### Non-goals of the current system

- Attempt workspaces are not containers, separate OS users, or syscall sandboxes.
- The benchmark dashboard is a localhost review surface, not a benchmark launcher,
  multi-user service, or authority that rewrites immutable result artifacts.
- Automated post-run analysis is not part of benchmark scoring.
- The chemistry inventory does not prescribe deterministic skill routing.
- The paper stages are not exposed as one transactional orchestration service.
- Benchmark cleanup does not prune retained sessions or historical run artifacts.

## 6. Specification and Runbook Index

### Normative project and benchmark contracts

- `AGENTS.md`: repository workflow, canonical document rule, test and commit
  requirements.
- `docs/superpowers/specs/2026-07-16-benchmark-attempt-workspace-behavior-and-adjudication-spec.md`:
  current attempt behavior, access policy, four-axis adjudication, and historical
  replay contract.
- `docs/superpowers/specs/2026-07-16-benchmark-forbidden-path-root-containment-spec.md`:
  protected-root containment and transcript path evidence.
- `docs/superpowers/specs/2026-07-23-benchmark-audit-error-allowlist-and-cancellation-spec.md`:
  typed audit recovery, EOF heredoc handling, owned-process cancellation, and
  persistent cancellation terminal states.
- `docs/superpowers/specs/2026-07-15-verifier-grounded-openclaw-single-llm-integration-usage-spec.md`:
  verifier-grounded dataset exposure and isolated scoring contract.
- `benchmarking/resources/verifier_grounded/release.json`: current pinned
  verifier-grounded release identity.

### Operational runbooks and component contracts

- `README.md`: verifier-grounded CLI usage and paper-processing operations.
- `docs/benchmark-dashboard-usage.md`: dashboard launch, data sources, and review
  workflow.
- `skills/debateclaw-v1/SKILL.md` and `skills/debateclaw-v1/references/`:
  DebateClaw presets, runtime conventions, model/slot mapping, and recovery.
- `skills/chemqa-review/SKILL.md` and
  `skills/chemqa-review/references/contracts.md`: ChemQA fixed-lane runtime and
  artifact contract.
- `skills/benchmark-cleanroom/SKILL.md` and
  `skills/benchmark-cleanroom/references/runtime-surfaces.md`: cleanup manifest,
  lease, and retention contract.
- Each provider skill's `SKILL.md` and optional `references/contracts.md` are the
  authority for that provider's request, dependency, and output contract.
