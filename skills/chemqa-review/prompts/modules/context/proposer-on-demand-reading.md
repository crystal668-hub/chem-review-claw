On-demand reading rules for `proposer-1`:

- Load only the skill doc or contract needed for the step you are executing right now.
- Default startup posture: do not read extra files until a concrete next action needs them.
- Read `paper-retrieval` only when planning or executing search.
- Read `paper-access` only after you have a candidate set of DOIs or URLs to resolve, and prefer batching multiple candidates into one access request.
- Read `paper-parse` after batched access produces readable local PDFs, or for the best available readable artifact when access coverage is limited.
- Read `debateclaw-v1` only when transport behavior is unclear or blocked.
- Do not preload all sibling skills, all contracts, or all references just to be safe.
- Do not scan `generated/prompt-bundles/`, `control/runplans/`, `control/run-status/`, or unrelated historical debate workdirs unless you are explicitly diagnosing a blocker.
