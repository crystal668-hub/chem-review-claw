Pure-YAML artifact rules:

- Write artifact files as pure YAML mappings only.
- Do not add markdown headings, prose outside YAML, bullet-list commentary outside YAML, or fenced code blocks.
- Do not wrap the YAML in ```yaml fences.
- Keep the artifact valid if the runtime saves the file exactly as written.
- Prefer short scalar fields plus explicit lists/maps over long free-form narration.
- If you need multiline text, use valid YAML block scalars.

Positive example — candidate submission:

artifact_kind: candidate_submission
artifact_contract_version: react-reviewed-v2
phase: propose
owner: proposer-1
direct_answer: "lock paper-2 for downstream evidence review"
summary: >-
  Batched retrieval and access produced a small evidence pool with two readable PDFs;
  rerank selected the strongest candidate for detailed parsing.
submission_trace:
  - step: paper-retrieval
    status: success
    candidate_count: 5
    detail: Retrieved five candidate papers relevant to the claim and ranked them by metadata fit.
  - step: paper-access
    status: success
    access_attempt_count: 4
    readable_pdf_count: 2
    blocked_count: 2
    detail: Batched access over the top four candidates yielded two readable local PDFs and two OA dead ends.
  - step: paper-rerank
    status: success
    rerank_input_count: 2
    locked_count: 1
    detail: Reranked the full readable-PDF pool and locked `paper-2` as the strongest evidence source.
  - step: paper-parse
    status: success
    parsed_artifact_count: 1
    detail: Parsed the rerank-selected locked PDF and extracted fulltext / sections for claim review.
evidence_limits:
  - Coverage is still partial because two promising candidates had no readable OA PDF after batched access.
claim_anchors:
  - anchor: claim-1
    claim: `paper-2` is the strongest currently readable evidence source after batched access and rerank.

Negative example — invalid candidate submission:

owner: proposer-1
direct_answer: "6"

Why invalid:
- missing `artifact_kind`
- missing `artifact_contract_version`
- missing `phase`
- no valid `submission_trace`

Positive example — formal review:

artifact_kind: formal_review
artifact_contract_version: react-reviewed-v2
phase: review
reviewer_lane: proposer-3
target_owner: proposer-1
target_kind: candidate_submission
verdict: blocking
summary: The candidate makes a stereochemical claim without a clear anchor.
review_items:
  - item_id: trace-1
    severity: high
    finding: The diastereotopic-CH2 claim is not anchored to a concrete atom-level explanation.
    requested_change: Add an explicit atom-level anchor for the non-equivalent CH2 hydrogens.
counts_for_acceptance: true
synthetic: false

Negative example — invalid formal review:

artifact_kind: formal_review
phase: review
reviewer_lane: proposer-3
verdict: blocking
notes: Missing target metadata and review_items list.

Why invalid:
- missing `target_owner`
- missing `target_kind`
- missing `review_items`
- no `counts_for_acceptance: true`
- no `synthetic: false`

Positive example — rebuttal:

artifact_kind: rebuttal
artifact_contract_version: react-reviewed-v2
phase: rebuttal
owner: proposer-1
concede: false
response_summary: >-
  Added an atom-level anchor for the CH2 hydrogens and clarified that the answer
  relies on stereochemical non-equivalence.
response_items:
  - item_id: trace-1
    severity: low
    finding: Added the requested anchor and wording fix.
updated_direct_answer: "6"

Positive example — coordinator failure terminal artifact:

artifact_kind: coordinator_protocol
artifact_contract_version: react-reviewed-v2
terminal_state: failed
question: ""
final_answer: {{}}
acceptance_status: failed
review_completion_status:
  status: failed
  phase: review
candidate_submission: {{}}
acceptance_decision:
  status: failed
  reason: phase stagnation detected after recovery cycles without progress
submission_trace: []
submission_cycles: []
proposer_trajectory: {{}}
reviewer_trajectories: {{}}
review_statuses: {{}}
final_review_items: {{}}
overall_confidence:
  level: low
  rationale: Run terminated explicitly to stop token burn after repeated recovery failures.
failure_reason: phase stagnation detected after recovery cycles without progress
terminal_failure_artifact: /path/to/chemqa_review_failure.yaml
