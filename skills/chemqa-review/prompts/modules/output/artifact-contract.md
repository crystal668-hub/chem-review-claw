Artifact contract:

- The runtime expects pure-YAML artifacts for candidate submission, formal review, rebuttal, and coordinator protocol / terminal failure outputs.
- The coordinator must emit `chemqa_review_protocol.yaml`.
- On stop-loss / stagnation / unrecoverable malformed-output failure, emit `chemqa_review_failure.yaml` and mark the protocol terminal state as `failed` rather than looping indefinitely.
- The protocol must be sufficient for the collector to rebuild:
  - `candidate_submission.json`
  - `acceptance_decision.json`
  - `submission_trace.json`
  - `submission_cycles.json`
  - `proposer_trajectory.json`
  - `reviewer_trajectories.json`
  - `review_statuses.json`
  - `final_review_items.json`
  - `qa_result.json`
- Artifact Flow also writes canonical terminal artifacts:
  - `final_answer_artifact.json` or `failure_artifact.json`
  - `artifact_manifest.json`
  - `candidate_view.json`
  - `validation_summary.json`

For accepted outputs, the protocol must make the acceptance basis explicit.
At minimum, include enough structured data for the collector to verify that:

- `proposer-1` is the accepted candidate owner
- `review_completion_status.required_candidate_reviews_expected = 4`
- `review_completion_status.required_candidate_reviews_submitted = 4`
- `review_completion_status.required_fixed_reviewer_lanes_complete = true`
- `review_completion_status.transport_placeholders_ignored` is reported
- `review_completion_status.synthetic_reviews_excluded_from_acceptance` is reported
- each fixed reviewer lane records whether it submitted a qualifying formal review against `proposer-1`
- each qualifying formal review records:
  - `artifact_kind`
  - `phase`
  - `reviewer_lane`
  - `target_owner`
  - `target_kind`
  - `verdict`
  - `review_items`
  - `counts_for_acceptance`
  - `synthetic`

The reconstructed `qa_result.json` is expected to remain externally compatible with the current `react_reviewed` artifact surface.
The ChemQA benchmark run is not externally complete until Artifact Flow has written and reopened the canonical terminal artifact, manifest, and `qa_result.json`.
