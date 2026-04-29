ChemQA fixed-lane review protocol rules:

- `proposer-1` is the only semantic candidate owner.
- `proposer-2` through `proposer-5` are fixed reviewer lanes, not alternate candidates.
- During `propose`, only `proposer-1` should produce a candidate submission.
- Reviewer lanes should wait for an explicit `review` request from `debate_state.py next-action`; they do not own candidate-generation work.
- During `review`, only the four fixed reviewer lanes review `proposer-1`.
- Review targets are never reviewer-to-reviewer and never proposer-main to reviewer lanes.
- A review counts toward ChemQA acceptance only when it targets `proposer-1` with `target_kind: candidate_submission`.
- During `rebuttal`, only `proposer-1` responds.
- Rebuttals must explicitly declare whether they are `response_only`, `answer_revision`, or `concession`; only `answer_revision` updates the current candidate view.
- The coordinator should treat one review round as complete only when all four reviewer lanes have submitted formal reviews for the current `proposer-1` candidate revision.
- The coordinator must not wait for reviewer-to-reviewer traffic or any proposer-lane cross-review fanout that is irrelevant to the fixed-lane protocol.
- At protocol terminal completion, the coordinator should use the model to refine and write the final `chemqa_review_protocol.yaml`, with deterministic reconstruction retained as fallback.
- Benchmark-visible terminal completion is published only after Artifact Flow writes canonical terminal artifacts and `qa_result.json`.
