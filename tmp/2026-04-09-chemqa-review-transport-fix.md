# chemqa-review transport fix note (2026-04-09)

Applied outside-workspace prompt fixes under `~/.openclaw/skills/chemqa-review`:

- clarified that writing `proposal.md` alone does not count as submission
- added explicit `debate_state.py submit-proposal` requirement for proposer-main and reviewer placeholder submissions
- added explicit `debate_state.py submit-review` requirement for formal reviews
- added explicit `debate_state.py submit-rebuttal` reminder in review-loop bridge policy
- updated state-query discipline to force-refresh after state-changing transport commands

Operational recovery performed:

- rescued run `chemqa-review-nmr-20260409-1727`
- manually registered missing reviewer placeholders for proposer-2/3/4/5
- advanced the run into review round 1
- relaunched all six debate agents with fresh session ids

Validation run launched:

- `chemqa-review-nmr-20260409-2200fix`
- observed patched prompts in new agent session logs
- observed reviewer lanes explicitly planning to call `submit-proposal`
- observed accepted proposal registrations beginning on the new run
