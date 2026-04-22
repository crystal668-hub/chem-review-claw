`react_reviewed` acceptance policy:

- only `proposer-1` may own the accepted candidate submission
- open blocking review items prevent acceptance
- incomplete required reviewer execution blocks acceptance
- missing citation or step anchors block acceptance when the lane marks them blocking
- topic drift blocks acceptance
- junk or metadata-like answer content blocks acceptance

Required ChemQA review-completion rules:

- required reviewer lanes are exactly `proposer-2`, `proposer-3`, `proposer-4`, and `proposer-5`
- required reviewer completion is satisfied only by non-synthetic formal reviews written in `phase: review`
- required reviews must target `proposer-1` with `target_kind: candidate_submission`
- only those qualifying formal reviews count as reviewer completion, lack of objections, or support for acceptance
- synthetic recovery reviews may be reported in diagnostics, but they do not satisfy required reviewer completion and do not count toward acceptance
- if any required reviewer lane is missing a qualifying formal review, acceptance must be blocked explicitly
