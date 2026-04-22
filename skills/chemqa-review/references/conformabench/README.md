# ConformaBench Authoring Templates

`ConformaBench` is the working benchmark name for open-ended chemistry construction
questions whose answers are judged by hidden RDKit geometry predicates.

The authoring model is intentionally split into two layers:

- `public_record.template.yaml`: the public task shown to the solver
- `hidden_judge_spec.template.yaml`: the hidden evaluation contract used only by the judge

This split keeps the prompt natural while preserving a strict machine-checkable
verification path.

## Design Goals

- Open constructive chemistry questions, not multiple choice
- Literature-informed reasoning, without turning the prompt into a bibliography hunt
- RDKit used only on the judge side
- A stable prompt shape across the benchmark
- Compatibility with the current `chemqa-review` runtime through a dedicated
  `conformabench_constructive` eval kind

## Naming

The benchmark name is currently `ConformaBench`.

Why this name:

- `Conforma` points at conformational and geometry-centric design
- `Bench` keeps it benchmark-oriented and extensible
- it does not overcommit the benchmark to a single force field, molecule class,
  or one narrow mechanism family

If the public benchmark name changes later, keep the two-layer schema and file
layout unchanged unless the runtime integration also changes.

## File Layout

- `public_record.template.yaml`
- `hidden_judge_spec.template.yaml`
- `compiled_frontierscience_research_record.template.yaml`

## Authoring Flow

1. Write the public prompt in `public_record`.
2. Write the hidden RDKit validation logic in `hidden_judge_spec`.
3. Compile the public rubric into the benchmark pool `answer` field used by
   `conformabench_constructive`.
4. Keep `hidden_judge_spec` out of any solver-visible bundle.

## Style Rules

- The public prompt should not contain author names, years, journal names, DOI,
  or directly identifying canonical example molecules.
- The public prompt should include mechanism, molecule-family, observable, and
  design-goal anchors.
- The final machine-readable answer must be on exactly one line:
  `FINAL ANSWER: <SMILES>`
- The hidden judge should gate on molecule validity before any explanation rubric.
