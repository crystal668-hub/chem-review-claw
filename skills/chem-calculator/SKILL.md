---
name: chem-calculator
description: Use for local, reproducible chemistry calculations that need deterministic numeric checking rather than web lookup.
---

# Chem Calculator

## Overview

`chem-calculator` is a local first-batch chemistry calculation toolbox. It covers common molar-mass, stoichiometry, concentration, Ksp, acid/base, gas-law, thermodynamics, redox, electrochemistry, unit-conversion, and answer-check tasks with structured JSON output.

## When to Use

Use this skill when:
- a chemistry question is primarily numerical
- a candidate numeric answer needs verification
- unit handling or tolerance checks matter
- the calculation can be handled with bounded local models instead of free-form reasoning

Do not use this skill for structure lookup, nomenclature resolution, or literature search.

## Execution

```bash
python3 <skill-root>/scripts/<capability>.py \
  --request-json /path/to/request.json \
  --output-dir /tmp/<skill-out> \
  --json
```

- `--output-dir` is required and will be created if missing.
- Every script writes `result.json` in the output directory.
- `--json` prints the same top-level payload written to `result.json`.

Read [contracts.md](/Users/xutao/.openclaw/workspace/skills/chem-calculator/references/contracts.md) for supported request modes and failure semantics.
