# Chem Calculator Contracts

## Shared CLI Contract

Every script supports:

```bash
python3 <skill-root>/scripts/<capability>.py \
  --request-json /path/to/request.json \
  --output-dir /tmp/<skill-out> \
  --json
```

Rules:
- `--request-json` is the canonical input file.
- `--output-dir` is required and created if missing.
- `--json` prints the same payload written to `result.json`.
- All scripts write one stable result file: `result.json`.

## Shared Output Shape

```json
{
  "status": "success|partial|error",
  "request": {},
  "primary_result": {},
  "candidates": [],
  "diagnostics": [],
  "warnings": [],
  "errors": [],
  "tool_trace": [],
  "source_trace": [],
  "provider_health": {}
}
```

Status semantics:
- `success`: the supported operation completed with usable output
- `partial`: some output is usable, but the request hit a documented limitation, ambiguity, unsupported feature, or compatibility warning
- `error`: no usable result is available for the requested operation

## Supported First-Batch Operations

### `molar_mass.py`

Request:

```json
{
  "operation": "molar_mass",
  "formula": "CuSO4·5H2O"
}
```

Supports simple formulas, parenthesized formulas, and hydrate dot notation using a small local element table.

### `stoichiometry.py`

Supported operations:
- `limiting_reagent`
- `combustion_analysis`
- `percent_yield`

Example:

```json
{
  "operation": "limiting_reagent",
  "reaction": {
    "reactants": [{"species": "H2", "coefficient": 2}],
    "products": [{"species": "H2O", "coefficient": 2}]
  },
  "known_amounts": [{"species": "H2", "value": 5.0, "unit": "mol"}],
  "target_species": "H2O",
  "target_unit": "mol"
}
```

### `concentration.py`

Supported operations:
- `dilution`
- `mix_solutions`

### `ksp_solver.py`

Supported operations:
- `precipitation_check`
- `residual_concentration`

`residual_concentration` currently supports monoatomic dissolved ions inferred from a simple local solid formula.

### `acid_base_solver.py`

Supported operations:
- `strong_acid_ph`
- `weak_base_ph`
- `buffer_ph`

### `gas_law.py`

Supported operations:
- `ideal_gas` with `solve_for = "moles"`
- `partial_pressure`

### `thermo_solver.py`

Supported operations:
- `delta_g`
- `equilibrium_constant_from_delta_g`

`delta_g` accepts either explicit unit-bearing objects or the shorthand fixture-style numeric fields.

### `redox_balance.py`

Supported operations:
- `oxidation_states`
- `electron_count`

`electron_count` is intentionally limited to simple related-species comparisons where one shared element changes oxidation state.

### `electrochemistry.py`

Supported operations:
- `nernst`
- `faraday`

### `unit_convert.py`

Supported operation:
- `convert`

Supports the local chemistry-first unit set used by the first-batch solvers, including temperature, pressure, mass, volume, amount, concentration, time, current, and basic thermo energy/entropy units.

### `answer_check.py`

Compares:
- expected value + unit
- candidate value + unit
- absolute or relative tolerance
- optional significant-figure floor

Common failure reasons:
- `incompatible_unit`
- `rounding_mismatch`
- `tolerance_mismatch`
