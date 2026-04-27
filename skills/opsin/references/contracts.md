# OPSIN Contract

## Standard CLI

```bash
python3 <skill-root>/scripts/<capability>.py \
  --request-json /path/to/request.json \
  --output-dir /tmp/<skill-out> \
  --json
```

- `--request-json` is required.
- `--output-dir` is required and created if missing.
- Each script writes one stable result file: `result.json`.
- `--json` prints the same top-level payload written to disk.

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

## Status Semantics

- `success`: usable result returned.
- `partial`: some usable result returned, but the batch contains failures or degradation.
- `error`: no usable result for the requested operation.

## `name_to_structure.py`

### Request

```json
{
  "name": "ethyl ethanoate",
  "timeout_seconds": 10.0,
  "retry_attempts": 1
}
```

### Notes

- Uses the EMBL-EBI OPSIN web service at `https://www.ebi.ac.uk/opsin/ws/<url-encoded-name>.json`.
- Retry policy is bounded and only applies to transient network failures.
- No fallback to PubChem or other providers.

### Result Kinds

- `structure`
- `no_result`
- `provider_failure`
- `invalid_request`

## `batch_name_to_structure.py`

### Request

```json
{
  "names": ["ethyl ethanoate", "aspirin"],
  "timeout_seconds": 10.0,
  "retry_attempts": 1
}
```

- Returns per-name payloads in `candidates`.
- Top-level `status` is `partial` when at least one lookup succeeds and at least one fails.

## `parse_diagnostics.py`

### Request

```json
{
  "diagnostics": [
    {
      "input_name": "xylene",
      "provider_status": "FAILURE",
      "provider_message": "Name appears to be ambiguous between multiple structures"
    }
  ]
}
```

### Categories

- `unsupported_syntax`
- `ambiguous_name`
- `non_systematic_name`
- `malformed_input`
- `parse_failure`

## `validate_with_rdkit.py`

### Request

```json
{
  "name": "ethanol",
  "opsin_result": {
    "smiles": "CCO",
    "stdinchi": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
  }
}
```

### Result Kinds

- `validated_structure`
- `invalid_structure`
- `dependency_missing`
- `invalid_request`
