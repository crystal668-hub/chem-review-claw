# PubChem Script Contracts

## Shared CLI

All scripts support:

```bash
python3 <skill-root>/scripts/<capability>.py \
  --request-json /path/to/request.json \
  --output-dir /tmp/<skill-out> \
  --json
```

- `--request-json`: required request payload path
- `--output-dir`: required directory for artifacts and result JSON
- `--json`: print the same top-level payload written to disk

Each script writes one stable result file in `--output-dir`.

## Shared Result Shape

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

- `success`: usable result returned
- `partial`: some useful data returned, but ambiguity or provider degradation
  remains
- `error`: no usable result returned

## Request Examples

### `name_to_cid.py`

```json
{
  "query": "aspirin",
  "max_candidates": 5,
  "timeout_seconds": 8.0,
  "retry_attempts": 1
}
```

### `cid_to_properties.py`

```json
{
  "cids": [2244],
  "properties": [
    "MolecularFormula",
    "MolecularWeight",
    "CanonicalSMILES",
    "IsomericSMILES",
    "InChI",
    "InChIKey",
    "Charge"
  ]
}
```

### `synonyms.py`

```json
{
  "cid": 2244,
  "max_synonyms": 20
}
```

### `formula_search.py`

```json
{
  "formula": "C9H8O4",
  "max_candidates": 10
}
```

### `similarity_search.py`

```json
{
  "query_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
  "threshold": 95,
  "max_records": 5
}
```

### `compound_summary.py`

```json
{
  "query": "aspirin",
  "max_candidates": 3,
  "synonym_limit": 10
}
```

## Output Files

- `name_to_cid.py` -> `name_to_cid_result.json`
- `cid_to_properties.py` -> `cid_to_properties_result.json`
- `synonyms.py` -> `synonyms_result.json`
- `formula_search.py` -> `formula_search_result.json`
- `similarity_search.py` -> `similarity_search_result.json`
- `compound_summary.py` -> `compound_summary_result.json`

## Failure and Health Rules

- Every network-backed response records provider URL, HTTP status, elapsed time,
  timeout status, and parse status in `source_trace` and `provider_health`.
- Retry is bounded and only used for transient HTTP status or transport errors.
- PubChem-only scripts do not silently fall back to any other provider.
