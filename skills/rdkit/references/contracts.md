# RDKit Skill Contracts

## Shared CLI Contract

Every script supports:

```bash
python3 <skill-root>/scripts/<capability>.py \
  --request-json /path/to/request.json \
  --output-dir /tmp/<skill-out> \
  --json
```

Behavior:

- `--request-json` is required and is the canonical input.
- `--output-dir` is required and is created automatically.
- Each script writes one stable output file: `result.json`.
- `--json` prints the exact same top-level payload written to `result.json`.
- Errors are returned as structured JSON. Scripts do not require network access.

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

Status meaning:

- `success`: the requested capability completed with usable output
- `partial`: output is usable but limited, ambiguous, or explicitly heuristic
- `error`: no usable output is available for the requested operation

## Structure Request Shape

Single-molecule scripts accept:

```json
{
  "molecule": {
    "format": "smiles|inchi",
    "value": "CCO"
  }
}
```

Optional flags:

- `strip_atom_maps`: boolean, used by `canonicalize.py`
- `num_conformers`: integer, used by `conformer_embed.py`
- `random_seed`: integer, used by `conformer_embed.py`

Multi-molecule scripts:

- `substructure.py`
  - `molecules`: list of structure objects
  - `query`: `{"smarts": "..."}` or `{"name": "<builtin>"}` 
- `similarity.py`
  - `query`: structure object
  - `candidates`: list of structure objects with optional `id`
- `reaction_smarts.py`
  - `reaction_smarts`: reaction SMARTS string
  - `reactants`: list of structure objects

## Capability Notes

- `canonicalize.py`: returns canonical and isomeric SMILES plus validation data.
- `descriptors.py`: returns a compact numeric descriptor summary.
- `functional_groups.py`: returns a curated match list for common functional
  groups used in first-batch chemistry reasoning.
- `substructure.py`: returns per-molecule match booleans and atom-index matches.
- `rings_aromaticity.py`: returns ring sizes, aromatic ring counts, aromatic
  atom counts, fused-ring hints, and heteroaromatic counts.
- `stereochemistry.py`: returns chiral-center and double-bond stereo summaries.
- `similarity.py`: ranks supplied candidates using deterministic Morgan
  fingerprint similarity.
- `reaction_smarts.py`: applies an explicit reaction transform and returns
  product candidates when the reactants match.
- `conformer_embed.py`: embeds and optimizes available conformers locally.
- `nmr_symmetry_heuristics.py`: returns graph-symmetry equivalence classes and
  always includes warnings about heuristic limitations.

## Failure Modes

- Invalid JSON request: structured `error`
- Missing required request fields: structured `error`
- Unsupported structure format: structured `error`
- Invalid SMILES or InChI: structured `error`
- Missing RDKit dependency: structured `error`
- No reaction application result: structured `error`
- Heuristic ambiguity or limited interpretation: structured `partial`
