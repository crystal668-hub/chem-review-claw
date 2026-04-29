Required sibling skills:

- `debateclaw-v1`
- `paper-retrieval`
- `paper-access`
- `paper-parse`
- `paper-rerank`
- `rdkit`
- `pubchem`
- `opsin`
- `chem-calculator`

Resolve these bundles from the same parent `skills/` directory as this bundle.
Do not assume repository-relative paths.

Routing table:

- numeric / stoichiometric / equilibrium / unit / concentration / acid-base / gas-law / electrochemistry / formula-math trigger -> `chem-calculator`
- SMILES / formula / ring / unsaturation / chirality / stereochemistry / substructure / conformer / structure-constraint trigger -> `rdkit`
- IUPAC / systematic name trigger -> `opsin`, then `rdkit` for structure-sensitive validation
- common name / CID / synonym / property / public compound identifier trigger -> `pubchem`, then `rdkit` for structure-sensitive validation
- literature or external-fact trigger not covered by local chemistry providers -> `paper-retrieval` -> `paper-access` -> `paper-rerank` -> `paper-parse`

When a route is triggered, use the listed skill instead of relying only on unaided reasoning. Record the generated provider result JSON artifact path or a structured `tool_trace` entry. If you skip a triggered route, record a `submission_trace` entry with `status: skipped`, the `trigger`, the `reason`, and the residual `risk`.
