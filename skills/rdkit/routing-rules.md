# RDKit Routing Rules

- Use `canonicalize.py` before any downstream RDKit operation when the input is
  a raw SMILES, InChI, or externally sourced structure.
- Use `descriptors.py` for formula, exact mass, molecular weight, charge,
  donor/acceptor counts, TPSA, logP, and quick molecule summaries.
- Use `functional_groups.py` when the question asks about chemical class,
  reactive handles, polymerizable groups, donor/acceptor behavior, or
  structure-driven option elimination.
- Use `substructure.py` when the prompt includes a structural motif, SMARTS
  constraint, or a named local motif check.
- Use `rings_aromaticity.py` for aromaticity, ring-system comparison, fused
  rings, and heteroaromatic analysis.
- Use `stereochemistry.py` for chirality, E/Z checks, enantiomer or
  diastereomer reasoning, and unspecified stereo detection.
- Use `similarity.py` for ranking supplied candidate molecules against a known
  structure using deterministic fingerprint similarity.
- Use `reaction_smarts.py` for reaction compatibility, product plausibility,
  and reaction-option filtering with explicit structural transforms.
- Use `conformer_embed.py` only when approximate 3D geometry matters; do not
  use it for name lookup or simple formula questions.
- Use `nmr_symmetry_heuristics.py` only as a heuristic. Its output should be
  treated as graph-symmetry guidance, not definitive NMR assignment.
