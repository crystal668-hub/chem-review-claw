# OPSIN Routing Rules

- Use `name_to_structure.py` for one clear systematic or IUPAC-like name.
- Use `batch_name_to_structure.py` when multiple systematic names or answer options should be resolved consistently.
- Use `parse_diagnostics.py` after OPSIN failure when the agent needs to distinguish unsupported syntax, ambiguity, malformed input, or non-systematic naming.
- Use `validate_with_rdkit.py` after any successful OPSIN structure before structural reasoning or downstream cheminformatics.
- Do not use OPSIN as a synonym or fact database.
- Prefer `pubchem` over OPSIN for trivial names, trade names, drug names, abbreviations, minerals, materials shorthand, and broad alias lookup.
