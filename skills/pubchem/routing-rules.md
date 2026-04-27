# PubChem Routing Rules

- Use `name_to_cid.py` when the prompt gives a common name, trivial name, drug
  name, material name, or free-form synonym.
- Use `cid_to_properties.py` when a PubChem CID is already known or a previous
  PubChem call returned candidate CIDs.
- Use `synonyms.py` when alias matching across prompt text, options, or cited
  sources matters.
- Use `formula_search.py` when the prompt provides only a molecular formula and
  asks for identity hints or candidate structures.
- Use `similarity_search.py` when a known structure should be compared against
  public PubChem analogs.
- Use `compound_summary.py` for a compact agent-facing lookup that resolves a
  single compound and includes basic metadata in one provider-only payload.
- After any PubChem structure lookup, call `rdkit canonicalize.py` or an
  equivalent RDKit validation step before using the structure in reasoning.
