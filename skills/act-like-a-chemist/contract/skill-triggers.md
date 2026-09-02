# Provider Skill Capability Reference

## Purpose

This optional reference describes provider capabilities for concrete Atomic Coverage Checklist atoms. Whether to use a provider, which provider to select, and how deeply to explore remain model decisions. When a tool is used, its result can be classified as `supports`, `partially supports`, `contradicts`, or `only verifies an intermediate step`.

Provider output is scoped evidence, not a verdict. An unexecuted skill is not evidence. When a provider skill contributes, cite its output path, structured tool trace, or retrieved source in the answer or artifact trace.

Concrete provider skill names must come from the single-agent-exposed provider inventory in `workspace/skills/chemistry-routing-matrix.json`. The inventory is a machine-readable skill catalog, not a deterministic router; runtime/orchestration skills are not provider routes and must not be selected for checklist atoms.

## Capability Areas

Provider skills cover numeric calculation, molecular structure, literature, materials databases, spectra, proteins, MD, HPC, ML, drug safety, and other explicit domains in the prompt.

The selected domain must match the atom's evidence need:

- Numeric calculation atoms need deterministic calculation, unit conversion, or algebraic checking.
- Molecular structure atoms need molecular identity, name resolution, descriptors, substructure, stereochemistry, or public compound metadata.
- Literature atoms need source-specific paper, protocol, target, assay, material-source, or experimental-condition evidence.
- Materials database atoms need crystal/material structures, phase stability, computed material properties, or federated materials records.
- Spectra atoms need provided XRD/IR data, spectroscopic exchange formats, plots, images, tables, or structure visualization.
- Protein atoms need protein structure, predicted structure, or pathway evidence.
- MD atoms need simulation setup, trajectory analysis, force-field assignment, or topology generation.
- HPC atoms need engine-specific quantum/DFT input authoring, execution debugging, or output parsing.
- ML atoms need featurization, model training, inference, force-field prediction, structure generation, or model benchmarking.
- Drug safety atoms need bioactivity, drug-likeness, purchasability, ADMET, toxicology, regulatory, or discovery-pipeline evidence.

## Provider Choice

No provider order is mandatory. Local deterministic tools, external databases, specialized engines, and workflow skills offer different evidence types, costs, inputs, and outputs; select among them according to the task and available context.

Specialized capabilities may be relevant for named databases, supplied CIF/PDB/PDF/ZIP/checkpoint files, API-specific evidence, engine-specific input/output workflows, systematic literature review, or ML training and inference.

If a provider path fails twice for the same verification target, mark that atom `blocked` and continue with bounded reasoning. Do not debug tool invocation style during a benchmark run.

## Capability Reference

| Capability area | Commonly relevant skills | Capability distinctions |
| --- | --- | --- |
| Numeric calculation / equations / units | `chem-calculator`, `xtb-cli`, `cclib`, `qc-output-analysis` | Deterministic chemistry math, local xTB execution from XYZ, or parsing existing quantum chemistry outputs. |
| Molecular structure / molecular identity | `rdkit`, `opsin`, `pubchem`, `datamol`, `pubchem-database`, `chemistry-query`, `blue-obelisk`, `cml` | Structure analysis, name resolution, public records, batch processing, reaction workflows, and format interoperability. |
| Literature / source evidence | `paper-retrieval`, `paper-access`, `paper-parse`, `openalex-database`, `pubmed-database`, `literature-review`, `synthesize-literature` | Discovery, accessible-artifact resolution, parsing, bibliometrics, and synthesis are separate capabilities. |
| Materials database / crystals / solid-state records | `pymatgen`, `materials-project`, `jarvis`, `cod`, `oqmd`, `optimade`, `optimade-python-tools`, `cccbdb`, `molssi-qca` | Crystal handling, named database lookup, phase stability, benchmark data, archive data, and federated search. |
| Spectra / formats / visualization | `spectral-analysis`, `jcamp-dx`, `cif`, `crystal-viewer`, `xtal2png` | Provided spectra interpretation, exchange formats, CIF validation, interactive views, and image encodings. |
| Protein / biological structure / pathways | `pdb-database`, `alphafold-database`, `reactome-database` | Experimental structures, predicted structures, and pathway evidence. |
| MD / force fields / atomistic setup | `molecular-dynamics`, `openmm`, `open-forcefield-toolkit`, `atb`, `ase` | Trajectory analysis, simulation setup, parameterization, topology generation, and atomistic conversion. |
| HPC / quantum / DFT workflows | `hpc-gaussian`, `hpc-orca`, `hpc-pyscf`, `hpc-xtb`, `hpc-cp2k`, `hpc-nwchem`, `hpc-vasp`, `hpc-quantum-espresso`, `q-chem`, `cclib`, `qc-output-analysis`, `xtb-cli` | Engine-specific authoring, execution, debugging, local xTB calculations, and output parsing. |
| ML / generative / property prediction | `matminer`, `molfeat`, `matbench`, `crabnet`, `modnet`, `xenonpy`, `matformer`, `chemprop`, `schnet`, `chgnet`, `mattersim`, `nequip`, `orb`, `mace`, `reann`, `torchmd-net`, `mattergen`, `diffcsp`, `crystalflow` | Featurization, training, inference, ML potentials, structure generation, screening, and benchmarking. |
| Drug safety / discovery / bioactivity | `chembl-database`, `medchem`, `zinc-database`, `tooluniverse-chemical-compound-retrieval`, `tooluniverse-chemical-safety`, `tooluniverse-small-molecule-discovery` | Bioactivity, drug-likeness, purchasability, ADMET, toxicology, and discovery workflows. |
| Workflow / orchestration / runtime | `benchmark-cleanroom`, `debateclaw-v1`, `chemqa-review` | Benchmark and ChemQA infrastructure; these bundles are excluded from single-agent provider exposure. |
