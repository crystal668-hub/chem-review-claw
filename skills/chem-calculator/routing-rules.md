# Chem Calculator Routing Rules

- Use `molar_mass.py` when a calculation depends on formula mass or molecular weight.
- Use `stoichiometry.py` for limiting reagent, reaction yield, combustion analysis, and simple mole/mass conversions around a balanced reaction.
- Use `concentration.py` for dilution, mixing, and simple molarity bookkeeping.
- Use `ksp_solver.py` when the task asks whether precipitation occurs or what residual dissolved ion concentration remains.
- Use `acid_base_solver.py` for pH, pOH, weak acid/base approximations, and Henderson-Hasselbalch style buffer checks.
- Use `gas_law.py` for ideal gas law and Dalton-law style partial pressure calculations.
- Use `thermo_solver.py` for Gibbs free energy calculations, equilibrium constants from thermodynamics, and unit-normalized thermo arithmetic.
- Use `redox_balance.py` for oxidation-state assignment and simple electron-count checks between related species.
- Use `electrochemistry.py` for Nernst equation and Faraday electrolysis calculations.
- Use `unit_convert.py` whenever units differ across givens or the target answer unit needs normalization.
- Use `answer_check.py` after a candidate answer exists and needs an independent unit/tolerance/rounding check.

If a request falls outside a documented operation mode, the script should return structured `partial` or `error` JSON instead of guessing.
