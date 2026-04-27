from __future__ import annotations

from chemcalc_core import ChemCalcError, as_float, build_result, parse_formula, require_mapping, require_text, run_cli, set_success


def _species_stoichiometric_count(solid_formula: str, ion_species: str) -> int:
    solid_counts, _ = parse_formula(solid_formula)
    ion_counts, _ = parse_formula(ion_species)
    if len(ion_counts) != 1:
        raise ChemCalcError(
            f"residual concentration supports monoatomic ion species, got `{ion_species}`",
            code="unsupported_request",
            status="partial",
        )
    element = next(iter(ion_counts))
    if element not in solid_counts:
        raise ChemCalcError(f"ion `{ion_species}` is not present in solid `{solid_formula}`", code="invalid_request")
    return solid_counts[element]


def handle(request: dict[str, object]) -> dict[str, object]:
    result = build_result(request, "ksp_solver")
    operation = require_text(request.get("operation"), "operation")
    ksp = as_float(request.get("ksp"), "ksp")
    if operation == "precipitation_check":
        ion_product = require_mapping(request.get("ion_product"), "ion_product")
        stoichiometry = require_mapping(request.get("stoichiometry"), "stoichiometry")
        q_value = 1.0
        for species, concentration in ion_product.items():
            exponent = as_float(stoichiometry.get(species), f"stoichiometry.{species}")
            q_value *= as_float(concentration, f"ion_product.{species}") ** exponent
        return set_success(
            result,
            {
                "reaction_quotient": q_value,
                "ksp": ksp,
                "will_precipitate": q_value > ksp,
            },
        )
    if operation == "residual_concentration":
        solid = require_text(request.get("solid"), "solid")
        known_ion = require_mapping(request.get("known_ion"), "known_ion")
        known_species = require_text(known_ion.get("species"), "known_ion.species")
        known_concentration = as_float(known_ion.get("concentration_molar"), "known_ion.concentration_molar")
        unknown_species = require_text(request.get("unknown_ion_species"), "unknown_ion_species")
        known_count = _species_stoichiometric_count(solid, known_species)
        unknown_count = _species_stoichiometric_count(solid, unknown_species)
        residual_unknown = (ksp / (known_concentration ** known_count)) ** (1.0 / unknown_count)
        label = "".join(character for character in unknown_species.lower() if character.isalpha())
        return set_success(
            result,
            {
                f"residual_{label}_molar": round(residual_unknown, 12),
                "known_ion_species": known_species,
                "unknown_ion_species": unknown_species,
            },
        )
    raise ChemCalcError(f"unsupported ksp operation `{operation}`", code="unsupported_request", status="partial")


if __name__ == "__main__":
    raise SystemExit(run_cli("ksp_solver", handle))
