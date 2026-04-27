from __future__ import annotations

from chemcalc_core import (
    ChemCalcError,
    as_float,
    build_result,
    convert_value,
    empirical_formula_from_moles,
    molar_mass_for_formula,
    normalize_unit,
    require_list,
    require_mapping,
    require_text,
    run_cli,
    set_success,
)


def _amount_to_moles(species: str, value: float, unit: str) -> float:
    canonical = normalize_unit(unit)
    if canonical == "mol":
        return value
    if canonical == "mmol":
        return convert_value(value, canonical, "mol")
    if canonical in {"g", "mg", "kg"}:
        mass_g = convert_value(value, canonical, "g")
        molar_mass, _ = molar_mass_for_formula(species)
        return mass_g / molar_mass
    raise ChemCalcError(
        f"unsupported stoichiometry amount unit `{unit}`",
        code="unsupported_unit",
        status="partial",
    )


def _moles_to_unit(species: str, moles: float, target_unit: str) -> float:
    canonical = normalize_unit(target_unit)
    if canonical == "mol":
        return moles
    if canonical == "mmol":
        return convert_value(moles, "mol", "mmol")
    if canonical in {"g", "mg", "kg"}:
        molar_mass, _ = molar_mass_for_formula(species)
        return convert_value(moles * molar_mass, "g", canonical)
    raise ChemCalcError(
        f"unsupported stoichiometry target unit `{target_unit}`",
        code="unsupported_unit",
        status="partial",
    )


def handle(request: dict[str, object]) -> dict[str, object]:
    result = build_result(request, "stoichiometry")
    operation = require_text(request.get("operation"), "operation")
    if operation == "limiting_reagent":
        reaction = require_mapping(request.get("reaction"), "reaction")
        reactants = require_list(reaction.get("reactants"), "reaction.reactants")
        products = require_list(reaction.get("products"), "reaction.products")
        known_amounts = require_list(request.get("known_amounts"), "known_amounts")
        target_species = require_text(request.get("target_species"), "target_species")
        target_unit = require_text(request.get("target_unit"), "target_unit")

        known_by_species: dict[str, float] = {}
        for index, item in enumerate(known_amounts):
            amount = require_mapping(item, f"known_amounts[{index}]")
            species = require_text(amount.get("species"), f"known_amounts[{index}].species")
            value = as_float(amount.get("value"), f"known_amounts[{index}].value")
            unit = require_text(amount.get("unit"), f"known_amounts[{index}].unit")
            known_by_species[species] = _amount_to_moles(species, value, unit)

        limiting_species = None
        limiting_extent = None
        for item in reactants:
            reactant = require_mapping(item, "reaction.reactants[]")
            species = require_text(reactant.get("species"), "reaction.reactants[].species")
            coefficient = as_float(reactant.get("coefficient"), "reaction.reactants[].coefficient")
            if species not in known_by_species:
                raise ChemCalcError(f"missing known amount for reactant `{species}`", code="invalid_request")
            extent = known_by_species[species] / coefficient
            if limiting_extent is None or extent < limiting_extent:
                limiting_extent = extent
                limiting_species = species
        assert limiting_species is not None and limiting_extent is not None

        product_coeff = None
        for item in products:
            product = require_mapping(item, "reaction.products[]")
            species = require_text(product.get("species"), "reaction.products[].species")
            if species == target_species:
                product_coeff = as_float(product.get("coefficient"), "reaction.products[].coefficient")
                break
        if product_coeff is None:
            raise ChemCalcError(f"target species `{target_species}` is not a listed product", code="invalid_request")
        product_moles = limiting_extent * product_coeff
        return set_success(
            result,
            {
                "limiting_reagent": {
                    "species": limiting_species,
                    "reaction_extent_mol": round(limiting_extent, 6),
                },
                "product_amount": {
                    "species": target_species,
                    "value": round(_moles_to_unit(target_species, product_moles, target_unit), 6),
                    "unit": normalize_unit(target_unit),
                },
            },
        )

    if operation == "combustion_analysis":
        sample_mass = as_float(request.get("sample_mass_g"), "sample_mass_g")
        products = require_mapping(request.get("products"), "products")
        carbon_dioxide_mass = as_float(products.get("CO2_mass_g"), "products.CO2_mass_g")
        water_mass = as_float(products.get("H2O_mass_g"), "products.H2O_mass_g")
        co2_molar_mass, _ = molar_mass_for_formula("CO2")
        h2o_molar_mass, _ = molar_mass_for_formula("H2O")
        carbon_moles = carbon_dioxide_mass / co2_molar_mass
        hydrogen_moles = 2.0 * (water_mass / h2o_molar_mass)
        oxygen_mass = sample_mass - carbon_moles * 12.011 - hydrogen_moles * 1.008
        if oxygen_mass < -1.0e-6:
            raise ChemCalcError("sample mass is inconsistent with combustion products", code="invalid_request")
        oxygen_moles = max(0.0, oxygen_mass / 15.999)
        empirical_formula = empirical_formula_from_moles(
            {
                "C": carbon_moles,
                "H": hydrogen_moles,
                "O": oxygen_moles,
            }
        )
        return set_success(
            result,
            {
                "empirical_formula": empirical_formula,
                "element_moles": {
                    "C": round(carbon_moles, 6),
                    "H": round(hydrogen_moles, 6),
                    "O": round(oxygen_moles, 6),
                },
            },
        )

    if operation == "percent_yield":
        theoretical = require_mapping(request.get("theoretical_yield"), "theoretical_yield")
        actual = require_mapping(request.get("actual_yield"), "actual_yield")
        theoretical_value = as_float(theoretical.get("value"), "theoretical_yield.value")
        theoretical_unit = require_text(theoretical.get("unit"), "theoretical_yield.unit")
        actual_value = as_float(actual.get("value"), "actual_yield.value")
        actual_unit = require_text(actual.get("unit"), "actual_yield.unit")
        actual_in_theoretical = convert_value(actual_value, actual_unit, theoretical_unit)
        percent_yield = actual_in_theoretical / theoretical_value * 100.0
        return set_success(result, {"percent_yield": round(percent_yield, 6)})

    raise ChemCalcError(f"unsupported stoichiometry operation `{operation}`", code="unsupported_request", status="partial")


if __name__ == "__main__":
    raise SystemExit(run_cli("stoichiometry", handle))
