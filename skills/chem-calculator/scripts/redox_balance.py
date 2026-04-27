from __future__ import annotations

from chemcalc_core import (
    ChemCalcError,
    build_result,
    oxidation_states_for_formula,
    require_text,
    run_cli,
    set_success,
)


def handle(request: dict[str, object]) -> dict[str, object]:
    result = build_result(request, "redox_balance")
    operation = require_text(request.get("operation"), "operation")
    if operation == "oxidation_states":
        formula = require_text(request.get("formula"), "formula")
        return set_success(result, {"formula": formula, "oxidation_states": oxidation_states_for_formula(formula)})
    if operation == "electron_count":
        reactant_formula = require_text(request.get("reactant_formula"), "reactant_formula")
        product_formula = require_text(request.get("product_formula"), "product_formula")
        reactant_states = oxidation_states_for_formula(reactant_formula)
        product_states = oxidation_states_for_formula(product_formula)
        shared_elements = sorted(set(reactant_states) & set(product_states))
        differing = [element for element in shared_elements if reactant_states[element] != product_states[element]]
        if len(differing) != 1:
            raise ChemCalcError(
                "electron counting supports one changing shared element",
                code="unsupported_request",
                status="partial",
            )
        element = differing[0]
        electrons = abs(product_states[element] - reactant_states[element])
        return set_success(
            result,
            {
                "element": element,
                "reactant_oxidation_state": reactant_states[element],
                "product_oxidation_state": product_states[element],
                "electrons_transferred": electrons,
            },
        )
    raise ChemCalcError(f"unsupported redox operation `{operation}`", code="unsupported_request", status="partial")


if __name__ == "__main__":
    raise SystemExit(run_cli("redox_balance", handle))
