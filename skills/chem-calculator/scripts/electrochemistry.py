from __future__ import annotations

import math

from chemcalc_core import (
    ChemCalcError,
    FARADAY_CONSTANT_C_PER_MOL,
    IDEAL_GAS_CONSTANT_J_PER_MOL_K,
    as_float,
    build_result,
    require_text,
    run_cli,
    set_success,
)


def handle(request: dict[str, object]) -> dict[str, object]:
    result = build_result(request, "electrochemistry")
    operation = require_text(request.get("operation"), "operation")
    if operation == "nernst":
        standard_potential = as_float(request.get("standard_potential_v"), "standard_potential_v")
        electrons = as_float(request.get("electrons_transferred"), "electrons_transferred")
        quotient = as_float(request.get("reaction_quotient"), "reaction_quotient")
        temperature = as_float(request.get("temperature_k"), "temperature_k")
        potential = standard_potential - (IDEAL_GAS_CONSTANT_J_PER_MOL_K * temperature / (electrons * FARADAY_CONSTANT_C_PER_MOL)) * math.log(quotient)
        return set_success(result, {"cell_potential_v": round(potential, 6)})
    if operation == "faraday":
        current = as_float(request.get("current_a"), "current_a")
        time_seconds = as_float(request.get("time_s"), "time_s")
        molar_mass = as_float(request.get("molar_mass_g_per_mol"), "molar_mass_g_per_mol")
        electrons_per_mole = as_float(request.get("electrons_per_mole"), "electrons_per_mole")
        charge = current * time_seconds
        deposited_mass = charge * molar_mass / (electrons_per_mole * FARADAY_CONSTANT_C_PER_MOL)
        reported_mass = math.floor(deposited_mass * 1000.0) / 1000.0
        return set_success(
            result,
            {
                "charge_c": round(charge, 6),
                "deposited_mass_g": round(reported_mass, 6),
            },
        )
    raise ChemCalcError(f"unsupported electrochemistry operation `{operation}`", code="unsupported_request", status="partial")


if __name__ == "__main__":
    raise SystemExit(run_cli("electrochemistry", handle))
