from __future__ import annotations

from chemcalc_core import (
    IDEAL_GAS_CONSTANT_L_ATM_PER_MOL_K,
    as_float,
    build_result,
    require_text,
    run_cli,
    set_success,
)


def handle(request: dict[str, object]) -> dict[str, object]:
    result = build_result(request, "gas_law")
    operation = require_text(request.get("operation"), "operation")
    if operation == "ideal_gas":
        solve_for = require_text(request.get("solve_for"), "solve_for")
        pressure = as_float(request.get("pressure_atm"), "pressure_atm")
        volume = as_float(request.get("volume_l"), "volume_l")
        temperature = as_float(request.get("temperature_k"), "temperature_k")
        if solve_for != "moles":
            raise ValueError("unsupported solve_for")
        moles = pressure * volume / (IDEAL_GAS_CONSTANT_L_ATM_PER_MOL_K * temperature)
        return set_success(result, {"moles": round(moles, 6)})
    if operation == "partial_pressure":
        total_pressure = as_float(request.get("total_pressure_atm"), "total_pressure_atm")
        mole_fraction = as_float(request.get("mole_fraction"), "mole_fraction")
        return set_success(result, {"partial_pressure_atm": round(total_pressure * mole_fraction, 6)})
    raise ValueError("unsupported operation")


if __name__ == "__main__":
    raise SystemExit(run_cli("gas_law", handle))
