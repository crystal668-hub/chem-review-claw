from __future__ import annotations

import math

from chemcalc_core import (
    IDEAL_GAS_CONSTANT_J_PER_MOL_K,
    as_float,
    build_result,
    convert_value,
    extract_quantity,
    require_text,
    run_cli,
    set_success,
)


def handle(request: dict[str, object]) -> dict[str, object]:
    result = build_result(request, "thermo_solver")
    operation = require_text(request.get("operation"), "operation")
    if operation == "delta_g":
        if "delta_h" in request:
            delta_h_value, delta_h_unit = extract_quantity(request.get("delta_h"), field_name="delta_h")
            delta_s_value, delta_s_unit = extract_quantity(request.get("delta_s"), field_name="delta_s")
            temperature_value, temperature_unit = extract_quantity(request.get("temperature"), field_name="temperature")
        else:
            delta_h_value, delta_h_unit = extract_quantity(request.get("delta_h_kj_per_mol"), default_unit="kJ/mol", field_name="delta_h_kj_per_mol")
            delta_s_value, delta_s_unit = extract_quantity(request.get("delta_s_j_per_mol_k"), default_unit="J/mol/K", field_name="delta_s_j_per_mol_k")
            temperature_value, temperature_unit = extract_quantity(request.get("temperature_k"), default_unit="K", field_name="temperature_k")
        delta_h_j = convert_value(delta_h_value, delta_h_unit, "J/mol")
        delta_s_j = convert_value(delta_s_value, delta_s_unit, "J/mol/K")
        temperature_k = convert_value(temperature_value, temperature_unit, "K")
        delta_g_j = delta_h_j - temperature_k * delta_s_j
        return set_success(result, {"delta_g_kj_per_mol": round(delta_g_j / 1000.0, 6)})
    if operation == "equilibrium_constant_from_delta_g":
        delta_g_value, delta_g_unit = extract_quantity(request.get("delta_g_kj_per_mol"), default_unit="kJ/mol", field_name="delta_g_kj_per_mol")
        temperature_value, temperature_unit = extract_quantity(request.get("temperature_k"), default_unit="K", field_name="temperature_k")
        delta_g_j = convert_value(delta_g_value, delta_g_unit, "J/mol")
        temperature_k = convert_value(temperature_value, temperature_unit, "K")
        equilibrium_constant = math.exp(-delta_g_j / (IDEAL_GAS_CONSTANT_J_PER_MOL_K * temperature_k))
        return set_success(result, {"equilibrium_constant": round(equilibrium_constant, 6)})
    raise ValueError("unsupported operation")


if __name__ == "__main__":
    raise SystemExit(run_cli("thermo_solver", handle))
