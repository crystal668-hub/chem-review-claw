from __future__ import annotations

import math

from chemcalc_core import ChemCalcError, WATER_ION_PRODUCT, as_float, build_result, require_text, run_cli, set_success


def handle(request: dict[str, object]) -> dict[str, object]:
    result = build_result(request, "acid_base_solver")
    operation = require_text(request.get("operation"), "operation")
    if operation == "strong_acid_ph":
        concentration = as_float(request.get("acid_concentration_molar"), "acid_concentration_molar")
        ph = -math.log10(concentration)
        return set_success(result, {"ph": round(ph, 6), "poh": round(14.0 - ph, 6)})
    if operation == "weak_base_ph":
        concentration = as_float(request.get("base_concentration_molar"), "base_concentration_molar")
        kb = as_float(request.get("kb"), "kb")
        hydroxide = math.sqrt(kb * concentration)
        poh = -math.log10(hydroxide)
        ph = -math.log10(WATER_ION_PRODUCT) + math.log10(hydroxide)
        return set_success(result, {"ph": round(ph, 6), "poh": round(poh, 6), "hydroxide_molar": round(hydroxide, 8)})
    if operation == "buffer_ph":
        pka = as_float(request.get("pka"), "pka")
        acid_concentration = as_float(request.get("acid_concentration_molar"), "acid_concentration_molar")
        base_concentration = as_float(request.get("base_concentration_molar"), "base_concentration_molar")
        ph = pka + math.log10(base_concentration / acid_concentration)
        return set_success(result, {"ph": round(ph, 6), "pka": pka})
    raise ChemCalcError(f"unsupported acid/base operation `{operation}`", code="unsupported_request", status="partial")


if __name__ == "__main__":
    raise SystemExit(run_cli("acid_base_solver", handle))
