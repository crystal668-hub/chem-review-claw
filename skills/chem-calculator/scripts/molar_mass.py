from __future__ import annotations

from chemcalc_core import build_result, molar_mass_for_formula, require_text, run_cli, set_success


def handle(request: dict[str, object]) -> dict[str, object]:
    result = build_result(request, "molar_mass")
    formula = require_text(request.get("formula"), "formula")
    molar_mass, composition = molar_mass_for_formula(formula)
    return set_success(
        result,
        {
            "formula": formula,
            "composition": composition,
            "molar_mass_g_per_mol": round(molar_mass, 6),
        },
    )


if __name__ == "__main__":
    raise SystemExit(run_cli("molar_mass", handle))
