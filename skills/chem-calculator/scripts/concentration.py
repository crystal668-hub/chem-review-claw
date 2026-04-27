from __future__ import annotations

from chemcalc_core import as_float, build_result, require_list, require_mapping, require_text, run_cli, set_success


def handle(request: dict[str, object]) -> dict[str, object]:
    result = build_result(request, "concentration")
    operation = require_text(request.get("operation"), "operation")
    if operation == "dilution":
        stock_concentration = as_float(request.get("stock_concentration_molar"), "stock_concentration_molar")
        stock_volume = as_float(request.get("stock_volume_l"), "stock_volume_l")
        final_volume = as_float(request.get("final_volume_l"), "final_volume_l")
        target = stock_concentration * stock_volume / final_volume
        return set_success(
            result,
            {
                "target_concentration_molar": round(target, 6),
                "moles_solute": round(stock_concentration * stock_volume, 6),
            },
        )
    if operation == "mix_solutions":
        solutions = require_list(request.get("solutions"), "solutions")
        total_moles = 0.0
        total_volume = 0.0
        for index, item in enumerate(solutions):
            solution = require_mapping(item, f"solutions[{index}]")
            concentration = as_float(solution.get("concentration_molar"), f"solutions[{index}].concentration_molar")
            volume = as_float(solution.get("volume_l"), f"solutions[{index}].volume_l")
            total_moles += concentration * volume
            total_volume += volume
        return set_success(
            result,
            {
                "total_moles": round(total_moles, 6),
                "final_volume_l": round(total_volume, 6),
                "final_concentration_molar": round(total_moles / total_volume, 6),
            },
        )
    raise ValueError("unsupported operation")


if __name__ == "__main__":
    raise SystemExit(run_cli("concentration", handle))
