from __future__ import annotations

from chemcalc_core import (
    ChemCalcError,
    as_float,
    build_result,
    convert_value,
    count_significant_figures,
    normalize_unit,
    require_mapping,
    run_cli,
    set_partial,
    set_success,
)


def handle(request: dict[str, object]) -> dict[str, object]:
    result = build_result(request, "answer_check")
    expected = require_mapping(request.get("expected"), "expected")
    candidate = require_mapping(request.get("candidate"), "candidate")
    tolerance = require_mapping(request.get("tolerance"), "tolerance")

    expected_value = as_float(expected.get("value"), "expected.value")
    expected_unit = normalize_unit(expected.get("unit"))
    candidate_value = as_float(candidate.get("value"), "candidate.value")
    candidate_unit = normalize_unit(candidate.get("unit"))

    try:
        candidate_converted = convert_value(candidate_value, candidate_unit, expected_unit)
    except ChemCalcError:
        return set_partial(
            result,
            {
                "is_correct": False,
                "failure_reason": "incompatible_unit",
                "expected_unit": expected_unit,
                "candidate_unit": candidate_unit,
            },
            code="incompatible_unit",
            message=f"candidate unit `{candidate_unit}` cannot be converted to `{expected_unit}`",
        )

    required_sig_figs = request.get("significant_figures")
    if required_sig_figs is not None:
        candidate_sig_figs = count_significant_figures(candidate.get("value"))
        if candidate_sig_figs < int(required_sig_figs):
            return set_partial(
                result,
                {
                    "is_correct": False,
                    "failure_reason": "rounding_mismatch",
                    "required_significant_figures": int(required_sig_figs),
                    "candidate_significant_figures": candidate_sig_figs,
                },
                code="rounding_mismatch",
                message="candidate answer does not preserve the required significant figures",
            )

    difference = abs(candidate_converted - expected_value)
    if "absolute" in tolerance:
        if difference > as_float(tolerance.get("absolute"), "tolerance.absolute"):
            return set_partial(
                result,
                {
                    "is_correct": False,
                    "failure_reason": "tolerance_mismatch",
                    "difference": round(difference, 6),
                },
                code="tolerance_mismatch",
                message="candidate answer is outside the allowed absolute tolerance",
            )
    elif "relative" in tolerance:
        relative = as_float(tolerance.get("relative"), "tolerance.relative")
        scale = abs(expected_value) if expected_value != 0 else 1.0
        if difference > relative * scale:
            return set_partial(
                result,
                {
                    "is_correct": False,
                    "failure_reason": "tolerance_mismatch",
                    "difference": round(difference, 6),
                },
                code="tolerance_mismatch",
                message="candidate answer is outside the allowed relative tolerance",
            )
    else:
        raise ChemCalcError("tolerance must include `absolute` or `relative`", code="invalid_request")

    return set_success(
        result,
        {
            "is_correct": True,
            "normalized_value": round(candidate_converted, 6),
            "unit": expected_unit,
        },
    )


if __name__ == "__main__":
    raise SystemExit(run_cli("answer_check", handle))
