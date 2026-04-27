from __future__ import annotations

from chemcalc_core import as_float, build_result, convert_value, normalize_unit, require_text, run_cli, set_success


def handle(request: dict[str, object]) -> dict[str, object]:
    result = build_result(request, "unit_convert")
    operation = require_text(request.get("operation"), "operation")
    if operation != "convert":
        raise ValueError("unsupported operation")
    value = as_float(request.get("value"), "value")
    from_unit = normalize_unit(require_text(request.get("from_unit"), "from_unit"))
    to_unit = normalize_unit(require_text(request.get("to_unit"), "to_unit"))
    converted = convert_value(value, from_unit, to_unit)
    return set_success(
        result,
        {
            "value": round(converted, 6),
            "from_unit": from_unit,
            "to_unit": to_unit,
        },
    )


if __name__ == "__main__":
    raise SystemExit(run_cli("unit_convert", handle))
