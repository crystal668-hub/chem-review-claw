from __future__ import annotations

import argparse
import json
import math
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable


RESULT_FILENAME = "result.json"
FARADAY_CONSTANT_C_PER_MOL = 96485.33212
IDEAL_GAS_CONSTANT_L_ATM_PER_MOL_K = 0.082057338
IDEAL_GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
WATER_ION_PRODUCT = 1.0e-14

ELEMENT_MASSES = {
    "Ag": 107.8682,
    "C": 12.011,
    "Cl": 35.45,
    "Cu": 63.546,
    "Fe": 55.845,
    "H": 1.00798,
    "Mn": 54.938044,
    "N": 14.007,
    "O": 15.9994,
    "S": 32.065,
}

COMMON_OXIDATION_STATES = {
    "F": -1,
    "O": -2,
    "H": 1,
    "Li": 1,
    "Na": 1,
    "K": 1,
    "Rb": 1,
    "Cs": 1,
    "Be": 2,
    "Mg": 2,
    "Ca": 2,
    "Sr": 2,
    "Ba": 2,
    "Cl": -1,
    "Br": -1,
    "I": -1,
}

UNIT_ALIASES = {
    "a": "A",
    "amp": "A",
    "amps": "A",
    "ampere": "A",
    "amperes": "A",
    "atm": "atm",
    "c": "C",
    "degc": "C",
    "g": "g",
    "gram": "g",
    "grams": "g",
    "j/mol": "J/mol",
    "j/mol/k": "J/mol/K",
    "j": "J",
    "k": "K",
    "kelvin": "K",
    "kg": "kg",
    "kj": "kJ",
    "kj/mol": "kJ/mol",
    "kj/mol/k": "kJ/mol/K",
    "l": "L",
    "liter": "L",
    "liters": "L",
    "litre": "L",
    "litres": "L",
    "m": "mol/L",
    "mg": "mg",
    "min": "min",
    "minute": "min",
    "minutes": "min",
    "ml": "mL",
    "mmol": "mmol",
    "mmol/l": "mmol/L",
    "mol": "mol",
    "mol/l": "mol/L",
    "pa": "Pa",
    "s": "s",
    "sec": "s",
    "second": "s",
    "seconds": "s",
    "torr": "torr",
    "kpa": "kPa",
}

LINEAR_UNITS = {
    "A": ("current", 1.0),
    "J": ("energy", 1.0),
    "J/mol": ("molar_energy", 1.0),
    "J/mol/K": ("molar_entropy", 1.0),
    "K": ("temperature", 1.0),
    "L": ("volume", 1.0),
    "Pa": ("pressure", 1.0),
    "atm": ("pressure", 101325.0),
    "g": ("mass", 1.0),
    "kJ": ("energy", 1000.0),
    "kJ/mol": ("molar_energy", 1000.0),
    "kJ/mol/K": ("molar_entropy", 1000.0),
    "kPa": ("pressure", 1000.0),
    "kg": ("mass", 1000.0),
    "mL": ("volume", 0.001),
    "mg": ("mass", 0.001),
    "min": ("time", 60.0),
    "mmol": ("amount", 0.001),
    "mmol/L": ("concentration", 0.001),
    "mol": ("amount", 1.0),
    "mol/L": ("concentration", 1.0),
    "s": ("time", 1.0),
    "torr": ("pressure", 101325.0 / 760.0),
}


class ChemCalcError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_request",
        status: str = "error",
        primary_result: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status
        self.primary_result = primary_result or {}


def build_result(request: dict[str, Any], tool_name: str) -> dict[str, Any]:
    return {
        "status": "error",
        "request": request,
        "primary_result": {},
        "candidates": [],
        "diagnostics": [],
        "warnings": [],
        "errors": [],
        "tool_trace": [
            {
                "tool": tool_name,
                "mode": "local",
            }
        ],
        "source_trace": [
            {
                "kind": "local_reference",
                "name": "chem-calculator-first-batch",
            }
        ],
        "provider_health": {
            "provider": "local",
            "mode": "stdlib",
            "available": True,
        },
    }


def add_message(result: dict[str, Any], bucket: str, code: str, message: str) -> None:
    result[bucket].append({"code": code, "message": message})


def set_success(
    result: dict[str, Any],
    primary_result: dict[str, Any],
    *,
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result["status"] = "success"
    result["primary_result"] = primary_result
    if candidates is not None:
        result["candidates"] = candidates
    return result


def set_partial(
    result: dict[str, Any],
    primary_result: dict[str, Any],
    *,
    code: str,
    message: str,
) -> dict[str, Any]:
    result["status"] = "partial"
    result["primary_result"] = primary_result
    add_message(result, "warnings", code, message)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="chem-calculator local chemistry tool")
    parser.add_argument("--request-json", required=True, help="Path to request JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for result output")
    parser.add_argument("--json", action="store_true", help="Print result JSON to stdout")
    return parser.parse_args(argv)


def dump_result(payload: dict[str, Any], output_dir: str | Path, emit_json: bool) -> int:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / RESULT_FILENAME
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    result_path.write_text(text, encoding="utf-8")
    if emit_json:
        sys.stdout.write(text)
        sys.stdout.write("\n")
    return 0


def run_cli(tool_name: str, handler: Callable[[dict[str, Any]], dict[str, Any]], argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    request = json.loads(Path(args.request_json).read_text(encoding="utf-8"))
    try:
        payload = handler(request)
    except ChemCalcError as exc:
        payload = build_result(request, tool_name)
        payload["status"] = exc.status
        payload["primary_result"] = exc.primary_result
        add_message(payload, "errors" if exc.status == "error" else "warnings", exc.code, exc.message)
    except Exception as exc:  # pragma: no cover - defensive path
        payload = build_result(request, tool_name)
        add_message(payload, "errors", "internal_error", str(exc))
    return dump_result(payload, args.output_dir, args.json)


def require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ChemCalcError(f"`{field_name}` must be an object", code="invalid_request")
    return value


def require_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ChemCalcError(f"`{field_name}` must be a list", code="invalid_request")
    return value


def as_float(value: Any, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ChemCalcError(f"`{field_name}` must be numeric", code="invalid_request") from exc
    if math.isnan(numeric) or math.isinf(numeric):
        raise ChemCalcError(f"`{field_name}` must be finite", code="invalid_request")
    return numeric


def require_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ChemCalcError(f"`{field_name}` is required", code="invalid_request")
    return text


def normalize_unit(unit: str) -> str:
    normalized = str(unit or "").strip()
    if not normalized:
        raise ChemCalcError("unit is required", code="invalid_request")
    alias = UNIT_ALIASES.get(normalized.lower())
    return alias or normalized


def unit_dimension(unit: str) -> str:
    canonical = normalize_unit(unit)
    if canonical == "C":
        return "temperature"
    if canonical in LINEAR_UNITS:
        return LINEAR_UNITS[canonical][0]
    raise ChemCalcError(f"unsupported unit `{unit}`", code="unsupported_unit", status="partial")


def convert_value(value: float, from_unit: str, to_unit: str) -> float:
    origin = normalize_unit(from_unit)
    target = normalize_unit(to_unit)
    if origin == target:
        return float(value)
    if {origin, target} <= {"C", "K"}:
        if origin == "C" and target == "K":
            return float(value) + 273.15
        if origin == "K" and target == "C":
            return float(value) - 273.15
    if origin == "C" or target == "C":
        raise ChemCalcError(
            f"unsupported temperature conversion `{from_unit}` -> `{to_unit}`",
            code="unsupported_unit",
            status="partial",
        )
    if origin not in LINEAR_UNITS or target not in LINEAR_UNITS:
        raise ChemCalcError(f"unsupported unit `{from_unit}` or `{to_unit}`", code="unsupported_unit", status="partial")
    origin_dimension, origin_scale = LINEAR_UNITS[origin]
    target_dimension, target_scale = LINEAR_UNITS[target]
    if origin_dimension != target_dimension:
        raise ChemCalcError(
            f"incompatible unit conversion `{from_unit}` -> `{to_unit}`",
            code="incompatible_unit",
            status="partial",
        )
    base_value = float(value) * origin_scale
    return base_value / target_scale


def extract_quantity(raw_value: Any, *, default_unit: str | None = None, field_name: str) -> tuple[float, str]:
    if isinstance(raw_value, dict):
        value = as_float(raw_value.get("value"), f"{field_name}.value")
        unit = normalize_unit(raw_value.get("unit"))
        return value, unit
    if default_unit is None:
        raise ChemCalcError(f"`{field_name}` must include a unit", code="invalid_request")
    return as_float(raw_value, field_name), normalize_unit(default_unit)


def normalize_species_label(species: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", str(species or "").strip())
    return cleaned.lower()


def _split_formula_and_charge(formula: str) -> tuple[str, int]:
    text = require_text(formula, "formula")
    if "^" in text:
        match = re.search(r"\^(\d+)?([+-])$", text)
        if match:
            magnitude = int(match.group(1) or "1")
            sign = 1 if match.group(2) == "+" else -1
            return text[: match.start()], sign * magnitude
    if text.endswith(("+", "-")):
        sign = 1 if text[-1] == "+" else -1
        base = text[:-1]
        single_ion = re.fullmatch(r"([A-Z][a-z]?)(\d+)", base)
        if single_ion:
            return single_ion.group(1), sign * int(single_ion.group(2))
        return base, sign
    return text, 0


def _parse_formula_segment(segment: str, start_index: int = 0) -> tuple[dict[str, int], int]:
    counts: dict[str, int] = {}
    index = start_index
    while index < len(segment):
        char = segment[index]
        if char in ")]":
            return counts, index + 1
        if char in "([":
            inner_counts, next_index = _parse_formula_segment(segment, index + 1)
            multiplier, index = _parse_integer(segment, next_index)
            _merge_counts(counts, inner_counts, multiplier)
            continue
        if char.isupper():
            end_index = index + 1
            while end_index < len(segment) and segment[end_index].islower():
                end_index += 1
            symbol = segment[index:end_index]
            multiplier, index = _parse_integer(segment, end_index)
            counts[symbol] = counts.get(symbol, 0) + multiplier
            continue
        raise ChemCalcError(f"unsupported formula token `{char}` in `{segment}`", code="unsupported_formula", status="partial")
    return counts, index


def _parse_integer(segment: str, start_index: int) -> tuple[int, int]:
    end_index = start_index
    while end_index < len(segment) and segment[end_index].isdigit():
        end_index += 1
    if end_index == start_index:
        return 1, start_index
    return int(segment[start_index:end_index]), end_index


def _merge_counts(target: dict[str, int], source: dict[str, int], multiplier: int) -> None:
    for element, count in source.items():
        target[element] = target.get(element, 0) + count * multiplier


def parse_formula(formula: str) -> tuple[dict[str, int], int]:
    base_formula, charge = _split_formula_and_charge(formula)
    total_counts: dict[str, int] = {}
    for part in re.split(r"[·.]", base_formula):
        segment = part.strip()
        if not segment:
            continue
        leading_multiplier = 1
        match = re.match(r"^(\d+)(.*)$", segment)
        if match:
            leading_multiplier = int(match.group(1))
            segment = match.group(2)
        part_counts, next_index = _parse_formula_segment(segment)
        if next_index != len(segment):
            raise ChemCalcError(f"unsupported formula syntax in `{formula}`", code="unsupported_formula", status="partial")
        _merge_counts(total_counts, part_counts, leading_multiplier)
    if not total_counts:
        raise ChemCalcError(f"unable to parse formula `{formula}`", code="unsupported_formula", status="partial")
    return total_counts, charge


def molar_mass_for_formula(formula: str) -> tuple[float, dict[str, int]]:
    composition, _ = parse_formula(formula)
    missing = sorted(element for element in composition if element not in ELEMENT_MASSES)
    if missing:
        raise ChemCalcError(
            f"missing element data for {', '.join(missing)}",
            code="missing_element_data",
            status="partial",
            primary_result={"missing_elements": missing},
        )
    total = 0.0
    for element, count in composition.items():
        total += ELEMENT_MASSES[element] * count
    return total, composition


def empirical_formula_from_moles(moles_by_element: dict[str, float]) -> str:
    positive = {element: value for element, value in moles_by_element.items() if value > 1.0e-12}
    if not positive:
        raise ChemCalcError("empirical formula requires positive mole amounts", code="invalid_request")
    smallest = min(positive.values())
    ratios = {element: value / smallest for element, value in positive.items()}
    multiplier = 1
    integers: dict[str, int] | None = None
    for candidate_multiplier in range(1, 9):
        scaled: dict[str, int] = {}
        valid = True
        for element, ratio in ratios.items():
            scaled_ratio = ratio * candidate_multiplier
            rounded = round(scaled_ratio)
            if abs(scaled_ratio - rounded) > 0.05:
                valid = False
                break
            scaled[element] = int(rounded)
        if valid:
            multiplier = candidate_multiplier
            integers = scaled
            break
    if integers is None:
        raise ChemCalcError("could not infer empirical formula", code="unsupported_request", status="partial")
    gcd = 0
    for count in integers.values():
        gcd = math.gcd(gcd, count)
    if gcd > 1:
        integers = {element: count // gcd for element, count in integers.items()}
    ordered = []
    for element in sorted(integers.keys(), key=lambda item: ("CH".find(item) if item in {"C", "H"} else 10, item)):
        count = integers[element]
        ordered.append(f"{element}{'' if count == 1 else count}")
    return "".join(ordered)


def oxidation_states_for_formula(formula: str) -> dict[str, int]:
    composition, charge = parse_formula(formula)
    if len(composition) == 1:
        only_element = next(iter(composition))
        return {only_element: int(charge / composition[only_element])}
    known_total = 0
    unknown_elements = []
    states: dict[str, int] = {}
    for element, count in composition.items():
        if element in COMMON_OXIDATION_STATES:
            state = COMMON_OXIDATION_STATES[element]
            states[element] = state
            known_total += state * count
        else:
            unknown_elements.append(element)
    if len(unknown_elements) != 1:
        raise ChemCalcError(
            f"oxidation state solver supports one unknown element, got {len(unknown_elements)}",
            code="unsupported_request",
            status="partial",
        )
    unknown = unknown_elements[0]
    unknown_count = composition[unknown]
    numerator = charge - known_total
    if numerator % unknown_count != 0:
        ratio = Fraction(numerator, unknown_count)
        raise ChemCalcError(
            f"non-integer oxidation state for `{unknown}`: {ratio}",
            code="unsupported_request",
            status="partial",
        )
    states[unknown] = numerator // unknown_count
    return states


def count_significant_figures(value: Any) -> int:
    text = str(value).strip()
    if not text:
        return 0
    if "e" in text.lower():
        mantissa = text.lower().split("e", 1)[0]
        return count_significant_figures(mantissa)
    text = text.lstrip("+-")
    if "." in text:
        integer_part, fractional_part = text.split(".", 1)
        digits = (integer_part + fractional_part).lstrip("0")
        return len(digits)
    digits = text.lstrip("0")
    return len(digits.rstrip("0")) if digits else 0
