#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROVIDER_TRACE_MODES = {"off", "audit", "enforce"}


@dataclass(frozen=True)
class ProviderTraceRequirement:
    skill: str
    trigger: str
    reason: str


@dataclass(frozen=True)
class ProviderTraceValidation:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    requirements: list[ProviderTraceRequirement] = field(default_factory=list)


def requirements_for_candidate(
    *,
    answer_kind: str,
    prompt: str = "",
    eval_kind: str = "",
    dataset: str = "",
) -> list[ProviderTraceRequirement]:
    text = " ".join([answer_kind, eval_kind, dataset, prompt]).lower()
    requirements: list[ProviderTraceRequirement] = []
    if answer_kind == "numeric_short_answer" or any(
        token in text
        for token in (
            "stoichiometric",
            "stoichiometry",
            "equilibrium",
            "acid-base",
            "gas-law",
            "unit-conversion",
            "concentration",
            "electrochemistry",
            "formula-math",
            "molar mass",
        )
    ):
        requirements.append(
            ProviderTraceRequirement(
                "chem-calculator",
                "numeric_or_formula_math",
                "Numeric or formula-math answer requires deterministic calculation trace.",
            )
        )
    if any(
        token in text
        for token in (
            "smiles",
            "inchi",
            "stereochemistry",
            "substructure",
            "conformer",
            "ring count",
            "chirality",
            "unsaturation",
            "structure-constraint",
        )
    ):
        requirements.append(
            ProviderTraceRequirement(
                "rdkit",
                "structure_check",
                "Structure-sensitive answer requires deterministic RDKit trace.",
            )
        )
    if any(token in text for token in ("iupac", "systematic name")):
        requirements.append(
            ProviderTraceRequirement(
                "opsin",
                "systematic_name",
                "Systematic-name answer requires OPSIN trace.",
            )
        )
        requirements.append(
            ProviderTraceRequirement(
                "rdkit",
                "opsin_structure_validation",
                "OPSIN-derived structures require RDKit validation trace.",
            )
        )
    if any(token in text for token in ("pubchem", "cid", "synonym", "common name")):
        requirements.append(
            ProviderTraceRequirement(
                "pubchem",
                "public_compound_lookup",
                "Public compound identity/property answer requires PubChem trace.",
            )
        )
    return _dedupe_requirements(requirements)


def validate_provider_traces(
    payload: dict[str, Any],
    *,
    answer_kind: str,
    prompt: str = "",
    eval_kind: str = "",
    dataset: str = "",
    require_existing_provider_paths: bool = False,
) -> ProviderTraceValidation:
    requirements = requirements_for_candidate(
        answer_kind=answer_kind,
        prompt=prompt,
        eval_kind=eval_kind,
        dataset=dataset,
    )
    errors: list[str] = []
    trace_entries = _provider_trace_entries(payload)
    for requirement in requirements:
        matching_entries = [entry for entry in trace_entries if _entry_mentions_skill(entry, requirement.skill)]
        if any(_entry_is_acceptable(entry, requirement, require_existing_provider_paths=require_existing_provider_paths) for entry in matching_entries):
            continue
        if matching_entries:
            errors.append(
                f"provider trace for `{requirement.skill}` is incomplete for trigger `{requirement.trigger}`; "
                "provide status success/partial with a provider result JSON artifact path or a skipped trace with trigger, reason, and risk."
            )
            continue
        errors.append(f"missing required provider trace for `{requirement.skill}` triggered by `{requirement.trigger}`: {requirement.reason}")
    return ProviderTraceValidation(errors=errors, warnings=[], requirements=requirements)


def _dedupe_requirements(requirements: list[ProviderTraceRequirement]) -> list[ProviderTraceRequirement]:
    seen: set[tuple[str, str]] = set()
    deduped: list[ProviderTraceRequirement] = []
    for requirement in requirements:
        key = (requirement.skill, requirement.trigger)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(requirement)
    return deduped


def _provider_trace_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in _as_list(payload.get("submission_trace")):
        if isinstance(item, dict):
            entries.append(dict(item))
    for anchor in _as_list(payload.get("claim_anchors")):
        if not isinstance(anchor, dict):
            continue
        tool_trace = anchor.get("tool_trace")
        if isinstance(tool_trace, dict):
            entries.append(dict(tool_trace))
        else:
            for item in _as_list(tool_trace):
                if isinstance(item, dict):
                    entries.append(dict(item))
    return entries


def _entry_mentions_skill(entry: dict[str, Any], skill: str) -> bool:
    skill_key = _normalize_skill(skill)
    fields = (
        entry.get("skill"),
        entry.get("tool"),
        entry.get("provider"),
        entry.get("step"),
        entry.get("name"),
    )
    return any(_normalize_skill(value) == skill_key or skill_key in _normalize_skill(value).split() for value in fields)


def _entry_is_acceptable(
    entry: dict[str, Any],
    requirement: ProviderTraceRequirement,
    *,
    require_existing_provider_paths: bool,
) -> bool:
    status = str(entry.get("status") or "").strip().lower()
    if status in {"success", "partial"}:
        if not require_existing_provider_paths:
            return True
        result_path = str(
            entry.get("result_path")
            or entry.get("output_path")
            or entry.get("artifact_path")
            or entry.get("path")
            or ""
        ).strip()
        return bool(result_path and Path(result_path).expanduser().is_file())
    if status == "skipped":
        return all(str(entry.get(key) or "").strip() for key in ("trigger", "reason", "risk"))
    return False


def _normalize_skill(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]
