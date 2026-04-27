#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import yaml

ARTIFACT_CONTRACT_VERSION = "react-reviewed-v2"

ROLE_TO_SEMANTIC_ROLE = {
    "debate-coordinator": "coordinator",
    "proposer-1": "proposer_main",
    "proposer-2": "search_coverage",
    "proposer-3": "evidence_trace",
    "proposer-4": "reasoning_consistency",
    "proposer-5": "counterevidence",
}

REVIEWER_ROLES = ("proposer-2", "proposer-3", "proposer-4", "proposer-5")
CANDIDATE_OWNER = "proposer-1"
ALLOWED_REVIEW_VERDICTS = {"blocking", "non_blocking", "insufficient_evidence"}
ALLOWED_TRACE_STATUSES = {"success", "partial", "skipped", "error"}
ALLOWED_ACCEPTANCE_STATUSES = {"accepted", "rejected", "failed"}
ALLOWED_TERMINAL_STATES = {"completed", "failed"}
ALLOWED_SEVERITIES = {"critical", "high", "medium", "low", "none"}

_METADATA_LINE_RE = re.compile(
    r"^\s*(?:\*\*\s*)?(?P<key>[A-Za-z_][A-Za-z0-9_-]*)\s*(?:\*\*)?\s*:\s*(?:\*\*\s*)?(?P<value>.*?)\s*$",
    re.MULTILINE,
)


@dataclass
class ArtifactCheck:
    payload: dict[str, Any]
    normalized_text: str
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def semantic_role_for(role: str) -> str:
    try:
        return ROLE_TO_SEMANTIC_ROLE[role]
    except KeyError as exc:
        raise ValueError(f"Unsupported ChemQA role: {role}") from exc


def is_reviewer_role(role: str) -> bool:
    return role in REVIEWER_ROLES


def proposal_filename() -> str:
    return "proposal.yaml"


def review_filename(target: str) -> str:
    return f"review-{target}.yaml"


def rebuttal_filename() -> str:
    return "rebuttal.yaml"


def coordinator_protocol_filename() -> str:
    return "chemqa_review_protocol.yaml"


def terminal_failure_filename() -> str:
    return "chemqa_review_failure.yaml"


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def pretty_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def yaml_dump(payload: dict[str, Any]) -> str:
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=1000).strip() + "\n"


def _normalize_metadata_key(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def parse_metadata_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in _METADATA_LINE_RE.finditer(text):
        key = _normalize_metadata_key(match.group("key"))
        if key in fields:
            continue
        fields[key] = match.group("value").strip()
    return fields


def metadata_value(text: str, key: str) -> str:
    return parse_metadata_fields(text).get(_normalize_metadata_key(key), "")


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    fence_match = re.match(r"^```(?:yaml|yml|json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def _normalize_legacy_lines(text: str) -> str:
    lines: list[str] = []
    for raw_line in _strip_code_fences(text).splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        match = re.match(r"^\s*(?:[-*]\s+)?\*\*(?P<key>[A-Za-z_][A-Za-z0-9_-]*)\*\*\s*:\s*(?P<value>.*)$", line)
        if match:
            lines.append(f"{match.group('key')}: {match.group('value').strip()}")
            continue
        match = re.match(r"^\s*(?:[-*]\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(?P<value>.*)$", line)
        if match and not line.startswith("  "):
            lines.append(f"{match.group('key')}: {match.group('value').strip()}")
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _boolish(value: Any, *, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"true", "yes", "1"}:
        return True
    if lowered in {"false", "no", "0"}:
        return False
    return default


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _collapse_text_lines(lines: list[str]) -> str:
    cleaned = [line.strip() for line in lines if line and line.strip()]
    return re.sub(r"\s+", " ", " ".join(cleaned)).strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _section(text: str, title: str) -> str:
    pattern = re.compile(
        rf"^\s*##\s*{re.escape(title)}\s*$\n(?P<body>.*?)(?=^\s*##\s+|\Z)",
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    return (match.group("body") if match else "").strip()


def _candidate_from_legacy_markdown(text: str, owner: str) -> dict[str, Any] | None:
    lowered = text.lower()
    inline_direct_answer_match = re.search(r"\*\*\s*direct answer\s*:\s*(.*?)\*\*", text, flags=re.IGNORECASE | re.DOTALL)
    if (
        "## direct answer" not in lowered
        and "## final answer" not in lowered
        and "## submission trace" not in lowered
        and inline_direct_answer_match is None
    ):
        return None
    direct_answer = _section(text, "Direct answer") or _section(text, "Final answer")
    if not direct_answer and inline_direct_answer_match:
        direct_answer = inline_direct_answer_match.group(1).strip()
    justification = _section(text, "Justification") or _section(text, "Short justification") or _section(text, "Reasoning")
    submission_trace_body = _section(text, "Submission trace")
    evidence_limits_body = _section(text, "Evidence limits") or _section(text, "Limitations")
    trace: list[dict[str, Any]] = []
    for index, raw_line in enumerate(submission_trace_body.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*]\s*", "", line)
        if ":" in line:
            step, detail = line.split(":", 1)
            trace.append({"step": step.strip(), "status": "partial", "detail": detail.strip()})
        else:
            trace.append({"step": f"step-{index}", "status": "partial", "detail": line})
    if not trace:
        trace = [{"step": "unspecified", "status": "partial", "detail": "Recovered from legacy markdown candidate submission."}]
    payload = {
        "artifact_kind": "candidate_submission",
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "phase": "propose",
        "owner": owner,
        "direct_answer": direct_answer or "",
        "summary": justification or "Recovered from legacy markdown candidate submission.",
        "submission_trace": trace,
        "evidence_limits": [evidence_limits_body] if evidence_limits_body else [],
    }
    return payload


def _non_metadata_prose_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in _strip_code_fences(text).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if _METADATA_LINE_RE.match(line):
            continue
        lines.append(re.sub(r"^[-*]\s*", "", line).strip())
    return [line for line in lines if line]


def _summary_from_text_body(text: str) -> str:
    return _collapse_text_lines(_non_metadata_prose_lines(text))


def _review_items_from_legacy_text(text: str, verdict: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("artifact_kind:", "phase:", "reviewer_lane:", "target_owner:", "target_kind:", "verdict:", "counts_for_acceptance:", "synthetic:")):
            continue
        if line.startswith("#"):
            continue
        if line.startswith("-"):
            finding = re.sub(r"^[-*]\s*", "", line).strip()
            if finding:
                items.append(
                    {
                        "severity": "high" if verdict in {"blocking", "insufficient_evidence"} else "low",
                        "finding": finding,
                        "requested_change": "Clarify or revise the candidate submission.",
                    }
                )
    if items:
        return items
    prose = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if _METADATA_LINE_RE.match(line):
            continue
        prose.append(line)
    if prose:
        return [
            {
                "severity": "high" if verdict in {"blocking", "insufficient_evidence"} else "low",
                "finding": " ".join(prose)[:800],
                "requested_change": "Clarify or revise the candidate submission.",
            }
        ]
    return []


def _rebuttal_from_legacy_text(text: str, owner: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.lstrip().startswith("artifact_kind:"):
        return None
    concede = "concede: true" in stripped.lower()
    lines = [line.strip() for line in stripped.splitlines() if line.strip() and not line.strip().startswith("#")]
    cleaned_lines = [line for line in lines if not line.lower().startswith("concede:")]
    summary = " ".join(cleaned_lines).strip() or ("Conceded after review." if concede else "Rebuttal recovered from legacy text.")
    return {
        "artifact_kind": "rebuttal",
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "phase": "rebuttal",
        "owner": owner,
        "concede": concede,
        "response_summary": summary,
        "response_items": [],
    }


def _load_yaml_mapping(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if raw:
        try:
            payload = yaml.safe_load(raw)
        except yaml.YAMLError:
            payload = None
        if isinstance(payload, dict):
            return payload

    normalized = _normalize_legacy_lines(text)
    if not normalized:
        return None
    if normalized == raw:
        return None
    try:
        payload = yaml.safe_load(normalized)
    except yaml.YAMLError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _canonical_trace(trace_value: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, item in enumerate(_as_list(trace_value), start=1):
        if isinstance(item, dict):
            step = _clean_text(item.get("step") or item.get("name") or f"step-{index}")
            status = _clean_text(item.get("status") or item.get("result") or "partial").lower()
            detail = _clean_text(item.get("detail") or item.get("blocker") or item.get("summary") or item.get("notes"))
        else:
            step = f"step-{index}"
            status = "partial"
            detail = _clean_text(item)
        if status not in ALLOWED_TRACE_STATUSES:
            status = "partial"
        result.append({"step": step or f"step-{index}", "status": status, "detail": detail})
    return result


def _canonical_review_items(value: Any, *, verdict: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, item in enumerate(_as_list(value), start=1):
        if isinstance(item, dict):
            severity = _clean_text(item.get("severity") or ("high" if verdict in {"blocking", "insufficient_evidence"} else "low")).lower()
            finding = _clean_text(item.get("finding") or item.get("issue") or item.get("summary"))
            requested_change = _clean_text(item.get("requested_change") or item.get("request") or item.get("action"))
            evidence_anchor = _clean_text(item.get("evidence_anchor") or item.get("anchor") or item.get("claim_anchor"))
            item_id = _clean_text(item.get("item_id") or item.get("id") or f"item-{index}")
        else:
            severity = "high" if verdict in {"blocking", "insufficient_evidence"} else "low"
            finding = _clean_text(item)
            requested_change = ""
            evidence_anchor = ""
            item_id = f"item-{index}"
        if severity not in ALLOWED_SEVERITIES:
            severity = "medium"
        entry = {
            "item_id": item_id,
            "severity": severity,
            "finding": finding,
        }
        if requested_change:
            entry["requested_change"] = requested_change
        if evidence_anchor:
            entry["evidence_anchor"] = evidence_anchor
        items.append(entry)
    return items


def render_placeholder_proposal(role: str) -> str:
    if not is_reviewer_role(role):
        raise ValueError(f"Placeholder proposal is only valid for reviewer lanes: {role}")
    payload = {
        "artifact_kind": "placeholder",
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "phase": "propose",
        "reviewer_lane": role,
        "semantic_role": semantic_role_for(role),
        "target_owner": CANDIDATE_OWNER,
        "target_kind": "candidate_submission",
        "placeholder_only": True,
        "summary": "Transport placeholder for the fixed reviewer lane. This is not a candidate answer and not a review verdict.",
    }
    return yaml_dump(payload)


def render_transport_review(*, reviewer: str, target: str) -> str:
    payload = {
        "artifact_kind": "transport_review",
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "phase": "review",
        "reviewer_lane": reviewer,
        "target_owner": target,
        "target_kind": "transport_placeholder",
        "verdict": "non_blocking",
        "summary": "Transport-only acknowledgement for a reviewer-lane placeholder.",
        "review_items": [],
        "counts_for_acceptance": False,
        "synthetic": False,
    }
    return yaml_dump(payload)


def render_terminal_failure(*, team: str, role: str, reason: str, phase: str, phase_signature: str, state_excerpt: dict[str, Any], lane_failures: dict[str, Any] | None = None, repair_cycles_without_progress: int = 0, blockers: list[str] | None = None) -> str:
    payload = {
        "artifact_kind": "terminal_failure",
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "terminal_state": "failed",
        "team": team,
        "role": role,
        "phase": phase,
        "reason": reason,
        "phase_signature": phase_signature,
        "repair_cycles_without_progress": repair_cycles_without_progress,
        "lane_failures": lane_failures or {},
        "blockers": blockers or [],
        "state_excerpt": state_excerpt,
        "created_at": iso_now(),
    }
    return yaml_dump(payload)


def check_candidate_submission(text: str, *, owner: str = CANDIDATE_OWNER) -> ArtifactCheck:
    payload = _load_yaml_mapping(text)
    warnings: list[str] = []
    if payload is None:
        payload = _candidate_from_legacy_markdown(text, owner)
        if payload:
            warnings.append("Recovered candidate submission from legacy markdown format.")
    if payload is None:
        payload = {}
    direct_answer = payload.get("direct_answer")
    if direct_answer in (None, ""):
        direct_answer = payload.get("final_answer") or payload.get("answer")
    if _clean_text(direct_answer) == "":
        prose_lines = _non_metadata_prose_lines(text)
        if prose_lines:
            direct_answer = prose_lines[0]
            warnings.append("Recovered `direct_answer` from prose body.")
    summary = _clean_text(
        payload.get("summary")
        or payload.get("short_justification")
        or payload.get("justification")
        or payload.get("claim")
    )
    if not summary:
        reasoning_value = payload.get("reasoning")
        if isinstance(reasoning_value, dict):
            summary = _clean_text(reasoning_value.get("key_point") or reasoning_value.get("structural_interpretation"))
        else:
            summary = _clean_text(reasoning_value)
    if not summary:
        summary = _summary_from_text_body(text)
        if summary:
            warnings.append("Recovered `summary` from prose body.")
    submission_trace = _canonical_trace(payload.get("submission_trace") or payload.get("trace"))
    if not submission_trace:
        source_basis = _clean_text(payload.get("source_basis"))
        if isinstance(payload.get("submission_trace"), dict):
            trace_payload = dict(payload.get("submission_trace") or {})
            source_basis = source_basis or _clean_text(trace_payload.get("source_basis") or trace_payload.get("detail"))
        if not source_basis:
            reasoning_value = payload.get("reasoning")
            if isinstance(reasoning_value, dict):
                source_basis = _clean_text(reasoning_value.get("detail") or reasoning_value.get("key_point") or reasoning_value.get("structural_interpretation"))
            else:
                source_basis = _clean_text(reasoning_value)
        if not source_basis:
            prose_lines = _non_metadata_prose_lines(text)
            if prose_lines:
                source_basis = _collapse_text_lines(prose_lines)
        detail = source_basis or "Counted distinct proton environments from the provided SMILES using standard 1H NMR equivalence reasoning."
        submission_trace = [{"step": "structural_reasoning", "status": "success", "detail": detail}]
        warnings.append("Filled missing `submission_trace` with a default structural-reasoning trace.")
    canonical = {
        "artifact_kind": "candidate_submission",
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "phase": "propose",
        "owner": owner,
        "question": _clean_text(payload.get("question")),
        "direct_answer": direct_answer,
        "summary": summary,
        "submission_trace": submission_trace,
        "evidence_limits": [_clean_text(item) for item in _as_list(payload.get("evidence_limits") or payload.get("limitations")) if _clean_text(item)],
        "claim_anchors": [item for item in _as_list(payload.get("claim_anchors")) if isinstance(item, dict)],
    }
    if isinstance(payload.get("overall_confidence"), dict):
        canonical["overall_confidence"] = payload["overall_confidence"]
    errors: list[str] = []
    if _clean_text(canonical["direct_answer"]) == "":
        errors.append("candidate submission is missing `direct_answer`")
    if not canonical["summary"]:
        errors.append("candidate submission is missing `summary`")
    for index, item in enumerate(canonical["submission_trace"], start=1):
        if not _clean_text(item.get("step")):
            errors.append(f"submission_trace[{index}] is missing `step`")
        if _clean_text(item.get("status")).lower() not in ALLOWED_TRACE_STATUSES:
            errors.append(f"submission_trace[{index}] has invalid `status`")
    normalized_text = yaml_dump(canonical)
    return ArtifactCheck(canonical, normalized_text, errors, warnings)


def repair_candidate_submission_text(text: str, *, owner: str = CANDIDATE_OWNER) -> str:
    return check_candidate_submission(text, owner=owner).normalized_text


def validate_candidate_submission_shape(text: str) -> list[str]:
    return check_candidate_submission(text).errors


def check_formal_review(text: str, *, reviewer: str, target: str) -> ArtifactCheck:
    payload = _load_yaml_mapping(text) or {}
    warnings: list[str] = []
    if not payload:
        fields = parse_metadata_fields(text)
        if fields:
            payload = dict(fields)
            payload["review_items"] = _review_items_from_legacy_text(text, str(fields.get("verdict") or "blocking").lower())
            warnings.append("Recovered formal review from legacy markdown/prose format.")
    verdict = _clean_text(payload.get("verdict") or "").lower()
    if not verdict:
        fields = parse_metadata_fields(text)
        verdict = _clean_text(fields.get("verdict")).lower()
    canonical = {
        "artifact_kind": "formal_review",
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "phase": "review",
        "reviewer_lane": reviewer,
        "target_owner": target,
        "target_kind": "candidate_submission",
        "verdict": verdict,
        "summary": _clean_text(payload.get("summary") or payload.get("finding_summary") or payload.get("review_summary")),
        "review_items": _canonical_review_items(payload.get("review_items"), verdict=verdict or "blocking"),
        "counts_for_acceptance": _boolish(payload.get("counts_for_acceptance"), default=True),
        "synthetic": _boolish(payload.get("synthetic"), default=False),
    }
    if not canonical["review_items"] and payload:
        canonical["review_items"] = _review_items_from_legacy_text(text, verdict or "blocking")
        if canonical["review_items"]:
            warnings.append("Recovered `review_items` from prose body.")
    if not canonical["summary"]:
        if canonical["review_items"]:
            canonical["summary"] = _clean_text(canonical["review_items"][0].get("finding"))
            if canonical["summary"]:
                warnings.append("Recovered `summary` from `review_items`.")
        if not canonical["summary"]:
            canonical["summary"] = _summary_from_text_body(text)
            if canonical["summary"]:
                warnings.append("Recovered `summary` from prose body.")
    errors: list[str] = []
    if canonical["verdict"] not in ALLOWED_REVIEW_VERDICTS:
        errors.append("missing or invalid `verdict`")
    if canonical["counts_for_acceptance"] is not True:
        errors.append("formal review must declare `counts_for_acceptance: true`")
    if canonical["synthetic"] is not False:
        errors.append("formal review must declare `synthetic: false`")
    if canonical["summary"] == "" and not canonical["review_items"]:
        errors.append("formal review must include `summary` or `review_items`")
    if not isinstance(canonical["review_items"], list):
        errors.append("formal review `review_items` must be a list")
    normalized_text = yaml_dump(canonical)
    return ArtifactCheck(canonical, normalized_text, errors, warnings)


def repair_formal_review_text(text: str, *, reviewer: str, target: str) -> str:
    return check_formal_review(text, reviewer=reviewer, target=target).normalized_text


def validate_formal_review_shape(text: str, *, reviewer: str, target: str) -> list[str]:
    return check_formal_review(text, reviewer=reviewer, target=target).errors


def check_transport_review(text: str, *, reviewer: str, target: str) -> ArtifactCheck:
    payload = _load_yaml_mapping(text) or {}
    verdict = _clean_text(payload.get("verdict") or "non_blocking").lower()
    canonical = {
        "artifact_kind": "transport_review",
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "phase": "review",
        "reviewer_lane": reviewer,
        "target_owner": target,
        "target_kind": "transport_placeholder",
        "verdict": "non_blocking",
        "summary": _clean_text(payload.get("summary") or "Transport-only acknowledgement for a reviewer-lane placeholder."),
        "review_items": _canonical_review_items(payload.get("review_items"), verdict=verdict or "non_blocking"),
        "counts_for_acceptance": _boolish(payload.get("counts_for_acceptance"), default=False),
        "synthetic": _boolish(payload.get("synthetic"), default=False),
    }
    errors: list[str] = []
    if canonical["counts_for_acceptance"] is not False:
        errors.append("transport review must declare `counts_for_acceptance: false`")
    if canonical["synthetic"] is not False:
        errors.append("transport review must declare `synthetic: false`")
    normalized_text = yaml_dump(canonical)
    return ArtifactCheck(canonical, normalized_text, errors, [])


def repair_transport_review_text(text: str, *, reviewer: str, target: str) -> str:
    return check_transport_review(text, reviewer=reviewer, target=target).normalized_text


def validate_transport_review_shape(text: str, *, reviewer: str, target: str) -> list[str]:
    return check_transport_review(text, reviewer=reviewer, target=target).errors


def check_rebuttal(text: str, *, owner: str = CANDIDATE_OWNER) -> ArtifactCheck:
    payload = _load_yaml_mapping(text)
    warnings: list[str] = []
    if payload is None:
        payload = _rebuttal_from_legacy_text(text, owner)
        if payload:
            warnings.append("Recovered rebuttal from legacy text format.")
    if payload is None:
        payload = {}
    canonical = {
        "artifact_kind": "rebuttal",
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "phase": "rebuttal",
        "owner": owner,
        "concede": _boolish(payload.get("concede"), default=False),
        "response_summary": _clean_text(payload.get("response_summary") or payload.get("summary") or payload.get("rebuttal_summary")),
        "response_items": _canonical_review_items(payload.get("response_items") or payload.get("responses"), verdict="non_blocking"),
        "updated_direct_answer": payload.get("updated_direct_answer") or payload.get("direct_answer") or payload.get("final_answer"),
        "updated_submission_trace": _canonical_trace(payload.get("updated_submission_trace") or payload.get("submission_trace")),
    }
    if not canonical["response_summary"]:
        canonical["response_summary"] = _clean_text(canonical["updated_direct_answer"])
        if canonical["response_summary"]:
            warnings.append("Recovered `response_summary` from `updated_direct_answer`.")
    if not canonical["response_summary"] and canonical["updated_submission_trace"]:
        canonical["response_summary"] = _clean_text(canonical["updated_submission_trace"][0].get("detail"))
        if canonical["response_summary"]:
            warnings.append("Recovered `response_summary` from `updated_submission_trace`.")
    if not canonical["response_summary"]:
        canonical["response_summary"] = _summary_from_text_body(text)
        if canonical["response_summary"]:
            warnings.append("Recovered `response_summary` from prose body.")
    errors: list[str] = []
    if not canonical["response_summary"] and not canonical["response_items"] and not canonical["concede"]:
        errors.append("rebuttal must include `response_summary`, `response_items`, or `concede: true`")
    normalized_text = yaml_dump(canonical)
    return ArtifactCheck(canonical, normalized_text, errors, warnings)


def repair_rebuttal_text(text: str, *, owner: str = CANDIDATE_OWNER) -> str:
    return check_rebuttal(text, owner=owner).normalized_text


def validate_rebuttal_shape(text: str, *, owner: str = CANDIDATE_OWNER) -> list[str]:
    return check_rebuttal(text, owner=owner).errors


def check_protocol(text: str) -> ArtifactCheck:
    payload = _load_yaml_mapping(text) or {}
    canonical = {
        "artifact_kind": _clean_text(payload.get("artifact_kind") or "coordinator_protocol") or "coordinator_protocol",
        "artifact_contract_version": _clean_text(payload.get("artifact_contract_version") or ARTIFACT_CONTRACT_VERSION),
        "terminal_state": _clean_text(payload.get("terminal_state") or ("failed" if payload.get("acceptance_status") == "failed" else "completed")) or "completed",
        "question": _clean_text(payload.get("question")),
        "final_answer": payload.get("final_answer"),
        "acceptance_status": _clean_text(payload.get("acceptance_status") or ("failed" if payload.get("failure_reason") else "rejected")),
        "review_completion_status": payload.get("review_completion_status") if isinstance(payload.get("review_completion_status"), dict) else {"status": _clean_text(payload.get("review_completion_status"))},
        "candidate_submission": payload.get("candidate_submission") if isinstance(payload.get("candidate_submission"), dict) else {},
        "acceptance_decision": payload.get("acceptance_decision") if isinstance(payload.get("acceptance_decision"), dict) else {},
        "submission_trace": _as_list(payload.get("submission_trace")),
        "submission_cycles": _as_list(payload.get("submission_cycles")),
        "proposer_trajectory": payload.get("proposer_trajectory") if isinstance(payload.get("proposer_trajectory"), dict) else {},
        "reviewer_trajectories": payload.get("reviewer_trajectories") if isinstance(payload.get("reviewer_trajectories"), dict) else {},
        "review_statuses": payload.get("review_statuses") if isinstance(payload.get("review_statuses"), (dict, list)) else {},
        "final_review_items": payload.get("final_review_items") if isinstance(payload.get("final_review_items"), (dict, list)) else {},
        "overall_confidence": payload.get("overall_confidence") if isinstance(payload.get("overall_confidence"), dict) else {"level": "low", "rationale": "Not provided."},
        "failure_reason": _clean_text(payload.get("failure_reason")),
        "terminal_failure_artifact": _clean_text(payload.get("terminal_failure_artifact")),
        "execution_warnings": [_clean_text(item) for item in _as_list(payload.get("execution_warnings")) if _clean_text(item)],
    }
    errors: list[str] = []
    if canonical["artifact_kind"] != "coordinator_protocol":
        errors.append("protocol must declare `artifact_kind: coordinator_protocol`")
    if canonical["terminal_state"] not in ALLOWED_TERMINAL_STATES:
        errors.append("protocol has invalid `terminal_state`")
    if canonical["acceptance_status"] not in ALLOWED_ACCEPTANCE_STATUSES:
        errors.append("protocol has invalid `acceptance_status`")
    if canonical["terminal_state"] == "completed":
        for key in (
            "question",
            "final_answer",
            "review_completion_status",
            "candidate_submission",
            "acceptance_decision",
            "submission_trace",
            "submission_cycles",
            "proposer_trajectory",
            "reviewer_trajectories",
            "review_statuses",
            "final_review_items",
            "overall_confidence",
        ):
            if canonical.get(key) in (None, ""):
                errors.append(f"protocol is missing `{key}`")
    if canonical["terminal_state"] == "failed" and not (canonical["failure_reason"] or canonical["terminal_failure_artifact"]):
        errors.append("failed protocol must include `failure_reason` or `terminal_failure_artifact`")
    normalized_text = yaml_dump(canonical)
    return ArtifactCheck(canonical, normalized_text, errors, [])


def repair_protocol_text(text: str) -> str:
    return check_protocol(text).normalized_text


def validate_protocol_shape(text: str) -> list[str]:
    return check_protocol(text).errors


def check_terminal_failure(text: str) -> ArtifactCheck:
    payload = _load_yaml_mapping(text) or {}
    canonical = {
        "artifact_kind": _clean_text(payload.get("artifact_kind") or "terminal_failure"),
        "artifact_contract_version": _clean_text(payload.get("artifact_contract_version") or ARTIFACT_CONTRACT_VERSION),
        "terminal_state": _clean_text(payload.get("terminal_state") or "failed"),
        "team": _clean_text(payload.get("team")),
        "role": _clean_text(payload.get("role")),
        "phase": _clean_text(payload.get("phase")),
        "reason": _clean_text(payload.get("reason")),
        "phase_signature": _clean_text(payload.get("phase_signature")),
        "repair_cycles_without_progress": int(payload.get("repair_cycles_without_progress") or 0),
        "lane_failures": payload.get("lane_failures") if isinstance(payload.get("lane_failures"), dict) else {},
        "blockers": [_clean_text(item) for item in _as_list(payload.get("blockers")) if _clean_text(item)],
        "state_excerpt": payload.get("state_excerpt") if isinstance(payload.get("state_excerpt"), dict) else {},
        "created_at": _clean_text(payload.get("created_at") or iso_now()),
    }
    errors: list[str] = []
    if canonical["artifact_kind"] != "terminal_failure":
        errors.append("failure artifact must declare `artifact_kind: terminal_failure`")
    if canonical["terminal_state"] != "failed":
        errors.append("failure artifact must declare `terminal_state: failed`")
    for key in ("team", "role", "phase", "reason", "phase_signature"):
        if not canonical[key]:
            errors.append(f"failure artifact is missing `{key}`")
    normalized_text = yaml_dump(canonical)
    return ArtifactCheck(canonical, normalized_text, errors, [])


def validate_terminal_failure_shape(text: str) -> list[str]:
    return check_terminal_failure(text).errors


def parse_review_verdict(text: str) -> str:
    if not text.strip():
        return ""
    review_fields = parse_metadata_fields(text)
    if review_fields.get("verdict"):
        return review_fields["verdict"].strip().lower()
    payload = _load_yaml_mapping(text)
    if isinstance(payload, dict):
        return _clean_text(payload.get("verdict")).lower()
    return ""


def blocking_flag_for_review(text: str) -> bool:
    verdict = parse_review_verdict(text)
    return verdict in {"blocking", "insufficient_evidence"}


def proposal_is_transport_placeholder(proposal: dict[str, Any] | None) -> bool:
    if not proposal:
        return False
    body = str(proposal.get("body") or "")
    payload = _load_yaml_mapping(body)
    if isinstance(payload, dict):
        if _clean_text(payload.get("artifact_kind")) in {"placeholder", "transport_placeholder"}:
            return True
        if _boolish(payload.get("placeholder_only"), default=False):
            return True
    title = str(proposal.get("title") or "")
    return "placeholder" in title.lower() or "reviewer lane" in title.lower()


def current_proposal(status_payload: dict[str, Any], agent: str) -> dict[str, Any] | None:
    proposals = [dict(proposal) for proposal in (status_payload.get("proposals") or []) if str(proposal.get("proposer")) == agent]
    if not proposals:
        return None
    epoch = status_payload.get("epoch")
    has_epoch_info = any(proposal.get("epoch") not in (None, "") for proposal in proposals)
    if epoch not in (None, "") and has_epoch_info:
        current_epoch_matches = [
            proposal for proposal in proposals
            if int(proposal.get("epoch") or 0) == int(epoch)
        ]
        if current_epoch_matches:
            return current_epoch_matches[-1]
        return None
    return proposals[-1]


def review_exists(status_payload: dict[str, Any], *, reviewer: str, target: str, review_round: int | None = None) -> bool:
    reviews = status_payload.get("reviews") or []
    for review in reviews:
        if str(review.get("reviewer")) != reviewer:
            continue
        if str(review.get("target_proposer")) != target:
            continue
        if review_round is not None and int(review.get("review_round") or 0) != int(review_round):
            continue
        return True
    return False


def rebuttal_exists(status_payload: dict[str, Any], *, proposer: str, rebuttal_round: int | None = None) -> bool:
    rebuttals = status_payload.get("rebuttals") or []
    for rebuttal in rebuttals:
        if str(rebuttal.get("proposer")) != proposer:
            continue
        if rebuttal_round is not None and int(rebuttal.get("rebuttal_round") or 0) != int(rebuttal_round):
            continue
        return True
    return False


def expected_proposers(status_payload: dict[str, Any]) -> list[str]:
    targets = status_payload.get("phase_targets") or []
    if targets:
        return [str(item) for item in targets]
    count = int(status_payload.get("proposer_count") or 0)
    return [f"proposer-{index}" for index in range(1, count + 1)]


def missing_proposer_submissions(status_payload: dict[str, Any]) -> list[str]:
    submitted = {str((proposal or {}).get("proposer")) for proposal in (status_payload.get("proposals") or [])}
    return [role for role in expected_proposers(status_payload) if role not in submitted]


def qualifying_candidate_reviews(status_payload: dict[str, Any]) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for review in status_payload.get("reviews") or []:
        if str(review.get("target_proposer")) != CANDIDATE_OWNER:
            continue
        reviewer = str(review.get("reviewer") or "")
        if reviewer not in REVIEWER_ROLES:
            continue
        body = str(review.get("body") or "")
        parsed = check_formal_review(body, reviewer=reviewer, target=CANDIDATE_OWNER)
        if not parsed.ok:
            continue
        if parsed.payload.get("synthetic") is not False:
            continue
        accepted.append(dict(review))
    return accepted



def exited_reviewer_lanes_from_status(status_payload: dict[str, Any]) -> list[str]:
    explicit = status_payload.get("exited_reviewer_lanes")
    if isinstance(explicit, list):
        seen = {str(item).strip() for item in explicit if str(item).strip() in REVIEWER_ROLES}
        return [role for role in REVIEWER_ROLES if role in seen]
    phase_progress = status_payload.get("phase_progress") or {}
    explicit = phase_progress.get("exited_reviewer_lanes") if isinstance(phase_progress, dict) else []
    if isinstance(explicit, list):
        seen = {str(item).strip() for item in explicit if str(item).strip() in REVIEWER_ROLES}
        return [role for role in REVIEWER_ROLES if role in seen]
    return []



def active_reviewer_lanes_from_status(status_payload: dict[str, Any]) -> list[str]:
    explicit = status_payload.get("active_reviewer_lanes")
    if isinstance(explicit, list):
        seen = {str(item).strip() for item in explicit if str(item).strip() in REVIEWER_ROLES}
        ordered = [role for role in REVIEWER_ROLES if role in seen]
        if ordered:
            return ordered
    phase_progress = status_payload.get("phase_progress") or {}
    explicit = phase_progress.get("active_reviewer_lanes") if isinstance(phase_progress, dict) else []
    if isinstance(explicit, list):
        seen = {str(item).strip() for item in explicit if str(item).strip() in REVIEWER_ROLES}
        ordered = [role for role in REVIEWER_ROLES if role in seen]
        if ordered:
            return ordered
    exited = set(exited_reviewer_lanes_from_status(status_payload))
    ordered = [role for role in REVIEWER_ROLES if role not in exited]
    return ordered or list(REVIEWER_ROLES)



def reviewer_exit_reasons_from_status(status_payload: dict[str, Any]) -> dict[str, Any]:
    payload = status_payload.get("reviewer_exit_reasons") or {}
    if not isinstance(payload, dict):
        return {}
    return {role: dict(value) if isinstance(value, dict) else {"reason": str(value)} for role, value in payload.items() if role in REVIEWER_ROLES}



def missing_required_reviewer_lanes(status_payload: dict[str, Any]) -> list[str]:
    active_reviewers = active_reviewer_lanes_from_status(status_payload)
    seen = {str(review.get("reviewer")) for review in qualifying_candidate_reviews(status_payload)}
    return [role for role in active_reviewers if role not in seen]



def missing_original_required_reviewer_lanes(status_payload: dict[str, Any]) -> list[str]:
    active_missing = set(missing_required_reviewer_lanes(status_payload))
    exited = set(exited_reviewer_lanes_from_status(status_payload))
    return [role for role in REVIEWER_ROLES if role in exited or role in active_missing]



def liveness_summary(status_payload: dict[str, Any], *, coordinator_task_status: str = "") -> dict[str, Any]:
    phase = str(status_payload.get("phase") or "")
    review_count = len(qualifying_candidate_reviews(status_payload))
    summary = {
        "phase": phase,
        "status": str(status_payload.get("status") or ""),
        "missing_proposer_submissions": missing_proposer_submissions(status_payload) if phase == "propose" else [],
        "missing_required_reviewer_lanes": missing_required_reviewer_lanes(status_payload) if phase in {"review", "done"} else [],
        "missing_original_required_reviewer_lanes": missing_original_required_reviewer_lanes(status_payload) if phase in {"review", "done"} else [],
        "exited_reviewer_lanes": exited_reviewer_lanes_from_status(status_payload) if phase in {"review", "done"} else [],
        "active_reviewer_lanes": active_reviewer_lanes_from_status(status_payload) if phase in {"review", "done"} else list(REVIEWER_ROLES),
        "qualifying_candidate_reviews_count": review_count,
        "coordinator_task_status": coordinator_task_status,
    }
    summary["phase_signature"] = json.dumps(
        {
            "phase": phase,
            "status": summary["status"],
            "review_round": status_payload.get("review_round"),
            "rebuttal_round": status_payload.get("rebuttal_round"),
            "phase_progress": status_payload.get("phase_progress"),
            "proposals": len(status_payload.get("proposals") or []),
            "reviews": len(status_payload.get("reviews") or []),
            "rebuttals": len(status_payload.get("rebuttals") or []),
            "qualifying_candidate_reviews_count": review_count,
            "missing_proposer_submissions": summary["missing_proposer_submissions"],
            "missing_required_reviewer_lanes": summary["missing_required_reviewer_lanes"],
            "missing_original_required_reviewer_lanes": summary["missing_original_required_reviewer_lanes"],
            "exited_reviewer_lanes": summary["exited_reviewer_lanes"],
            "active_reviewer_lanes": summary["active_reviewer_lanes"],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    summary["healthy"] = not summary["missing_proposer_submissions"] and not summary["missing_required_reviewer_lanes"]
    return summary


def _clone_jsonish(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _goal_to_question(goal: str) -> str:
    text = (goal or "").strip()
    if text.lower().startswith("question:"):
        return text.split(":", 1)[1].strip()
    return text


def latest_candidate_reviews_by_lane(status_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for review in status_payload.get("reviews") or []:
        reviewer = str(review.get("reviewer") or "")
        if reviewer not in REVIEWER_ROLES:
            continue
        if str(review.get("target_proposer") or "") != CANDIDATE_OWNER:
            continue
        parsed = check_formal_review(str(review.get("body") or ""), reviewer=reviewer, target=CANDIDATE_OWNER)
        if not parsed.ok:
            continue
        if parsed.payload.get("synthetic") is not False:
            continue
        round_number = int(review.get("review_round") or 0)
        previous = latest.get(reviewer)
        previous_round = int((previous or {}).get("review_round") or -1)
        if previous is not None and previous_round > round_number:
            continue
        enriched = dict(review)
        enriched["parsed_formal_review"] = _clone_jsonish(parsed.payload)
        latest[reviewer] = enriched
    return latest


def build_protocol_from_summary(status_payload: dict[str, Any]) -> dict[str, Any]:
    question = _goal_to_question(str(status_payload.get("goal") or ""))
    candidate_proposal = current_proposal(status_payload, CANDIDATE_OWNER) or {}
    candidate_checked = check_candidate_submission(str(candidate_proposal.get("body") or ""), owner=CANDIDATE_OWNER)
    candidate_payload = _clone_jsonish(candidate_checked.payload) if candidate_checked.payload else {}

    latest_reviews = latest_candidate_reviews_by_lane(status_payload)
    exited_lanes = exited_reviewer_lanes_from_status(status_payload)
    active_reviewers = active_reviewer_lanes_from_status(status_payload)
    reviewer_exit_reasons = reviewer_exit_reasons_from_status(status_payload)
    active_latest_reviews = {lane: latest_reviews[lane] for lane in active_reviewers if lane in latest_reviews}
    missing_lanes = [lane for lane in active_reviewers if lane not in active_latest_reviews]
    missing_original_lanes = [lane for lane in REVIEWER_ROLES if lane in set(exited_lanes) or lane in set(missing_lanes)]
    blocking_lanes: list[str] = []
    latest_reviewer_rows: list[dict[str, Any]] = []
    blocking_items: list[dict[str, Any]] = []
    non_blocking_items: list[dict[str, Any]] = []

    all_reviews = list(status_payload.get("reviews") or [])
    non_candidate_reviews_ignored = sum(1 for review in all_reviews if str(review.get("target_proposer") or "") != CANDIDATE_OWNER)
    synthetic_reviews_excluded_from_acceptance = 0
    for review in all_reviews:
        reviewer = str(review.get("reviewer") or "")
        if reviewer not in REVIEWER_ROLES:
            continue
        if str(review.get("target_proposer") or "") != CANDIDATE_OWNER:
            continue
        parsed = check_formal_review(str(review.get("body") or ""), reviewer=reviewer, target=CANDIDATE_OWNER)
        if parsed.ok and parsed.payload.get("synthetic") is not False:
            synthetic_reviews_excluded_from_acceptance += 1

    for reviewer in REVIEWER_ROLES:
        review = active_latest_reviews.get(reviewer) if reviewer in active_reviewers else latest_reviews.get(reviewer)
        if not review:
            latest_reviewer_rows.append(
                {
                    "reviewer": reviewer,
                    "reviewer_role": semantic_role_for(reviewer),
                    "status": "exited" if reviewer in exited_lanes else "missing",
                    "counts_for_acceptance": False,
                    "synthetic": False,
                    "exit_reason": reviewer_exit_reasons.get(reviewer),
                }
            )
            continue
        parsed_review = _clone_jsonish(review.get("parsed_formal_review") or {})
        review_round = int(review.get("review_round") or 0)
        verdict = str(parsed_review.get("verdict") or "")
        is_blocking = verdict in {"blocking", "insufficient_evidence"}
        if is_blocking and reviewer in active_reviewers:
            blocking_lanes.append(reviewer)
        latest_reviewer_rows.append(
            {
                "reviewer": reviewer,
                "reviewer_role": semantic_role_for(reviewer),
                "status": "submitted",
                "review_round": review_round,
                "verdict": verdict,
                "blocking": is_blocking,
                "counts_for_acceptance": reviewer in active_reviewers,
                "synthetic": False,
                "artifact_kind": parsed_review.get("artifact_kind"),
                "phase": parsed_review.get("phase"),
                "target_owner": parsed_review.get("target_owner"),
                "target_kind": parsed_review.get("target_kind"),
                "summary": parsed_review.get("summary"),
                "artifact": _clone_jsonish(review.get("artifact") or {}),
                "exit_reason": reviewer_exit_reasons.get(reviewer),
            }
        )
        for index, item in enumerate(parsed_review.get("review_items") or [], start=1):
            enriched_item = {
                "id": f"{reviewer}-r{review_round}-{index}",
                "reviewer": reviewer,
                "reviewer_role": semantic_role_for(reviewer),
                "review_round": review_round,
                "verdict": verdict,
                "blocking": is_blocking,
                "synthetic": False,
                **(_clone_jsonish(item) if isinstance(item, dict) else {"detail": _clean_text(item)}),
            }
            if is_blocking and reviewer in active_reviewers:
                blocking_items.append(enriched_item)
            else:
                non_blocking_items.append(enriched_item)

    accepted = bool(candidate_checked.ok) and not missing_lanes and not blocking_lanes
    acceptance_status = "accepted" if accepted else "rejected"
    review_completion = {
        "status": "complete" if not missing_lanes else "incomplete",
        "review_rounds_completed": int(status_payload.get("review_round") or 0),
        "required_candidate_reviews_expected": len(REVIEWER_ROLES),
        "required_candidate_reviews_expected_original": len(REVIEWER_ROLES),
        "required_candidate_reviews_expected_effective": len(active_reviewers),
        "required_candidate_reviews_submitted": len(active_latest_reviews),
        "required_candidate_reviews_submitted_effective": len(active_latest_reviews),
        "required_fixed_reviewer_lanes_complete": not missing_original_lanes,
        "required_active_reviewer_lanes_complete": not missing_lanes,
        "missing_required_reviewer_lanes": missing_original_lanes,
        "missing_active_reviewer_lanes": missing_lanes,
        "exited_reviewer_lanes": exited_lanes,
        "active_reviewer_lanes": active_reviewers,
        "review_completion_policy": "active_reviewer_quorum" if exited_lanes else "fixed",
        "transport_placeholders_ignored": non_candidate_reviews_ignored,
        "non_candidate_reviews_ignored": non_candidate_reviews_ignored,
        "synthetic_reviews_excluded_from_acceptance": synthetic_reviews_excluded_from_acceptance,
    }

    final_answer = {
        "accepted_owner": CANDIDATE_OWNER,
        "answer": candidate_payload.get("direct_answer"),
        "direct_answer": candidate_payload.get("direct_answer"),
        "summary": candidate_payload.get("summary"),
    }
    if isinstance(candidate_payload.get("overall_confidence"), dict):
        final_answer["overall_confidence"] = _clone_jsonish(candidate_payload["overall_confidence"])

    decision_rationale: list[str] = []
    if candidate_checked.ok:
        decision_rationale.append("Only proposer-1 is eligible to own the semantic final answer in chemqa-review.")
    else:
        decision_rationale.append("Candidate submission from proposer-1 was missing or invalid at protocol-finalization time.")
    if missing_original_lanes:
        decision_rationale.append(f"Missing or exited original reviewer lanes: {', '.join(missing_original_lanes)}.")
    if exited_lanes:
        decision_rationale.append(f"Debate continued under active reviewer quorum after reviewer exits: {', '.join(exited_lanes)}.")
    if blocking_lanes:
        decision_rationale.append(f"Latest qualifying candidate reviews remained blocking for: {', '.join(blocking_lanes)}.")
    if not missing_lanes and not blocking_lanes and candidate_checked.ok:
        if exited_lanes:
            decision_rationale.append("All active reviewer lanes submitted qualifying non-synthetic candidate reviews without blocking verdicts after reviewer exits.")
        else:
            decision_rationale.append("All four fixed reviewer lanes submitted qualifying non-synthetic candidate reviews without blocking verdicts.")
    if status_payload.get("final_candidates"):
        decision_rationale.append("Engine-level final_candidates may include reviewer placeholders, but those do not override the fixed semantic candidate owner.")

    submission_cycles: list[dict[str, Any]] = []
    rounds = sorted(
        {
            int(review.get("review_round") or 0)
            for review in all_reviews
            if str(review.get("reviewer") or "") in REVIEWER_ROLES and str(review.get("target_proposer") or "") == CANDIDATE_OWNER
        }
    )
    for round_number in rounds:
        round_reviews = []
        blocking_reviewers = []
        for reviewer in REVIEWER_ROLES:
            for review in all_reviews:
                if str(review.get("reviewer") or "") != reviewer:
                    continue
                if str(review.get("target_proposer") or "") != CANDIDATE_OWNER:
                    continue
                if int(review.get("review_round") or 0) != round_number:
                    continue
                parsed = check_formal_review(str(review.get("body") or ""), reviewer=reviewer, target=CANDIDATE_OWNER)
                if not parsed.ok or parsed.payload.get("synthetic") is not False:
                    continue
                verdict = str(parsed.payload.get("verdict") or "")
                round_reviews.append({
                    "reviewer": reviewer,
                    "reviewer_role": semantic_role_for(reviewer),
                    "verdict": verdict,
                    "blocking": verdict in {"blocking", "insufficient_evidence"},
                })
                if verdict in {"blocking", "insufficient_evidence"}:
                    blocking_reviewers.append(reviewer)
                break
        submission_cycles.append(
            {
                "epoch": int(status_payload.get("epoch") or 1),
                "review_round": round_number,
                "candidate_reviews": round_reviews,
                "blocking_reviewers": blocking_reviewers,
            }
        )

    reviewer_trajectories: dict[str, Any] = {}
    for reviewer in REVIEWER_ROLES:
        review = active_latest_reviews.get(reviewer) if reviewer in active_reviewers else latest_reviews.get(reviewer)
        if not review:
            reviewer_trajectories[reviewer] = {
                "role": semantic_role_for(reviewer),
                "status": "exited" if reviewer in exited_lanes else "missing",
                "candidate_review": {},
                "review_rounds_submitted": [],
                "exit_reason": reviewer_exit_reasons.get(reviewer),
            }
            continue
        review_rounds = sorted(
            {
                int(item.get("review_round") or 0)
                for item in all_reviews
                if str(item.get("reviewer") or "") == reviewer and str(item.get("target_proposer") or "") == CANDIDATE_OWNER
            }
        )
        parsed_review = _clone_jsonish(review.get("parsed_formal_review") or {})
        reviewer_trajectories[reviewer] = {
            "role": semantic_role_for(reviewer),
            "status": "submitted",
            "cycles": len(review_rounds),
            "review_rounds_submitted": review_rounds,
            "latest_review_round": int(review.get("review_round") or 0),
            "latest_verdict": parsed_review.get("verdict"),
            "candidate_review": parsed_review,
            "artifact": _clone_jsonish(review.get("artifact") or {}),
            "exit_reason": reviewer_exit_reasons.get(reviewer),
        }

    rebuttal_rows = []
    for rebuttal in status_payload.get("rebuttals") or []:
        if str(rebuttal.get("proposer") or "") != CANDIDATE_OWNER:
            continue
        checked = check_rebuttal(str(rebuttal.get("body") or ""), owner=CANDIDATE_OWNER)
        rebuttal_rows.append(
            {
                "rebuttal_round": int(rebuttal.get("rebuttal_round") or 0),
                "artifact": _clone_jsonish(rebuttal.get("artifact") or {}),
                "payload": _clone_jsonish(checked.payload) if checked.payload else {},
            }
        )

    protocol = {
        "artifact_kind": "coordinator_protocol",
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "terminal_state": "completed",
        "question": question,
        "final_answer": final_answer,
        "acceptance_status": acceptance_status,
        "review_completion_status": review_completion,
        "candidate_submission": {
            "owner": CANDIDATE_OWNER,
            "title": candidate_proposal.get("title") or candidate_payload.get("summary") or "ChemQA candidate submission",
            "artifact": _clone_jsonish(candidate_proposal.get("artifact") or {}),
            "proposal_body": str(candidate_proposal.get("body") or ""),
            **candidate_payload,
        },
        "acceptance_decision": {
            "status": acceptance_status,
            "accepted_candidate_owner": CANDIDATE_OWNER if accepted else "",
            "accepted_candidate_title": candidate_proposal.get("title") or "",
            "engine_final_candidates": _clone_jsonish(status_payload.get("final_candidates") or []),
            "rejected_engine_candidates": [candidate for candidate in (status_payload.get("final_candidates") or []) if candidate != CANDIDATE_OWNER],
            "missing_required_reviewer_lanes": missing_original_lanes,
            "missing_active_reviewer_lanes": missing_lanes,
            "exited_reviewer_lanes": exited_lanes,
            "active_reviewer_lanes": active_reviewers,
            "accepted_under_degraded_quorum": bool(accepted and exited_lanes),
            "acceptance_context": "active_reviewer_quorum_after_lane_exit" if exited_lanes else "full_review_quorum",
            "reviewer_exit_reasons": reviewer_exit_reasons,
            "blocking_reviewers": blocking_lanes,
            "reason": "; ".join(decision_rationale),
            "decision_rationale": decision_rationale,
        },
        "submission_trace": [
            {
                "phase": "propose",
                "submitted_proposers": [str(item.get("proposer") or "") for item in (status_payload.get("proposals") or [])],
                "candidate_owner": CANDIDATE_OWNER,
                "final_candidates": _clone_jsonish(status_payload.get("final_candidates") or []),
            },
            {
                "phase": "review",
                "required_candidate_reviews_expected": len(REVIEWER_ROLES),
                "required_candidate_reviews_expected_effective": len(active_reviewers),
                "required_candidate_reviews_submitted": len(active_latest_reviews),
                "missing_required_reviewer_lanes": missing_original_lanes,
                "missing_active_reviewer_lanes": missing_lanes,
                "exited_reviewer_lanes": exited_lanes,
                "active_reviewer_lanes": active_reviewers,
                "blocking_reviewers": blocking_lanes,
            },
            {
                "phase": "rebuttal",
                "rebuttal_rounds_completed": int(status_payload.get("rebuttal_round") or 0),
            },
        ],
        "submission_cycles": submission_cycles,
        "proposer_trajectory": {
            "role": semantic_role_for(CANDIDATE_OWNER),
            "status": "submitted" if candidate_checked.ok else "invalid",
            "candidate_submission": candidate_payload,
            "artifact": _clone_jsonish(candidate_proposal.get("artifact") or {}),
            "rebuttals": rebuttal_rows,
            "final_candidate": CANDIDATE_OWNER in (status_payload.get("final_candidates") or []),
        },
        "reviewer_trajectories": reviewer_trajectories,
        "review_statuses": {
            CANDIDATE_OWNER: {
                "target_owner": CANDIDATE_OWNER,
                "status": review_completion["status"],
                "required_candidate_reviews_expected": len(REVIEWER_ROLES),
                "required_candidate_reviews_expected_effective": len(active_reviewers),
                "required_candidate_reviews_submitted": len(active_latest_reviews),
                "required_candidate_reviews_submitted_effective": len(active_latest_reviews),
                "missing_required_reviewer_lanes": missing_original_lanes,
                "missing_active_reviewer_lanes": missing_lanes,
                "exited_reviewer_lanes": exited_lanes,
                "active_reviewer_lanes": active_reviewers,
                "review_completion_policy": "active_reviewer_quorum" if exited_lanes else "fixed",
                "reviewers": latest_reviewer_rows,
            }
        },
        "final_review_items": {
            "blocking_items": blocking_items,
            "non_blocking_items": non_blocking_items,
        },
        "overall_confidence": _clone_jsonish(candidate_payload.get("overall_confidence") or {
            "level": "medium" if accepted else "low",
            "rationale": "; ".join(decision_rationale) or "Protocol reconstructed from terminal debate summary.",
        }),
        "execution_warnings": [
            warning
            for warning in [
                "Engine final_candidates includes reviewer placeholders and should not be interpreted as multiple semantic answers."
                if status_payload.get("final_candidates")
                else "",
                f"Missing active reviewer lanes: {', '.join(missing_lanes)}." if missing_lanes else "",
                f"Exited reviewer lanes: {', '.join(exited_lanes)}." if exited_lanes else "",
                f"Blocking candidate reviews remain from: {', '.join(blocking_lanes)}." if blocking_lanes else "",
            ]
            if warning
        ],
    }
    return protocol



def apply_forced_missing_review_completion(
    protocol_payload: dict[str, Any],
    *,
    reason: str,
    missing_lanes: list[str],
    blockers: list[str] | None = None,
    recovery_cycles_without_progress: int = 0,
) -> dict[str, Any]:
    protocol = _clone_jsonish(protocol_payload)
    blocker_list = [str(item).strip() for item in (blockers or []) if str(item).strip()]
    missing_lane_set = {str(item).strip() for item in missing_lanes if str(item).strip()}
    ordered_missing_lanes = [lane for lane in REVIEWER_ROLES if lane in missing_lane_set]

    review_completion = protocol.setdefault("review_completion_status", {})
    existing_submitted = int(review_completion.get("required_candidate_reviews_submitted") or 0)
    review_completion["status"] = "incomplete"
    review_completion["required_candidate_reviews_expected"] = len(REVIEWER_ROLES)
    review_completion["required_candidate_reviews_expected_original"] = len(REVIEWER_ROLES)
    review_completion["required_candidate_reviews_expected_effective"] = max(0, len(REVIEWER_ROLES) - len(ordered_missing_lanes))
    review_completion["required_candidate_reviews_submitted"] = min(existing_submitted, len(REVIEWER_ROLES))
    review_completion["required_candidate_reviews_submitted_effective"] = min(existing_submitted, max(0, len(REVIEWER_ROLES) - len(ordered_missing_lanes)))
    review_completion["required_fixed_reviewer_lanes_complete"] = not ordered_missing_lanes
    review_completion["required_active_reviewer_lanes_complete"] = not ordered_missing_lanes
    review_completion["missing_required_reviewer_lanes"] = ordered_missing_lanes
    review_completion["missing_active_reviewer_lanes"] = ordered_missing_lanes
    review_completion["exited_reviewer_lanes"] = []
    review_completion["active_reviewer_lanes"] = [lane for lane in REVIEWER_ROLES if lane not in ordered_missing_lanes]
    review_completion["review_completion_policy"] = "forced_missing_review_completion"
    review_completion["forced_completion"] = True
    review_completion["forced_completion_reason"] = reason
    review_completion["recovery_cycles_without_progress"] = int(recovery_cycles_without_progress)

    acceptance_decision = protocol.setdefault("acceptance_decision", {})
    acceptance_decision["forced_completion"] = True
    acceptance_decision["forced_completion_reason"] = reason
    acceptance_decision["missing_required_reviewer_lanes"] = ordered_missing_lanes
    acceptance_decision["missing_active_reviewer_lanes"] = ordered_missing_lanes
    acceptance_decision["acceptance_context"] = "forced_missing_review_completion"
    decision_rationale = [str(item).strip() for item in _as_list(acceptance_decision.get("decision_rationale")) if str(item).strip()]
    if reason not in decision_rationale:
        decision_rationale.append(reason)
    acceptance_decision["decision_rationale"] = decision_rationale
    acceptance_decision["reason"] = "; ".join(decision_rationale)

    final_answer = protocol.get("final_answer")
    if isinstance(final_answer, dict):
        final_answer["forced_completion"] = True
        final_answer["forced_completion_reason"] = reason
        if ordered_missing_lanes:
            final_answer["missing_required_reviewer_lanes"] = ordered_missing_lanes

    overall_confidence = protocol.get("overall_confidence")
    if not isinstance(overall_confidence, dict):
        overall_confidence = {}
        protocol["overall_confidence"] = overall_confidence
    overall_confidence["level"] = "low"
    rationale_parts = [str(overall_confidence.get("rationale") or "").strip(), reason]
    if blocker_list:
        rationale_parts.append("Recovery blockers: " + "; ".join(blocker_list))
    overall_confidence["rationale"] = "; ".join(part for part in rationale_parts if part)

    execution_warnings = [str(item).strip() for item in _as_list(protocol.get("execution_warnings")) if str(item).strip()]
    if ordered_missing_lanes:
        missing_warning = f"Forced completion with missing required reviewer lanes: {', '.join(ordered_missing_lanes)}."
        if missing_warning not in execution_warnings:
            execution_warnings.append(missing_warning)
    forced_warning = f"Coordinator forced degraded completion after recovery attempts without full reviewer coverage: {reason}"
    if forced_warning not in execution_warnings:
        execution_warnings.append(forced_warning)
    for blocker in blocker_list:
        warning = f"Recovery blocker: {blocker}"
        if warning not in execution_warnings:
            execution_warnings.append(warning)
    protocol["execution_warnings"] = execution_warnings
    return protocol
