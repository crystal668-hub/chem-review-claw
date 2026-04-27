from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote


RESULT_FILENAME = "result.json"
OPSIN_BASE_URL = "https://www.ebi.ac.uk/opsin/ws"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_RETRY_ATTEMPTS = 1
MAX_RETRY_ATTEMPTS = 2
RETRYABLE_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}


def compact_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()


def ensure_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [compact_text(item) for item in value if compact_text(item)]
    text = compact_text(value)
    return [text] if text else []


def parse_args(argv: list[str] | None, description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--request-json", required=True, help="Path to request JSON payload")
    parser.add_argument("--output-dir", required=True, help="Directory for result.json")
    parser.add_argument("--json", action="store_true", help="Print result JSON to stdout")
    return parser.parse_args(argv)


def load_request(path: str) -> dict[str, Any]:
    request_path = Path(path).expanduser().resolve()
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request JSON must be an object")
    return payload


def init_payload(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "error",
        "request": request,
        "primary_result": {},
        "candidates": [],
        "diagnostics": [],
        "warnings": [],
        "errors": [],
        "tool_trace": [],
        "source_trace": [],
        "provider_health": {},
    }


def finalize_payload(payload: dict[str, Any], *, output_dir: str | Path, emit_json: bool) -> None:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    result_path = output_path / RESULT_FILENAME
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if emit_json:
        print(json.dumps(payload, indent=2, sort_keys=True))


def exit_code_for_status(status: str) -> int:
    return 0 if status in {"success", "partial"} else 1


def invalid_request_payload(request: dict[str, Any], message: str) -> dict[str, Any]:
    payload = init_payload(request)
    payload["status"] = "error"
    payload["primary_result"] = {
        "result_kind": "invalid_request",
        "validation_status": "invalid",
        "validation_errors": [message],
    }
    payload["errors"] = [{"code": "invalid_request", "message": message}]
    return payload


def build_provider_url(name: str, base_url: str = OPSIN_BASE_URL) -> str:
    return f"{base_url.rstrip('/')}/{quote(name, safe='')}.json"


def normalize_diagnostic(
    *,
    input_name: str,
    provider_status: str | None,
    provider_message: Any,
) -> dict[str, Any]:
    message = compact_text(provider_message)
    lowered = message.lower()
    category = "parse_failure"
    severity = "error"
    likely_action = "review_name"

    if "unsupported" in lowered:
        category = "unsupported_syntax"
        severity = "warning"
        likely_action = "rewrite_with_supported_iupac_syntax"
    elif "ambiguous" in lowered or "multiple structures" in lowered:
        category = "ambiguous_name"
        severity = "warning"
        likely_action = "choose_a_more_specific_name"
    elif "non-systematic" in lowered or "non systematic" in lowered or "common name" in lowered or "trivial name" in lowered:
        category = "non_systematic_name"
        severity = "warning"
        likely_action = "use_a_systematic_name_or_pubchem"
    elif "malformed" in lowered or "cannot parse" in lowered or "invalid" in lowered:
        category = "malformed_input"
        severity = "error"
        likely_action = "fix_name_syntax"

    return {
        "input_name": input_name,
        "provider": "opsin",
        "provider_status": compact_text(provider_status) or None,
        "provider_message": message,
        "category": category,
        "severity": severity,
        "likely_action": likely_action,
    }


def _base_provider_health(provider_url: str, timeout_seconds: float, retry_attempts: int) -> dict[str, Any]:
    return {
        "provider_url": provider_url,
        "timeout_seconds": timeout_seconds,
        "retry_attempts": retry_attempts,
        "attempts_made": 0,
        "retries_used": 0,
        "http_status": None,
        "elapsed_ms": None,
        "timeout": False,
        "parse_status": None,
        "status": "idle",
        "message": None,
    }


def run_opsin_lookup(
    request: dict[str, Any],
    *,
    requests_get: Callable[..., Any],
) -> dict[str, Any]:
    payload = init_payload(request)
    name = compact_text(request.get("name"))
    if not name:
        return invalid_request_payload(request, "request.name is required")

    timeout_seconds = float(request.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS)
    retry_attempts = min(MAX_RETRY_ATTEMPTS, max(0, int(request.get("retry_attempts") or DEFAULT_RETRY_ATTEMPTS)))
    provider_url = build_provider_url(name, compact_text(request.get("provider_base_url")) or OPSIN_BASE_URL)
    provider_health = _base_provider_health(provider_url, timeout_seconds, retry_attempts)
    payload["provider_health"] = {"opsin": provider_health}

    max_attempts = retry_attempts + 1
    for attempt in range(1, max_attempts + 1):
        provider_health["attempts_made"] = attempt
        started = time.perf_counter()
        try:
            response = requests_get(
                provider_url,
                headers={"Accept": "application/json"},
                timeout=timeout_seconds,
            )
            provider_health["http_status"] = getattr(response, "status_code", None)
            response.raise_for_status()
            response_payload = response.json()
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
            parse_status = compact_text(response_payload.get("status")).upper() or "UNKNOWN"
            provider_health["elapsed_ms"] = elapsed_ms
            provider_health["parse_status"] = parse_status
            provider_health["status"] = "healthy"
            provider_health["message"] = compact_text(response_payload.get("message")) or None

            payload["source_trace"].append(
                {
                    "provider": "opsin",
                    "provider_url": provider_url,
                    "attempt": attempt,
                    "http_status": provider_health["http_status"],
                    "elapsed_ms": elapsed_ms,
                    "timeout": False,
                    "parse_status": parse_status,
                }
            )

            for warning in ensure_string_list(response_payload.get("warnings")):
                payload["warnings"].append(
                    {
                        "provider": "opsin",
                        "input_name": name,
                        "message": warning,
                    }
                )

            smiles = compact_text(response_payload.get("smiles")) or None
            stdinchi = compact_text(response_payload.get("stdinchi")) or None
            stdinchikey = compact_text(response_payload.get("stdinchikey")) or None
            inchi = compact_text(response_payload.get("inchi")) or None
            cml = compact_text(response_payload.get("cml")) or None
            message = compact_text(response_payload.get("message")) or None

            if parse_status == "SUCCESS" and any([smiles, stdinchi, stdinchikey, inchi, cml]):
                payload["status"] = "success"
                payload["primary_result"] = {
                    "input_name": name,
                    "result_kind": "structure",
                    "smiles": smiles,
                    "stdinchi": stdinchi,
                    "stdinchikey": stdinchikey,
                    "inchi": inchi,
                    "cml": cml,
                    "provider_message": message,
                    "validation_status": "not_validated",
                    "validation_errors": [],
                }
                return payload

            payload["status"] = "error"
            payload["primary_result"] = {
                "input_name": name,
                "result_kind": "no_result",
                "provider_message": message,
                "validation_status": "not_applicable",
                "validation_errors": [],
            }
            payload["diagnostics"].append(
                normalize_diagnostic(
                    input_name=name,
                    provider_status=parse_status,
                    provider_message=message or "OPSIN returned no structure",
                )
            )
            return payload
        except Exception as exc:  # requests exception types are resolved in wrapper scope
            from requests import exceptions as request_exceptions  # local import keeps helper lightweight

            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
            provider_health["elapsed_ms"] = elapsed_ms
            provider_health["message"] = compact_text(str(exc)) or exc.__class__.__name__

            error_code = "provider_request_error"
            transient = False
            if isinstance(exc, request_exceptions.Timeout):
                provider_health["status"] = "timeout"
                provider_health["timeout"] = True
                error_code = "provider_timeout"
                transient = True
            elif isinstance(exc, request_exceptions.HTTPError):
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                provider_health["http_status"] = status_code
                provider_health["status"] = "degraded"
                error_code = "provider_http_error"
                transient = status_code in RETRYABLE_HTTP_STATUSES
            elif isinstance(exc, request_exceptions.RequestException):
                provider_health["status"] = "degraded"
                error_code = "provider_request_error"
                transient = True
            else:
                provider_health["status"] = "degraded"
                error_code = "provider_parse_error"

            payload["source_trace"].append(
                {
                    "provider": "opsin",
                    "provider_url": provider_url,
                    "attempt": attempt,
                    "http_status": provider_health["http_status"],
                    "elapsed_ms": elapsed_ms,
                    "timeout": provider_health["timeout"],
                    "parse_status": provider_health["parse_status"],
                }
            )

            if transient and attempt < max_attempts:
                provider_health["retries_used"] = attempt
                payload["tool_trace"].append(
                    {
                        "tool": "opsin_lookup",
                        "attempt": attempt,
                        "action": "retry",
                        "reason": error_code,
                    }
                )
                continue

            payload["status"] = "error"
            payload["primary_result"] = {
                "input_name": name,
                "result_kind": "provider_failure",
                "provider_message": provider_health["message"],
                "validation_status": "not_applicable",
                "validation_errors": [],
            }
            payload["errors"].append(
                {
                    "code": error_code,
                    "message": provider_health["message"],
                    "provider": "opsin",
                }
            )
            return payload

    payload["status"] = "error"
    payload["primary_result"] = {
        "input_name": name,
        "result_kind": "provider_failure",
        "provider_message": "OPSIN lookup exhausted retries",
        "validation_status": "not_applicable",
        "validation_errors": [],
    }
    payload["errors"].append(
        {
            "code": "provider_retry_exhausted",
            "message": "OPSIN lookup exhausted retries",
            "provider": "opsin",
        }
    )
    return payload
