from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote

import requests


PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
DEFAULT_PROPERTIES = [
    "MolecularFormula",
    "MolecularWeight",
    "CanonicalSMILES",
    "IsomericSMILES",
    "InChI",
    "InChIKey",
    "Charge",
]
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
Requester = Callable[..., Any]


@dataclass
class PubChemHttpError(RuntimeError):
    message: str
    http_status: Optional[int] = None
    payload: Any = None
    url: Optional[str] = None
    timed_out: bool = False
    parse_status: str = "not_attempted"
    trace: list[dict[str, Any]] | None = None

    def __str__(self) -> str:
        return self.message


def _compact_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()


def _ensure_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def parse_cli_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PubChem provider skill")
    parser.add_argument("--request-json", required=True, help="Path to request JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for emitted artifacts")
    parser.add_argument("--json", action="store_true", help="Print the final result JSON to stdout")
    return parser.parse_args(argv)


def load_request(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    raise ValueError("request payload must be a JSON object")


def base_result(request: dict[str, Any]) -> dict[str, Any]:
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
        "provider_health": {
            "pubchem": {
                "provider": "pubchem",
                "base_url": PUBCHEM_BASE_URL,
                "status": "idle",
                "calls": 0,
                "successes": 0,
                "retries": 0,
                "retry_exhausted": False,
                "last_url": None,
                "last_http_status": None,
                "last_elapsed_ms": None,
                "last_timeout": False,
                "last_parse_status": "not_attempted",
                "last_error": None,
            }
        },
    }


def write_result(result: dict[str, Any], output_dir: str | Path, filename: str) -> Path:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / filename
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result_path


def finalize_success(
    result: dict[str, Any],
    *,
    output_dir: str | Path,
    filename: str,
    emit_json: bool = False,
) -> dict[str, Any]:
    write_result(result, output_dir, filename)
    if emit_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def handle_exception(
    request: dict[str, Any],
    *,
    output_dir: str | Path,
    filename: str,
    exc: Exception,
    emit_json: bool = False,
) -> dict[str, Any]:
    result = base_result(request)
    message = str(exc) or exc.__class__.__name__
    result["errors"].append({"message": message, "type": exc.__class__.__name__})
    result["provider_health"]["pubchem"]["status"] = "error"
    result["provider_health"]["pubchem"]["last_error"] = message
    write_result(result, output_dir, filename)
    if emit_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


class PubChemClient:
    def __init__(
        self,
        *,
        timeout_seconds: float = 8.0,
        retry_attempts: int = 1,
        requester: Optional[Requester] = None,
    ) -> None:
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.retry_attempts = max(0, int(retry_attempts))
        self.requester = requester or requests.request
        self.health = base_result({})["provider_health"]["pubchem"]

    def _record_attempt(
        self,
        *,
        url: str,
        outcome: str,
        elapsed_ms: float,
        http_status: Optional[int],
        timed_out: bool,
        parse_status: str,
    ) -> dict[str, Any]:
        self.health["calls"] += 1
        self.health["last_url"] = url
        self.health["last_http_status"] = http_status
        self.health["last_elapsed_ms"] = round(elapsed_ms, 3)
        self.health["last_timeout"] = timed_out
        self.health["last_parse_status"] = parse_status
        return {
            "provider": "pubchem",
            "url": url,
            "http_status": http_status,
            "elapsed_ms": round(elapsed_ms, 3),
            "timed_out": timed_out,
            "parse_status": parse_status,
            "outcome": outcome,
        }

    def request_json(self, path: str, *, params: Optional[dict[str, Any]] = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        encoded_path = "/".join(quote(part, safe=",") for part in path.split("/"))
        url = f"{PUBCHEM_BASE_URL}/{encoded_path}/JSON"
        trace: list[dict[str, Any]] = []
        attempts = self.retry_attempts + 1
        for attempt in range(1, attempts + 1):
            timed_out = False
            http_status: Optional[int] = None
            parse_status = "not_attempted"
            started = time.perf_counter()
            try:
                response = self.requester("GET", url, params=params, timeout=self.timeout_seconds)
                elapsed_ms = (time.perf_counter() - started) * 1000
                http_status = int(getattr(response, "status_code", 200))
                payload: Any = None
                try:
                    payload = response.json()
                    parse_status = "ok"
                except ValueError as exc:
                    parse_status = "invalid_json"
                    raise PubChemHttpError(
                        "PubChem returned invalid JSON",
                        http_status=http_status,
                        payload=getattr(response, "text", None),
                        url=str(getattr(response, "url", url)),
                        parse_status=parse_status,
                        trace=list(trace),
                    ) from exc
                if 200 <= http_status < 300:
                    self.health["successes"] += 1
                    self.health["status"] = "healthy"
                    self.health["last_error"] = None
                    trace.append(
                        self._record_attempt(
                            url=str(getattr(response, "url", url)),
                            outcome="success",
                            elapsed_ms=elapsed_ms,
                            http_status=http_status,
                            timed_out=False,
                            parse_status=parse_status,
                        )
                    )
                    return payload if isinstance(payload, dict) else {"payload": payload}, trace
                fault = payload.get("Fault") if isinstance(payload, dict) else None
                message = _compact_text((fault or {}).get("Message")) or f"PubChem HTTP {http_status}"
                outcome = "not_found" if http_status == 404 else "http_error"
                trace.append(
                    self._record_attempt(
                        url=str(getattr(response, "url", url)),
                        outcome=outcome,
                        elapsed_ms=elapsed_ms,
                        http_status=http_status,
                        timed_out=False,
                        parse_status=parse_status,
                    )
                )
                if http_status in RETRYABLE_STATUS_CODES and attempt < attempts:
                    self.health["retries"] += 1
                    time.sleep(min(0.2 * attempt, 0.5))
                    continue
                self.health["status"] = "error"
                self.health["last_error"] = message
                self.health["retry_exhausted"] = attempt >= attempts and http_status in RETRYABLE_STATUS_CODES
                raise PubChemHttpError(
                    message,
                    http_status=http_status,
                    payload=payload,
                    url=str(getattr(response, "url", url)),
                    parse_status=parse_status,
                    trace=list(trace),
                )
            except requests.exceptions.Timeout as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000
                timed_out = True
                trace.append(
                    self._record_attempt(
                        url=url,
                        outcome="timeout",
                        elapsed_ms=elapsed_ms,
                        http_status=http_status,
                        timed_out=timed_out,
                        parse_status=parse_status,
                    )
                )
                if attempt < attempts:
                    self.health["retries"] += 1
                    time.sleep(min(0.2 * attempt, 0.5))
                    continue
                self.health["status"] = "error"
                self.health["last_error"] = str(exc)
                self.health["retry_exhausted"] = True
                raise PubChemHttpError(
                    str(exc) or "PubChem timeout",
                    http_status=http_status,
                    payload=None,
                    url=url,
                    timed_out=True,
                    parse_status=parse_status,
                    trace=list(trace),
                ) from exc
            except requests.exceptions.RequestException as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000
                trace.append(
                    self._record_attempt(
                        url=url,
                        outcome="request_error",
                        elapsed_ms=elapsed_ms,
                        http_status=http_status,
                        timed_out=timed_out,
                        parse_status=parse_status,
                    )
                )
                if attempt < attempts:
                    self.health["retries"] += 1
                    time.sleep(min(0.2 * attempt, 0.5))
                    continue
                self.health["status"] = "error"
                self.health["last_error"] = str(exc)
                self.health["retry_exhausted"] = True
                raise PubChemHttpError(
                    str(exc),
                    http_status=http_status,
                    payload=None,
                    url=url,
                    timed_out=False,
                    parse_status=parse_status,
                    trace=list(trace),
                ) from exc
        raise PubChemHttpError("PubChem request failed")

    def clone_health(self) -> dict[str, Any]:
        return dict(self.health)


def status_from_parts(*, usable: bool, partial: bool) -> str:
    if usable and partial:
        return "partial"
    if usable:
        return "success"
    return "error"


def merge_health(result: dict[str, Any], client: PubChemClient) -> None:
    result["provider_health"]["pubchem"] = client.clone_health()


def apply_http_error(result: dict[str, Any], exc: PubChemHttpError) -> dict[str, Any]:
    result["errors"].append(
        {
            "message": str(exc),
            "http_status": exc.http_status,
            "timed_out": exc.timed_out,
            "url": exc.url,
        }
    )
    return result


def is_pubchem_error(exc: Exception) -> bool:
    return exc.__class__.__name__ == "PubChemHttpError" and hasattr(exc, "parse_status") and hasattr(exc, "timed_out")


def require_fields(request: dict[str, Any], fields: list[str]) -> None:
    missing = [name for name in fields if request.get(name) in (None, "", [])]
    if missing:
        raise ValueError(f"missing required request fields: {', '.join(missing)}")


def normalize_cids(value: Any) -> list[int]:
    numbers: list[int] = []
    for item in _ensure_list(value):
        try:
            numbers.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid CID value: {item!r}") from exc
    if not numbers:
        raise ValueError("at least one CID is required")
    return numbers


def normalize_properties(value: Any) -> list[str]:
    props = [_compact_text(item) for item in _ensure_list(value) if _compact_text(item)]
    return props or list(DEFAULT_PROPERTIES)
