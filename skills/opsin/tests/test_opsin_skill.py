from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
import requests

from .conftest import RESULT_FILENAME, load_script_module, write_request


class FakeResponse:
    def __init__(self, *, status_code: int = 200, payload: dict | None = None, text: str | None = None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text if text is not None else json.dumps(self._payload)

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}", response=self)


def run_cli(module, tmp_path: Path, request_payload: dict, capsys) -> tuple[int, dict]:
    request_path = write_request(tmp_path, request_payload)
    output_dir = tmp_path / "output"
    exit_code = module.main(
        [
            "--request-json",
            str(request_path),
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )
    stdout = capsys.readouterr().out.strip()
    payload = json.loads(stdout)
    written_payload = json.loads((output_dir / RESULT_FILENAME).read_text(encoding="utf-8"))
    assert written_payload == payload
    return exit_code, payload


def test_successful_systematic_name_resolution(tmp_path: Path, capsys) -> None:
    module = load_script_module("name_to_structure")
    request_payload = {"name": "ethyl ethanoate"}
    fake_payload = {
        "status": "SUCCESS",
        "smiles": "CCOC(=O)C",
        "stdinchi": "InChI=1S/C4H8O2/c1-3-6-4(2)5/h3H2,1-2H3",
        "stdinchikey": "XEKOWRVHYACXOJ-UHFFFAOYSA-N",
        "inchi": "InChI=1/C4H8O2/c1-3-6-4(2)5/h3H2,1-2H3",
        "warnings": [],
        "message": "",
    }

    with mock.patch.object(module.requests, "get", return_value=FakeResponse(payload=fake_payload)):
        exit_code, payload = run_cli(module, tmp_path, request_payload, capsys)

    assert exit_code == 0
    assert payload["status"] == "success"
    assert payload["primary_result"]["input_name"] == "ethyl ethanoate"
    assert payload["primary_result"]["smiles"] == "CCOC(=O)C"
    assert payload["primary_result"]["result_kind"] == "structure"
    assert payload["provider_health"]["opsin"]["http_status"] == 200
    assert payload["provider_health"]["opsin"]["timeout"] is False
    assert payload["provider_health"]["opsin"]["parse_status"] == "SUCCESS"
    assert payload["source_trace"][0]["provider_url"].endswith("/ethyl%20ethanoate.json")


def test_unparseable_common_name_case(tmp_path: Path, capsys) -> None:
    module = load_script_module("name_to_structure")
    request_payload = {"name": "aspirin"}
    fake_payload = {
        "status": "FAILURE",
        "message": "Name appears to be a non-systematic or ambiguous chemical name",
        "warnings": [],
    }

    with mock.patch.object(module.requests, "get", return_value=FakeResponse(payload=fake_payload)):
        exit_code, payload = run_cli(module, tmp_path, request_payload, capsys)

    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["primary_result"]["result_kind"] == "no_result"
    assert payload["diagnostics"][0]["category"] in {"non_systematic_name", "ambiguous_name"}
    assert payload["provider_health"]["opsin"]["status"] == "healthy"
    assert payload["errors"] == []


def test_batch_input_with_mixed_success_failure(tmp_path: Path, capsys) -> None:
    module = load_script_module("batch_name_to_structure")
    request_payload = {"names": ["ethyl ethanoate", "aspirin"]}
    responses = [
        FakeResponse(
            payload={
                "status": "SUCCESS",
                "smiles": "CCOC(=O)C",
                "stdinchi": "InChI=1S/C4H8O2/c1-3-6-4(2)5/h3H2,1-2H3",
                "stdinchikey": "XEKOWRVHYACXOJ-UHFFFAOYSA-N",
                "message": "",
                "warnings": [],
            }
        ),
        FakeResponse(
            payload={
                "status": "FAILURE",
                "message": "Name appears to be a non-systematic chemical name",
                "warnings": [],
            }
        ),
    ]

    with mock.patch.object(module.requests, "get", side_effect=responses):
        exit_code, payload = run_cli(module, tmp_path, request_payload, capsys)

    assert exit_code == 0
    assert payload["status"] == "partial"
    assert payload["primary_result"]["success_count"] == 1
    assert payload["primary_result"]["failure_count"] == 1
    assert len(payload["candidates"]) == 2
    assert payload["candidates"][0]["status"] == "success"
    assert payload["candidates"][1]["primary_result"]["result_kind"] == "no_result"


@pytest.mark.parametrize(
    ("request_setup", "expected_code", "expected_http_status", "expected_timeout"),
    [
        ({"side_effect": requests.exceptions.Timeout("timed out")}, "provider_timeout", None, True),
        ({"return_value": FakeResponse(status_code=503, payload={"status": "FAILURE", "message": "service unavailable"})}, "provider_http_error", 503, False),
    ],
)
def test_provider_timeout_and_http_error_response(
    tmp_path: Path,
    capsys,
    request_setup,
    expected_code: str,
    expected_http_status: int | None,
    expected_timeout: bool,
) -> None:
    module = load_script_module("name_to_structure")
    request_payload = {"name": "ethyl ethanoate", "timeout_seconds": 0.01, "retry_attempts": 1}

    with mock.patch.object(module.requests, "get", **request_setup):
        exit_code, payload = run_cli(module, tmp_path, request_payload, capsys)

    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["primary_result"]["result_kind"] == "provider_failure"
    assert payload["errors"][0]["code"] == expected_code
    assert payload["provider_health"]["opsin"]["timeout"] is expected_timeout
    assert payload["provider_health"]["opsin"]["http_status"] == expected_http_status


def test_diagnostics_normalization_for_unsupported_or_ambiguous_names(tmp_path: Path, capsys) -> None:
    module = load_script_module("parse_diagnostics")
    request_payload = {
        "diagnostics": [
            {
                "input_name": "eta-cyclopentadienyl iron",
                "provider_status": "FAILURE",
                "provider_message": "Unsupported nomenclature feature: eta bonding notation",
            },
            {
                "input_name": "xylene",
                "provider_status": "FAILURE",
                "provider_message": "Name appears to be ambiguous between multiple structures",
            },
        ]
    }

    exit_code, payload = run_cli(module, tmp_path, request_payload, capsys)

    assert exit_code == 0
    assert payload["status"] == "success"
    categories = [item["category"] for item in payload["diagnostics"]]
    assert categories == ["unsupported_syntax", "ambiguous_name"]
    assert payload["primary_result"]["normalized_count"] == 2


def test_rdkit_validation_success_from_mocked_opsin_output(tmp_path: Path, capsys) -> None:
    module = load_script_module("validate_with_rdkit")
    request_payload = {
        "name": "ethanol",
        "opsin_result": {
            "smiles": "CCO",
            "stdinchi": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
            "stdinchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        },
    }

    exit_code, payload = run_cli(module, tmp_path, request_payload, capsys)

    assert exit_code == 0
    assert payload["status"] == "success"
    assert payload["primary_result"]["validation_status"] == "valid"
    assert payload["primary_result"]["canonical_smiles"] == "CCO"
    assert payload["errors"] == []


def test_rdkit_validation_failure_from_mocked_opsin_output(tmp_path: Path, capsys) -> None:
    module = load_script_module("validate_with_rdkit")
    request_payload = {
        "name": "broken",
        "opsin_result": {
            "smiles": "not-a-smiles",
        },
    }

    exit_code, payload = run_cli(module, tmp_path, request_payload, capsys)

    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["primary_result"]["validation_status"] == "invalid"
    assert payload["primary_result"]["result_kind"] == "invalid_structure"
    assert payload["errors"][0]["code"] == "rdkit_parse_failed"


def test_result_json_distinguishes_no_result_from_provider_failure(tmp_path: Path, capsys) -> None:
    module = load_script_module("name_to_structure")

    no_result_payload = {
        "status": "FAILURE",
        "message": "Name appears to be a non-systematic chemical name",
        "warnings": [],
    }
    with mock.patch.object(module.requests, "get", return_value=FakeResponse(payload=no_result_payload)):
        _, no_result = run_cli(module, tmp_path / "no-result", {"name": "aspirin"}, capsys)

    with mock.patch.object(module.requests, "get", side_effect=requests.exceptions.Timeout("timed out")):
        _, provider_failure = run_cli(module, tmp_path / "provider-failure", {"name": "ethyl ethanoate"}, capsys)

    assert no_result["primary_result"]["result_kind"] == "no_result"
    assert provider_failure["primary_result"]["result_kind"] == "provider_failure"
    assert no_result["provider_health"]["opsin"]["status"] == "healthy"
    assert provider_failure["provider_health"]["opsin"]["status"] == "timeout"
