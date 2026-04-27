from __future__ import annotations

from pathlib import Path

import requests


class DummyResponse:
    def __init__(self, *, status_code: int, payload: dict, url: str) -> None:
        self.status_code = status_code
        self._payload = payload
        self.url = url
        self.text = ""

    def json(self) -> dict:
        return self._payload


def test_name_to_cid_mocked_success(load_module, tmp_path: Path) -> None:
    module = load_module("name_to_cid")

    def fake_requester(method: str, url: str, **_: object) -> DummyResponse:
        assert method == "GET"
        return DummyResponse(
            status_code=200,
            payload={"IdentifierList": {"CID": [2244]}},
            url=url,
        )

    payload = module.run({"query": "aspirin", "max_candidates": 5}, output_dir=tmp_path, requester=fake_requester)

    assert payload["status"] == "success"
    assert payload["primary_result"]["cid"] == 2244
    assert payload["candidates"][0]["cid"] == 2244
    assert payload["provider_health"]["pubchem"]["status"] == "healthy"
    assert payload["source_trace"][0]["url"].endswith("/compound/name/aspirin/cids/JSON")
    assert (tmp_path / "name_to_cid_result.json").is_file()


def test_name_to_cid_no_hit_response(load_module, tmp_path: Path) -> None:
    module = load_module("name_to_cid")

    def fake_requester(method: str, url: str, **_: object) -> DummyResponse:
        return DummyResponse(
            status_code=404,
            payload={"Fault": {"Code": "PUGREST.NotFound", "Message": "No CID found"}},
            url=url,
        )

    payload = module.run({"query": "notarealcompound"}, output_dir=tmp_path, requester=fake_requester)

    assert payload["status"] == "error"
    assert payload["primary_result"] == {}
    assert payload["candidates"] == []
    assert payload["errors"]
    assert payload["source_trace"][0]["outcome"] == "not_found"
    assert payload["provider_health"]["pubchem"]["last_http_status"] == 404


def test_name_to_cid_multiple_candidates_yields_partial(load_module, tmp_path: Path) -> None:
    module = load_module("name_to_cid")

    def fake_requester(method: str, url: str, **_: object) -> DummyResponse:
        return DummyResponse(
            status_code=200,
            payload={"IdentifierList": {"CID": [702, 6322, 11125]}},
            url=url,
        )

    payload = module.run({"query": "mustard gas", "max_candidates": 3}, output_dir=tmp_path, requester=fake_requester)

    assert payload["status"] == "partial"
    assert payload["primary_result"]["cid"] == 702
    assert [item["cid"] for item in payload["candidates"]] == [702, 6322, 11125]
    assert payload["warnings"]


def test_name_to_cid_timeout_response(load_module, tmp_path: Path, monkeypatch) -> None:
    module = load_module("name_to_cid")
    common = load_module("pubchem_common")
    monkeypatch.setattr(common.time, "sleep", lambda *_: None)
    calls: list[str] = []

    def fake_requester(method: str, url: str, **_: object) -> DummyResponse:
        calls.append(url)
        raise requests.exceptions.Timeout("timed out")

    payload = module.run(
        {"query": "aspirin", "retry_attempts": 1, "timeout_seconds": 0.01},
        output_dir=tmp_path,
        requester=fake_requester,
    )

    assert payload["status"] == "error"
    assert len(calls) == 2
    assert payload["provider_health"]["pubchem"]["status"] == "error"
    assert payload["provider_health"]["pubchem"]["last_timeout"] is True
    assert payload["source_trace"][-1]["timed_out"] is True
