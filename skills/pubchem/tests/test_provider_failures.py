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


def test_http_error_response_is_structured(load_module, tmp_path: Path) -> None:
    module = load_module("name_to_cid")

    def fake_requester(method: str, url: str, **_: object) -> DummyResponse:
        return DummyResponse(
            status_code=503,
            payload={"Fault": {"Code": "PUGREST.ServerBusy", "Message": "Busy"}},
            url=url,
        )

    payload = module.run({"query": "aspirin", "retry_attempts": 0}, output_dir=tmp_path, requester=fake_requester)

    assert payload["status"] == "error"
    assert payload["errors"]
    assert payload["provider_health"]["pubchem"]["status"] == "error"
    assert payload["provider_health"]["pubchem"]["last_http_status"] == 503
    assert payload["source_trace"][-1]["outcome"] == "http_error"
