from __future__ import annotations

def test_formula_search_multiple_candidates(load_module, tmp_path) -> None:
    module = load_module("formula_search")
    calls: list[tuple[str, str, dict | None]] = []

    class DummyResponse:
        def __init__(self, *, status_code: int, payload: dict, url: str) -> None:
            self.status_code = status_code
            self._payload = payload
            self.url = url
            self.text = ""

        def json(self) -> dict:
            return self._payload

    responses = [
        DummyResponse(
            status_code=200,
            payload={"IdentifierList": {"CID": [702, 962]}},
            url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastformula/C2H6O/cids/JSON",
        ),
        DummyResponse(
            status_code=200,
            payload={
                "PropertyTable": {
                    "Properties": [
                        {"CID": 702, "MolecularFormula": "C2H6O", "CanonicalSMILES": "CCO", "MolecularWeight": 46.07},
                        {"CID": 962, "MolecularFormula": "C2H6O", "CanonicalSMILES": "COC", "MolecularWeight": 46.07},
                    ]
                }
            },
            url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/702,962/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,InChI,InChIKey,Charge/JSON",
        ),
    ]

    def fake_requester(method: str, url: str, **kwargs):
        calls.append((method, url, kwargs.get("params")))
        return responses.pop(0)

    payload = module.run({"formula": "C2H6O", "max_candidates": 2}, output_dir=tmp_path, requester=fake_requester)

    assert payload["status"] == "success"
    assert len(payload["candidates"]) == 2
    assert payload["primary_result"]["cid"] == 702
    assert calls[0][1].endswith("/compound/fastformula/C2H6O/cids/JSON")
    assert calls[1][1].endswith("/compound/cid/702,962/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,InChI,InChIKey,Charge/JSON")


def test_similarity_search_request_construction_and_partial_failure(load_module, tmp_path) -> None:
    module = load_module("similarity_search")
    calls: list[tuple[str, str, dict | None]] = []

    class DummyResponse:
        def __init__(self, *, status_code: int, payload: dict, url: str) -> None:
            self.status_code = status_code
            self._payload = payload
            self.url = url
            self.text = ""

        def json(self) -> dict:
            return self._payload

    responses = [
        DummyResponse(
            status_code=200,
            payload={"IdentifierList": {"CID": [702, 962]}},
            url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/CCO/cids/JSON",
        ),
        DummyResponse(
            status_code=503,
            payload={"Fault": {"Code": "PUGREST.ServerBusy", "Message": "Busy"}},
            url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/702,962/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,InChI,InChIKey,Charge/JSON",
        ),
    ]

    def fake_requester(method: str, url: str, **kwargs):
        calls.append((method, url, kwargs.get("params")))
        return responses.pop(0)

    payload = module.run(
        {"query_smiles": "CCO", "threshold": 95, "max_records": 2, "retry_attempts": 0},
        output_dir=tmp_path,
        requester=fake_requester,
    )

    assert calls[0][1].endswith("/compound/fastsimilarity_2d/smiles/CCO/cids/JSON")
    assert calls[0][2]["Threshold"] == 95
    assert calls[0][2]["MaxRecords"] == 2
    assert payload["status"] == "partial"
    assert [item["cid"] for item in payload["candidates"]] == [702, 962]
    assert payload["warnings"]
    assert payload["provider_health"]["pubchem"]["last_http_status"] == 503
