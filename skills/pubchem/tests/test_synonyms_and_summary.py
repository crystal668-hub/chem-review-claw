from __future__ import annotations


def test_synonyms_include_provider_health_and_source_trace(load_module, tmp_path) -> None:
    module = load_module("synonyms")

    class DummyResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self.url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/synonyms/JSON"
            self.text = ""

        def json(self) -> dict:
            return {
                "InformationList": {
                    "Information": [
                        {
                            "CID": 2244,
                            "Synonym": ["Aspirin", "2-Acetoxybenzoic acid", "Acetylsalicylic acid"],
                        }
                    ]
                }
            }

    payload = module.run({"cid": 2244, "max_synonyms": 2}, output_dir=tmp_path, requester=lambda *args, **kwargs: DummyResponse())

    assert payload["status"] == "success"
    assert payload["primary_result"]["cid"] == 2244
    assert payload["primary_result"]["synonyms"] == ["Aspirin", "2-Acetoxybenzoic acid"]
    assert payload["provider_health"]["pubchem"]["status"] == "healthy"
    assert payload["source_trace"][0]["url"].endswith("/compound/cid/2244/synonyms/JSON")


def test_compound_summary_chains_resolution_and_properties(load_module, tmp_path) -> None:
    module = load_module("compound_summary")

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
            payload={"IdentifierList": {"CID": [2244]}},
            url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/cids/JSON",
        ),
        DummyResponse(
            status_code=200,
            payload={
                "PropertyTable": {
                    "Properties": [
                        {
                            "CID": 2244,
                            "MolecularFormula": "C9H8O4",
                            "MolecularWeight": 180.16,
                            "CanonicalSMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
                        }
                    ]
                }
            },
            url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,InChI,InChIKey,Charge/JSON",
        ),
        DummyResponse(
            status_code=200,
            payload={"InformationList": {"Information": [{"CID": 2244, "Synonym": ["Aspirin", "Acetylsalicylic acid"]}]}},
            url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/synonyms/JSON",
        ),
    ]

    payload = module.run({"query": "aspirin", "synonym_limit": 2}, output_dir=tmp_path, requester=lambda *args, **kwargs: responses.pop(0))

    assert payload["status"] in {"success", "partial"}
    assert payload["primary_result"]["cid"] == 2244
    assert payload["primary_result"]["molecular_formula"] == "C9H8O4"
    assert payload["primary_result"]["synonyms"] == ["Aspirin", "Acetylsalicylic acid"]
