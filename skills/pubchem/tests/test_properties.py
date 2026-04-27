from __future__ import annotations

def test_property_parsing_complete_response(load_module, tmp_path) -> None:
    module = load_module("cid_to_properties")

    class DummyResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self.url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,InChI,InChIKey,Charge/JSON"
            self.text = ""

        def json(self) -> dict:
            return {
                "PropertyTable": {
                    "Properties": [
                        {
                            "CID": 2244,
                            "MolecularFormula": "C9H8O4",
                            "MolecularWeight": 180.16,
                            "CanonicalSMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
                            "IsomericSMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
                            "InChI": "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)",
                            "InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                            "Charge": 0,
                        }
                    ]
                }
            }

    payload = module.run({"cids": [2244]}, output_dir=tmp_path, requester=lambda *args, **kwargs: DummyResponse())

    assert payload["status"] == "success"
    assert payload["primary_result"]["cid"] == 2244
    assert payload["primary_result"]["molecular_formula"] == "C9H8O4"
    assert payload["primary_result"]["canonical_smiles"].startswith("CC(=O)")
    assert payload["warnings"] == []


def test_property_parsing_partial_response(load_module, tmp_path) -> None:
    module = load_module("cid_to_properties")

    class DummyResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self.url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,InChI,InChIKey,Charge/JSON"
            self.text = ""

        def json(self) -> dict:
            return {
                "PropertyTable": {
                    "Properties": [
                        {
                            "CID": 2244,
                            "MolecularFormula": "C9H8O4",
                            "CanonicalSMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
                        }
                    ]
                }
            }

    payload = module.run({"cids": [2244]}, output_dir=tmp_path, requester=lambda *args, **kwargs: DummyResponse())

    assert payload["status"] == "partial"
    assert payload["primary_result"]["cid"] == 2244
    assert payload["primary_result"]["missing_properties"]
    assert payload["warnings"]
