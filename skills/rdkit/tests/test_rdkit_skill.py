from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import shutil
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SKILL_ROOT = ROOT / "skills" / "rdkit"
SCRIPTS_ROOT = SKILL_ROOT / "scripts"


def run_script(script_name: str, request: dict) -> tuple[dict, Path]:
    temp_path = Path(tempfile.mkdtemp(prefix="rdkit-skill-test-"))
    request_path = temp_path / "request.json"
    output_dir = temp_path / "out"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / script_name),
            "--request-json",
            str(request_path),
            "--output-dir",
            str(output_dir),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout = completed.stdout.strip()
    payload = json.loads(stdout or "{}")
    result_path = output_dir / "result.json"
    if result_path.exists():
        disk_payload = json.loads(result_path.read_text(encoding="utf-8"))
        assert payload == disk_payload
    payload["_exit_code"] = completed.returncode
    return payload, result_path


class RdkitSkillLayoutTests(unittest.TestCase):
    def test_skill_layout_files_exist(self) -> None:
        expected = [
            SKILL_ROOT / "SKILL.md",
            SKILL_ROOT / "routing-rules.md",
            SKILL_ROOT / "references" / "contracts.md",
        ]
        for path in expected:
            self.assertTrue(path.is_file(), f"missing required file: {path}")

    def test_all_required_scripts_exist(self) -> None:
        expected = [
            "canonicalize.py",
            "descriptors.py",
            "functional_groups.py",
            "substructure.py",
            "rings_aromaticity.py",
            "stereochemistry.py",
            "similarity.py",
            "reaction_smarts.py",
            "conformer_embed.py",
            "nmr_symmetry_heuristics.py",
        ]
        for name in expected:
            self.assertTrue((SCRIPTS_ROOT / name).is_file(), f"missing script: {name}")


class CanonicalizeTests(unittest.TestCase):
    def test_valid_smiles_canonicalization(self) -> None:
        payload, result_path = run_script(
            "canonicalize.py",
            {"molecule": {"format": "smiles", "value": "OCC"}},
        )

        self.assertEqual("success", payload["status"])
        self.assertTrue(result_path.is_file())
        primary = payload["primary_result"]
        self.assertEqual("CCO", primary["canonical_smiles"])
        self.assertEqual("CCO", primary["isomeric_smiles"])
        self.assertTrue(primary["valid"])
        self.assertEqual([], payload["errors"])

    def test_invalid_smiles_returns_structured_error(self) -> None:
        payload, _ = run_script(
            "canonicalize.py",
            {"molecule": {"format": "smiles", "value": "C1CC"}},
        )

        self.assertEqual("error", payload["status"])
        self.assertFalse(payload["primary_result"]["valid"])
        self.assertTrue(payload["errors"])
        self.assertEqual(0, payload["_exit_code"])


class RingsAromaticityTests(unittest.TestCase):
    def test_aromatic_ring_example(self) -> None:
        payload, _ = run_script(
            "rings_aromaticity.py",
            {"molecule": {"format": "smiles", "value": "c1ccccc1"}},
        )

        self.assertEqual("success", payload["status"])
        primary = payload["primary_result"]
        self.assertEqual(1, primary["ring_count"])
        self.assertEqual(1, primary["aromatic_ring_count"])
        self.assertEqual([6], primary["ring_sizes"])
        self.assertEqual(6, primary["aromatic_atom_count"])

    def test_aliphatic_ring_example(self) -> None:
        payload, _ = run_script(
            "rings_aromaticity.py",
            {"molecule": {"format": "smiles", "value": "C1CCCCC1"}},
        )

        self.assertEqual("success", payload["status"])
        primary = payload["primary_result"]
        self.assertEqual(1, primary["ring_count"])
        self.assertEqual(0, primary["aromatic_ring_count"])
        self.assertEqual([6], primary["ring_sizes"])
        self.assertEqual(0, primary["aromatic_atom_count"])


class StereochemistryTests(unittest.TestCase):
    def test_chiral_molecule(self) -> None:
        payload, _ = run_script(
            "stereochemistry.py",
            {"molecule": {"format": "smiles", "value": "F[C@H](Cl)Br"}},
        )

        self.assertEqual("success", payload["status"])
        primary = payload["primary_result"]
        self.assertEqual(1, primary["chiral_center_count"])
        self.assertEqual(1, primary["specified_chiral_center_count"])
        self.assertEqual(0, primary["unspecified_chiral_center_count"])

    def test_achiral_molecule(self) -> None:
        payload, _ = run_script(
            "stereochemistry.py",
            {"molecule": {"format": "smiles", "value": "CCO"}},
        )

        self.assertEqual("success", payload["status"])
        primary = payload["primary_result"]
        self.assertEqual(0, primary["chiral_center_count"])
        self.assertEqual(0, primary["double_bond_stereo_count"])


class FunctionalGroupTests(unittest.TestCase):
    def test_curated_functional_group_matches(self) -> None:
        cases = {
            "CCO": {"alcohol"},
            "CCN": {"amine"},
            "CC=O": {"carbonyl"},
            "CC(=O)O": {"carbonyl", "carboxylic_acid"},
            "CC(=O)OC": {"carbonyl", "ester"},
            "CC(=O)NC": {"carbonyl", "amide"},
            "C=CC": {"alkene"},
            "CC#C": {"alkyne"},
            "Fc1ccccc1": {"aryl_halide"},
            "CC#N": {"nitrile"},
            "C[N+](=O)[O-]": {"nitro"},
            "CCS": {"thiol"},
        }

        for smiles, expected in cases.items():
            with self.subTest(smiles=smiles):
                payload, _ = run_script(
                    "functional_groups.py",
                    {"molecule": {"format": "smiles", "value": smiles}},
                )
                self.assertEqual("success", payload["status"])
                found = {item["name"] for item in payload["candidates"] if item["matched"]}
                self.assertTrue(expected.issubset(found), f"{smiles}: expected {expected}, got {found}")


class SubstructureTests(unittest.TestCase):
    def test_substructure_positive(self) -> None:
        payload, _ = run_script(
            "substructure.py",
            {
                "molecules": [{"format": "smiles", "value": "CCO"}],
                "query": {"smarts": "[OX2H]"},
            },
        )

        self.assertEqual("success", payload["status"])
        self.assertTrue(payload["primary_result"]["matches"][0]["matched"])

    def test_substructure_negative(self) -> None:
        payload, _ = run_script(
            "substructure.py",
            {
                "molecules": [{"format": "smiles", "value": "CCC"}],
                "query": {"smarts": "[OX2H]"},
            },
        )

        self.assertEqual("success", payload["status"])
        self.assertFalse(payload["primary_result"]["matches"][0]["matched"])


class SimilarityTests(unittest.TestCase):
    def test_similarity_ranking_is_deterministic(self) -> None:
        payload, _ = run_script(
            "similarity.py",
            {
                "query": {"format": "smiles", "value": "CCO"},
                "candidates": [
                    {"id": "ethanol", "format": "smiles", "value": "CCO"},
                    {"id": "ethanethiol", "format": "smiles", "value": "CCS"},
                    {"id": "propanol", "format": "smiles", "value": "CCCO"},
                    {"id": "dimethyl_ether", "format": "smiles", "value": "COC"},
                ],
            },
        )

        self.assertEqual("success", payload["status"])
        ranked = [item["id"] for item in payload["candidates"]]
        self.assertEqual(["ethanol", "propanol", "ethanethiol", "dimethyl_ether"], ranked)


class ReactionSmartsTests(unittest.TestCase):
    def test_reaction_smarts_success(self) -> None:
        payload, _ = run_script(
            "reaction_smarts.py",
            {
                "reaction_smarts": "[C:1]=[O:2]>>[C:1][O:2]",
                "reactants": [{"format": "smiles", "value": "CC=O"}],
            },
        )

        self.assertEqual("success", payload["status"])
        products = payload["primary_result"]["product_sets"]
        self.assertTrue(products)
        self.assertEqual("CCO", products[0]["products"][0]["canonical_smiles"])

    def test_reaction_smarts_failure(self) -> None:
        payload, _ = run_script(
            "reaction_smarts.py",
            {
                "reaction_smarts": "[C:1]=[O:2]>>[C:1][O:2]",
                "reactants": [{"format": "smiles", "value": "CCC"}],
            },
        )

        self.assertEqual("error", payload["status"])
        self.assertEqual([], payload["primary_result"]["product_sets"])
        self.assertTrue(payload["errors"])


class ConformerEmbedTests(unittest.TestCase):
    def test_conformer_embedding_success(self) -> None:
        payload, _ = run_script(
            "conformer_embed.py",
            {
                "molecule": {"format": "smiles", "value": "CCO"},
                "num_conformers": 1,
            },
        )

        self.assertEqual("success", payload["status"])
        primary = payload["primary_result"]
        self.assertGreaterEqual(primary["embedded_conformer_count"], 1)
        self.assertIn(primary["force_field"], {"UFF", "MMFF94", "MMFF94s"})

    def test_conformer_embedding_failure(self) -> None:
        payload, _ = run_script(
            "conformer_embed.py",
            {
                "molecule": {"format": "smiles", "value": "CCO"},
                "num_conformers": 0,
            },
        )

        self.assertEqual("error", payload["status"])
        self.assertEqual(0, payload["primary_result"]["embedded_conformer_count"])
        self.assertTrue(payload["errors"])


class NmrSymmetryHeuristicTests(unittest.TestCase):
    def test_benzene_equivalence_classes_include_uncertainty_note(self) -> None:
        payload, _ = run_script(
            "nmr_symmetry_heuristics.py",
            {"molecule": {"format": "smiles", "value": "c1ccccc1"}},
        )

        self.assertEqual("partial", payload["status"])
        primary = payload["primary_result"]
        self.assertEqual(1, primary["proton_equivalence_class_count"])
        self.assertEqual(1, primary["carbon_equivalence_class_count"])
        self.assertTrue(payload["warnings"])

    def test_toluene_has_multiple_proton_classes_and_documented_uncertainty(self) -> None:
        payload, _ = run_script(
            "nmr_symmetry_heuristics.py",
            {"molecule": {"format": "smiles", "value": "Cc1ccccc1"}},
        )

        self.assertEqual("partial", payload["status"])
        primary = payload["primary_result"]
        self.assertEqual(4, primary["proton_equivalence_class_count"])
        self.assertEqual(5, primary["carbon_equivalence_class_count"])
        self.assertTrue(any("heuristic" in warning["message"].lower() for warning in payload["warnings"]))


if __name__ == "__main__":
    unittest.main()
