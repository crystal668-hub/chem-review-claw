from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SKILL_ROOT / "scripts"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
EXPECTED_TOP_LEVEL_KEYS = {
    "status",
    "request",
    "primary_result",
    "candidates",
    "diagnostics",
    "warnings",
    "errors",
    "tool_trace",
    "source_trace",
    "provider_health",
}


def run_skill(script_name: str, fixture_name: str) -> tuple[dict[str, object], Path]:
    request_path = FIXTURES_DIR / fixture_name
    with tempfile.TemporaryDirectory(prefix="chem-calculator-test-") as temp_dir_name:
        output_dir = Path(temp_dir_name)
        command = [
            sys.executable,
            str(SCRIPTS_DIR / script_name),
            "--request-json",
            str(request_path),
            "--output-dir",
            str(output_dir),
            "--json",
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        stdout_payload = json.loads(completed.stdout)
        result_path = output_dir / "result.json"
        file_payload = json.loads(result_path.read_text(encoding="utf-8"))
        if stdout_payload != file_payload:
            raise AssertionError(f"stdout payload mismatch for {script_name}")
        return stdout_payload, result_path


class CliContractTests(unittest.TestCase):
    def test_every_script_supports_required_cli_contract(self) -> None:
        fixture_map = {
            "molar_mass.py": "molar_mass_simple.json",
            "stoichiometry.py": "stoichiometry_limiting_reagent.json",
            "concentration.py": "concentration_dilution.json",
            "ksp_solver.py": "ksp_precipitation.json",
            "acid_base_solver.py": "acid_base_strong_acid.json",
            "gas_law.py": "gas_law_ideal.json",
            "thermo_solver.py": "thermo_delta_g.json",
            "redox_balance.py": "redox_oxidation_states.json",
            "electrochemistry.py": "electrochemistry_nernst.json",
            "unit_convert.py": "unit_convert_temperature.json",
            "answer_check.py": "answer_check_correct.json",
        }

        for script_name, fixture_name in fixture_map.items():
            with self.subTest(script=script_name):
                payload, result_path = run_skill(script_name, fixture_name)
                self.assertEqual(result_path.name, "result.json")
                self.assertTrue(EXPECTED_TOP_LEVEL_KEYS.issubset(payload.keys()))
                self.assertIn(payload["status"], {"success", "partial", "error"})
                self.assertIsInstance(payload["request"], dict)
                self.assertIsInstance(payload["candidates"], list)
                self.assertIsInstance(payload["diagnostics"], list)
                self.assertIsInstance(payload["warnings"], list)
                self.assertIsInstance(payload["errors"], list)
                self.assertIsInstance(payload["tool_trace"], list)
                self.assertIsInstance(payload["source_trace"], list)
                self.assertIsInstance(payload["provider_health"], dict)


class MolarMassTests(unittest.TestCase):
    def test_simple_formula(self) -> None:
        payload, _ = run_skill("molar_mass.py", "molar_mass_simple.json")
        self.assertEqual(payload["status"], "success")
        result = payload["primary_result"]
        self.assertEqual(result["formula"], "H2O")
        self.assertAlmostEqual(result["molar_mass_g_per_mol"], 18.015, places=3)

    def test_parenthesized_formula(self) -> None:
        payload, _ = run_skill("molar_mass.py", "molar_mass_parenthesized.json")
        self.assertEqual(payload["status"], "success")
        self.assertAlmostEqual(payload["primary_result"]["molar_mass_g_per_mol"], 399.878, places=3)

    def test_hydrated_formula(self) -> None:
        payload, _ = run_skill("molar_mass.py", "molar_mass_hydrate.json")
        self.assertEqual(payload["status"], "success")
        self.assertAlmostEqual(payload["primary_result"]["molar_mass_g_per_mol"], 249.685, places=3)


class StoichiometryTests(unittest.TestCase):
    def test_limiting_reagent(self) -> None:
        payload, _ = run_skill("stoichiometry.py", "stoichiometry_limiting_reagent.json")
        self.assertEqual(payload["status"], "success")
        result = payload["primary_result"]
        self.assertEqual(result["limiting_reagent"]["species"], "O2")
        self.assertAlmostEqual(result["product_amount"]["value"], 4.0, places=6)
        self.assertEqual(result["product_amount"]["unit"], "mol")

    def test_combustion_analysis(self) -> None:
        payload, _ = run_skill("stoichiometry.py", "stoichiometry_combustion_analysis.json")
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["primary_result"]["empirical_formula"], "CH2O")

    def test_percent_yield(self) -> None:
        payload, _ = run_skill("stoichiometry.py", "stoichiometry_percent_yield.json")
        self.assertEqual(payload["status"], "success")
        self.assertAlmostEqual(payload["primary_result"]["percent_yield"], 80.0, places=6)


class ConcentrationTests(unittest.TestCase):
    def test_dilution(self) -> None:
        payload, _ = run_skill("concentration.py", "concentration_dilution.json")
        self.assertEqual(payload["status"], "success")
        self.assertAlmostEqual(payload["primary_result"]["target_concentration_molar"], 0.5, places=6)

    def test_mixing(self) -> None:
        payload, _ = run_skill("concentration.py", "concentration_mixing.json")
        self.assertEqual(payload["status"], "success")
        self.assertAlmostEqual(payload["primary_result"]["final_concentration_molar"], 0.75, places=6)


class KspTests(unittest.TestCase):
    def test_precipitation(self) -> None:
        payload, _ = run_skill("ksp_solver.py", "ksp_precipitation.json")
        self.assertEqual(payload["status"], "success")
        self.assertTrue(payload["primary_result"]["will_precipitate"])

    def test_residual_concentration(self) -> None:
        payload, _ = run_skill("ksp_solver.py", "ksp_residual_concentration.json")
        self.assertEqual(payload["status"], "success")
        self.assertAlmostEqual(payload["primary_result"]["residual_ag_molar"], 1.8e-8, places=12)


class AcidBaseTests(unittest.TestCase):
    def test_strong_acid(self) -> None:
        payload, _ = run_skill("acid_base_solver.py", "acid_base_strong_acid.json")
        self.assertEqual(payload["status"], "success")
        self.assertAlmostEqual(payload["primary_result"]["ph"], 1.0, places=6)

    def test_weak_base(self) -> None:
        payload, _ = run_skill("acid_base_solver.py", "acid_base_weak_base.json")
        self.assertEqual(payload["status"], "success")
        self.assertAlmostEqual(payload["primary_result"]["ph"], 11.13, places=2)

    def test_buffer(self) -> None:
        payload, _ = run_skill("acid_base_solver.py", "acid_base_buffer.json")
        self.assertEqual(payload["status"], "success")
        self.assertAlmostEqual(payload["primary_result"]["ph"], 4.74, places=2)


class GasLawTests(unittest.TestCase):
    def test_ideal_gas(self) -> None:
        payload, _ = run_skill("gas_law.py", "gas_law_ideal.json")
        self.assertEqual(payload["status"], "success")
        self.assertAlmostEqual(payload["primary_result"]["moles"], 1.0, places=4)

    def test_partial_pressure(self) -> None:
        payload, _ = run_skill("gas_law.py", "gas_law_partial_pressure.json")
        self.assertEqual(payload["status"], "success")
        self.assertAlmostEqual(payload["primary_result"]["partial_pressure_atm"], 0.7, places=6)


class ThermoTests(unittest.TestCase):
    def test_delta_g(self) -> None:
        payload, _ = run_skill("thermo_solver.py", "thermo_delta_g.json")
        self.assertEqual(payload["status"], "success")
        self.assertAlmostEqual(payload["primary_result"]["delta_g_kj_per_mol"], -22.409, places=3)

    def test_equilibrium_relation(self) -> None:
        payload, _ = run_skill("thermo_solver.py", "thermo_equilibrium_relation.json")
        self.assertEqual(payload["status"], "success")
        self.assertAlmostEqual(payload["primary_result"]["equilibrium_constant"], 88.03, places=2)

    def test_unit_handling(self) -> None:
        payload, _ = run_skill("thermo_solver.py", "thermo_unit_handling.json")
        self.assertEqual(payload["status"], "success")
        self.assertAlmostEqual(payload["primary_result"]["delta_g_kj_per_mol"], -10.0, places=6)


class RedoxTests(unittest.TestCase):
    def test_oxidation_states(self) -> None:
        payload, _ = run_skill("redox_balance.py", "redox_oxidation_states.json")
        self.assertEqual(payload["status"], "success")
        oxidation_states = payload["primary_result"]["oxidation_states"]
        self.assertEqual(oxidation_states["Mn"], 7)
        self.assertEqual(oxidation_states["O"], -2)

    def test_electron_count(self) -> None:
        payload, _ = run_skill("redox_balance.py", "redox_electron_count.json")
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["primary_result"]["electrons_transferred"], 5)


class ElectrochemistryTests(unittest.TestCase):
    def test_nernst(self) -> None:
        payload, _ = run_skill("electrochemistry.py", "electrochemistry_nernst.json")
        self.assertEqual(payload["status"], "success")
        self.assertAlmostEqual(payload["primary_result"]["cell_potential_v"], 1.041, places=3)

    def test_faraday(self) -> None:
        payload, _ = run_skill("electrochemistry.py", "electrochemistry_faraday.json")
        self.assertEqual(payload["status"], "success")
        self.assertAlmostEqual(payload["primary_result"]["deposited_mass_g"], 0.592, places=3)


class AnswerCheckTests(unittest.TestCase):
    def test_correct_value(self) -> None:
        payload, _ = run_skill("answer_check.py", "answer_check_correct.json")
        self.assertEqual(payload["status"], "success")
        self.assertTrue(payload["primary_result"]["is_correct"])

    def test_wrong_unit(self) -> None:
        payload, _ = run_skill("answer_check.py", "answer_check_wrong_unit.json")
        self.assertEqual(payload["status"], "partial")
        self.assertFalse(payload["primary_result"]["is_correct"])
        self.assertEqual(payload["primary_result"]["failure_reason"], "incompatible_unit")

    def test_rounding_mismatch(self) -> None:
        payload, _ = run_skill("answer_check.py", "answer_check_rounding_mismatch.json")
        self.assertEqual(payload["status"], "partial")
        self.assertFalse(payload["primary_result"]["is_correct"])
        self.assertEqual(payload["primary_result"]["failure_reason"], "rounding_mismatch")

    def test_tolerance_mismatch(self) -> None:
        payload, _ = run_skill("answer_check.py", "answer_check_tolerance_mismatch.json")
        self.assertEqual(payload["status"], "partial")
        self.assertFalse(payload["primary_result"]["is_correct"])
        self.assertEqual(payload["primary_result"]["failure_reason"], "tolerance_mismatch")


if __name__ == "__main__":
    unittest.main()
