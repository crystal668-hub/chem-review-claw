from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


def test_skill_files_exist(skill_root: Path) -> None:
    assert (skill_root / "SKILL.md").is_file()
    assert (skill_root / "routing-rules.md").is_file()
    assert (skill_root / "references" / "contracts.md").is_file()


@pytest.mark.parametrize(
    "script_name,result_name",
    [
        ("name_to_cid.py", "name_to_cid_result.json"),
        ("cid_to_properties.py", "cid_to_properties_result.json"),
        ("synonyms.py", "synonyms_result.json"),
        ("formula_search.py", "formula_search_result.json"),
        ("similarity_search.py", "similarity_search_result.json"),
        ("compound_summary.py", "compound_summary_result.json"),
    ],
)
def test_cli_writes_structured_error_on_invalid_request(
    skill_root: Path,
    tmp_path: Path,
    write_request,
    script_name: str,
    result_name: str,
) -> None:
    request_path = write_request({})
    output_dir = tmp_path / "out"
    script_path = skill_root / "scripts" / script_name

    completed = subprocess.run(
        ["python3", str(script_path), "--request-json", str(request_path), "--output-dir", str(output_dir), "--json"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "error"
    assert payload["request"] == {}
    assert isinstance(payload["errors"], list)
    assert (output_dir / result_name).is_file()
