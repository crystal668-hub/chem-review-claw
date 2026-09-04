from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarking.core.datasets import load_records
from benchmarking.runtime.vgb_bridge import load_release_config
from scripts.sync_verifier_grounded_datasets import (
    REFERENCE_PLACEHOLDER,
    RESOURCE_DATASET_ROOT,
    build_dataset_records,
    sync_datasets,
)

FORBIDDEN_KEYS = {
    "example",
    "gold",
    "sample_answer",
    "sample_answers",
    "source_repo",
    "task",
    "verifier_specs",
}


def _description() -> dict[str, Any]:
    config = load_release_config()
    schemas = {
        "rdkit": {
            "format": "final_answer_line",
            "final_answer_prefix": "FINAL ANSWER:",
            "value_type": "smiles",
        },
        "xtb": {
            "format": "final_answer_block",
            "final_answer_prefix": "FINAL ANSWER:",
            "value_type": "xyz",
            "fence_language": "xyz",
        },
        "property_calculation_advanced": {
            "format": "final_answer_line",
            "final_answer_prefix": "FINAL ANSWER:",
            "value_type": "json",
        },
        "property_calculation_basic": {
            "format": "final_answer_line",
            "final_answer_prefix": "FINAL ANSWER:",
            "value_type": "json",
        },
    }
    return {
        "package_version": config.version,
        "tracks": {
            name: {
                "version": config.version,
                "prompts": [
                    {
                        "track": name,
                        "task_id": task_id,
                        "prompt": f"Public prompt for {task_id}. FINAL ANSWER:",
                        "answer_schema": schemas[name],
                    }
                    for task_id in track["task_ids"]
                ],
            }
            for name, track in config.tracks.items()
        },
    }


def _assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        assert FORBIDDEN_KEYS.isdisjoint(value)
        for item in value.values():
            _assert_no_forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_forbidden_keys(item)


def test_build_dataset_records_exposes_only_public_scoring_identity() -> None:
    config = load_release_config()
    description = _description()

    for track_name, track_config in config.tracks.items():
        rows = build_dataset_records(
            config=config,
            description=description,
            track_name=track_name,
        )
        assert len(rows) == track_config["task_count"]
        assert [row["id"] for row in rows] == track_config["task_ids"]
        assert all(row["answer"] == REFERENCE_PLACEHOLDER for row in rows)
        assert all(row["verifier_grounded"]["release"] == config.identity for row in rows)
        assert all(row["verifier_grounded"]["track"] == track_name for row in rows)
        assert all("example" not in row["verifier_grounded"]["answer_schema"] for row in rows)
        _assert_no_forbidden_keys(rows)


def test_sync_datasets_writes_tracked_and_runtime_copies(tmp_path: Path) -> None:
    config = load_release_config()
    resource_root = tmp_path / "resources"
    benchmarks_root = tmp_path / "formal-benchmarks"

    written = sync_datasets(
        config=config,
        description=_description(),
        resource_root=resource_root,
        benchmarks_root=benchmarks_root,
    )

    assert len(written) == 8
    for track in config.tracks.values():
        dataset = track["dataset"]
        resource_path = resource_root / f"{dataset}.jsonl"
        runtime_path = benchmarks_root / dataset / "data" / f"{dataset}.jsonl"
        assert resource_path.read_bytes() == runtime_path.read_bytes()


def test_checked_in_datasets_match_pinned_release_inventory() -> None:
    config = load_release_config()
    for track_name, track_config in config.tracks.items():
        dataset = track_config["dataset"]
        path = RESOURCE_DATASET_ROOT / f"{dataset}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [row["id"] for row in rows] == track_config["task_ids"]
        assert len(rows) == track_config["task_count"]
        assert all(row["verifier_grounded"]["track"] == track_name for row in rows)
        _assert_no_forbidden_keys(rows)

        records = load_records([path])
        assert len(records) == track_config["task_count"]
        assert all(record.grading.kind == "verifier_grounded" for record in records)


def test_rdkit_chain_distance_prompt_exposes_smarts_and_uff_protocol() -> None:
    path = RESOURCE_DATASET_ROOT / "verifier_grounded_rdkit.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    record = next(row for row in rows if row["id"] == "rdkit_chain_end_to_end_max_013")

    assert (
        "[C;X4;!R]-[C;X4;!R]-[C;X4;!R]-[C;X4;!R]-[C;X4;!R]-[C;X4;!R]"
        in record["prompt"]
    )
    assert "Universal Force Field (UFF)" in record["prompt"]
    assert "lowest-energy converged UFF conformer" in record["prompt"]
