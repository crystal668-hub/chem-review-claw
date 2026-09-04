from __future__ import annotations

from unittest.mock import patch

import pytest

from benchmarking.runtime import vgb_bridge as bridge


def test_release_config_pins_version_hash_and_complete_inventory() -> None:
    config = bridge.load_release_config()

    assert config.version == "0.9.1"
    assert config.source_tag == "v0.9.1"
    assert config.source_commit == "a651ff5124e81a516419295e708b1b15f32fd3b9"
    assert config.wheel_sha256 == "d061e3c002076ee93f4a2f8b195df91b95f8fc1fb2eceded97b71791dc8611e9"
    assert config.wheel_size == 185193
    assert {name: track["task_count"] for name, track in config.tracks.items()} == {
        "property_calculation_advanced": 20,
        "property_calculation_basic": 51,
        "rdkit": 14,
        "xtb": 20,
    }
    assert all(track["task_count"] == len(track["task_ids"]) for track in config.tracks.values())


def test_runtime_environment_does_not_inherit_agent_python_paths(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("PYTHONPATH", "/agent/source")
    monkeypatch.setenv("VIRTUAL_ENV", "/agent/venv")

    env = bridge._runtime_env()

    assert env["PATH"] == "/usr/bin"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in env
    assert "VIRTUAL_ENV" not in env


def test_evaluate_answer_rejects_unpinned_release_before_subprocess() -> None:
    with (
        patch.object(bridge, "_invoke_api") as invoke,
        pytest.raises(bridge.VerifierGroundedRuntimeError, match="does not match"),
    ):
        bridge.evaluate_answer(
            track="rdkit",
            task_id="rdkit_qed_max_001",
            answer_text="FINAL ANSWER: CCO",
            release_identity={"package": "wrong", "version": "0", "wheel_sha256": "0"},
        )
    invoke.assert_not_called()


def test_evaluate_answer_calls_public_api_runtime_with_track_and_task() -> None:
    config = bridge.load_release_config()
    expected = {"task_id": "rdkit_qed_max_001", "status": "scored", "scores": {"score": 0.5}}
    with patch.object(bridge, "_invoke_api", return_value=expected) as invoke:
        result = bridge.evaluate_answer(
            track="rdkit",
            task_id="rdkit_qed_max_001",
            answer_text="FINAL ANSWER: CCO",
            release_identity=config.identity,
        )

    assert result == expected
    payload = invoke.call_args.args[1]
    assert payload == {
        "action": "evaluate_one",
        "track": "rdkit",
        "task_id": "rdkit_qed_max_001",
        "answer_text": "FINAL ANSWER: CCO",
    }
    assert "source_repo" not in payload
    assert "verifier_specs" not in payload


def test_evaluate_answer_uses_invocation_release_after_default_changes() -> None:
    invocation_config = bridge.load_release_config()
    changed_default = bridge.ReleaseConfig(
        **{
            **invocation_config.__dict__,
            "version": "future",
        }
    )
    expected = {"task_id": "rdkit_qed_max_001", "status": "scored", "scores": {"score": 0.5}}
    with (
        patch.object(bridge, "load_release_config", return_value=changed_default),
        patch.object(bridge, "_invoke_api", return_value=expected) as invoke,
    ):
        result = bridge.evaluate_answer(
            track="rdkit",
            task_id="rdkit_qed_max_001",
            answer_text="FINAL ANSWER: CCO",
            release_identity=invocation_config.identity,
            release_config=invocation_config,
        )

    assert result == expected
    assert invoke.call_args.args[0] is invocation_config


def test_load_public_reference_answers_calls_public_api_runtime() -> None:
    config = bridge.load_release_config()
    task_ids = config.tracks["property_calculation_advanced"]["task_ids"]
    expected = [{"task_id": task_id, "answer": 0.0} for task_id in task_ids]
    with patch.object(
        bridge, "_invoke_api", return_value={"reference_answers": expected}
    ) as invoke:
        result = bridge.load_public_reference_answers("property_calculation_advanced")

    assert result == expected
    assert invoke.call_args.args[1] == {
        "action": "reference_answers",
        "track": "property_calculation_advanced",
        "task_ids": task_ids,
    }


def test_load_public_reference_answers_rejects_incomplete_pinned_inventory() -> None:
    with patch.object(
        bridge,
        "_invoke_api",
        return_value={
            "reference_answers": [
                {
                    "task_id": "property_calculation_advanced_001_free_energy",
                    "answer": 0.258031679,
                    "unit": "kJ/mol",
                }
            ]
        },
    ), pytest.raises(bridge.VerifierGroundedRuntimeError, match="inventory"):
        bridge.load_public_reference_answers("property_calculation_advanced")
