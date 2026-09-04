from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from benchmarking.dashboard.app import create_app
from tests.test_benchmark_dashboard import write_demo_run


def test_dashboard_app_module_imports_without_fastapi_extra() -> None:
    module = importlib.import_module("benchmarking.dashboard.app")

    assert "uv run --extra web-ui" in module.FASTAPI_EXTRA_MESSAGE


def test_dashboard_static_frontend_contains_dashboard_shell() -> None:
    static_root = Path(__file__).resolve().parents[1] / "benchmarking" / "dashboard" / "static"
    index = (static_root / "index.html").read_text(encoding="utf-8")
    script = (static_root / "app.js").read_text(encoding="utf-8")
    styles = (static_root / "styles.css").read_text(encoding="utf-8")

    assert "Benchmark Dashboard" in index
    assert "run-list" in index
    assert "record-list" in index
    assert "/static/app.js?v=20260904-subset-display" in index
    assert "setInterval(refreshProgress" in script
    assert "function renderInlineMarkdown" in script
    assert "asset-image" in script
    assert "dataset-filter" in index
    assert "subset-filter" in index
    assert "function renderRunFacets" in script
    assert 'Subset: ${escapeHtml(subsets)}' in script
    assert "scopedRuns" in script
    assert "hide-run" in index
    assert 'class="refresh-icon"' in index
    assert 'button.classList.add("is-refreshing")' in script
    assert 'button.setAttribute("aria-busy", "true")' in script
    assert "animation: refresh-spin 700ms linear infinite" in styles
    assert "@media (prefers-reduced-motion: reduce)" in styles
    assert "api/annotations" in script
    assert "function renderRecordScoreBadges" in script
    assert "score-badge-strip" in script
    assert "average_normalized_score" not in script
    assert "avg ${score}" not in script
    assert "function renderRunScoreComparison" in script
    assert "single_llm_skills_on" in script
    assert "single_llm_skills_off" in script
    assert "run-score-line" in script
    assert "Δ" in script
    assert ".run-score-line .positive" in styles
    assert ".run-score-line .negative" in styles
    assert "Exec calls:" in script
    assert "${!group.skills_enabled ? `<span>Exec calls:" in script
    assert "group.skills_enabled" in script
    assert "Skill calls:" in script
    assert "Skill failures:" in script
    assert "function renderGroupDuration" in script
    assert "agent_duration_seconds" in script
    assert "answer ${Math.round(agentDuration)}s" in script


def test_dashboard_static_assets_disable_browser_cache(tmp_path: Path) -> None:
    testclient = pytest.importorskip("fastapi.testclient")
    app = create_app(run_roots=[tmp_path], annotation_db=tmp_path / "dashboard.sqlite")
    client = testclient.TestClient(app)

    index = client.get("/")
    script = client.get("/static/app.js")

    assert index.status_code == 200
    assert script.status_code == 200
    assert "no-store" in index.headers.get("cache-control", "")
    assert "no-store" in script.headers.get("cache-control", "")


def test_dashboard_api_supports_run_metadata_and_annotation_crud(tmp_path: Path) -> None:
    run_root = write_demo_run(tmp_path)
    testclient = pytest.importorskip("fastapi.testclient")

    app = create_app(run_roots=[tmp_path], annotation_db=tmp_path / "dashboard.sqlite")
    client = testclient.TestClient(app)

    patched = client.patch(f"/api/runs/{run_root.name}", json={"alias": "Smoke", "favorite": True, "hidden": True})
    assert patched.status_code == 200
    assert patched.json()["alias"] == "Smoke"
    assert client.get("/api/runs").json() == []
    visible = client.get("/api/runs?include_hidden=true").json()
    assert visible[0]["alias"] == "Smoke"
    assert visible[0]["hidden"] is True

    created = client.post(
        "/api/annotations",
        json={
            "run_id": run_root.name,
            "record_id": "r1",
            "group_id": "single_llm_skills_on",
            "note": "needs review",
            "status": "needs_review",
            "tags": ["manual"],
            "manual_verdict": "uncertain",
        },
    )
    assert created.status_code == 200
    annotation_id = created.json()["id"]
    updated = client.patch(f"/api/annotations/{annotation_id}", json={"note": "checked", "tags": ["done"]})
    assert updated.status_code == 200
    assert updated.json()["note"] == "checked"
    record = client.get(f"/api/runs/{run_root.name}/records/r1").json()
    assert record["annotations"][0]["tags"] == ["done"]
    deleted = client.delete(f"/api/annotations/{annotation_id}")
    assert deleted.status_code == 200
    assert client.get(f"/api/runs/{run_root.name}/records/r1").json()["annotations"] == []


def test_dashboard_asset_api_rejects_path_traversal(tmp_path: Path) -> None:
    run_root = write_demo_run(tmp_path)
    testclient = pytest.importorskip("fastapi.testclient")

    app = create_app(run_roots=[tmp_path], annotation_db=tmp_path / "dashboard.sqlite")
    client = testclient.TestClient(app)

    response = client.get(f"/api/runs/{run_root.name}/assets/../outside.txt")

    assert response.status_code == 404
